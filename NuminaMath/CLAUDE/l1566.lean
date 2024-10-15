import Mathlib

namespace NUMINAMATH_CALUDE_team_a_finishes_faster_l1566_156655

/-- Proves that Team A finishes 3 hours faster than Team R given the specified conditions --/
theorem team_a_finishes_faster (course_distance : ℝ) (team_r_speed : ℝ) (speed_difference : ℝ) :
  course_distance = 300 →
  team_r_speed = 20 →
  speed_difference = 5 →
  let team_a_speed := team_r_speed + speed_difference
  let team_r_time := course_distance / team_r_speed
  let team_a_time := course_distance / team_a_speed
  team_r_time - team_a_time = 3 := by
  sorry

end NUMINAMATH_CALUDE_team_a_finishes_faster_l1566_156655


namespace NUMINAMATH_CALUDE_tim_took_25_rulers_l1566_156621

/-- The number of rulers Tim took from the drawer -/
def rulers_taken (initial : ℕ) (remaining : ℕ) : ℕ := initial - remaining

/-- Proof that Tim took 25 rulers from the drawer -/
theorem tim_took_25_rulers :
  let initial_rulers : ℕ := 46
  let remaining_rulers : ℕ := 21
  rulers_taken initial_rulers remaining_rulers = 25 := by
  sorry

end NUMINAMATH_CALUDE_tim_took_25_rulers_l1566_156621


namespace NUMINAMATH_CALUDE_max_area_at_120_l1566_156651

/-- Represents a rectangular cow pasture -/
structure Pasture where
  fence_length : ℝ
  barn_length : ℝ

/-- Calculates the area of the pasture given the length of the side perpendicular to the barn -/
def pasture_area (p : Pasture) (x : ℝ) : ℝ :=
  x * (p.fence_length - 2 * x)

/-- Theorem stating that the maximum area occurs when the side parallel to the barn is 120 feet -/
theorem max_area_at_120 (p : Pasture) (h1 : p.fence_length = 240) (h2 : p.barn_length = 500) :
  ∃ (max_x : ℝ), (∀ (x : ℝ), pasture_area p x ≤ pasture_area p max_x) ∧ p.fence_length - 2 * max_x = 120 := by
  sorry


end NUMINAMATH_CALUDE_max_area_at_120_l1566_156651


namespace NUMINAMATH_CALUDE_triangle_formation_l1566_156633

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function that checks if a set of three real numbers can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 4 5 6 ∧
  ¬ can_form_triangle 1 2 3 ∧
  ¬ can_form_triangle 1 1.5 3 ∧
  ¬ can_form_triangle 3 4 8 :=
sorry

end NUMINAMATH_CALUDE_triangle_formation_l1566_156633


namespace NUMINAMATH_CALUDE_marked_elements_not_unique_l1566_156612

/-- Represents the table with 4 rows and 10 columns --/
def Table := Fin 4 → Fin 10 → Fin 10

/-- The table where each row is shifted by one position --/
def shiftedTable : Table :=
  λ i j => (j + i) % 10

/-- A marking of 10 elements in the table --/
def Marking := Fin 10 → Fin 4 × Fin 10

/-- Predicate to check if a marking is valid (one per row and column) --/
def isValidMarking (m : Marking) : Prop :=
  (∀ i : Fin 4, ∃! j : Fin 10, (i, j) ∈ Set.range m) ∧
  (∀ j : Fin 10, ∃! i : Fin 4, (i, j) ∈ Set.range m)

theorem marked_elements_not_unique (t : Table) (m : Marking) 
  (h : isValidMarking m) : 
  ∃ i j : Fin 10, i ≠ j ∧ t (m i).1 (m i).2 = t (m j).1 (m j).2 :=
sorry

end NUMINAMATH_CALUDE_marked_elements_not_unique_l1566_156612


namespace NUMINAMATH_CALUDE_fraction_of_powers_equals_500_l1566_156669

theorem fraction_of_powers_equals_500 : (0.5 : ℝ)^4 / (0.05 : ℝ)^3 = 500 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_powers_equals_500_l1566_156669


namespace NUMINAMATH_CALUDE_heart_ratio_two_four_four_two_l1566_156602

def heart (n m : ℕ) : ℕ := n^2 * m^3

theorem heart_ratio_two_four_four_two :
  (heart 2 4) / (heart 4 2) = 2 := by sorry

end NUMINAMATH_CALUDE_heart_ratio_two_four_four_two_l1566_156602


namespace NUMINAMATH_CALUDE_nick_sold_fewer_bottles_l1566_156650

/-- Proves that Nick sold 6 fewer bottles of soda than Remy in the morning -/
theorem nick_sold_fewer_bottles (remy_morning : ℕ) (price : ℚ) (evening_sales : ℚ) (evening_increase : ℚ) :
  remy_morning = 55 →
  price = 1/2 →
  evening_sales = 55 →
  evening_increase = 3 →
  ∃ (nick_morning : ℕ), remy_morning - nick_morning = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_nick_sold_fewer_bottles_l1566_156650


namespace NUMINAMATH_CALUDE_subset_condition_intersection_condition_l1566_156600

open Set Real

-- Define set A
def A : Set ℝ := {x : ℝ | |x + 2| < 3}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x : ℝ | (x - m) * (x - 2) < 0}

-- Theorem for part 1
theorem subset_condition (m : ℝ) : A ⊆ B m → m ≤ -5 := by sorry

-- Theorem for part 2
theorem intersection_condition (m n : ℝ) : A ∩ B m = Ioo (-1) n → m = -1 ∧ n = 1 := by sorry

end NUMINAMATH_CALUDE_subset_condition_intersection_condition_l1566_156600


namespace NUMINAMATH_CALUDE_percent_relation_l1566_156642

theorem percent_relation (a b c : ℝ) (x : ℝ) 
  (h1 : c = 0.20 * a) 
  (h2 : b = 2.00 * a) 
  (h3 : c = (x / 100) * b) : 
  x = 10 := by
sorry

end NUMINAMATH_CALUDE_percent_relation_l1566_156642


namespace NUMINAMATH_CALUDE_saras_golf_balls_l1566_156630

-- Define the number of dozens Sara has
def saras_dozens : ℕ := 9

-- Define the number of items in a dozen
def items_per_dozen : ℕ := 12

-- Theorem stating that Sara's total number of golf balls is 108
theorem saras_golf_balls : saras_dozens * items_per_dozen = 108 := by
  sorry

end NUMINAMATH_CALUDE_saras_golf_balls_l1566_156630


namespace NUMINAMATH_CALUDE_composition_equality_l1566_156661

theorem composition_equality (f g : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = x / 3 + 2) →
  (∀ x, g x = 5 - 2 * x) →
  f (g a) = 4 →
  a = -1/2 := by
sorry

end NUMINAMATH_CALUDE_composition_equality_l1566_156661


namespace NUMINAMATH_CALUDE_adjacent_fractions_property_l1566_156643

-- Define the type for our rational numbers
def IrreducibleRational := {q : ℚ // q > 0 ∧ Irreducible q ∧ q.num * q.den < 1988}

-- Define the property of being adjacent in the sequence
def Adjacent (q1 q2 : IrreducibleRational) : Prop :=
  q1.val < q2.val ∧ ∀ q : IrreducibleRational, q.val ≤ q1.val ∨ q2.val ≤ q.val

-- State the theorem
theorem adjacent_fractions_property (q1 q2 : IrreducibleRational) 
  (h : Adjacent q1 q2) : 
  q1.val.den * q2.val.num - q1.val.num * q2.val.den = 1 :=
sorry

end NUMINAMATH_CALUDE_adjacent_fractions_property_l1566_156643


namespace NUMINAMATH_CALUDE_number_of_baskets_l1566_156648

def apples_per_basket : ℕ := 17
def total_apples : ℕ := 629

theorem number_of_baskets : 
  total_apples / apples_per_basket = 37 := by sorry

end NUMINAMATH_CALUDE_number_of_baskets_l1566_156648


namespace NUMINAMATH_CALUDE_units_digit_of_expression_l1566_156657

-- Define a function to get the unit's place digit
def unitsDigit (n : ℕ) : ℕ := n % 10

-- State the theorem
theorem units_digit_of_expression : unitsDigit ((3^34 * 7^21) + 5^17) = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_expression_l1566_156657


namespace NUMINAMATH_CALUDE_cistern_wet_surface_area_l1566_156694

/-- Calculate the total wet surface area of a rectangular cistern -/
def total_wet_surface_area (length width depth : ℝ) : ℝ :=
  let bottom_area := length * width
  let long_sides_area := 2 * length * depth
  let short_sides_area := 2 * width * depth
  bottom_area + long_sides_area + short_sides_area

/-- Theorem stating that the total wet surface area of the given cistern is 83 square meters -/
theorem cistern_wet_surface_area :
  total_wet_surface_area 7 4 1.25 = 83 := by
  sorry

end NUMINAMATH_CALUDE_cistern_wet_surface_area_l1566_156694


namespace NUMINAMATH_CALUDE_product_equals_one_l1566_156691

theorem product_equals_one :
  (∀ a b c : ℝ, a * b * c = Real.sqrt ((a + 2) * (b + 3)) / (c + 1)) →
  6 * 15 * 11 = 1 := by
sorry

end NUMINAMATH_CALUDE_product_equals_one_l1566_156691


namespace NUMINAMATH_CALUDE_smallest_area_of_2020th_square_l1566_156623

theorem smallest_area_of_2020th_square :
  ∀ (n : ℕ),
  (∃ (a : ℕ), n^2 = 2019 + a ∧ a ≠ 1) →
  (∀ (a : ℕ), n^2 = 2019 + a ∧ a ≠ 1 → a ≥ 112225) :=
by sorry

end NUMINAMATH_CALUDE_smallest_area_of_2020th_square_l1566_156623


namespace NUMINAMATH_CALUDE_theater_lost_revenue_l1566_156608

/-- Calculates the lost revenue for a movie theater given its capacity, ticket price, and actual tickets sold. -/
theorem theater_lost_revenue (capacity : ℕ) (ticket_price : ℚ) (tickets_sold : ℕ) :
  capacity = 50 →
  ticket_price = 8 →
  tickets_sold = 24 →
  (capacity : ℚ) * ticket_price - (tickets_sold : ℚ) * ticket_price = 208 := by
  sorry

end NUMINAMATH_CALUDE_theater_lost_revenue_l1566_156608


namespace NUMINAMATH_CALUDE_intersection_M_N_l1566_156687

def M : Set ℝ := {x | x^2 + x - 2 < 0}
def N : Set ℝ := {x | Real.log x / Real.log (1/2) > -1}

theorem intersection_M_N : M ∩ N = {x : ℝ | 0 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l1566_156687


namespace NUMINAMATH_CALUDE_inequality_solution_l1566_156680

theorem inequality_solution (x : ℝ) : 
  (2 * Real.sqrt ((4 * x - 9)^2) + Real.sqrt (3 * Real.sqrt x - 5 + 2 * |x - 2|) ≤ 18 - 8 * x) ↔ 
  (x = 0) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l1566_156680


namespace NUMINAMATH_CALUDE_spadesuit_calculation_l1566_156654

-- Define the spadesuit operation
def spadesuit (a b : ℝ) : ℝ := |a - b|

-- State the theorem
theorem spadesuit_calculation : spadesuit 3 (spadesuit 5 (spadesuit 7 10)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_spadesuit_calculation_l1566_156654


namespace NUMINAMATH_CALUDE_wilson_payment_is_17_10_l1566_156604

/-- Calculates the total payment for Wilson's fast-food order --/
def wilsonPayment (hamburgerPrice fryPrice colaPrice sundaePrice couponDiscount loyaltyDiscount : ℚ) : ℚ :=
  let subtotal := 2 * hamburgerPrice + 3 * colaPrice + fryPrice + sundaePrice
  let afterCoupon := subtotal - couponDiscount
  afterCoupon * (1 - loyaltyDiscount)

/-- Theorem stating that Wilson's payment is $17.10 --/
theorem wilson_payment_is_17_10 :
  wilsonPayment 5 3 2 4 4 (1/10) = 171/10 := by
  sorry

end NUMINAMATH_CALUDE_wilson_payment_is_17_10_l1566_156604


namespace NUMINAMATH_CALUDE_defective_probability_l1566_156628

/-- Represents a box of components -/
structure Box where
  total : ℕ
  defective : ℕ

/-- The probability of selecting a box -/
def boxProb : ℚ := 1 / 2

/-- The probability of selecting a defective component from a given box -/
def defectiveProb (box : Box) : ℚ := box.defective / box.total

/-- The two boxes of components -/
def box1 : Box := ⟨10, 2⟩
def box2 : Box := ⟨20, 3⟩

/-- The main theorem stating the probability of selecting a defective component -/
theorem defective_probability : 
  boxProb * defectiveProb box1 + boxProb * defectiveProb box2 = 7 / 40 := by
  sorry

end NUMINAMATH_CALUDE_defective_probability_l1566_156628


namespace NUMINAMATH_CALUDE_complement_of_hit_at_least_once_l1566_156622

/-- Represents the outcome of a single shot -/
inductive ShotOutcome
| Hit
| Miss

/-- Represents the outcomes of two shots -/
def TwoShots := (ShotOutcome × ShotOutcome)

/-- The event of hitting the target at least once in two shots -/
def HitAtLeastOnce (shots : TwoShots) : Prop :=
  shots.1 = ShotOutcome.Hit ∨ shots.2 = ShotOutcome.Hit

/-- The event of missing the target both times -/
def MissBothTimes (shots : TwoShots) : Prop :=
  shots.1 = ShotOutcome.Miss ∧ shots.2 = ShotOutcome.Miss

/-- Theorem stating that MissBothTimes is the complement of HitAtLeastOnce -/
theorem complement_of_hit_at_least_once :
  ∀ (shots : TwoShots), ¬(HitAtLeastOnce shots) ↔ MissBothTimes shots :=
sorry


end NUMINAMATH_CALUDE_complement_of_hit_at_least_once_l1566_156622


namespace NUMINAMATH_CALUDE_complex_equation_implies_ab_eight_l1566_156635

theorem complex_equation_implies_ab_eight (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + b * i) * (3 + i) = 10 + 10 * i →
  a * b = 8 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_ab_eight_l1566_156635


namespace NUMINAMATH_CALUDE_arccos_minus_one_equals_pi_l1566_156624

theorem arccos_minus_one_equals_pi : Real.arccos (-1) = π := by
  sorry

end NUMINAMATH_CALUDE_arccos_minus_one_equals_pi_l1566_156624


namespace NUMINAMATH_CALUDE_other_communities_count_l1566_156677

theorem other_communities_count (total_boys : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ) :
  total_boys = 850 →
  muslim_percent = 34/100 →
  hindu_percent = 28/100 →
  sikh_percent = 10/100 →
  (total_boys : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 238 := by
  sorry

end NUMINAMATH_CALUDE_other_communities_count_l1566_156677


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1566_156693

theorem complex_equation_solution (z : ℂ) : (3 + Complex.I) * z = 4 - 2 * Complex.I → z = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1566_156693


namespace NUMINAMATH_CALUDE_polynomial_factorization_l1566_156620

theorem polynomial_factorization (a x : ℝ) : a * x^2 - a * x - 2 * a = a * (x - 2) * (x + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l1566_156620


namespace NUMINAMATH_CALUDE_p_and_q_properties_l1566_156681

-- Define proposition p
def p : Prop := ∃ x : ℝ, Real.sqrt x = Real.sqrt (2 * x + 1)

-- Define proposition q
def q : Prop := ∀ x : ℝ, x > 0 → x^2 < x^3

-- Theorem stating the properties of p and q
theorem p_and_q_properties :
  (∃ x : ℝ, Real.sqrt x = Real.sqrt (2 * x + 1)) ∧  -- p is existential
  ¬p ∧                                             -- p is false
  (∀ x : ℝ, x > 0 → x^2 < x^3) ∧                   -- q is universal
  ¬q                                               -- q is false
  := by sorry

end NUMINAMATH_CALUDE_p_and_q_properties_l1566_156681


namespace NUMINAMATH_CALUDE_possible_set_A_l1566_156626

-- Define the set B
def B : Set ℝ := {x | x ≥ 0}

-- Define the theorem
theorem possible_set_A (A : Set ℝ) (h1 : A ∩ B = A) : 
  ∃ A', A' = {1, 2} ∧ A' ∩ B = A' :=
sorry

end NUMINAMATH_CALUDE_possible_set_A_l1566_156626


namespace NUMINAMATH_CALUDE_novels_on_ends_l1566_156601

theorem novels_on_ends (total_books : ℕ) (novels : ℕ) (other_books : ℕ) 
  (h1 : total_books = 5)
  (h2 : novels = 2)
  (h3 : other_books = 3)
  (h4 : total_books = novels + other_books) :
  (other_books.factorial * novels.factorial) = 12 :=
by sorry

end NUMINAMATH_CALUDE_novels_on_ends_l1566_156601


namespace NUMINAMATH_CALUDE_negative_product_sum_l1566_156683

theorem negative_product_sum (a b : ℚ) (h1 : a * b > 0) (h2 : a + b < 0) : a < 0 ∧ b < 0 := by
  sorry

end NUMINAMATH_CALUDE_negative_product_sum_l1566_156683


namespace NUMINAMATH_CALUDE_investment_earnings_l1566_156606

/-- Calculates the earnings from a stock investment --/
def calculate_earnings (investment : ℕ) (dividend_rate : ℕ) (market_price : ℕ) (face_value : ℕ) : ℕ :=
  let shares := investment / market_price
  let total_face_value := shares * face_value
  (dividend_rate * total_face_value) / 100

/-- Theorem stating that the given investment yields the expected earnings --/
theorem investment_earnings : 
  calculate_earnings 5760 1623 64 100 = 146070 := by
  sorry

end NUMINAMATH_CALUDE_investment_earnings_l1566_156606


namespace NUMINAMATH_CALUDE_coin_flip_probability_l1566_156675

theorem coin_flip_probability :
  let n : ℕ := 6  -- total number of coins
  let k : ℕ := 3  -- number of specific coins we're interested in
  let total_outcomes : ℕ := 2^n
  let favorable_outcomes : ℕ := 2^(n - k)
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 8 :=
by sorry

end NUMINAMATH_CALUDE_coin_flip_probability_l1566_156675


namespace NUMINAMATH_CALUDE_largest_power_of_two_dividing_3_512_minus_1_l1566_156644

theorem largest_power_of_two_dividing_3_512_minus_1 :
  (∃ (n : ℕ), 2^n ∣ (3^512 - 1) ∧ ∀ (m : ℕ), 2^m ∣ (3^512 - 1) → m ≤ n) ∧
  (∀ (n : ℕ), (2^n ∣ (3^512 - 1) ∧ ∀ (m : ℕ), 2^m ∣ (3^512 - 1) → m ≤ n) → n = 11) :=
sorry

end NUMINAMATH_CALUDE_largest_power_of_two_dividing_3_512_minus_1_l1566_156644


namespace NUMINAMATH_CALUDE_alex_sandwiches_l1566_156631

/-- The number of different sandwiches Alex can make -/
def num_sandwiches (num_meats : ℕ) (num_cheeses : ℕ) : ℕ :=
  (num_meats.choose 2) * num_cheeses

/-- Theorem stating the number of sandwiches Alex can make -/
theorem alex_sandwiches :
  num_sandwiches 8 7 = 196 :=
by sorry

end NUMINAMATH_CALUDE_alex_sandwiches_l1566_156631


namespace NUMINAMATH_CALUDE_parabola_translation_correct_l1566_156679

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Represents a 2D translation -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- The original parabola y = x^2 -/
def originalParabola : Parabola := { a := 1, b := 0, c := 0 }

/-- The translation of 1 unit right and 2 units up -/
def givenTranslation : Translation := { dx := 1, dy := 2 }

/-- Function to apply a translation to a parabola -/
def applyTranslation (p : Parabola) (t : Translation) : Parabola :=
  { a := p.a
    b := -2 * p.a * t.dx + p.b
    c := p.a * t.dx^2 - p.b * t.dx + p.c + t.dy }

theorem parabola_translation_correct :
  applyTranslation originalParabola givenTranslation = { a := 1, b := -2, c := 3 } := by
  sorry

end NUMINAMATH_CALUDE_parabola_translation_correct_l1566_156679


namespace NUMINAMATH_CALUDE_waiter_tables_l1566_156634

theorem waiter_tables (initial_customers : ℕ) (left_customers : ℕ) (people_per_table : ℕ) 
  (h1 : initial_customers = 22)
  (h2 : left_customers = 14)
  (h3 : people_per_table = 4) :
  (initial_customers - left_customers) / people_per_table = 2 := by
  sorry

end NUMINAMATH_CALUDE_waiter_tables_l1566_156634


namespace NUMINAMATH_CALUDE_box_bottles_count_l1566_156652

-- Define the number of items in a dozen
def dozen : ℕ := 12

-- Define the number of water bottles
def water_bottles : ℕ := 2 * dozen

-- Define the number of additional apple bottles
def additional_apple_bottles : ℕ := dozen / 2

-- Define the total number of apple bottles
def apple_bottles : ℕ := water_bottles + additional_apple_bottles

-- Define the total number of bottles
def total_bottles : ℕ := water_bottles + apple_bottles

-- Theorem statement
theorem box_bottles_count : total_bottles = 54 := by
  sorry

end NUMINAMATH_CALUDE_box_bottles_count_l1566_156652


namespace NUMINAMATH_CALUDE_subcommittee_count_l1566_156663

theorem subcommittee_count (total_members : ℕ) (subcommittees_per_member : ℕ) (members_per_subcommittee : ℕ) :
  total_members = 360 →
  subcommittees_per_member = 3 →
  members_per_subcommittee = 6 →
  (total_members * subcommittees_per_member) / members_per_subcommittee = 180 :=
by sorry

end NUMINAMATH_CALUDE_subcommittee_count_l1566_156663


namespace NUMINAMATH_CALUDE_logical_equivalence_l1566_156690

theorem logical_equivalence (P Q R : Prop) :
  (¬P ∧ ¬Q → R) ↔ (P ∨ Q ∨ R) := by sorry

end NUMINAMATH_CALUDE_logical_equivalence_l1566_156690


namespace NUMINAMATH_CALUDE_curve_intersection_minimum_a_l1566_156646

theorem curve_intersection_minimum_a (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ a * x^2 = Real.exp x) →
  a ≥ Real.exp 2 / 4 :=
by sorry

end NUMINAMATH_CALUDE_curve_intersection_minimum_a_l1566_156646


namespace NUMINAMATH_CALUDE_inequality_always_true_l1566_156618

theorem inequality_always_true (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : 
  (a + c > b + d) ∧ 
  ¬(∀ a b c d : ℝ, a > b → c > d → a - c > b - d) ∧ 
  ¬(∀ a b c d : ℝ, a > b → c > d → a * c > b * d) ∧ 
  ¬(∀ a b c d : ℝ, a > b → c > d → a / c > b / d) :=
by sorry

end NUMINAMATH_CALUDE_inequality_always_true_l1566_156618


namespace NUMINAMATH_CALUDE_characterize_valid_k_l1566_156610

/-- A coloring of the complete graph on n vertices using k colors -/
def GraphColoring (n : ℕ) (k : ℕ) := Fin n → Fin n → Fin k

/-- Property: for any k vertices, all edges between them have different colors -/
def HasUniqueColors (n : ℕ) (k : ℕ) (coloring : GraphColoring n k) : Prop :=
  ∀ (vertices : Finset (Fin n)), vertices.card = k →
    (∀ (i j : Fin n), i ∈ vertices → j ∈ vertices → i ≠ j →
      ∀ (x y : Fin n), x ∈ vertices → y ∈ vertices → x ≠ y → (x, y) ≠ (i, j) →
        coloring i j ≠ coloring x y)

/-- The set of valid k values for a 10-vertex graph -/
def ValidK : Set ℕ := {k | k ≥ 5 ∧ k ≤ 10}

/-- Main theorem: characterization of valid k for a 10-vertex graph -/
theorem characterize_valid_k :
  ∀ k, k ∈ ValidK ↔ ∃ (coloring : GraphColoring 10 k), HasUniqueColors 10 k coloring :=
sorry

end NUMINAMATH_CALUDE_characterize_valid_k_l1566_156610


namespace NUMINAMATH_CALUDE_distinct_reals_integer_combination_l1566_156665

theorem distinct_reals_integer_combination (x y : ℝ) (h : x ≠ y) :
  ∃ (m n : ℤ), m * x + n * y > 0 ∧ n * x + m * y < 0 := by
  sorry

end NUMINAMATH_CALUDE_distinct_reals_integer_combination_l1566_156665


namespace NUMINAMATH_CALUDE_race_distance_p_l1566_156676

/-- The distance P runs in a race where:
  1. P's speed is 20% faster than Q's speed
  2. Q starts 300 meters ahead of P
  3. P and Q finish the race at the same time
-/
theorem race_distance_p (vq : ℝ) : ∃ dp : ℝ,
  let vp := 1.2 * vq
  let dq := dp - 300
  dp / vp = dq / vq ∧ dp = 1800 := by
  sorry

end NUMINAMATH_CALUDE_race_distance_p_l1566_156676


namespace NUMINAMATH_CALUDE_min_max_values_min_value_three_variables_l1566_156671

-- Problem 1
theorem min_max_values (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (a + 1 / a^2)^2 + (b + 1 / b^2)^2 ≥ 25/4 ∧ (a + 1/a) * (b + 1/b) ≤ 25/4 := by
  sorry

-- Problem 2
theorem min_value_three_variables (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (habc : a + b + c = 1) :
  (a + 1/a)^3 + (b + 1/b)^3 + (c + 1/c)^3 ≥ 1000/9 := by
  sorry

end NUMINAMATH_CALUDE_min_max_values_min_value_three_variables_l1566_156671


namespace NUMINAMATH_CALUDE_m_range_for_monotonic_function_l1566_156632

-- Define a monotonically increasing function on ℝ
def MonotonicIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

-- State the theorem
theorem m_range_for_monotonic_function (f : ℝ → ℝ) (m : ℝ) 
  (h1 : MonotonicIncreasing f) (h2 : f (m^2) > f (-m)) : 
  m ∈ Set.Ioi 0 ∪ Set.Iic (-1) :=
sorry

end NUMINAMATH_CALUDE_m_range_for_monotonic_function_l1566_156632


namespace NUMINAMATH_CALUDE_tournament_points_l1566_156678

-- Define the type for teams
inductive Team : Type
  | A | B | C | D | E

-- Define the function for points
def points : Team → ℕ
  | Team.A => 7
  | Team.B => 6
  | Team.C => 4
  | Team.D => 5
  | Team.E => 2

-- Define the properties of the tournament
axiom different_points : ∀ t1 t2 : Team, t1 ≠ t2 → points t1 ≠ points t2
axiom a_most_points : ∀ t : Team, t ≠ Team.A → points Team.A > points t
axiom b_beat_a : points Team.B > points Team.A
axiom b_no_loss : ∀ t : Team, t ≠ Team.B → points Team.B ≥ points t
axiom c_no_loss : ∀ t : Team, t ≠ Team.C → points Team.C ≥ points t
axiom d_more_than_c : points Team.D > points Team.C

-- Theorem to prove
theorem tournament_points : 
  (points Team.A = 7 ∧ 
   points Team.B = 6 ∧ 
   points Team.C = 4 ∧ 
   points Team.D = 5 ∧ 
   points Team.E = 2) := by
  sorry

end NUMINAMATH_CALUDE_tournament_points_l1566_156678


namespace NUMINAMATH_CALUDE_equation_solutions_l1566_156627

-- Define the equation
def equation (x y : ℝ) : Prop :=
  (36 / Real.sqrt (abs x)) + (9 / Real.sqrt (abs y)) = 
  42 - 9 * (if x < 0 then Complex.I * Real.sqrt (abs x) else Real.sqrt x) - 
  (if y < 0 then Complex.I * Real.sqrt (abs y) else Real.sqrt y)

-- Define the set of solutions
def solutions : Set (ℝ × ℝ) :=
  {(4, 9), (-4, 873 + 504 * Real.sqrt 3), (-4, 873 - 504 * Real.sqrt 3), 
   ((62 + 14 * Real.sqrt 13) / 9, -9), ((62 - 14 * Real.sqrt 13) / 9, -9)}

-- Theorem statement
theorem equation_solutions :
  ∀ x y : ℝ, equation x y ↔ (x, y) ∈ solutions :=
sorry

end NUMINAMATH_CALUDE_equation_solutions_l1566_156627


namespace NUMINAMATH_CALUDE_simplify_expression_1_l1566_156689

theorem simplify_expression_1 : 
  Real.sqrt 8 + Real.sqrt (1/3) - 2 * Real.sqrt 2 = Real.sqrt 3 / 3 := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_l1566_156689


namespace NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l1566_156613

/-- The area of a stripe on a cylindrical silo -/
theorem stripe_area_on_cylindrical_silo 
  (diameter : ℝ) 
  (stripe_width : ℝ) 
  (revolutions : ℕ) 
  (h_diameter : diameter = 20) 
  (h_stripe_width : stripe_width = 4) 
  (h_revolutions : revolutions = 4) : 
  stripe_width * revolutions * (π * diameter) = 640 * π := by
sorry

end NUMINAMATH_CALUDE_stripe_area_on_cylindrical_silo_l1566_156613


namespace NUMINAMATH_CALUDE_max_value_of_product_sum_l1566_156638

theorem max_value_of_product_sum (w x y z : ℝ) 
  (nonneg_w : 0 ≤ w) (nonneg_x : 0 ≤ x) (nonneg_y : 0 ≤ y) (nonneg_z : 0 ≤ z)
  (sum_condition : w + x + y + z = 200) :
  w * x + w * y + y * z + z * x ≤ 10000 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_product_sum_l1566_156638


namespace NUMINAMATH_CALUDE_barbed_wire_cost_l1566_156607

theorem barbed_wire_cost (field_area : ℝ) (wire_cost_per_meter : ℝ) (gate_width : ℝ) (num_gates : ℕ) : 
  field_area = 3136 ∧ 
  wire_cost_per_meter = 1.4 ∧ 
  gate_width = 1 ∧ 
  num_gates = 2 → 
  (Real.sqrt field_area * 4 - (gate_width * num_gates)) * wire_cost_per_meter = 310.8 := by
  sorry

end NUMINAMATH_CALUDE_barbed_wire_cost_l1566_156607


namespace NUMINAMATH_CALUDE_volleyball_team_lineups_l1566_156682

def team_size : ℕ := 16
def quadruplet_size : ℕ := 4
def starter_size : ℕ := 6

def valid_lineups : ℕ := Nat.choose team_size starter_size - Nat.choose (team_size - quadruplet_size) (starter_size - quadruplet_size)

theorem volleyball_team_lineups : valid_lineups = 7942 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_team_lineups_l1566_156682


namespace NUMINAMATH_CALUDE_floor_sqrt_80_l1566_156611

theorem floor_sqrt_80 : ⌊Real.sqrt 80⌋ = 8 := by sorry

end NUMINAMATH_CALUDE_floor_sqrt_80_l1566_156611


namespace NUMINAMATH_CALUDE_badminton_purchase_costs_l1566_156692

/-- Represents the cost calculation for badminton equipment purchases --/
structure BadmintonPurchase where
  num_rackets : ℕ
  num_shuttlecocks : ℕ
  racket_price : ℕ
  shuttlecock_price : ℕ
  store_a_promotion : Bool
  store_b_discount : ℚ

/-- Calculates the cost at Store A --/
def cost_store_a (p : BadmintonPurchase) : ℕ :=
  p.num_rackets * p.racket_price + (p.num_shuttlecocks - p.num_rackets) * p.shuttlecock_price

/-- Calculates the cost at Store B --/
def cost_store_b (p : BadmintonPurchase) : ℚ :=
  ((p.num_rackets * p.racket_price + p.num_shuttlecocks * p.shuttlecock_price : ℚ) * (1 - p.store_b_discount))

/-- The main theorem to prove --/
theorem badminton_purchase_costs 
  (x : ℕ) 
  (h : x > 16) :
  let p : BadmintonPurchase := {
    num_rackets := 16,
    num_shuttlecocks := x,
    racket_price := 150,
    shuttlecock_price := 40,
    store_a_promotion := true,
    store_b_discount := 1/5
  }
  cost_store_a p = 1760 + 40 * x ∧ 
  cost_store_b p = 1920 + 32 * x := by
  sorry

#check badminton_purchase_costs

end NUMINAMATH_CALUDE_badminton_purchase_costs_l1566_156692


namespace NUMINAMATH_CALUDE_line_segment_intersection_l1566_156647

/-- Given a line ax + y + 2 = 0 and points P(-2, 1) and Q(3, 2), 
    if the line intersects with the line segment PQ, 
    then a ≤ -4/3 or a ≥ 3/2 -/
theorem line_segment_intersection (a : ℝ) : 
  (∃ (x y : ℝ), a * x + y + 2 = 0 ∧ 
    ((x = -2 ∧ y = 1) ∨ 
     (x = 3 ∧ y = 2) ∨ 
     (∃ (t : ℝ), 0 < t ∧ t < 1 ∧ 
       x = -2 + 5*t ∧ 
       y = 1 + t))) → 
  (a ≤ -4/3 ∨ a ≥ 3/2) :=
by sorry

end NUMINAMATH_CALUDE_line_segment_intersection_l1566_156647


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1566_156641

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution (w : ℂ) :
  w - 1 = (1 + w) * i → w = i := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1566_156641


namespace NUMINAMATH_CALUDE_f_neg_one_eq_zero_l1566_156603

def f (x : ℝ) : ℝ := x^2 - 1

theorem f_neg_one_eq_zero : f (-1) = 0 := by sorry

end NUMINAMATH_CALUDE_f_neg_one_eq_zero_l1566_156603


namespace NUMINAMATH_CALUDE_janes_number_l1566_156605

theorem janes_number (x : ℝ) : 5 * (2 * x + 15) = 175 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_janes_number_l1566_156605


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l1566_156666

theorem triangle_abc_properties (A : Real) (h : Real.sin A + Real.cos A = 1/5) :
  (Real.sin A * Real.cos A = -12/25) ∧
  (π/2 < A ∧ A < π) ∧
  (Real.tan A = -4/3) := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l1566_156666


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1566_156695

theorem complex_equation_solution (x : ℝ) : 
  (Complex.I * (x + Complex.I) : ℂ) = -1 + 2 * Complex.I → x = 2 := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1566_156695


namespace NUMINAMATH_CALUDE_f_neg_two_value_l1566_156615

/-- Given a function f(x) = -ax^5 - x^3 + bx - 7, if f(2) = -9, then f(-2) = -5 -/
theorem f_neg_two_value (a b : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ -a * x^5 - x^3 + b * x - 7
  f 2 = -9 → f (-2) = -5 := by
sorry

end NUMINAMATH_CALUDE_f_neg_two_value_l1566_156615


namespace NUMINAMATH_CALUDE_one_root_in_interval_l1566_156616

theorem one_root_in_interval : ∃! x : ℝ, 0 < x ∧ x < 2 ∧ 2 * x^3 - 6 * x^2 + 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_one_root_in_interval_l1566_156616


namespace NUMINAMATH_CALUDE_third_month_sale_l1566_156659

def average_sale : ℕ := 6500
def num_months : ℕ := 6
def sales : List ℕ := [6400, 7000, 7200, 6500, 5100]

theorem third_month_sale :
  (num_months * average_sale - sales.sum) = 6800 :=
sorry

end NUMINAMATH_CALUDE_third_month_sale_l1566_156659


namespace NUMINAMATH_CALUDE_room_tiling_theorem_l1566_156656

/-- Calculates the number of tiles needed for a room with given dimensions and tile specifications -/
def tiles_needed (room_length room_width border_width : ℕ) : ℕ :=
  let border_tiles := 2 * (room_length + room_width - 4 * border_width) + 4 * border_width * border_width
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let inner_area := inner_length * inner_width
  let large_tiles := (inner_area + 8) / 9  -- Ceiling division
  border_tiles + large_tiles

/-- The theorem stating that 80 tiles are needed for the given room specifications -/
theorem room_tiling_theorem : tiles_needed 18 14 2 = 80 := by
  sorry

end NUMINAMATH_CALUDE_room_tiling_theorem_l1566_156656


namespace NUMINAMATH_CALUDE_toy_store_revenue_l1566_156668

theorem toy_store_revenue (D : ℚ) (D_pos : D > 0) : 
  let nov := (2 / 5 : ℚ) * D
  let jan := (1 / 5 : ℚ) * nov
  let avg := (nov + jan) / 2
  D / avg = 25 / 6 := by
sorry

end NUMINAMATH_CALUDE_toy_store_revenue_l1566_156668


namespace NUMINAMATH_CALUDE_evaluate_expression_l1566_156629

theorem evaluate_expression : -(16 / 4 * 8 - 70 + 4 * 7) = 10 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l1566_156629


namespace NUMINAMATH_CALUDE_sequence_convergence_l1566_156696

def sequence_property (r : ℝ) (a : ℕ → ℤ) : Prop :=
  r ≥ 0 ∧ ∀ n, a n ≤ a (n + 2) ∧ (a (n + 2) : ℝ)^2 ≤ (a n : ℝ)^2 + r * (a (n + 1) : ℝ)

theorem sequence_convergence (r : ℝ) (a : ℕ → ℤ) (h : sequence_property r a) :
  (r ≤ 2 → ∃ N, ∀ n ≥ N, a (n + 2) = a n) ∧
  (r > 2 → ∃ a : ℕ → ℤ, sequence_property r a ∧ ∀ N, ∃ n ≥ N, a (n + 2) ≠ a n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_convergence_l1566_156696


namespace NUMINAMATH_CALUDE_largest_two_digit_number_with_one_l1566_156699

def digits : List Nat := [1, 5, 6, 9]

def is_valid_number (n : Nat) : Prop :=
  n ≥ 10 ∧ n < 100 ∧
  n % 10 = 1 ∧
  (n / 10) ∈ digits ∧
  1 ∈ digits

theorem largest_two_digit_number_with_one :
  ∀ n : Nat, is_valid_number n → n ≤ 91 :=
by sorry

end NUMINAMATH_CALUDE_largest_two_digit_number_with_one_l1566_156699


namespace NUMINAMATH_CALUDE_prove_total_workers_l1566_156674

def total_workers : ℕ := 9
def other_workers : ℕ := 7
def chosen_workers : ℕ := 2

theorem prove_total_workers :
  (total_workers = other_workers + 2) →
  (Nat.choose total_workers chosen_workers = 36) →
  (1 / (Nat.choose total_workers chosen_workers : ℚ) = 1 / 36) →
  total_workers = 9 := by
sorry

end NUMINAMATH_CALUDE_prove_total_workers_l1566_156674


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1566_156686

/-- Given that x and y are positive real numbers, 3x² and y vary inversely,
    y = 18 when x = 3, and y = 2400, prove that x = 9√6 / 85. -/
theorem inverse_variation_problem (x y : ℝ) (hx : x > 0) (hy : y > 0)
    (h_inverse : ∃ k, k > 0 ∧ ∀ x' y', x' > 0 → y' > 0 → 3 * x'^2 * y' = k)
    (h_initial : 3 * 3^2 * 18 = 3 * x^2 * 2400) :
    x = 9 * Real.sqrt 6 / 85 := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1566_156686


namespace NUMINAMATH_CALUDE_product_of_three_numbers_l1566_156697

theorem product_of_three_numbers (x y z m : ℚ) : 
  x + y + z = 240 ∧ 
  9 * x = m ∧ 
  y - 11 = m ∧ 
  z + 11 = m ∧ 
  x < y ∧ 
  x < z → 
  x * y * z = 7514700 / 9 := by
sorry

end NUMINAMATH_CALUDE_product_of_three_numbers_l1566_156697


namespace NUMINAMATH_CALUDE_remainder_sum_l1566_156658

theorem remainder_sum (a b : ℤ) 
  (ha : a % 60 = 58) 
  (hb : b % 90 = 84) : 
  (a + b) % 30 = 22 := by
sorry

end NUMINAMATH_CALUDE_remainder_sum_l1566_156658


namespace NUMINAMATH_CALUDE_division_problem_l1566_156698

theorem division_problem : (107.8 : ℝ) / 11 = 9.8 := by sorry

end NUMINAMATH_CALUDE_division_problem_l1566_156698


namespace NUMINAMATH_CALUDE_soda_price_l1566_156653

/-- The cost of a burger in cents -/
def burger_cost : ℚ := sorry

/-- The cost of a soda in cents -/
def soda_cost : ℚ := sorry

/-- Uri's purchase: 3 burgers and 1 soda for 360 cents -/
axiom uri_purchase : 3 * burger_cost + soda_cost = 360

/-- Gen's purchase: 1 burger and 3 sodas for 330 cents -/
axiom gen_purchase : burger_cost + 3 * soda_cost = 330

theorem soda_price : soda_cost = 78.75 := by sorry

end NUMINAMATH_CALUDE_soda_price_l1566_156653


namespace NUMINAMATH_CALUDE_banana_tree_problem_l1566_156619

/-- The number of bananas initially on the tree -/
def initial_bananas : ℕ := 1180

/-- The number of bananas left on the tree after Raj cut some -/
def bananas_left : ℕ := 500

/-- The number of bananas Raj has eaten -/
def bananas_eaten : ℕ := 170

/-- The number of bananas remaining in Raj's basket -/
def bananas_in_basket : ℕ := 3 * bananas_eaten

theorem banana_tree_problem :
  initial_bananas = bananas_left + bananas_eaten + bananas_in_basket :=
by sorry

end NUMINAMATH_CALUDE_banana_tree_problem_l1566_156619


namespace NUMINAMATH_CALUDE_max_k_inequality_k_max_is_tight_l1566_156614

theorem max_k_inequality (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  ∀ k : ℝ, k ≤ 174960 →
    (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2*d)^5) ≥ k * a * b * c * d^3 :=
by sorry

theorem k_max_is_tight :
  ∃ a b c d : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
    (a + b + c) * (3^4 * (a + b + c + d)^5 + 2^4 * (a + b + c + 2*d)^5) = 174960 * a * b * c * d^3 :=
by sorry

end NUMINAMATH_CALUDE_max_k_inequality_k_max_is_tight_l1566_156614


namespace NUMINAMATH_CALUDE_no_solution_l1566_156672

def connection (a b : ℕ+) : ℚ :=
  (Nat.lcm a.val b.val : ℚ) / (a.val * b.val)

theorem no_solution : ¬ ∃ y : ℕ+, y.val < 50 ∧ connection y 13 = 3/5 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_l1566_156672


namespace NUMINAMATH_CALUDE_min_attempts_correct_l1566_156645

/-- Represents the minimum number of attempts to make a lamp work given a set of batteries. -/
def min_attempts (total : ℕ) (good : ℕ) (bad : ℕ) : ℕ :=
  if total = 2 * good - 1 then good else good - 1

theorem min_attempts_correct (n : ℕ) (h : n > 2) :
  (min_attempts (2 * n + 1) (n + 1) n = n + 1) ∧
  (min_attempts (2 * n) n n = n) :=
by sorry

#check min_attempts_correct

end NUMINAMATH_CALUDE_min_attempts_correct_l1566_156645


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l1566_156660

theorem tan_45_degrees_equals_one : Real.tan (π / 4) = 1 := by sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l1566_156660


namespace NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1566_156639

/-- For a parabola with equation x^2 = (1/2)y, the distance from its focus to its directrix is 1/4 -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), x^2 = (1/2) * y → 
  ∃ (focus_x focus_y directrix_y : ℝ),
    (focus_x = 0 ∧ 
     focus_y = 1/8 ∧ 
     directrix_y = -1/8 ∧
     focus_y - directrix_y = 1/4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_focus_directrix_distance_l1566_156639


namespace NUMINAMATH_CALUDE_pattern_1010_is_BCDA_l1566_156640

/-- Represents the four vertices of a square -/
inductive Vertex
| A
| B
| C
| D

/-- Represents a square configuration -/
def Square := List Vertex

/-- The initial square configuration -/
def initial_square : Square := [Vertex.A, Vertex.B, Vertex.C, Vertex.D]

/-- Performs a 90-degree counterclockwise rotation on a square -/
def rotate (s : Square) : Square := 
  match s with
  | [a, b, c, d] => [b, c, d, a]
  | _ => s

/-- Reflects a square over its horizontal line of symmetry -/
def reflect (s : Square) : Square :=
  match s with
  | [a, b, c, d] => [d, c, b, a]
  | _ => s

/-- Applies the alternating pattern of rotation and reflection n times -/
def apply_pattern (s : Square) (n : Nat) : Square :=
  match n with
  | 0 => s
  | n + 1 => if n % 2 == 0 then rotate (apply_pattern s n) else reflect (apply_pattern s n)

theorem pattern_1010_is_BCDA : 
  apply_pattern initial_square 1010 = [Vertex.B, Vertex.C, Vertex.D, Vertex.A] := by
  sorry

end NUMINAMATH_CALUDE_pattern_1010_is_BCDA_l1566_156640


namespace NUMINAMATH_CALUDE_ants_in_park_l1566_156662

-- Define the dimensions of the park in meters
def park_width : ℝ := 100
def park_length : ℝ := 130

-- Define the ant density per square centimeter
def ants_per_sq_cm : ℝ := 1.2

-- Define the conversion factor from meters to centimeters
def cm_per_meter : ℝ := 100

-- Theorem statement
theorem ants_in_park :
  let park_area_sq_cm := park_width * park_length * cm_per_meter^2
  let total_ants := park_area_sq_cm * ants_per_sq_cm
  total_ants = 156000000 := by
  sorry

end NUMINAMATH_CALUDE_ants_in_park_l1566_156662


namespace NUMINAMATH_CALUDE_factorial_101_102_is_perfect_square_factorial_100_101_not_perfect_square_factorial_100_102_not_perfect_square_factorial_101_103_not_perfect_square_factorial_102_103_not_perfect_square_l1566_156625

/-- Definition of factorial -/
def factorial (n : ℕ) : ℕ := Nat.factorial n

/-- Definition of perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

/-- Theorem: 101! · 102! is a perfect square -/
theorem factorial_101_102_is_perfect_square :
  is_perfect_square (factorial 101 * factorial 102) := by sorry

/-- Theorem: 100! · 101! is not a perfect square -/
theorem factorial_100_101_not_perfect_square :
  ¬ is_perfect_square (factorial 100 * factorial 101) := by sorry

/-- Theorem: 100! · 102! is not a perfect square -/
theorem factorial_100_102_not_perfect_square :
  ¬ is_perfect_square (factorial 100 * factorial 102) := by sorry

/-- Theorem: 101! · 103! is not a perfect square -/
theorem factorial_101_103_not_perfect_square :
  ¬ is_perfect_square (factorial 101 * factorial 103) := by sorry

/-- Theorem: 102! · 103! is not a perfect square -/
theorem factorial_102_103_not_perfect_square :
  ¬ is_perfect_square (factorial 102 * factorial 103) := by sorry

end NUMINAMATH_CALUDE_factorial_101_102_is_perfect_square_factorial_100_101_not_perfect_square_factorial_100_102_not_perfect_square_factorial_101_103_not_perfect_square_factorial_102_103_not_perfect_square_l1566_156625


namespace NUMINAMATH_CALUDE_sequence_problem_l1566_156685

theorem sequence_problem (a b : ℕ → ℝ) : 
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- arithmetic sequence condition
  (∀ n, b (n + 1) / b n = b 2 / b 1) →  -- geometric sequence condition
  a 1 + a 2 = 10 →
  a 4 - a 3 = 2 →
  b 2 = a 3 →
  b 3 = a 7 →
  b 5 = 64 := by
sorry

end NUMINAMATH_CALUDE_sequence_problem_l1566_156685


namespace NUMINAMATH_CALUDE_largest_initial_number_l1566_156636

theorem largest_initial_number :
  ∃ (a b c d e : ℕ),
    189 + a + b + c + d + e = 200 ∧
    ¬(189 ∣ a) ∧ ¬(189 ∣ b) ∧ ¬(189 ∣ c) ∧ ¬(189 ∣ d) ∧ ¬(189 ∣ e) ∧
    ∀ (n : ℕ), n > 189 →
      ¬∃ (x y z w v : ℕ),
        n + x + y + z + w + v = 200 ∧
        ¬(n ∣ x) ∧ ¬(n ∣ y) ∧ ¬(n ∣ z) ∧ ¬(n ∣ w) ∧ ¬(n ∣ v) :=
by sorry

end NUMINAMATH_CALUDE_largest_initial_number_l1566_156636


namespace NUMINAMATH_CALUDE_pecan_weight_in_mixture_l1566_156684

/-- A mixture of pecans and cashews -/
structure NutMixture where
  pecan_price : ℝ
  cashew_price : ℝ
  cashew_weight : ℝ
  total_weight : ℝ

/-- The amount of pecans in the mixture -/
def pecan_weight (m : NutMixture) : ℝ :=
  m.total_weight - m.cashew_weight

/-- Theorem stating the amount of pecans in the specific mixture -/
theorem pecan_weight_in_mixture (m : NutMixture) 
  (h1 : m.pecan_price = 5.60)
  (h2 : m.cashew_price = 3.50)
  (h3 : m.cashew_weight = 2)
  (h4 : m.total_weight = 3.33333333333) :
  pecan_weight m = 1.33333333333 := by
  sorry

#check pecan_weight_in_mixture

end NUMINAMATH_CALUDE_pecan_weight_in_mixture_l1566_156684


namespace NUMINAMATH_CALUDE_inscribed_sphere_volume_l1566_156637

/-- The volume of a sphere inscribed in a cube with edge length 10 inches -/
theorem inscribed_sphere_volume :
  let cube_edge : ℝ := 10
  let sphere_radius : ℝ := cube_edge / 2
  let sphere_volume : ℝ := (4 / 3) * π * sphere_radius ^ 3
  sphere_volume = (500 / 3) * π := by
sorry


end NUMINAMATH_CALUDE_inscribed_sphere_volume_l1566_156637


namespace NUMINAMATH_CALUDE_tan_alpha_value_l1566_156688

theorem tan_alpha_value (α : Real) (h : Real.tan (α / 2) = 2) :
  Real.tan α = -4/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l1566_156688


namespace NUMINAMATH_CALUDE_john_works_five_days_l1566_156617

/-- Represents the number of widgets John can make per hour -/
def widgets_per_hour : ℕ := 20

/-- Represents the number of hours John works per day -/
def hours_per_day : ℕ := 8

/-- Represents the total number of widgets John makes per week -/
def widgets_per_week : ℕ := 800

/-- Calculates the number of days John works per week -/
def days_worked_per_week : ℕ :=
  widgets_per_week /(widgets_per_hour * hours_per_day)

/-- Theorem stating that John works 5 days per week -/
theorem john_works_five_days :
  days_worked_per_week = 5 := by
  sorry

end NUMINAMATH_CALUDE_john_works_five_days_l1566_156617


namespace NUMINAMATH_CALUDE_parametric_represents_curve_l1566_156609

-- Define the curve
def curve (x : ℝ) : ℝ := x^2

-- Define the parametric equations
def parametric_x (t : ℝ) : ℝ := t
def parametric_y (t : ℝ) : ℝ := t^2

-- Theorem statement
theorem parametric_represents_curve :
  ∀ (t : ℝ), curve (parametric_x t) = parametric_y t :=
sorry

end NUMINAMATH_CALUDE_parametric_represents_curve_l1566_156609


namespace NUMINAMATH_CALUDE_girl_pairs_in_circular_arrangement_l1566_156670

/-- 
Given a circular arrangement of boys and girls:
- n_boys: number of boys
- n_girls: number of girls
- boy_pairs: number of pairs of boys sitting next to each other
- girl_pairs: number of pairs of girls sitting next to each other
-/
def circular_arrangement (n_boys n_girls boy_pairs girl_pairs : ℕ) : Prop :=
  n_boys + n_girls > 0 ∧ boy_pairs ≤ n_boys ∧ girl_pairs ≤ n_girls

theorem girl_pairs_in_circular_arrangement 
  (n_boys n_girls boy_pairs girl_pairs : ℕ) 
  (h_arrangement : circular_arrangement n_boys n_girls boy_pairs girl_pairs)
  (h_boys : n_boys = 10)
  (h_girls : n_girls = 15)
  (h_boy_pairs : boy_pairs = 5) :
  girl_pairs = 10 := by
  sorry

end NUMINAMATH_CALUDE_girl_pairs_in_circular_arrangement_l1566_156670


namespace NUMINAMATH_CALUDE_cube_volume_from_face_diagonal_l1566_156664

theorem cube_volume_from_face_diagonal (d : ℝ) (h : d = 6 * Real.sqrt 2) :
  let s := d / Real.sqrt 2
  s ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_face_diagonal_l1566_156664


namespace NUMINAMATH_CALUDE_max_value_polynomial_l1566_156673

theorem max_value_polynomial (x y : ℝ) (h : x + y = 5) :
  x^4*y + x^3*y + x^2*y + x*y + x*y^2 + x*y^3 + x*y^4 ≤ 6084/17 ∧
  ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 5 ∧ 
    x₀^4*y₀ + x₀^3*y₀ + x₀^2*y₀ + x₀*y₀ + x₀*y₀^2 + x₀*y₀^3 + x₀*y₀^4 = 6084/17 :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l1566_156673


namespace NUMINAMATH_CALUDE_boiling_temperature_calculation_boiling_temperature_proof_l1566_156667

theorem boiling_temperature_calculation (initial_temp : ℝ) (temp_increase : ℝ) 
  (pasta_time : ℝ) (total_time : ℝ) : ℝ :=
  let mixing_time := pasta_time / 3
  let cooking_and_mixing_time := pasta_time + mixing_time
  let time_to_boil := total_time - cooking_and_mixing_time
  let temp_increase_total := time_to_boil * temp_increase
  initial_temp + temp_increase_total

theorem boiling_temperature_proof :
  boiling_temperature_calculation 41 3 12 73 = 212 := by
  sorry

end NUMINAMATH_CALUDE_boiling_temperature_calculation_boiling_temperature_proof_l1566_156667


namespace NUMINAMATH_CALUDE_equation_solution_l1566_156649

theorem equation_solution : 
  ∃ y : ℝ, (2 / y + (3 / y) / (6 / y) = 1.2) ∧ y = 20 / 7 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1566_156649
