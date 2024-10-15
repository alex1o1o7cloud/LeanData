import Mathlib

namespace NUMINAMATH_CALUDE_hidden_sea_portion_l1986_198675

/-- Represents the composition of the landscape visible from an airplane window -/
structure Landscape where
  cloud : ℚ  -- Fraction of landscape covered by cloud
  island : ℚ  -- Fraction of landscape occupied by island
  sea : ℚ    -- Fraction of landscape occupied by sea

/-- The conditions of the landscape as described in the problem -/
def airplane_view : Landscape where
  cloud := 1/2
  island := 1/3
  sea := 2/3

theorem hidden_sea_portion (L : Landscape) 
  (h1 : L.cloud = 1/2)
  (h2 : L.island = 1/3)
  (h3 : L.cloud + L.island + L.sea = 1) :
  L.cloud * L.sea = 5/12 := by
  sorry

#check hidden_sea_portion

end NUMINAMATH_CALUDE_hidden_sea_portion_l1986_198675


namespace NUMINAMATH_CALUDE_special_triangle_area_l1986_198624

/-- A triangle with specific properties -/
structure SpecialTriangle where
  /-- The height of the triangle -/
  height : ℝ
  /-- The smaller part of the base -/
  smaller_base : ℝ
  /-- The ratio of the divided angle -/
  angle_ratio : ℝ
  /-- The height is 2 -/
  height_is_two : height = 2
  /-- The smaller part of the base is 1 -/
  smaller_base_is_one : smaller_base = 1
  /-- The height divides the angle in the ratio 2:1 -/
  angle_ratio_is_two_to_one : angle_ratio = 2/1

/-- The area of the SpecialTriangle is 11/3 -/
theorem special_triangle_area (t : SpecialTriangle) : 
  (1/2) * t.height * (t.smaller_base + (8/3)) = 11/3 := by
  sorry


end NUMINAMATH_CALUDE_special_triangle_area_l1986_198624


namespace NUMINAMATH_CALUDE_perfect_square_trinomial_l1986_198627

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + m*x + 9 = (x + a)^2) → (m = 6 ∨ m = -6) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_trinomial_l1986_198627


namespace NUMINAMATH_CALUDE_car_journey_speed_l1986_198642

/-- Represents the average speed between two towns -/
structure AverageSpeed where
  value : ℝ
  unit : String

/-- Represents the distance between two towns -/
structure Distance where
  value : ℝ
  unit : String

/-- Theorem: Given the conditions of the car journey, prove that the average speed from Town C to Town D is 36 mph -/
theorem car_journey_speed (d_ab d_bc d_cd : Distance) (s_ab s_bc s_total : AverageSpeed) :
  d_ab.value = 120 ∧ d_ab.unit = "miles" →
  d_bc.value = 60 ∧ d_bc.unit = "miles" →
  d_cd.value = 90 ∧ d_cd.unit = "miles" →
  s_ab.value = 40 ∧ s_ab.unit = "mph" →
  s_bc.value = 30 ∧ s_bc.unit = "mph" →
  s_total.value = 36 ∧ s_total.unit = "mph" →
  ∃ (s_cd : AverageSpeed), s_cd.value = 36 ∧ s_cd.unit = "mph" := by
  sorry


end NUMINAMATH_CALUDE_car_journey_speed_l1986_198642


namespace NUMINAMATH_CALUDE_range_of_m_l1986_198603

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + m*x₁ + 1 = 0 ∧ x₂^2 + m*x₂ + 1 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the theorem
theorem range_of_m :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ Set.Ioo 1 2 ∪ Set.Ici 3 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1986_198603


namespace NUMINAMATH_CALUDE_variance_binomial_8_3_4_l1986_198618

/-- The variance of a binomial distribution B(n, p) with n trials and probability p of success. -/
def binomialVariance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

/-- Proof that the variance of X ~ B(8, 3/4) is 3/2 -/
theorem variance_binomial_8_3_4 :
  binomialVariance 8 (3/4) = 3/2 := by
  sorry

#check variance_binomial_8_3_4

end NUMINAMATH_CALUDE_variance_binomial_8_3_4_l1986_198618


namespace NUMINAMATH_CALUDE_correct_scientific_notation_l1986_198695

/-- Scientific notation representation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  coeff_range : 1 ≤ coefficient ∧ coefficient < 10

/-- Check if a ScientificNotation represents a given number -/
def represents (sn : ScientificNotation) (n : ℝ) : Prop :=
  sn.coefficient * (10 : ℝ) ^ sn.exponent = n

/-- The number we want to represent (5.81 million) -/
def target_number : ℝ := 5.81e6

/-- The proposed scientific notation representation -/
def proposed_notation : ScientificNotation :=
  { coefficient := 5.81
    exponent := 6
    coeff_range := by sorry }

theorem correct_scientific_notation :
  represents proposed_notation target_number :=
sorry

end NUMINAMATH_CALUDE_correct_scientific_notation_l1986_198695


namespace NUMINAMATH_CALUDE_pet_store_earnings_l1986_198657

theorem pet_store_earnings :
  let num_kittens : ℕ := 2
  let num_puppies : ℕ := 1
  let kitten_price : ℕ := 6
  let puppy_price : ℕ := 5
  (num_kittens * kitten_price + num_puppies * puppy_price : ℕ) = 17 := by
  sorry

end NUMINAMATH_CALUDE_pet_store_earnings_l1986_198657


namespace NUMINAMATH_CALUDE_monotonic_decreasing_range_l1986_198664

def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*m*x + 1

theorem monotonic_decreasing_range (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₂ < 1 → f m x₁ > f m x₂) →
  m ≥ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_monotonic_decreasing_range_l1986_198664


namespace NUMINAMATH_CALUDE_smallest_rectangle_containing_circle_l1986_198650

theorem smallest_rectangle_containing_circle (r : ℝ) (h : r = 6) :
  (2 * r) * (2 * r) = 144 := by sorry

end NUMINAMATH_CALUDE_smallest_rectangle_containing_circle_l1986_198650


namespace NUMINAMATH_CALUDE_intersection_when_a_is_3_range_of_a_when_intersection_empty_l1986_198681

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x | 2 - a ≤ x ∧ x ≤ 2 + a}
def B : Set ℝ := {x | x ≤ 1 ∨ x ≥ 4}

-- Theorem for part 1
theorem intersection_when_a_is_3 :
  A 3 ∩ B = {x | -1 ≤ x ∧ x ≤ 1} ∪ {x | 4 ≤ x ∧ x ≤ 5} :=
by sorry

-- Theorem for part 2
theorem range_of_a_when_intersection_empty :
  ∀ a : ℝ, a > 0 → (A a ∩ B = ∅) → (0 < a ∧ a < 1) :=
by sorry

end NUMINAMATH_CALUDE_intersection_when_a_is_3_range_of_a_when_intersection_empty_l1986_198681


namespace NUMINAMATH_CALUDE_multiplier_value_l1986_198659

def f (x : ℝ) : ℝ := 3 * x - 5

theorem multiplier_value (x : ℝ) (h : x = 3) :
  ∃ m : ℝ, m * f x - 10 = f (x - 2) ∧ m = 2 := by
  sorry

end NUMINAMATH_CALUDE_multiplier_value_l1986_198659


namespace NUMINAMATH_CALUDE_find_a_value_l1986_198661

-- Define the polynomial expansion
def polynomial_expansion (n : ℕ) (a b c : ℤ) (x : ℝ) : Prop :=
  (x + 2) ^ n = x ^ n + a * x ^ (n - 1) + (b * x + c)

-- State the theorem
theorem find_a_value (n : ℕ) (a b c : ℤ) :
  n ≥ 3 →
  polynomial_expansion n a b c x →
  b = 4 * c →
  a = 16 := by
  sorry


end NUMINAMATH_CALUDE_find_a_value_l1986_198661


namespace NUMINAMATH_CALUDE_sin_sixty_degrees_l1986_198639

theorem sin_sixty_degrees : Real.sin (π / 3) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_sixty_degrees_l1986_198639


namespace NUMINAMATH_CALUDE_least_possible_third_side_l1986_198679

theorem least_possible_third_side (a b : ℝ) (ha : a = 7) (hb : b = 24) :
  let c := Real.sqrt (b^2 - a^2)
  c = Real.sqrt 527 ∧ c ≤ a ∧ c ≤ b := by sorry

end NUMINAMATH_CALUDE_least_possible_third_side_l1986_198679


namespace NUMINAMATH_CALUDE_pizza_cost_is_80_l1986_198623

/-- The total cost of pizzas given the number of pizzas, pieces per pizza, and cost per piece. -/
def total_cost (num_pizzas : ℕ) (pieces_per_pizza : ℕ) (cost_per_piece : ℕ) : ℕ :=
  num_pizzas * pieces_per_pizza * cost_per_piece

/-- Theorem stating that the total cost of pizzas is $80 under the given conditions. -/
theorem pizza_cost_is_80 :
  total_cost 4 5 4 = 80 := by
  sorry

end NUMINAMATH_CALUDE_pizza_cost_is_80_l1986_198623


namespace NUMINAMATH_CALUDE_wage_difference_proof_l1986_198690

/-- Proves that the difference between hourly wages of two candidates is $5 
    given specific conditions about their pay and work hours. -/
theorem wage_difference_proof (total_pay hours_p hours_q wage_p wage_q : ℝ) 
  (h1 : total_pay = 300)
  (h2 : wage_p = 1.5 * wage_q)
  (h3 : hours_q = hours_p + 10)
  (h4 : wage_p * hours_p = total_pay)
  (h5 : wage_q * hours_q = total_pay) :
  wage_p - wage_q = 5 := by
sorry

end NUMINAMATH_CALUDE_wage_difference_proof_l1986_198690


namespace NUMINAMATH_CALUDE_apple_distribution_l1986_198643

theorem apple_distribution (total_apples : ℕ) (red_percentage : ℚ) (classmates : ℕ) (extra_red : ℕ) : 
  total_apples = 80 →
  red_percentage = 3/5 →
  classmates = 6 →
  extra_red = 3 →
  (↑(total_apples) * red_percentage - extra_red) / classmates = 7.5 →
  ∃ (apples_per_classmate : ℕ), apples_per_classmate = 7 ∧ 
    apples_per_classmate * classmates ≤ ↑(total_apples) * red_percentage - extra_red ∧
    (apples_per_classmate + 1) * classmates > ↑(total_apples) * red_percentage - extra_red :=
by sorry

end NUMINAMATH_CALUDE_apple_distribution_l1986_198643


namespace NUMINAMATH_CALUDE_no_daughters_count_l1986_198665

def berthas_family (num_daughters : ℕ) (total_descendants : ℕ) (daughters_with_children : ℕ) : Prop :=
  num_daughters = 8 ∧
  total_descendants = 40 ∧
  daughters_with_children * 4 = total_descendants - num_daughters

theorem no_daughters_count (num_daughters : ℕ) (total_descendants : ℕ) (daughters_with_children : ℕ) :
  berthas_family num_daughters total_descendants daughters_with_children →
  total_descendants - num_daughters = 32 :=
by sorry

end NUMINAMATH_CALUDE_no_daughters_count_l1986_198665


namespace NUMINAMATH_CALUDE_smaller_root_of_quadratic_l1986_198698

theorem smaller_root_of_quadratic (x : ℝ) : 
  (x - 2/3)^2 + (x - 2/3)*(x - 1/3) = 0 → 
  (x = 1/2 ∨ x = 2/3) ∧ 1/2 < 2/3 := by
  sorry

end NUMINAMATH_CALUDE_smaller_root_of_quadratic_l1986_198698


namespace NUMINAMATH_CALUDE_sine_function_problem_l1986_198606

theorem sine_function_problem (a b c : ℝ) :
  let f : ℝ → ℝ := λ x ↦ a * Real.sin x + b * x + c
  (f 0 = -2) → (f (Real.pi / 2) = 1) → (f (-Real.pi / 2) = -5) := by
  sorry

end NUMINAMATH_CALUDE_sine_function_problem_l1986_198606


namespace NUMINAMATH_CALUDE_pages_read_per_year_l1986_198630

/-- The number of pages read in a year given monthly reading habits and book lengths -/
theorem pages_read_per_year
  (novels_per_month : ℕ)
  (pages_per_novel : ℕ)
  (months_per_year : ℕ)
  (h1 : novels_per_month = 4)
  (h2 : pages_per_novel = 200)
  (h3 : months_per_year = 12) :
  novels_per_month * pages_per_novel * months_per_year = 9600 :=
by sorry

end NUMINAMATH_CALUDE_pages_read_per_year_l1986_198630


namespace NUMINAMATH_CALUDE_faulty_meter_theorem_l1986_198612

/-- A shopkeeper sells goods using a faulty meter -/
structure Shopkeeper where
  profit_percent : ℝ
  supposed_weight : ℝ
  actual_weight : ℝ

/-- Calculate the weight difference of the faulty meter -/
def faulty_meter_weight (s : Shopkeeper) : ℝ :=
  s.supposed_weight - s.actual_weight

/-- Theorem stating the weight of the faulty meter -/
theorem faulty_meter_theorem (s : Shopkeeper) 
  (h1 : s.profit_percent = 11.11111111111111 / 100)
  (h2 : s.supposed_weight = 1000)
  (h3 : s.actual_weight = (1 - s.profit_percent) * s.supposed_weight) :
  faulty_meter_weight s = 100 := by
  sorry

end NUMINAMATH_CALUDE_faulty_meter_theorem_l1986_198612


namespace NUMINAMATH_CALUDE_cone_slant_height_l1986_198686

/-- Given a cone with base radius 2 cm and an unfolded side forming a sector
    with central angle 120°, prove that its slant height is 6 cm. -/
theorem cone_slant_height (r : ℝ) (θ : ℝ) (x : ℝ) 
    (h_r : r = 2)
    (h_θ : θ = 120)
    (h_arc_length : θ / 360 * (2 * Real.pi * x) = 2 * Real.pi * r) :
  x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cone_slant_height_l1986_198686


namespace NUMINAMATH_CALUDE_probability_is_three_tenths_l1986_198614

/-- A bag containing 5 balls numbered from 1 to 5 -/
def Bag : Finset ℕ := {1, 2, 3, 4, 5}

/-- The set of all possible pairs of balls -/
def AllPairs : Finset (ℕ × ℕ) := (Bag.product Bag).filter (fun p => p.1 < p.2)

/-- The set of pairs whose sum is either 3 or 6 -/
def FavorablePairs : Finset (ℕ × ℕ) := AllPairs.filter (fun p => p.1 + p.2 = 3 ∨ p.1 + p.2 = 6)

/-- The probability of drawing a pair with sum 3 or 6 -/
def ProbabilityOfSum3Or6 : ℚ := (FavorablePairs.card : ℚ) / (AllPairs.card : ℚ)

theorem probability_is_three_tenths : ProbabilityOfSum3Or6 = 3 / 10 := by
  sorry

end NUMINAMATH_CALUDE_probability_is_three_tenths_l1986_198614


namespace NUMINAMATH_CALUDE_triangle_properties_l1986_198620

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def satisfies_conditions (t : Triangle) : Prop :=
  t.b * Real.cos t.A = (2 * t.c + t.a) * Real.cos (Real.pi - t.B) ∧
  t.b = 4 ∧
  (1 / 2) * t.a * t.c * Real.sin t.B = Real.sqrt 3

-- Theorem statement
theorem triangle_properties (t : Triangle) 
  (h : satisfies_conditions t) : 
  t.B = (2 / 3) * Real.pi ∧ t.a + t.c = 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1986_198620


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l1986_198673

/-- Given a circle D with equation x^2 + 14x + y^2 - 8y = -64,
    prove that the sum of its center coordinates and radius is -2 -/
theorem circle_center_radius_sum :
  ∀ (c d s : ℝ),
  (∀ (x y : ℝ), x^2 + 14*x + y^2 - 8*y = -64 ↔ (x - c)^2 + (y - d)^2 = s^2) →
  c + d + s = -2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l1986_198673


namespace NUMINAMATH_CALUDE_cloud_9_diving_bookings_l1986_198619

/-- Cloud 9 Diving Company bookings problem -/
theorem cloud_9_diving_bookings 
  (total_after_cancellations : ℕ) 
  (group_bookings : ℕ) 
  (cancellation_returns : ℕ) 
  (h1 : total_after_cancellations = 26400)
  (h2 : group_bookings = 16000)
  (h3 : cancellation_returns = 1600) :
  total_after_cancellations + cancellation_returns - group_bookings = 12000 :=
by sorry

end NUMINAMATH_CALUDE_cloud_9_diving_bookings_l1986_198619


namespace NUMINAMATH_CALUDE_original_selling_price_with_loss_l1986_198651

-- Define the selling price with 10% gain
def selling_price_with_gain : ℝ := 660

-- Define the gain percentage
def gain_percentage : ℝ := 0.1

-- Define the loss percentage
def loss_percentage : ℝ := 0.1

-- Theorem to prove
theorem original_selling_price_with_loss :
  let cost_price := selling_price_with_gain / (1 + gain_percentage)
  let selling_price_with_loss := cost_price * (1 - loss_percentage)
  selling_price_with_loss = 540 := by sorry

end NUMINAMATH_CALUDE_original_selling_price_with_loss_l1986_198651


namespace NUMINAMATH_CALUDE_multiply_and_simplify_l1986_198635

theorem multiply_and_simplify (x : ℝ) :
  (x^6 + 64*x^3 + 4096) * (x^3 - 64) = x^9 - 262144 := by sorry

end NUMINAMATH_CALUDE_multiply_and_simplify_l1986_198635


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1986_198660

theorem negation_of_existence (P : ℝ → Prop) :
  (¬ ∃ x > 0, P x) ↔ (∀ x > 0, ¬ P x) :=
by sorry

theorem negation_of_quadratic_equation :
  (¬ ∃ x > 0, x^2 + 3*x - 5 = 0) ↔ (∀ x > 0, x^2 + 3*x - 5 ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_quadratic_equation_l1986_198660


namespace NUMINAMATH_CALUDE_exam_average_score_l1986_198654

/-- Given an exam with a maximum score and the percentages scored by three students,
    calculate the average mark scored by all three students. -/
theorem exam_average_score (max_score : ℕ) (amar_percent bhavan_percent chetan_percent : ℕ) :
  max_score = 900 ∧ amar_percent = 64 ∧ bhavan_percent = 36 ∧ chetan_percent = 44 →
  (amar_percent * max_score / 100 + bhavan_percent * max_score / 100 + chetan_percent * max_score / 100) / 3 = 432 :=
by sorry

end NUMINAMATH_CALUDE_exam_average_score_l1986_198654


namespace NUMINAMATH_CALUDE_halfway_fraction_l1986_198637

theorem halfway_fraction : ∃ (n d : ℕ), d ≠ 0 ∧ (n : ℚ) / d = (1 : ℚ) / 3 / 2 + (3 : ℚ) / 4 / 2 ∧ n = 13 ∧ d = 24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l1986_198637


namespace NUMINAMATH_CALUDE_root_in_interval_implies_k_range_l1986_198645

theorem root_in_interval_implies_k_range :
  ∀ k : ℝ, 
  (∃ x : ℝ, x ∈ (Set.Ioo 2 3) ∧ x^2 + (1-k)*x - 2*(k+1) = 0) →
  k ∈ Set.Ioo 1 2 :=
by sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_k_range_l1986_198645


namespace NUMINAMATH_CALUDE_total_red_pencils_l1986_198662

/-- The number of pencil packs bought -/
def total_packs : ℕ := 15

/-- The number of red pencils in a normal pack -/
def red_per_normal_pack : ℕ := 1

/-- The number of packs with extra red pencils -/
def packs_with_extra : ℕ := 3

/-- The number of extra red pencils in special packs -/
def extra_red_per_special_pack : ℕ := 2

/-- Theorem stating the total number of red pencils bought -/
theorem total_red_pencils : 
  total_packs * red_per_normal_pack + packs_with_extra * extra_red_per_special_pack = 21 :=
by
  sorry


end NUMINAMATH_CALUDE_total_red_pencils_l1986_198662


namespace NUMINAMATH_CALUDE_total_frog_eyes_l1986_198605

/-- The number of frogs in the pond -/
def num_frogs : ℕ := 6

/-- The number of eyes each frog has -/
def eyes_per_frog : ℕ := 2

/-- Theorem: The total number of frog eyes in the pond is equal to the product of the number of frogs and the number of eyes per frog -/
theorem total_frog_eyes : num_frogs * eyes_per_frog = 12 := by
  sorry

end NUMINAMATH_CALUDE_total_frog_eyes_l1986_198605


namespace NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l1986_198663

/-- Given an arithmetic sequence {a_n} where a₁ + a₅ + a₉ = 8π, 
    prove that cos(a₃ + a₇) = -1/2 -/
theorem arithmetic_sequence_cosine (a : ℕ → ℝ) 
    (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1))
    (h_sum : a 1 + a 5 + a 9 = 8 * Real.pi) :
    Real.cos (a 3 + a 7) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_cosine_l1986_198663


namespace NUMINAMATH_CALUDE_special_polygon_area_l1986_198653

/-- A polygon with special properties -/
structure SpecialPolygon where
  sides : ℕ
  perimeter : ℝ
  is_decomposable_into_rectangles : Prop
  all_sides_congruent : Prop
  sides_perpendicular : Prop

/-- The area of a special polygon -/
def area (p : SpecialPolygon) : ℝ := sorry

/-- Theorem stating the area of the specific polygon described in the problem -/
theorem special_polygon_area :
  ∀ (p : SpecialPolygon),
    p.sides = 24 ∧
    p.perimeter = 48 ∧
    p.is_decomposable_into_rectangles ∧
    p.all_sides_congruent ∧
    p.sides_perpendicular →
    area p = 32 := by sorry

end NUMINAMATH_CALUDE_special_polygon_area_l1986_198653


namespace NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_18_l1986_198631

theorem least_five_digit_congruent_to_7_mod_18 :
  ∃ (n : ℕ), 
    (n ≥ 10000 ∧ n < 100000) ∧ 
    (n % 18 = 7) ∧
    (∀ m : ℕ, m ≥ 10000 ∧ m < 100000 ∧ m % 18 = 7 → m ≥ n) ∧
    n = 10015 :=
by sorry

end NUMINAMATH_CALUDE_least_five_digit_congruent_to_7_mod_18_l1986_198631


namespace NUMINAMATH_CALUDE_solve_equation_l1986_198678

theorem solve_equation (x : ℚ) : 5 * (x - 10) = 3 * (3 - 3 * x) + 9 → x = 34 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1986_198678


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_quadratic_equations_all_solutions_l1986_198667

theorem quadratic_equations_solutions :
  (∃ x : ℝ, x^2 - 2*x - 1 = 0) ∧
  (∃ x : ℝ, (x - 2)^2 = 2*x - 4) :=
by
  constructor
  · use 1 + Real.sqrt 2
    sorry
  · use 2
    sorry

theorem quadratic_equations_all_solutions :
  (∀ x : ℝ, x^2 - 2*x - 1 = 0 ↔ (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2)) ∧
  (∀ x : ℝ, (x - 2)^2 = 2*x - 4 ↔ (x = 2 ∨ x = 4)) :=
by
  constructor
  · intro x
    sorry
  · intro x
    sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_quadratic_equations_all_solutions_l1986_198667


namespace NUMINAMATH_CALUDE_gym_distance_proof_l1986_198611

/-- The distance between Wang Lei's home and the gym --/
def gym_distance : ℕ := 1500

/-- Wang Lei's walking speed in meters per minute --/
def wang_lei_speed : ℕ := 40

/-- The older sister's walking speed in meters per minute --/
def older_sister_speed : ℕ := wang_lei_speed + 20

/-- Time taken by the older sister to reach the gym in minutes --/
def time_to_gym : ℕ := 25

/-- Distance from the meeting point to the gym in meters --/
def meeting_point_distance : ℕ := 300

theorem gym_distance_proof :
  gym_distance = older_sister_speed * time_to_gym ∧
  gym_distance = wang_lei_speed * (time_to_gym + meeting_point_distance / wang_lei_speed) :=
by sorry

end NUMINAMATH_CALUDE_gym_distance_proof_l1986_198611


namespace NUMINAMATH_CALUDE_percentage_to_pass_l1986_198616

/-- Given a student's marks and passing conditions, prove the percentage needed to pass -/
theorem percentage_to_pass
  (marks_obtained : ℕ)
  (marks_to_pass : ℕ)
  (max_marks : ℕ)
  (h1 : marks_obtained = 130)
  (h2 : marks_to_pass = marks_obtained + 14)
  (h3 : max_marks = 400) :
  (marks_to_pass : ℚ) / max_marks * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_pass_l1986_198616


namespace NUMINAMATH_CALUDE_cookies_needed_to_fill_bags_l1986_198699

/-- Represents the number of cookies needed to fill a bag completely -/
def bagCapacity : ℕ := 16

/-- Represents the total number of cookies Edgar bought -/
def totalCookies : ℕ := 292

/-- Represents the number of chocolate chip cookies Edgar bought -/
def chocolateChipCookies : ℕ := 154

/-- Represents the number of oatmeal raisin cookies Edgar bought -/
def oatmealRaisinCookies : ℕ := 86

/-- Represents the number of sugar cookies Edgar bought -/
def sugarCookies : ℕ := 52

/-- Calculates the number of additional cookies needed to fill the last bag completely -/
def additionalCookiesNeeded (cookieCount : ℕ) : ℕ :=
  bagCapacity - (cookieCount % bagCapacity)

theorem cookies_needed_to_fill_bags :
  additionalCookiesNeeded chocolateChipCookies = 6 ∧
  additionalCookiesNeeded oatmealRaisinCookies = 10 ∧
  additionalCookiesNeeded sugarCookies = 12 :=
by
  sorry

#check cookies_needed_to_fill_bags

end NUMINAMATH_CALUDE_cookies_needed_to_fill_bags_l1986_198699


namespace NUMINAMATH_CALUDE_workday_end_time_l1986_198646

-- Define the start time of the workday
def start_time : Nat := 8

-- Define the lunch break start time
def lunch_start : Nat := 13

-- Define the duration of the workday in hours (excluding lunch)
def workday_duration : Nat := 8

-- Define the duration of the lunch break in hours
def lunch_duration : Nat := 1

-- Theorem to prove the end time of the workday
theorem workday_end_time :
  start_time + workday_duration + lunch_duration = 17 := by
  sorry

#check workday_end_time

end NUMINAMATH_CALUDE_workday_end_time_l1986_198646


namespace NUMINAMATH_CALUDE_cookie_baking_time_l1986_198666

/-- Represents the cookie-making process with given times -/
structure CookieProcess where
  total_time : ℕ
  white_icing_time : ℕ
  chocolate_icing_time : ℕ

/-- Calculates the remaining time for batter, baking, and cooling -/
def remaining_time (process : CookieProcess) : ℕ :=
  process.total_time - (process.white_icing_time + process.chocolate_icing_time)

/-- Theorem: The remaining time for batter, baking, and cooling is 60 minutes -/
theorem cookie_baking_time (process : CookieProcess)
    (h1 : process.total_time = 120)
    (h2 : process.white_icing_time = 30)
    (h3 : process.chocolate_icing_time = 30) :
    remaining_time process = 60 := by
  sorry

#eval remaining_time { total_time := 120, white_icing_time := 30, chocolate_icing_time := 30 }

end NUMINAMATH_CALUDE_cookie_baking_time_l1986_198666


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l1986_198617

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y = 1 / 3) :
  (1 / x + 1 / y) ≥ 9 + 6 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l1986_198617


namespace NUMINAMATH_CALUDE_certain_number_value_l1986_198672

theorem certain_number_value : ∀ (t k certain_number : ℝ),
  t = 5 / 9 * (k - certain_number) →
  t = 75 →
  k = 167 →
  certain_number = 32 := by
sorry

end NUMINAMATH_CALUDE_certain_number_value_l1986_198672


namespace NUMINAMATH_CALUDE_system_solution_l1986_198683

theorem system_solution (a b c x y z : ℝ) 
  (h1 : x + y + z = 0)
  (h2 : c * x + a * y + b * z = 0)
  (h3 : (x + b)^2 + (y + c)^2 + (z + a)^2 = a^2 + b^2 + c^2)
  (h4 : a ≠ b)
  (h5 : b ≠ c)
  (h6 : a ≠ c) :
  ((x = 0 ∧ y = 0 ∧ z = 0) ∨ 
   (x = a - b ∧ y = b - c ∧ z = c - a)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1986_198683


namespace NUMINAMATH_CALUDE_tree_distance_l1986_198668

theorem tree_distance (yard_length : ℝ) (num_trees : ℕ) 
  (h1 : yard_length = 320)
  (h2 : num_trees = 47)
  (h3 : num_trees ≥ 2) :
  let distance := yard_length / (num_trees - 1)
  distance = 320 / 46 := by
sorry

end NUMINAMATH_CALUDE_tree_distance_l1986_198668


namespace NUMINAMATH_CALUDE_polynomial_remainder_remainder_theorem_cube_area_is_six_probability_half_tan_135_is_negative_one_l1986_198629

-- Problem 1
theorem polynomial_remainder : Int → Int := 
  fun x ↦ 2 * x^3 - 3 * x^2 + x - 1

theorem remainder_theorem (p : Int → Int) (a : Int) :
  p (-1) = -7 → ∃ q : Int → Int, ∀ x, p x = (x + 1) * q x + -7 := by sorry

-- Problem 2
def cube_surface_area (side_length : ℝ) : ℝ := 6 * side_length^2

theorem cube_area_is_six : cube_surface_area 1 = 6 := by sorry

-- Problem 3
def probability_white (red white : ℕ) : ℚ := white / (red + white)

theorem probability_half : probability_white 10 10 = 1/2 := by sorry

-- Problem 4
theorem tan_135_is_negative_one : Real.tan (135 * π / 180) = -1 := by sorry

end NUMINAMATH_CALUDE_polynomial_remainder_remainder_theorem_cube_area_is_six_probability_half_tan_135_is_negative_one_l1986_198629


namespace NUMINAMATH_CALUDE_tan_four_implies_expression_equals_21_68_l1986_198604

theorem tan_four_implies_expression_equals_21_68 (θ : Real) (h : Real.tan θ = 4) :
  (Real.sin θ + Real.cos θ) / (17 * Real.sin θ) + Real.sin θ^2 / 4 = 21/68 := by
  sorry

end NUMINAMATH_CALUDE_tan_four_implies_expression_equals_21_68_l1986_198604


namespace NUMINAMATH_CALUDE_condition_relationship_l1986_198687

theorem condition_relationship (x : ℝ) :
  (∀ x, x > 2 → x^2 > 4) ∧ 
  (∃ x, x^2 > 4 ∧ ¬(x > 2)) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l1986_198687


namespace NUMINAMATH_CALUDE_sum_of_squares_nonzero_iff_one_nonzero_l1986_198613

theorem sum_of_squares_nonzero_iff_one_nonzero (a b : ℝ) :
  a^2 + b^2 ≠ 0 ↔ a ≠ 0 ∨ b ≠ 0 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_nonzero_iff_one_nonzero_l1986_198613


namespace NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l1986_198633

/-- Given a parallelogram with opposite vertices at (2, -4) and (14, 10),
    the coordinates of the point where the diagonals intersect are (8, 3). -/
theorem parallelogram_diagonal_intersection :
  let v1 : ℝ × ℝ := (2, -4)
  let v2 : ℝ × ℝ := (14, 10)
  let midpoint : ℝ × ℝ := ((v1.1 + v2.1) / 2, (v1.2 + v2.2) / 2)
  midpoint = (8, 3) := by sorry

end NUMINAMATH_CALUDE_parallelogram_diagonal_intersection_l1986_198633


namespace NUMINAMATH_CALUDE_linear_equation_condition_l1986_198656

theorem linear_equation_condition (a : ℝ) : 
  (|a - 1| = 1 ∧ a - 2 ≠ 0) ↔ a = 0 :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_condition_l1986_198656


namespace NUMINAMATH_CALUDE_inequality_proof_l1986_198684

theorem inequality_proof (x y : ℝ) (h1 : 0 ≤ x) (h2 : 0 ≤ y) (h3 : x^2 + y^2 ≤ 1) :
  |x^2 + 2*x*y - y^2| ≤ Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1986_198684


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l1986_198677

/-- The distance between the foci of a hyperbola given by the equation 3x^2 - 18x - 2y^2 - 4y = 48 -/
theorem hyperbola_foci_distance : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, 3 * x^2 - 18 * x - 2 * y^2 - 4 * y = 48) →
    (a^2 = 53 / 3) →
    (b^2 = 53 / 6) →
    (c^2 = a^2 + b^2) →
    (2 * c = 2 * Real.sqrt (53 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l1986_198677


namespace NUMINAMATH_CALUDE_women_in_room_l1986_198652

theorem women_in_room (initial_men : ℕ) (initial_women : ℕ) : 
  initial_men * 5 = initial_women * 4 →
  (initial_men + 2) = 14 →
  2 * (initial_women - 3) = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_women_in_room_l1986_198652


namespace NUMINAMATH_CALUDE_non_shaded_perimeter_l1986_198638

/-- Given a rectangle with dimensions and a shaded area, calculate the perimeter of the non-shaded region --/
theorem non_shaded_perimeter (large_width large_height ext_width ext_height shaded_area : ℝ) :
  large_width = 12 →
  large_height = 8 →
  ext_width = 5 →
  ext_height = 2 →
  shaded_area = 104 →
  let total_area := large_width * large_height + ext_width * ext_height
  let non_shaded_area := total_area - shaded_area
  let non_shaded_width := ext_height
  let non_shaded_height := non_shaded_area / non_shaded_width
  2 * (non_shaded_width + non_shaded_height) = 6 :=
by sorry

end NUMINAMATH_CALUDE_non_shaded_perimeter_l1986_198638


namespace NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l1986_198682

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

def has_no_prime_factor_less_than (n k : ℕ) : Prop :=
  ∀ m : ℕ, m < k → is_prime m → ¬(n % m = 0)

theorem smallest_non_prime_non_square_no_small_factors :
  ∀ n : ℕ, n > 0 →
    (¬is_prime n ∧ ¬is_perfect_square n ∧ has_no_prime_factor_less_than n 70) →
    n ≥ 5183 :=
sorry

end NUMINAMATH_CALUDE_smallest_non_prime_non_square_no_small_factors_l1986_198682


namespace NUMINAMATH_CALUDE_committee_selection_with_fixed_member_l1986_198609

/-- The number of ways to select a committee with a fixed member -/
def select_committee (total : ℕ) (committee_size : ℕ) (fixed_members : ℕ) : ℕ :=
  Nat.choose (total - fixed_members) (committee_size - fixed_members)

/-- Theorem: Selecting a 4-person committee from 12 people with one fixed member -/
theorem committee_selection_with_fixed_member :
  select_committee 12 4 1 = 165 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_with_fixed_member_l1986_198609


namespace NUMINAMATH_CALUDE_binomial_expansion_fourth_fifth_terms_sum_zero_l1986_198625

/-- Given a binomial expansion (a-b)^n where n ≥ 2, ab ≠ 0, and a = mb with m = k + 2 and k a positive integer,
    prove that n = 2m + 3 makes the sum of the fourth and fifth terms zero. -/
theorem binomial_expansion_fourth_fifth_terms_sum_zero 
  (n : ℕ) (a b : ℝ) (m k : ℕ) :
  n ≥ 2 →
  a ≠ 0 →
  b ≠ 0 →
  k > 0 →
  m = k + 2 →
  a = m * b →
  (n = 2 * m + 3 ↔ 
    (Nat.choose n 3) * (a - b)^(n - 3) * b^3 + 
    (Nat.choose n 4) * (a - b)^(n - 4) * b^4 = 0) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_fourth_fifth_terms_sum_zero_l1986_198625


namespace NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l1986_198644

/-- Given that 0.overline{02} = 2/99, prove that 2.overline{06} = 68/33 -/
theorem recurring_decimal_to_fraction :
  (∃ (x : ℚ), x = 2 / 99 ∧ (∀ n : ℕ, (x * 10^(3*n) - ⌊x * 10^(3*n)⌋ = 0.02))) →
  (∃ (y : ℚ), y = 68 / 33 ∧ (∀ n : ℕ, (y - 2 - ⌊y - 2⌋ = 0.06))) :=
by sorry

end NUMINAMATH_CALUDE_recurring_decimal_to_fraction_l1986_198644


namespace NUMINAMATH_CALUDE_pencils_per_child_l1986_198636

/-- Given a group of children with pencils, prove that each child has 2 pencils. -/
theorem pencils_per_child (num_children : ℕ) (total_pencils : ℕ) 
  (h1 : num_children = 8) 
  (h2 : total_pencils = 16) : 
  total_pencils / num_children = 2 := by
  sorry

end NUMINAMATH_CALUDE_pencils_per_child_l1986_198636


namespace NUMINAMATH_CALUDE_student_group_assignment_l1986_198669

/-- The number of ways to assign students to groups -/
def assignment_count (num_students : ℕ) (num_groups : ℕ) : ℕ :=
  num_groups ^ num_students

/-- Theorem: The number of ways to assign 4 students to 3 groups is 3^4 -/
theorem student_group_assignment :
  assignment_count 4 3 = 3^4 := by
  sorry

end NUMINAMATH_CALUDE_student_group_assignment_l1986_198669


namespace NUMINAMATH_CALUDE_units_digit_of_seven_power_l1986_198621

theorem units_digit_of_seven_power : ∃ n : ℕ, 7^(6^5) ≡ 1 [ZMOD 10] :=
by sorry

end NUMINAMATH_CALUDE_units_digit_of_seven_power_l1986_198621


namespace NUMINAMATH_CALUDE_man_money_problem_l1986_198602

theorem man_money_problem (x : ℝ) : 
  (((2 * (2 * (2 * (2 * x - 50) - 60) - 70) - 80) = 0) ↔ (x = 53.75)) := by
  sorry

end NUMINAMATH_CALUDE_man_money_problem_l1986_198602


namespace NUMINAMATH_CALUDE_probability_at_least_one_woman_l1986_198622

def total_people : ℕ := 15
def num_men : ℕ := 9
def num_women : ℕ := 6
def committee_size : ℕ := 4

theorem probability_at_least_one_woman :
  let prob_all_men := (num_men / total_people) *
                      ((num_men - 1) / (total_people - 1)) *
                      ((num_men - 2) / (total_people - 2)) *
                      ((num_men - 3) / (total_people - 3))
  (1 : ℚ) - prob_all_men = 59 / 65 := by
  sorry

end NUMINAMATH_CALUDE_probability_at_least_one_woman_l1986_198622


namespace NUMINAMATH_CALUDE_pen_pencil_ratio_l1986_198670

/-- Proves that given 30 pencils and 5 more pencils than pens, the ratio of pens to pencils is 5:6 -/
theorem pen_pencil_ratio :
  ∀ (num_pens num_pencils : ℕ),
    num_pencils = 30 →
    num_pencils = num_pens + 5 →
    (num_pens : ℚ) / (num_pencils : ℚ) = 5 / 6 := by
  sorry

end NUMINAMATH_CALUDE_pen_pencil_ratio_l1986_198670


namespace NUMINAMATH_CALUDE_circle_roll_path_length_l1986_198628

/-- The total path length of a point on a circle rolling without slipping -/
theorem circle_roll_path_length
  (r : ℝ) -- radius of the circle
  (θ_flat : ℝ) -- angle rolled on flat surface in radians
  (θ_slope : ℝ) -- angle rolled on slope in radians
  (h_radius : r = 4 / Real.pi)
  (h_flat : θ_flat = 3 * Real.pi / 2)
  (h_slope : θ_slope = Real.pi / 2)
  (h_total : θ_flat + θ_slope = 2 * Real.pi) :
  2 * Real.pi * r = 8 :=
sorry

end NUMINAMATH_CALUDE_circle_roll_path_length_l1986_198628


namespace NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1986_198607

theorem closest_integer_to_cube_root : ∃ (n : ℤ), 
  n = 8 ∧ ∀ (m : ℤ), |m - (5^3 + 7^3)^(1/3)| ≥ |n - (5^3 + 7^3)^(1/3)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_cube_root_l1986_198607


namespace NUMINAMATH_CALUDE_line_points_k_value_l1986_198658

/-- Given a line containing the points (-1, 6), (6, k), and (20, 3), prove that k = 5 -/
theorem line_points_k_value :
  ∀ k : ℝ,
  (∃ m b : ℝ,
    (m * (-1) + b = 6) ∧
    (m * 6 + b = k) ∧
    (m * 20 + b = 3)) →
  k = 5 :=
by sorry

end NUMINAMATH_CALUDE_line_points_k_value_l1986_198658


namespace NUMINAMATH_CALUDE_circle_area_difference_l1986_198626

theorem circle_area_difference (π : ℝ) : 
  let r1 : ℝ := 30
  let d2 : ℝ := 30
  let r2 : ℝ := d2 / 2
  let area1 : ℝ := π * r1^2
  let area2 : ℝ := π * r2^2
  area1 - area2 = 675 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_difference_l1986_198626


namespace NUMINAMATH_CALUDE_factorization_2x_cubed_minus_8x_l1986_198689

theorem factorization_2x_cubed_minus_8x (x : ℝ) : 2*x^3 - 8*x = 2*x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_2x_cubed_minus_8x_l1986_198689


namespace NUMINAMATH_CALUDE_teacher_periods_per_day_l1986_198601

/-- Represents the number of periods a teacher teaches per day -/
def periods_per_day : ℕ := 5

/-- Represents the number of working days per month -/
def days_per_month : ℕ := 24

/-- Represents the payment per period in dollars -/
def payment_per_period : ℕ := 5

/-- Represents the number of months worked -/
def months_worked : ℕ := 6

/-- Represents the total earnings in dollars -/
def total_earnings : ℕ := 3600

/-- Theorem stating that given the conditions, the teacher teaches 5 periods per day -/
theorem teacher_periods_per_day :
  periods_per_day * days_per_month * months_worked * payment_per_period = total_earnings :=
sorry

end NUMINAMATH_CALUDE_teacher_periods_per_day_l1986_198601


namespace NUMINAMATH_CALUDE_number_problem_l1986_198697

theorem number_problem (x : ℝ) : (0.2 * x = 0.2 * 650 + 190) → x = 1600 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l1986_198697


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1986_198692

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 (a > 0, b > 0) and right focus F at (c, 0),
    if a line perpendicular to y = -bx/a passes through F and intersects the left branch of the hyperbola
    at point B such that vector FB = 2 * vector FA (where A is the foot of the perpendicular),
    then the eccentricity of the hyperbola is √5. -/
theorem hyperbola_eccentricity (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let F : ℝ × ℝ := (c, 0)
  let perpendicular_line := {(x, y) : ℝ × ℝ | y = a / b * (x - c)}
  let hyperbola := {(x, y) : ℝ × ℝ | x^2 / a^2 - y^2 / b^2 = 1}
  let A := (a^2 / c, a * b / c)
  ∃ B : ℝ × ℝ, B.1 < 0 ∧ B ∈ hyperbola ∧ B ∈ perpendicular_line ∧
    (B.1 - F.1, B.2 - F.2) = (2 * (A.1 - F.1), 2 * (A.2 - F.2)) →
  c^2 / a^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1986_198692


namespace NUMINAMATH_CALUDE_largest_non_representable_l1986_198676

def is_composite (n : ℕ) : Prop := ∃ m k, 1 < m ∧ 1 < k ∧ n = m * k

def is_representable (n : ℕ) : Prop :=
  ∃ (a b : ℕ), 0 < a ∧ is_composite b ∧ n = 36 * a + b

theorem largest_non_representable : 
  (∀ n > 187, is_representable n) ∧ ¬is_representable 187 := by sorry

end NUMINAMATH_CALUDE_largest_non_representable_l1986_198676


namespace NUMINAMATH_CALUDE_total_distance_two_trains_l1986_198600

/-- Given two trains A and B traveling for 15 minutes, with speeds of 70 kmph and 90 kmph respectively,
    the total distance covered by both trains is 40 kilometers. -/
theorem total_distance_two_trains (speed_A speed_B : ℝ) (time : ℝ) : 
  speed_A = 70 → speed_B = 90 → time = 0.25 → 
  (speed_A * time + speed_B * time) = 40 := by
sorry

end NUMINAMATH_CALUDE_total_distance_two_trains_l1986_198600


namespace NUMINAMATH_CALUDE_two_heads_five_coins_l1986_198634

/-- The probability of getting exactly k heads when tossing n fair coins -/
def coinTossProbability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

/-- Theorem: The probability of getting exactly two heads when tossing five fair coins is 5/16 -/
theorem two_heads_five_coins : coinTossProbability 5 2 = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_two_heads_five_coins_l1986_198634


namespace NUMINAMATH_CALUDE_sixteen_million_scientific_notation_l1986_198610

/-- Given a number n, returns true if it's in scientific notation -/
def is_scientific_notation (n : ℝ) : Prop :=
  ∃ (a : ℝ) (b : ℤ), 1 ≤ a ∧ a < 10 ∧ n = a * (10 : ℝ) ^ b

theorem sixteen_million_scientific_notation :
  is_scientific_notation 16000000 ∧
  16000000 = 1.6 * (10 : ℝ) ^ 7 :=
sorry

end NUMINAMATH_CALUDE_sixteen_million_scientific_notation_l1986_198610


namespace NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1986_198640

-- Define the equation
def equation (x : ℝ) : Prop := x^2 - 6*x + 8 = 0

-- Define an isosceles triangle type
structure IsoscelesTriangle where
  side1 : ℝ
  side2 : ℝ
  base : ℝ
  h_isosceles : side1 = side2
  h_equation1 : equation side1
  h_equation2 : equation side2
  h_triangle_inequality : side1 + side2 > base ∧ side1 + base > side2 ∧ side2 + base > side1

-- Theorem statement
theorem isosceles_triangle_perimeter (t : IsoscelesTriangle) : t.side1 + t.side2 + t.base = 10 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_perimeter_l1986_198640


namespace NUMINAMATH_CALUDE_gcd_1443_999_l1986_198671

theorem gcd_1443_999 : Nat.gcd 1443 999 = 111 := by
  sorry

end NUMINAMATH_CALUDE_gcd_1443_999_l1986_198671


namespace NUMINAMATH_CALUDE_f_decreasing_on_interval_l1986_198685

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Theorem statement
theorem f_decreasing_on_interval :
  ∀ x ∈ Set.Ioo 0 2, f' x < 0 :=
sorry

end NUMINAMATH_CALUDE_f_decreasing_on_interval_l1986_198685


namespace NUMINAMATH_CALUDE_sum_mod_thirteen_l1986_198655

theorem sum_mod_thirteen : (10247 + 10248 + 10249 + 10250) % 13 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_mod_thirteen_l1986_198655


namespace NUMINAMATH_CALUDE_unique_solution_l1986_198674

theorem unique_solution : ∃! x : ℝ, 
  (3 * x^2) / (x - 2) - (3 * x + 9) / 4 + (6 - 9 * x) / (x - 2) + 2 = 0 ∧ 
  x^3 ≠ 3 * x + 1 ∧
  x = 13 / 9 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1986_198674


namespace NUMINAMATH_CALUDE_commercial_time_calculation_l1986_198608

theorem commercial_time_calculation (num_programs : ℕ) (program_duration : ℕ) (commercial_fraction : ℚ) : 
  num_programs = 6 → 
  program_duration = 30 → 
  commercial_fraction = 1/4 → 
  (↑num_programs * ↑program_duration : ℚ) * commercial_fraction = 45 := by
  sorry

end NUMINAMATH_CALUDE_commercial_time_calculation_l1986_198608


namespace NUMINAMATH_CALUDE_problem_solution_l1986_198680

theorem problem_solution (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_prod : x * y * z = 1)
  (h_eq1 : x + 1 / z = 6)
  (h_eq2 : y + 1 / x = 30) :
  z + 1 / y = 38 / 179 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1986_198680


namespace NUMINAMATH_CALUDE_percentage_problem_l1986_198691

theorem percentage_problem (P : ℝ) : 
  (P * 100 + 60 = 100) → P = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l1986_198691


namespace NUMINAMATH_CALUDE_no_base_square_l1986_198647

theorem no_base_square (b : ℕ) : b > 1 → ¬∃ (n : ℕ), 2 * b^2 + 3 * b + 2 = n^2 := by
  sorry

end NUMINAMATH_CALUDE_no_base_square_l1986_198647


namespace NUMINAMATH_CALUDE_mean_of_four_numbers_with_given_variance_l1986_198688

/-- Given a set of four positive real numbers with a specific variance, prove that their mean is 2. -/
theorem mean_of_four_numbers_with_given_variance 
  (x₁ x₂ x₃ x₄ : ℝ) 
  (pos₁ : 0 < x₁) (pos₂ : 0 < x₂) (pos₃ : 0 < x₃) (pos₄ : 0 < x₄)
  (variance_eq : (1/4) * (x₁^2 + x₂^2 + x₃^2 + x₄^2 - 16) = 
                 (1/4) * ((x₁ - (x₁ + x₂ + x₃ + x₄)/4)^2 + 
                          (x₂ - (x₁ + x₂ + x₃ + x₄)/4)^2 + 
                          (x₃ - (x₁ + x₂ + x₃ + x₄)/4)^2 + 
                          (x₄ - (x₁ + x₂ + x₃ + x₄)/4)^2)) :
  (x₁ + x₂ + x₃ + x₄) / 4 = 2 := by
  sorry

end NUMINAMATH_CALUDE_mean_of_four_numbers_with_given_variance_l1986_198688


namespace NUMINAMATH_CALUDE_bell_pepper_slices_l1986_198693

theorem bell_pepper_slices (num_peppers : ℕ) (slices_per_pepper : ℕ) (smaller_pieces : ℕ) : 
  num_peppers = 5 →
  slices_per_pepper = 20 →
  smaller_pieces = 3 →
  let total_slices := num_peppers * slices_per_pepper
  let large_slices := total_slices / 2
  let small_pieces := large_slices * smaller_pieces
  total_slices - large_slices + small_pieces = 200 := by
  sorry

end NUMINAMATH_CALUDE_bell_pepper_slices_l1986_198693


namespace NUMINAMATH_CALUDE_deer_meat_content_deer_meat_content_is_200_l1986_198648

/-- Proves that each deer contains 200 pounds of meat given the hunting conditions -/
theorem deer_meat_content (hunting_wolves : ℕ) (additional_wolves : ℕ) 
  (meat_per_wolf_per_day : ℕ) (hunting_days : ℕ) (deer_per_hunting_wolf : ℕ) : ℕ :=
  let total_wolves := hunting_wolves + additional_wolves
  let total_meat_needed := total_wolves * meat_per_wolf_per_day * hunting_days
  let total_deer := hunting_wolves * deer_per_hunting_wolf
  total_meat_needed / total_deer

#check deer_meat_content 4 16 8 5 1 = 200

/-- Theorem stating that under the given conditions, each deer contains 200 pounds of meat -/
theorem deer_meat_content_is_200 : 
  deer_meat_content 4 16 8 5 1 = 200 := by
  sorry

end NUMINAMATH_CALUDE_deer_meat_content_deer_meat_content_is_200_l1986_198648


namespace NUMINAMATH_CALUDE_marilyn_bottle_caps_l1986_198641

/-- The number of bottle caps Marilyn starts with -/
def initial_caps : ℕ := 51

/-- The number of bottle caps Marilyn shares with Nancy -/
def shared_caps : ℕ := 36

/-- The number of bottle caps Marilyn ends up with -/
def remaining_caps : ℕ := initial_caps - shared_caps

theorem marilyn_bottle_caps : remaining_caps = 15 := by
  sorry

end NUMINAMATH_CALUDE_marilyn_bottle_caps_l1986_198641


namespace NUMINAMATH_CALUDE_inequality_of_reciprocals_l1986_198694

theorem inequality_of_reciprocals (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1/(2*a) + 1/(2*b) + 1/(2*c) ≥ 1/(a+b) + 1/(b+c) + 1/(c+a) := by
  sorry

end NUMINAMATH_CALUDE_inequality_of_reciprocals_l1986_198694


namespace NUMINAMATH_CALUDE_jessie_current_weight_l1986_198632

def initial_weight : ℝ := 69
def weight_lost : ℝ := 35

theorem jessie_current_weight : 
  initial_weight - weight_lost = 34 := by sorry

end NUMINAMATH_CALUDE_jessie_current_weight_l1986_198632


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1986_198615

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation that z satisfies
def equation (z : ℂ) : Prop := (-1 + i) * z = (1 + i)^2

-- Define what it means for a complex number to be in the fourth quadrant
def in_fourth_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im < 0

-- Theorem statement
theorem z_in_fourth_quadrant :
  ∃ z : ℂ, equation z ∧ in_fourth_quadrant z :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l1986_198615


namespace NUMINAMATH_CALUDE_lisa_photos_l1986_198649

theorem lisa_photos (animal_photos : ℕ) 
  (h1 : animal_photos + 3 * animal_photos + (3 * animal_photos - 10) = 45) : 
  animal_photos = 7 := by
sorry

end NUMINAMATH_CALUDE_lisa_photos_l1986_198649


namespace NUMINAMATH_CALUDE_mitzel_allowance_percentage_l1986_198696

theorem mitzel_allowance_percentage (spent : ℝ) (left : ℝ) : 
  spent = 14 → left = 26 → (spent / (spent + left)) * 100 = 35 := by
  sorry

end NUMINAMATH_CALUDE_mitzel_allowance_percentage_l1986_198696
