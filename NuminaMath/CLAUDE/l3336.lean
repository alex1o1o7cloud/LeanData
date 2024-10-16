import Mathlib

namespace NUMINAMATH_CALUDE_special_number_between_18_and_57_l3336_333675

theorem special_number_between_18_and_57 :
  ∃! n : ℕ, 18 ≤ n ∧ n ≤ 57 ∧ 
  7 ∣ n ∧ 
  (∀ p : ℕ, Prime p → p ≠ 7 → ¬(p ∣ n)) ∧
  n = 49 ∧
  Real.sqrt n = 7 := by
sorry

end NUMINAMATH_CALUDE_special_number_between_18_and_57_l3336_333675


namespace NUMINAMATH_CALUDE_abc_inequality_l3336_333671

theorem abc_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a * b * c ≥ (b + c - a) * (a + c - b) * (a + b - c) := by sorry

end NUMINAMATH_CALUDE_abc_inequality_l3336_333671


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3336_333630

/-- An arithmetic sequence with given properties -/
structure ArithmeticSequence where
  a : ℕ → ℤ  -- The sequence
  first_term : a 1 = 1
  third_term : a 3 = -3
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1

/-- Theorem about the general formula and sum of the sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ, seq.a n = 3 - 2 * n) ∧
  (∃ k : ℕ, k * (seq.a 1 + seq.a k) / 2 = -35 ∧ k = 7) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3336_333630


namespace NUMINAMATH_CALUDE_divisibility_implies_equality_l3336_333602

theorem divisibility_implies_equality (a b n : ℕ) :
  (∀ k : ℕ, k ≠ b → (b - k) ∣ (a - k^n)) →
  a = b^n :=
by sorry

end NUMINAMATH_CALUDE_divisibility_implies_equality_l3336_333602


namespace NUMINAMATH_CALUDE_coin_problem_l3336_333661

theorem coin_problem (x y z : ℕ) : 
  x + y + z = 900 →
  x + 2*y + 5*z = 1950 →
  z = x / 2 →
  y = 450 := by
sorry

end NUMINAMATH_CALUDE_coin_problem_l3336_333661


namespace NUMINAMATH_CALUDE_sports_club_overlap_l3336_333643

theorem sports_club_overlap (total : ℕ) (badminton : ℕ) (tennis : ℕ) (neither : ℕ) 
  (h1 : total = 42)
  (h2 : badminton = 20)
  (h3 : tennis = 23)
  (h4 : neither = 6)
  : (badminton + tennis) - (total - neither) = 7 := by
  sorry

end NUMINAMATH_CALUDE_sports_club_overlap_l3336_333643


namespace NUMINAMATH_CALUDE_similar_triangles_leg_length_l3336_333680

theorem similar_triangles_leg_length :
  ∀ (y : ℝ),
  (12 : ℝ) / y = 9 / 6 →
  y = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_leg_length_l3336_333680


namespace NUMINAMATH_CALUDE_trivia_team_groups_l3336_333698

theorem trivia_team_groups (total : ℕ) (not_picked : ℕ) (num_groups : ℕ) 
  (h1 : total = 17) 
  (h2 : not_picked = 5) 
  (h3 : num_groups = 3) :
  (total - not_picked) / num_groups = 4 :=
by sorry

end NUMINAMATH_CALUDE_trivia_team_groups_l3336_333698


namespace NUMINAMATH_CALUDE_acute_angle_range_l3336_333607

def a : Fin 2 → ℝ := ![1, 2]
def b (x : ℝ) : Fin 2 → ℝ := ![x, 4]

def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

def is_acute_angle (v w : Fin 2 → ℝ) : Prop := dot_product v w > 0

theorem acute_angle_range (x : ℝ) :
  is_acute_angle a (b x) ↔ x ∈ Set.Ioo (-8 : ℝ) 2 ∪ Set.Ioi 2 := by sorry

end NUMINAMATH_CALUDE_acute_angle_range_l3336_333607


namespace NUMINAMATH_CALUDE_work_completion_time_l3336_333664

theorem work_completion_time 
  (total_work : ℝ) 
  (p_q_together_time : ℝ) 
  (p_alone_time : ℝ) 
  (h1 : p_q_together_time = 6)
  (h2 : p_alone_time = 15)
  : ∃ q_alone_time : ℝ, q_alone_time = 10 ∧ 
    (1 / p_q_together_time = 1 / p_alone_time + 1 / q_alone_time) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3336_333664


namespace NUMINAMATH_CALUDE_sum_c_plus_d_l3336_333690

theorem sum_c_plus_d (a b c d : ℝ) 
  (h1 : a + b = 5)
  (h2 : b + c = 6)
  (h3 : a + d = 2) :
  c + d = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_c_plus_d_l3336_333690


namespace NUMINAMATH_CALUDE_trig_product_equals_one_sixteenth_l3336_333613

theorem trig_product_equals_one_sixteenth : 
  Real.cos (15 * π / 180) * Real.sin (30 * π / 180) * 
  Real.cos (75 * π / 180) * Real.sin (150 * π / 180) = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_trig_product_equals_one_sixteenth_l3336_333613


namespace NUMINAMATH_CALUDE_book_distribution_theorem_l3336_333638

def distribute_books (n : ℕ) : ℕ :=
  if n ≥ 2 then n - 1 else 0

theorem book_distribution_theorem :
  distribute_books 8 = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_book_distribution_theorem_l3336_333638


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3336_333679

/-- A point (x, y) is in the second quadrant if x < 0 and y > 0 -/
def in_second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- For any real number m, the point (-1, m^2 + 1) is in the second quadrant -/
theorem point_in_second_quadrant (m : ℝ) : in_second_quadrant (-1) (m^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3336_333679


namespace NUMINAMATH_CALUDE_parabola_c_value_l3336_333626

/-- A parabola with equation x = ay² + by + c, vertex at (5, 3), and passing through (3, 5) has c = 1/2 -/
theorem parabola_c_value (a b c : ℝ) : 
  (∀ y : ℝ, 5 = a * 3^2 + b * 3 + c) →  -- vertex condition
  (∀ y : ℝ, 3 = a * 5^2 + b * 5 + c) →  -- point condition
  c = 1/2 := by sorry

end NUMINAMATH_CALUDE_parabola_c_value_l3336_333626


namespace NUMINAMATH_CALUDE_amount_problem_l3336_333655

theorem amount_problem (a b : ℝ) 
  (h1 : a + b = 1210)
  (h2 : (4/5) * a = (2/3) * b) :
  b = 453.75 := by
sorry

end NUMINAMATH_CALUDE_amount_problem_l3336_333655


namespace NUMINAMATH_CALUDE_division_problem_l3336_333672

theorem division_problem (A : ℕ) (h : A % 7 = 3 ∧ A / 7 = 5) : A = 38 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l3336_333672


namespace NUMINAMATH_CALUDE_weight_per_hour_is_correct_l3336_333659

/-- Represents the types of coins Jim finds --/
inductive CoinType
| Gold
| Silver
| Bronze

/-- Represents a bag of coins --/
structure CoinBag where
  coinType : CoinType
  count : ℕ

def hours_spent : ℕ := 8

def coin_weight (ct : CoinType) : ℕ :=
  match ct with
  | CoinType.Gold => 10
  | CoinType.Silver => 5
  | CoinType.Bronze => 2

def treasure_chest : CoinBag := ⟨CoinType.Gold, 100⟩
def smaller_bags : List CoinBag := [⟨CoinType.Gold, 50⟩, ⟨CoinType.Gold, 50⟩]
def other_bags : List CoinBag := [⟨CoinType.Gold, 30⟩, ⟨CoinType.Gold, 20⟩, ⟨CoinType.Gold, 10⟩]
def silver_coins : CoinBag := ⟨CoinType.Silver, 30⟩
def bronze_coins : CoinBag := ⟨CoinType.Bronze, 50⟩

def all_bags : List CoinBag :=
  [treasure_chest] ++ smaller_bags ++ other_bags ++ [silver_coins, bronze_coins]

def total_weight (bags : List CoinBag) : ℕ :=
  bags.foldl (fun acc bag => acc + bag.count * coin_weight bag.coinType) 0

theorem weight_per_hour_is_correct :
  (total_weight all_bags : ℚ) / hours_spent = 356.25 := by sorry

end NUMINAMATH_CALUDE_weight_per_hour_is_correct_l3336_333659


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l3336_333603

theorem sum_of_two_numbers (x y : ℝ) (h1 : x - y = 7) (h2 : x^2 + y^2 = 130) : x + y = -7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l3336_333603


namespace NUMINAMATH_CALUDE_eggs_needed_proof_l3336_333678

def recipe_eggs : ℕ := 2
def recipe_people : ℕ := 4
def target_people : ℕ := 8
def available_eggs : ℕ := 3

theorem eggs_needed_proof : 
  (target_people / recipe_people * recipe_eggs) - available_eggs = 1 := by
sorry

end NUMINAMATH_CALUDE_eggs_needed_proof_l3336_333678


namespace NUMINAMATH_CALUDE_field_length_is_96_l3336_333640

/-- Proves that the length of a rectangular field is 96 meters given the specified conditions. -/
theorem field_length_is_96 (l w : ℝ) (h1 : l = 2 * w) (h2 : 64 = (1 / 72) * (l * w)) : l = 96 := by
  sorry

end NUMINAMATH_CALUDE_field_length_is_96_l3336_333640


namespace NUMINAMATH_CALUDE_delivery_fee_calculation_delivery_fee_is_twenty_l3336_333668

theorem delivery_fee_calculation (sandwich_price : ℝ) (num_sandwiches : ℕ) 
  (tip_percentage : ℝ) (total_received : ℝ) (delivery_fee : ℝ) : Prop :=
  sandwich_price = 5 →
  num_sandwiches = 18 →
  tip_percentage = 0.1 →
  total_received = 121 →
  delivery_fee = 20 →
  total_received = (sandwich_price * num_sandwiches) + delivery_fee + 
    (tip_percentage * (sandwich_price * num_sandwiches + delivery_fee))

-- Proof
theorem delivery_fee_is_twenty :
  ∃ (delivery_fee : ℝ),
    delivery_fee_calculation 5 18 0.1 121 delivery_fee :=
by
  sorry

end NUMINAMATH_CALUDE_delivery_fee_calculation_delivery_fee_is_twenty_l3336_333668


namespace NUMINAMATH_CALUDE_defeat_crab_ways_l3336_333657

/-- Represents the number of claws on the giant enemy crab -/
def num_claws : ℕ := 2

/-- Represents the number of legs on the giant enemy crab -/
def num_legs : ℕ := 6

/-- Represents the minimum number of legs that must be cut before claws can be cut -/
def min_legs_before_claws : ℕ := 3

/-- The number of ways to defeat the giant enemy crab -/
def ways_to_defeat_crab : ℕ := num_legs.factorial * num_claws.factorial * (Nat.choose (num_legs + num_claws - min_legs_before_claws) num_claws)

/-- Theorem stating the number of ways to defeat the giant enemy crab -/
theorem defeat_crab_ways : ways_to_defeat_crab = 14400 := by
  sorry

end NUMINAMATH_CALUDE_defeat_crab_ways_l3336_333657


namespace NUMINAMATH_CALUDE_sales_problem_l3336_333615

-- Define the sales revenue function
def sales_revenue (x : ℝ) : ℝ := 1000 * x

-- Define the sales cost function
def sales_cost (x : ℝ) : ℝ := 500 * x + 2000

-- State the theorem
theorem sales_problem :
  -- Condition 1: When x = 0, sales cost is 2000
  sales_cost 0 = 2000 ∧
  -- Condition 2: When x = 2, sales revenue is 2000 and sales cost is 3000
  sales_revenue 2 = 2000 ∧ sales_cost 2 = 3000 ∧
  -- Condition 3: Sales revenue is directly proportional to x (already satisfied by definition)
  -- Condition 4: Sales cost is a linear function of x (already satisfied by definition)
  -- Proof goals:
  -- 1. The functions satisfy all conditions (implicitly proved by the above)
  -- 2. Sales revenue equals sales cost at 4 tons
  (∃ x : ℝ, x = 4 ∧ sales_revenue x = sales_cost x) ∧
  -- 3. Profit at 10 tons is 3000 yuan
  sales_revenue 10 - sales_cost 10 = 3000 :=
by sorry

end NUMINAMATH_CALUDE_sales_problem_l3336_333615


namespace NUMINAMATH_CALUDE_ellipse_hyperbola_foci_coincide_l3336_333696

/-- The squared semi-major axis of the ellipse -/
def a_squared_ellipse : ℝ := 25

/-- The squared semi-major axis of the hyperbola -/
def a_squared_hyperbola : ℝ := 196

/-- The squared semi-minor axis of the hyperbola -/
def b_squared_hyperbola : ℝ := 121

/-- The equation of the ellipse -/
def ellipse_equation (x y b : ℝ) : Prop :=
  x^2 / a_squared_ellipse + y^2 / b^2 = 1

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  x^2 / a_squared_hyperbola - y^2 / b_squared_hyperbola = 1/49

/-- The theorem stating that if the foci of the ellipse and hyperbola coincide,
    then the squared semi-minor axis of the ellipse is 908/49 -/
theorem ellipse_hyperbola_foci_coincide :
  ∃ b : ℝ, (∀ x y : ℝ, ellipse_equation x y b ↔ hyperbola_equation x y) →
    b^2 = 908/49 := by sorry

end NUMINAMATH_CALUDE_ellipse_hyperbola_foci_coincide_l3336_333696


namespace NUMINAMATH_CALUDE_no_valid_division_l3336_333673

/-- The total weight of all stones -/
def total_weight : ℕ := (77 * 78) / 2

/-- The weight of the heaviest group for a given k -/
def heaviest_group_weight (k : ℕ) : ℕ := 
  (total_weight + k - 1) / k

/-- The number of stones in the heaviest group for a given k -/
def stones_in_heaviest_group (k : ℕ) : ℕ := 
  (heaviest_group_weight k + 76) / 77

/-- The total number of stones in all groups for a given k -/
def total_stones (k : ℕ) : ℕ := 
  k * (stones_in_heaviest_group k + (k - 1) / 2)

/-- The set of possible values for k -/
def possible_k : Finset ℕ := {9, 10, 11, 12}

theorem no_valid_division : 
  ∀ k ∈ possible_k, total_stones k > 77 := by sorry

end NUMINAMATH_CALUDE_no_valid_division_l3336_333673


namespace NUMINAMATH_CALUDE_competition_outcomes_l3336_333697

/-- The number of possible outcomes for champions in a competition -/
def num_outcomes (num_students : ℕ) (num_events : ℕ) : ℕ :=
  num_students ^ num_events

/-- Theorem: Given 3 students competing in 2 events, where each event has one champion,
    the total number of possible outcomes for the champions is 9. -/
theorem competition_outcomes :
  num_outcomes 3 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_competition_outcomes_l3336_333697


namespace NUMINAMATH_CALUDE_product_divisible_by_sum_iff_not_odd_prime_l3336_333685

theorem product_divisible_by_sum_iff_not_odd_prime (n : ℕ) : 
  (∃ k : ℕ, n.factorial = k * (n * (n + 1) / 2)) ↔ ¬(Nat.Prime (n + 1) ∧ Odd (n + 1)) :=
sorry

end NUMINAMATH_CALUDE_product_divisible_by_sum_iff_not_odd_prime_l3336_333685


namespace NUMINAMATH_CALUDE_fifth_power_sum_l3336_333681

theorem fifth_power_sum (a b x y : ℝ) 
  (h1 : a*x + b*y = 5)
  (h2 : a*x^2 + b*y^2 = 9)
  (h3 : a*x^3 + b*y^3 = 22)
  (h4 : a*x^4 + b*y^4 = 60) :
  a*x^5 + b*y^5 = 97089/203 := by
sorry

end NUMINAMATH_CALUDE_fifth_power_sum_l3336_333681


namespace NUMINAMATH_CALUDE_cos_alpha_for_point_P_l3336_333662

theorem cos_alpha_for_point_P (α : Real) :
  let P : ℝ × ℝ := (-3, 4)
  (∃ t : ℝ, t > 0 ∧ P = (t * Real.cos α, t * Real.sin α)) →
  Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_alpha_for_point_P_l3336_333662


namespace NUMINAMATH_CALUDE_expression_equals_503_l3336_333654

theorem expression_equals_503 : 2015 * (1999/2015) * (1/4) - 2011/2015 = 503 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_503_l3336_333654


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l3336_333689

/-- The equation of a line passing through (1,1) and tangent to the circle x^2 - 2x + y^2 = 0 is y = 1 -/
theorem tangent_line_to_circle (x y : ℝ) : 
  (∃ k : ℝ, y - 1 = k * (x - 1)) ∧ 
  (x^2 - 2*x + y^2 = 0 → (x - 1)^2 + (y - 0)^2 = 1) →
  y = 1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l3336_333689


namespace NUMINAMATH_CALUDE_equation_solutions_l3336_333614

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = (2 + Real.sqrt 7) / 3 ∧ x₂ = (2 - Real.sqrt 7) / 3 ∧
    3 * x₁^2 - 1 = 4 * x₁ ∧ 3 * x₂^2 - 1 = 4 * x₂) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -4 ∧ x₂ = 1 ∧
    (x₁ + 4)^2 = 5 * (x₁ + 4) ∧ (x₂ + 4)^2 = 5 * (x₂ + 4)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3336_333614


namespace NUMINAMATH_CALUDE_orchestra_price_is_12_l3336_333647

/-- Represents the pricing and sales of theater tickets --/
structure TheaterSales where
  orchestra_price : ℝ
  balcony_price : ℝ
  orchestra_tickets : ℕ
  balcony_tickets : ℕ

/-- Theorem stating the price of orchestra seats given the conditions --/
theorem orchestra_price_is_12 (sales : TheaterSales) :
  sales.balcony_price = 8 ∧
  sales.orchestra_tickets + sales.balcony_tickets = 380 ∧
  sales.orchestra_price * sales.orchestra_tickets + sales.balcony_price * sales.balcony_tickets = 3320 ∧
  sales.balcony_tickets = sales.orchestra_tickets + 240
  → sales.orchestra_price = 12 := by
  sorry


end NUMINAMATH_CALUDE_orchestra_price_is_12_l3336_333647


namespace NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l3336_333606

theorem smallest_n_multiple_of_seven (x y : ℤ) 
  (hx : 7 ∣ (x + 2)) 
  (hy : 7 ∣ (y - 2)) : 
  (∃ n : ℕ+, 7 ∣ (x^2 - x*y + y^2 + n) ∧ 
    ∀ m : ℕ+, 7 ∣ (x^2 - x*y + y^2 + m) → n ≤ m) ∧
  (7 ∣ (x^2 - x*y + y^2 + 2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_multiple_of_seven_l3336_333606


namespace NUMINAMATH_CALUDE_total_ways_is_eight_l3336_333682

/-- The number of course options available --/
def num_courses : Nat := 2

/-- The number of students choosing courses --/
def num_students : Nat := 3

/-- Calculates the total number of ways students can choose courses --/
def total_ways : Nat := num_courses ^ num_students

/-- Theorem stating that the total number of ways to choose courses is 8 --/
theorem total_ways_is_eight : total_ways = 8 := by
  sorry

end NUMINAMATH_CALUDE_total_ways_is_eight_l3336_333682


namespace NUMINAMATH_CALUDE_parabola_max_value_l3336_333658

/-- A parabola that opens downward and has its vertex at (2, -3) has a maximum value of -3 -/
theorem parabola_max_value (a b c : ℝ) (h_downward : a < 0) 
  (h_vertex : ∀ x, a * x^2 + b * x + c ≤ a * 2^2 + b * 2 + c) 
  (h_vertex_y : a * 2^2 + b * 2 + c = -3) : 
  ∀ x, a * x^2 + b * x + c ≤ -3 :=
sorry

end NUMINAMATH_CALUDE_parabola_max_value_l3336_333658


namespace NUMINAMATH_CALUDE_paper_plates_and_cups_cost_l3336_333635

theorem paper_plates_and_cups_cost (plate_cost cup_cost : ℝ) : 
  100 * plate_cost + 200 * cup_cost = 7.5 → 
  20 * plate_cost + 40 * cup_cost = 1.5 := by
sorry

end NUMINAMATH_CALUDE_paper_plates_and_cups_cost_l3336_333635


namespace NUMINAMATH_CALUDE_square_root_of_nine_three_is_square_root_of_nine_l3336_333687

theorem square_root_of_nine (x : ℝ) : x ^ 2 = 9 → x = 3 ∨ x = -3 := by
  sorry

theorem three_is_square_root_of_nine : ∃ x : ℝ, x ^ 2 = 9 ∧ x = 3 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_nine_three_is_square_root_of_nine_l3336_333687


namespace NUMINAMATH_CALUDE_equation_roots_theorem_l3336_333666

/-- 
Given an equation (x² - px) / (kx - d) = (n - 2) / (n + 2),
where the roots are numerically equal but opposite in sign and their product is 1,
prove that n = 2(k - p) / (k + p).
-/
theorem equation_roots_theorem (p k d n : ℝ) (x : ℝ → ℝ) :
  (∀ x, (x^2 - p*x) / (k*x - d) = (n - 2) / (n + 2)) →
  (∃ r : ℝ, x r = r ∧ x (-r) = -r) →
  (∃ r : ℝ, x r * x (-r) = 1) →
  n = 2*(k - p) / (k + p) := by
  sorry

end NUMINAMATH_CALUDE_equation_roots_theorem_l3336_333666


namespace NUMINAMATH_CALUDE_y_squared_mod_30_l3336_333616

theorem y_squared_mod_30 (y : ℤ) (h1 : 6 * y ≡ 12 [ZMOD 30]) (h2 : 5 * y ≡ 25 [ZMOD 30]) :
  y^2 ≡ 19 [ZMOD 30] := by
  sorry

end NUMINAMATH_CALUDE_y_squared_mod_30_l3336_333616


namespace NUMINAMATH_CALUDE_room_assignment_count_l3336_333692

/-- The number of rooms in the lodge -/
def num_rooms : ℕ := 6

/-- The number of friends checking in -/
def num_friends : ℕ := 7

/-- The maximum number of friends allowed per room -/
def max_per_room : ℕ := 2

/-- The minimum number of unoccupied rooms -/
def min_unoccupied : ℕ := 1

/-- A function that calculates the number of ways to assign friends to rooms -/
def assign_rooms : ℕ := sorry

/-- Theorem stating that the number of ways to assign rooms is 128520 -/
theorem room_assignment_count : assign_rooms = 128520 := by sorry

end NUMINAMATH_CALUDE_room_assignment_count_l3336_333692


namespace NUMINAMATH_CALUDE_eighteen_wheeler_toll_l3336_333695

/-- Calculate the toll for a truck given the number of axles -/
def toll (axles : ℕ) : ℚ :=
  1.50 + 0.50 * (axles - 2)

/-- Calculate the number of axles for a truck given the total number of wheels -/
def axles_count (wheels : ℕ) : ℕ :=
  wheels / 2

theorem eighteen_wheeler_toll :
  let wheels : ℕ := 18
  let axles : ℕ := axles_count wheels
  toll axles = 5 := by sorry

end NUMINAMATH_CALUDE_eighteen_wheeler_toll_l3336_333695


namespace NUMINAMATH_CALUDE_call_center_team_b_fraction_l3336_333694

/-- Represents the fraction of calls processed by Team B given the relative
    call processing rates and team sizes of two teams in a call center. -/
theorem call_center_team_b_fraction :
  -- Each member of Team A processes 6/5 calls compared to Team B
  ∀ (call_rate_a call_rate_b : ℚ),
  call_rate_a = 6 / 5 * call_rate_b →
  -- Team A has 5/8 as many agents as Team B
  ∀ (team_size_a team_size_b : ℚ),
  team_size_a = 5 / 8 * team_size_b →
  -- The fraction of calls processed by Team B
  (team_size_b * call_rate_b) /
    (team_size_a * call_rate_a + team_size_b * call_rate_b) = 4 / 7 :=
by sorry

end NUMINAMATH_CALUDE_call_center_team_b_fraction_l3336_333694


namespace NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l3336_333600

/-- Given vectors a and b in ℝ², if a + b = (5, -10) and a - b = (3, 6),
    then the cosine of the angle between a and b is 2√13/13. -/
theorem cosine_of_angle_between_vectors (a b : ℝ × ℝ) 
  (h1 : a + b = (5, -10)) 
  (h2 : a - b = (3, 6)) : 
  (a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)) = 2 * Real.sqrt 13 / 13 := by
  sorry

end NUMINAMATH_CALUDE_cosine_of_angle_between_vectors_l3336_333600


namespace NUMINAMATH_CALUDE_min_max_abs_quadratic_minus_linear_l3336_333641

theorem min_max_abs_quadratic_minus_linear (y : ℝ) :
  ∃ (y : ℝ), ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 2 → |x^2 - x*y| ≤ 0 ∧
  (∀ (y : ℝ), ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 2 ∧ |x^2 - x*y| ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_min_max_abs_quadratic_minus_linear_l3336_333641


namespace NUMINAMATH_CALUDE_path_area_l3336_333611

/-- The area of a path surrounding a rectangular field -/
theorem path_area (field_length field_width path_width : ℝ) 
  (h1 : field_length = 75) 
  (h2 : field_width = 55) 
  (h3 : path_width = 2.8) : 
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - 
  field_length * field_width = 759.36 := by
  sorry

#check path_area

end NUMINAMATH_CALUDE_path_area_l3336_333611


namespace NUMINAMATH_CALUDE_blocks_differing_in_two_ways_l3336_333651

/-- Represents the properties of a block -/
structure Block :=
  (material : Fin 3)
  (size : Fin 3)
  (color : Fin 4)
  (shape : Fin 5)

/-- The set of all blocks -/
def AllBlocks : Finset Block := sorry

/-- The reference block (wood small blue hexagon) -/
def ReferenceBlock : Block := ⟨0, 0, 0, 1⟩

/-- Function to count differences between two blocks -/
def countDifferences (b1 b2 : Block) : Nat := sorry

/-- Theorem stating the number of blocks differing in exactly 2 ways -/
theorem blocks_differing_in_two_ways :
  (AllBlocks.filter (fun b => countDifferences b ReferenceBlock = 2)).card = 44 := by
  sorry

end NUMINAMATH_CALUDE_blocks_differing_in_two_ways_l3336_333651


namespace NUMINAMATH_CALUDE_rectangle_area_invariance_l3336_333699

theorem rectangle_area_invariance (x y : ℝ) :
  (x + 5/2) * (y - 2/3) = (x - 5/2) * (y + 4/3) ∧ 
  (x + 5/2) * (y - 2/3) = x * y →
  x * y = 20 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_invariance_l3336_333699


namespace NUMINAMATH_CALUDE_triangle_formation_l3336_333627

/-- Triangle inequality check for three sides -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if a set of three numbers can form a triangle -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 7 12 17 ∧
  ¬ can_form_triangle 3 3 7 ∧
  ¬ can_form_triangle 4 5 9 ∧
  ¬ can_form_triangle 5 8 15 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l3336_333627


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3336_333649

theorem quadratic_inequality_solution (a : ℝ) (x₁ x₂ : ℝ) : 
  a > 0 → 
  (∀ x, x^2 - 2*a*x - 8*a^2 < 0 ↔ x₁ < x ∧ x < x₂) → 
  x₂ + x₁ = 15 → 
  a = 15/2 := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3336_333649


namespace NUMINAMATH_CALUDE_calculate_expression_l3336_333639

theorem calculate_expression : (18 / (3 + 9 - 6)) * 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l3336_333639


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3336_333693

def A : Set (ℝ × ℝ) := {p | p.2 = 2 * p.1 + 5}
def B : Set (ℝ × ℝ) := {p | p.2 = 1 - 2 * p.1}

theorem intersection_of_A_and_B : A ∩ B = {(-1, 3)} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3336_333693


namespace NUMINAMATH_CALUDE_average_monthly_increase_l3336_333617

/-- Represents the monthly growth rate as a real number between 0 and 1 -/
def monthly_growth_rate : ℝ := sorry

/-- The initial turnover in January in millions of yuan -/
def initial_turnover : ℝ := 2

/-- The turnover in March in millions of yuan -/
def march_turnover : ℝ := 2.88

/-- The number of months between January and March -/
def months_passed : ℕ := 2

theorem average_monthly_increase :
  initial_turnover * (1 + monthly_growth_rate) ^ months_passed = march_turnover ∧
  monthly_growth_rate = 0.2 := by sorry

end NUMINAMATH_CALUDE_average_monthly_increase_l3336_333617


namespace NUMINAMATH_CALUDE_linear_system_solution_l3336_333667

theorem linear_system_solution (x₁ x₂ x₃ x₄ : ℝ) : 
  (x₁ - 2*x₂ + x₄ = -3 ∧
   3*x₁ - x₂ - 2*x₃ = 1 ∧
   2*x₁ + x₂ - 2*x₃ - x₄ = 4 ∧
   x₁ + 3*x₂ - 2*x₃ - 2*x₄ = 7) →
  (∃ t u : ℝ, x₁ = -3 + 2*x₂ - x₄ ∧
              x₂ = 2 + (2/5)*t + (3/5)*u ∧
              x₃ = t ∧
              x₄ = u) :=
by sorry

end NUMINAMATH_CALUDE_linear_system_solution_l3336_333667


namespace NUMINAMATH_CALUDE_exam_maximum_marks_l3336_333644

theorem exam_maximum_marks (percentage : ℝ) (scored_marks : ℝ) (max_marks : ℝ) : 
  percentage = 92 / 100 → 
  scored_marks = 460 → 
  percentage * max_marks = scored_marks → 
  max_marks = 500 := by
sorry

end NUMINAMATH_CALUDE_exam_maximum_marks_l3336_333644


namespace NUMINAMATH_CALUDE_semicircle_circumference_from_rectangle_perimeter_l3336_333621

def rectangle_length : ℝ := 16
def rectangle_breadth : ℝ := 14

theorem semicircle_circumference_from_rectangle_perimeter :
  let rectangle_perimeter := 2 * (rectangle_length + rectangle_breadth)
  let square_side := rectangle_perimeter / 4
  let semicircle_circumference := (π * square_side) / 2 + square_side
  ∃ ε > 0, |semicircle_circumference - 38.55| < ε := by
  sorry

end NUMINAMATH_CALUDE_semicircle_circumference_from_rectangle_perimeter_l3336_333621


namespace NUMINAMATH_CALUDE_initial_deficit_calculation_l3336_333663

/-- Represents the score difference at the start of the final quarter -/
def initial_deficit : ℤ := sorry

/-- Liz's free throw points -/
def free_throw_points : ℕ := 5

/-- Liz's three-pointer points -/
def three_pointer_points : ℕ := 9

/-- Liz's jump shot points -/
def jump_shot_points : ℕ := 8

/-- Other team's points in the final quarter -/
def other_team_points : ℕ := 10

/-- Final score difference (negative means Liz's team lost) -/
def final_score_difference : ℤ := -8

theorem initial_deficit_calculation :
  initial_deficit = 20 :=
by sorry

end NUMINAMATH_CALUDE_initial_deficit_calculation_l3336_333663


namespace NUMINAMATH_CALUDE_final_student_count_l3336_333674

theorem final_student_count (initial_students : ℕ) (students_left : ℕ) (new_students : ℕ)
  (h1 : initial_students = 33)
  (h2 : students_left = 18)
  (h3 : new_students = 14) :
  initial_students - students_left + new_students = 29 := by
  sorry

end NUMINAMATH_CALUDE_final_student_count_l3336_333674


namespace NUMINAMATH_CALUDE_overlap_length_correct_l3336_333610

/-- The length of each overlapping segment in centimeters -/
def x : ℝ := 2.5

/-- The total length of all red segments (including overlaps) in centimeters -/
def total_length : ℝ := 98

/-- The actual distance from edge to edge (without overlaps) in centimeters -/
def actual_distance : ℝ := 83

/-- The number of overlapping regions -/
def num_overlaps : ℕ := 6

/-- Theorem stating that x is the correct length of each overlapping segment -/
theorem overlap_length_correct : x = (total_length - actual_distance) / num_overlaps := by
  sorry

end NUMINAMATH_CALUDE_overlap_length_correct_l3336_333610


namespace NUMINAMATH_CALUDE_vikki_earnings_insurance_deduction_l3336_333660

/-- Vikki's weekly earnings and deductions -/
def weekly_earnings_problem (hours_worked : ℕ) (hourly_rate : ℚ) (tax_rate : ℚ) 
  (union_dues : ℚ) (take_home_pay : ℚ) : Prop :=
  let gross_earnings := hours_worked * hourly_rate
  let tax_deduction := gross_earnings * tax_rate
  let after_tax_and_dues := gross_earnings - tax_deduction - union_dues
  let insurance_deduction := after_tax_and_dues - take_home_pay
  let insurance_percentage := insurance_deduction / gross_earnings * 100
  insurance_percentage = 5

theorem vikki_earnings_insurance_deduction :
  weekly_earnings_problem 42 10 (1/5) 5 310 :=
sorry

end NUMINAMATH_CALUDE_vikki_earnings_insurance_deduction_l3336_333660


namespace NUMINAMATH_CALUDE_team_formation_proof_l3336_333634

def number_of_teams (total_girls : ℕ) (total_boys : ℕ) (team_girls : ℕ) (team_boys : ℕ) (mandatory_girl : ℕ) : ℕ :=
  Nat.choose (total_girls - mandatory_girl) (team_girls - mandatory_girl) * Nat.choose total_boys team_boys

theorem team_formation_proof :
  let total_girls : ℕ := 5
  let total_boys : ℕ := 7
  let team_girls : ℕ := 2
  let team_boys : ℕ := 2
  let mandatory_girl : ℕ := 1
  number_of_teams total_girls total_boys team_girls team_boys mandatory_girl = 84 :=
by
  sorry

end NUMINAMATH_CALUDE_team_formation_proof_l3336_333634


namespace NUMINAMATH_CALUDE_hyperbola_properties_l3336_333676

/-- Represents a hyperbola with foci on the x-axis -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  e : ℝ

/-- The equation of the hyperbola -/
def hyperbola_equation (h : Hyperbola) (x y : ℝ) : Prop :=
  x^2 / h.a^2 - y^2 / h.b^2 = 1

/-- A line that intersects the hyperbola -/
def intersecting_line (m : ℝ) (x y : ℝ) : Prop :=
  x - y + m = 0

/-- Two points are perpendicular from the origin -/
def perpendicular_from_origin (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  x₁ * x₂ + y₁ * y₂ = 0

theorem hyperbola_properties (h : Hyperbola) (m : ℝ) :
  h.e = Real.sqrt 3 →
  h.b = Real.sqrt 2 →
  h.e = (Real.sqrt (h.a^2 + h.b^2)) / h.a →
  (∀ x y, hyperbola_equation h x y ↔ x^2 - y^2 / 2 = 1) ∧
  (∃ x₁ y₁ x₂ y₂, 
    hyperbola_equation h x₁ y₁ ∧
    hyperbola_equation h x₂ y₂ ∧
    intersecting_line m x₁ y₁ ∧
    intersecting_line m x₂ y₂ ∧
    perpendicular_from_origin x₁ y₁ x₂ y₂ →
    m = 2 ∨ m = -2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l3336_333676


namespace NUMINAMATH_CALUDE_statues_painted_l3336_333608

theorem statues_painted (total_paint : ℚ) (paint_per_statue : ℚ) :
  total_paint = 7/8 ∧ paint_per_statue = 1/8 → total_paint / paint_per_statue = 7 := by
  sorry

end NUMINAMATH_CALUDE_statues_painted_l3336_333608


namespace NUMINAMATH_CALUDE_strongest_teams_in_tournament_l3336_333620

/-- Represents a volleyball team in the tournament -/
structure Team :=
  (name : String)
  (wins : Nat)
  (losses : Nat)

/-- Represents the tournament results -/
structure TournamentResults :=
  (teams : List Team)
  (numTeams : Nat)
  (roundRobin : Bool)
  (bestOfThree : Bool)

/-- Determines if a team is one of the two strongest teams -/
def isStrongestTeam (team : Team) (results : TournamentResults) : Prop :=
  ∃ (otherTeam : Team),
    otherTeam ∈ results.teams ∧
    team ∈ results.teams ∧
    team ≠ otherTeam ∧
    ∀ (t : Team), t ∈ results.teams →
      (t.wins < team.wins ∨ (t.wins = team.wins ∧ t.losses ≥ team.losses)) ∨
      (t.wins < otherTeam.wins ∨ (t.wins = otherTeam.wins ∧ t.losses ≥ otherTeam.losses))

theorem strongest_teams_in_tournament
  (results : TournamentResults)
  (h1 : results.numTeams = 6)
  (h2 : results.roundRobin = true)
  (h3 : results.bestOfThree = true)
  (first : Team)
  (second : Team)
  (fourth : Team)
  (fifth : Team)
  (sixth : Team)
  (h4 : first ∈ results.teams ∧ first.wins = 2 ∧ first.losses = 3)
  (h5 : second ∈ results.teams ∧ second.wins = 4 ∧ second.losses = 1)
  (h6 : fourth ∈ results.teams ∧ fourth.wins = 0 ∧ fourth.losses = 5)
  (h7 : fifth ∈ results.teams ∧ fifth.wins = 4 ∧ fifth.losses = 1)
  (h8 : sixth ∈ results.teams ∧ sixth.wins = 4 ∧ sixth.losses = 1)
  : isStrongestTeam fifth results ∧ isStrongestTeam sixth results :=
sorry

end NUMINAMATH_CALUDE_strongest_teams_in_tournament_l3336_333620


namespace NUMINAMATH_CALUDE_fraction_decomposition_l3336_333683

-- Define the polynomials
def p (x : ℝ) : ℝ := 6 * x - 15
def q (x : ℝ) : ℝ := 3 * x^3 - 2 * x^2 - 5 * x + 6
def r (x : ℝ) : ℝ := x - 1
def s (x : ℝ) : ℝ := 3 * x^2 - x - 6

-- Define the equality condition
def equality_condition (A B : ℝ → ℝ) : Prop :=
  ∀ x, x ≠ 1 ∧ s x ≠ 0 → p x / q x = A x / r x + B x / s x

-- Theorem statement
theorem fraction_decomposition :
  ∀ A B, equality_condition A B →
    (∀ x, A x = 0) ∧ (∀ x, B x = 6 * x - 15) :=
sorry

end NUMINAMATH_CALUDE_fraction_decomposition_l3336_333683


namespace NUMINAMATH_CALUDE_science_fair_teams_l3336_333623

theorem science_fair_teams (total_students : Nat) (red_hats : Nat) (green_hats : Nat) 
  (total_teams : Nat) (red_red_teams : Nat) : 
  total_students = 144 →
  red_hats = 63 →
  green_hats = 81 →
  total_teams = 72 →
  red_red_teams = 28 →
  red_hats + green_hats = total_students →
  ∃ (green_green_teams : Nat), green_green_teams = 37 ∧ 
    green_green_teams + red_red_teams + (total_students - 2 * red_red_teams - 2 * green_green_teams) / 2 = total_teams :=
by
  sorry

end NUMINAMATH_CALUDE_science_fair_teams_l3336_333623


namespace NUMINAMATH_CALUDE_rectangle_area_error_l3336_333670

/-- The percentage error in the calculated area of a rectangle when its sides are measured with errors -/
theorem rectangle_area_error (x y : ℝ) : 
  let actual_area := (a : ℝ) * (b : ℝ)
  let measured_side1 := a * (1 + x / 100)
  let measured_side2 := b * (1 + y / 100)
  let calculated_area := measured_side1 * measured_side2
  let percentage_error := (calculated_area - actual_area) / actual_area * 100
  percentage_error = x + y + (x * y / 100) :=
by sorry


end NUMINAMATH_CALUDE_rectangle_area_error_l3336_333670


namespace NUMINAMATH_CALUDE_parallelogram_intersection_ratio_l3336_333619

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (W : Point) (X : Point) (Y : Point) (Z : Point)

/-- The theorem statement -/
theorem parallelogram_intersection_ratio 
  (WXYZ : Parallelogram) 
  (M : Point) (N : Point) (P : Point) :
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ M = Point.mk ((1 - t) * WXYZ.W.x + t * WXYZ.Z.x) ((1 - t) * WXYZ.W.y + t * WXYZ.Z.y) ∧ t = 3/100) →
  (∃ s : ℝ, s ∈ Set.Icc 0 1 ∧ N = Point.mk ((1 - s) * WXYZ.W.x + s * WXYZ.Y.x) ((1 - s) * WXYZ.W.y + s * WXYZ.Y.y) ∧ s = 3/251) →
  (∃ r : ℝ, r ∈ Set.Icc 0 1 ∧ P = Point.mk ((1 - r) * WXYZ.W.x + r * WXYZ.Y.x) ((1 - r) * WXYZ.W.y + r * WXYZ.Y.y)) →
  (∃ q : ℝ, q ∈ Set.Icc 0 1 ∧ P = Point.mk ((1 - q) * M.x + q * N.x) ((1 - q) * M.y + q * N.y)) →
  (WXYZ.Y.x - WXYZ.W.x) / (P.x - WXYZ.W.x) = 2 ∧ (WXYZ.Y.y - WXYZ.W.y) / (P.y - WXYZ.W.y) = 2 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_intersection_ratio_l3336_333619


namespace NUMINAMATH_CALUDE_point_movement_l3336_333645

/-- Given a point P in a Cartesian coordinate system, moving it upwards and to the left results in the expected new coordinates. -/
theorem point_movement (x y dx dy : ℤ) :
  let P : ℤ × ℤ := (x, y)
  let P' : ℤ × ℤ := (x - dx, y + dy)
  (P = (-2, 5) ∧ dx = 1 ∧ dy = 3) → P' = (-3, 8) := by
  sorry

#check point_movement

end NUMINAMATH_CALUDE_point_movement_l3336_333645


namespace NUMINAMATH_CALUDE_odd_periodic_function_properties_l3336_333612

-- Define an odd function f that satisfies f(x+1) = f(x-1)
def OddPeriodicFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ (∀ x, f (x + 1) = f (x - 1))

-- Theorem stating the two correct properties
theorem odd_periodic_function_properties (f : ℝ → ℝ) (h : OddPeriodicFunction f) :
  (∀ x, f (x + 2) = f x) ∧ (∀ k : ℤ, ∀ x, f (k + x) = -f (k - x)) := by
  sorry


end NUMINAMATH_CALUDE_odd_periodic_function_properties_l3336_333612


namespace NUMINAMATH_CALUDE_system_of_equations_solution_l3336_333669

theorem system_of_equations_solution (a b c : ℝ) :
  ∃ (x y z : ℝ), 
    (x + y + 2*z = a) ∧ 
    (x + 2*y + z = b) ∧ 
    (2*x + y + z = c) ∧
    (x = (3*c - a - b) / 4) ∧
    (y = (3*b - a - c) / 4) ∧
    (z = (3*a - b - c) / 4) := by
  sorry

end NUMINAMATH_CALUDE_system_of_equations_solution_l3336_333669


namespace NUMINAMATH_CALUDE_quadratic_range_solution_set_l3336_333636

/-- A quadratic function with real coefficients -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^2 + a*x + b

/-- The theorem stating the value of c given the conditions -/
theorem quadratic_range_solution_set (a b m : ℝ) :
  (∀ x, f a b x ≥ 0) →  -- range of f is [0, +∞)
  (∃ c, ∀ x, f a b x < c ↔ m < x ∧ x < m + 6) →  -- solution set of f(x) < c is (m, m+6)
  ∃ c, c = 9 := by sorry

end NUMINAMATH_CALUDE_quadratic_range_solution_set_l3336_333636


namespace NUMINAMATH_CALUDE_circle_intersection_existence_l3336_333648

theorem circle_intersection_existence :
  ∃ n : ℝ, 0 < n ∧ n < 2 ∧
  (∃ x y : ℝ, x^2 + y^2 - 2*n*x + 2*n*y + 2*n^2 - 8 = 0 ∧
              (x+1)^2 + (y-1)^2 = 2) ∧
  (∃ x' y' : ℝ, x' ≠ x ∨ y' ≠ y ∧
              x'^2 + y'^2 - 2*n*x' + 2*n*y' + 2*n^2 - 8 = 0 ∧
              (x'+1)^2 + (y'-1)^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_existence_l3336_333648


namespace NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l3336_333637

theorem consecutive_integers_product_812_sum_57 :
  ∀ x : ℕ, x > 0 ∧ x * (x + 1) = 812 → x + (x + 1) = 57 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_812_sum_57_l3336_333637


namespace NUMINAMATH_CALUDE_fraction_equality_l3336_333605

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0)
  (h : (x^2 + 4*x*y) / (y^2 - 4*x*y) = 3) :
  ∃ z : ℝ, (x^2 - 4*x*y) / (y^2 + 4*x*y) = z :=
by sorry

end NUMINAMATH_CALUDE_fraction_equality_l3336_333605


namespace NUMINAMATH_CALUDE_money_division_l3336_333642

theorem money_division (a b c : ℝ) : 
  a = (1/3) * (b + c) →
  b = (2/7) * (a + c) →
  a = b + 10 →
  a + b + c = 360 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l3336_333642


namespace NUMINAMATH_CALUDE_tangent_circles_locus_l3336_333650

-- Define the circles
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the tangency relation
def isTangent (c1 c2 : Circle) : Prop :=
  let d := Real.sqrt ((c1.center.1 - c2.center.1)^2 + (c1.center.2 - c2.center.2)^2)
  d = c1.radius + c2.radius ∨ d = |c1.radius - c2.radius|

-- Define the locus of points
inductive Locus
  | Hyperbola
  | StraightLine

-- Theorem statement
theorem tangent_circles_locus 
  (O₁ O₂ P : Circle) 
  (h_separate : O₁.center ≠ O₂.center) 
  (h_tangent₁ : isTangent O₁ P) 
  (h_tangent₂ : isTangent O₂ P) :
  (∃ l₁ l₂ : Locus, l₁ = Locus.Hyperbola ∧ l₂ = Locus.Hyperbola) ∨
  (∃ l₁ l₂ : Locus, l₁ = Locus.Hyperbola ∧ l₂ = Locus.StraightLine) :=
sorry

end NUMINAMATH_CALUDE_tangent_circles_locus_l3336_333650


namespace NUMINAMATH_CALUDE_smallest_three_digit_odd_multiple_of_three_l3336_333628

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def first_digit (n : ℕ) : ℕ := n / 100

theorem smallest_three_digit_odd_multiple_of_three :
  ∃ (n : ℕ), is_three_digit n ∧ 
             Odd (first_digit n) ∧ 
             n % 3 = 0 ∧
             (∀ m : ℕ, is_three_digit m ∧ Odd (first_digit m) ∧ m % 3 = 0 → n ≤ m) ∧
             n = 102 :=
sorry

end NUMINAMATH_CALUDE_smallest_three_digit_odd_multiple_of_three_l3336_333628


namespace NUMINAMATH_CALUDE_prism_volume_l3336_333633

/-- A right rectangular prism with face areas 12, 18, and 24 square inches has a volume of 72 cubic inches -/
theorem prism_volume (l w h : ℝ) (hl : l > 0) (hw : w > 0) (hh : h > 0)
  (area1 : l * w = 12) (area2 : w * h = 18) (area3 : l * h = 24) :
  l * w * h = 72 := by sorry

end NUMINAMATH_CALUDE_prism_volume_l3336_333633


namespace NUMINAMATH_CALUDE_expression_value_l3336_333653

theorem expression_value (x y : ℝ) (h : y = Real.sqrt (x - 2) + Real.sqrt (2 - x) + 1) :
  (Real.sqrt (48 * y) + Real.sqrt (8 * x)) * (4 * Real.sqrt (3 * y) - 2 * Real.sqrt (2 * x)) - x * y = 30 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l3336_333653


namespace NUMINAMATH_CALUDE_doughnut_sharing_l3336_333691

/-- The number of doughnuts in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens Samuel bought -/
def samuel_dozens : ℕ := 2

/-- The number of dozens Cathy bought -/
def cathy_dozens : ℕ := 3

/-- The number of doughnuts each person received -/
def doughnuts_per_person : ℕ := 6

/-- The number of people who bought the doughnuts (Samuel and Cathy) -/
def buyers : ℕ := 2

theorem doughnut_sharing :
  let total_doughnuts := samuel_dozens * dozen + cathy_dozens * dozen
  let total_people := total_doughnuts / doughnuts_per_person
  total_people - buyers = 8 := by
  sorry

end NUMINAMATH_CALUDE_doughnut_sharing_l3336_333691


namespace NUMINAMATH_CALUDE_points_per_bag_l3336_333632

theorem points_per_bag (total_bags : ℕ) (unrecycled_bags : ℕ) (total_points : ℕ) : 
  total_bags = 11 → unrecycled_bags = 2 → total_points = 45 → 
  (total_points / (total_bags - unrecycled_bags) : ℚ) = 5 := by
  sorry

end NUMINAMATH_CALUDE_points_per_bag_l3336_333632


namespace NUMINAMATH_CALUDE_male_students_count_l3336_333618

/-- Given a school with a total of 1200 students, where a sample of 200 students
    contains 85 females, prove that the number of male students in the school is 690. -/
theorem male_students_count (total : ℕ) (sample : ℕ) (females_in_sample : ℕ)
    (h1 : total = 1200)
    (h2 : sample = 200)
    (h3 : females_in_sample = 85) :
    total - (females_in_sample * (total / sample)) = 690 := by
  sorry

end NUMINAMATH_CALUDE_male_students_count_l3336_333618


namespace NUMINAMATH_CALUDE_rectangle_area_l3336_333629

/-- Given a rectangle composed of 24 congruent squares arranged in a 6x4 format
    with a diagonal of 10 cm, the total area of the rectangle is 2400/13 square cm. -/
theorem rectangle_area (squares : ℕ) (rows cols : ℕ) (diagonal : ℝ) :
  squares = 24 →
  rows = 6 →
  cols = 4 →
  diagonal = 10 →
  (rows * cols : ℝ) * (diagonal^2 / (rows^2 + cols^2)) = 2400 / 13 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3336_333629


namespace NUMINAMATH_CALUDE_geometric_sequence_special_case_l3336_333646

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- The theorem statement -/
theorem geometric_sequence_special_case (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 6 * a 6 - 8 * a 6 + 4 = 0) →
  (a 10 * a 10 - 8 * a 10 + 4 = 0) →
  (a 8 = 2 ∨ a 8 = -2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_special_case_l3336_333646


namespace NUMINAMATH_CALUDE_spinner_probability_l3336_333601

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_C = 1/6 → p_A + p_B + p_C + p_D = 1 → p_D = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3336_333601


namespace NUMINAMATH_CALUDE_absolute_difference_l3336_333652

theorem absolute_difference (m n : ℝ) (h1 : m * n = 6) (h2 : m + n = 10) : |m - n| = 2 * Real.sqrt 19 := by
  sorry

end NUMINAMATH_CALUDE_absolute_difference_l3336_333652


namespace NUMINAMATH_CALUDE_number_not_perfect_square_l3336_333604

theorem number_not_perfect_square (n : ℕ) 
  (h : ∃ k, n = 6 * (10^600 - 1) / 9 + k * 10^600 ∧ k ≥ 0) : 
  ¬∃ m : ℕ, n = m^2 := by
sorry

end NUMINAMATH_CALUDE_number_not_perfect_square_l3336_333604


namespace NUMINAMATH_CALUDE_quadratic_independent_of_x_squared_l3336_333665

/-- For a quadratic polynomial -3x^2 + mx^2 - x + 3, if its value is independent of the quadratic term of x, then m = 3 -/
theorem quadratic_independent_of_x_squared (m : ℝ) : 
  (∀ x : ℝ, ∃ k : ℝ, -3*x^2 + m*x^2 - x + 3 = -x + k) → m = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_independent_of_x_squared_l3336_333665


namespace NUMINAMATH_CALUDE_powderman_distance_approximation_l3336_333625

/-- The speed of the powderman in yards per second -/
def powderman_speed : ℝ := 8

/-- The time in seconds when the powderman hears the blast -/
def time_of_hearing : ℝ := 30.68

/-- The distance the powderman runs in yards -/
def distance_run : ℝ := powderman_speed * time_of_hearing

theorem powderman_distance_approximation :
  ∃ ε > 0, abs (distance_run - 245) < ε := by sorry

end NUMINAMATH_CALUDE_powderman_distance_approximation_l3336_333625


namespace NUMINAMATH_CALUDE_sock_probability_theorem_l3336_333656

def gray_socks : ℕ := 12
def white_socks : ℕ := 10
def blue_socks : ℕ := 6

def total_socks : ℕ := gray_socks + white_socks + blue_socks

def probability_matching_or_different_colors : ℚ :=
  let total_combinations := Nat.choose total_socks 3
  let matching_gray := Nat.choose gray_socks 2 * (white_socks + blue_socks)
  let matching_white := Nat.choose white_socks 2 * (gray_socks + blue_socks)
  let matching_blue := Nat.choose blue_socks 2 * (gray_socks + white_socks)
  let all_different := gray_socks * white_socks * blue_socks
  let favorable_outcomes := matching_gray + matching_white + matching_blue + all_different
  (favorable_outcomes : ℚ) / total_combinations

theorem sock_probability_theorem :
  probability_matching_or_different_colors = 81 / 91 :=
by sorry

end NUMINAMATH_CALUDE_sock_probability_theorem_l3336_333656


namespace NUMINAMATH_CALUDE_range_of_expression_l3336_333677

theorem range_of_expression (a b c : ℝ) 
  (h1 : -3 < b) (h2 : b < a) (h3 : a < -1) 
  (h4 : -2 < c) (h5 : c < -1) : 
  0 < (a - b) * c^2 ∧ (a - b) * c^2 < 8 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expression_l3336_333677


namespace NUMINAMATH_CALUDE_min_value_x_plus_y_l3336_333686

theorem min_value_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 9/y = 2) :
  ∀ a b : ℝ, a > 0 → b > 0 → 1/a + 9/b = 2 → x + y ≤ a + b ∧ ∃ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/x + 9/y = 2 ∧ x + y = 8 :=
sorry

end NUMINAMATH_CALUDE_min_value_x_plus_y_l3336_333686


namespace NUMINAMATH_CALUDE_second_smallest_dimension_is_twelve_l3336_333624

/-- Represents the dimensions of a rectangular crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a cylindrical pillar -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder can fit upright in a crate -/
def cylinderFitsInCrate (c : Cylinder) (d : CrateDimensions) : Prop :=
  (2 * c.radius ≤ d.length ∧ 2 * c.radius ≤ d.width) ∨
  (2 * c.radius ≤ d.length ∧ 2 * c.radius ≤ d.height) ∨
  (2 * c.radius ≤ d.width ∧ 2 * c.radius ≤ d.height)

/-- The theorem stating that the second smallest dimension of the crate is 12 feet -/
theorem second_smallest_dimension_is_twelve
  (d : CrateDimensions)
  (h1 : d.length = 6)
  (h2 : d.height = 12)
  (h3 : d.width > 0)
  (c : Cylinder)
  (h4 : c.radius = 6)
  (h5 : cylinderFitsInCrate c d) :
  d.width = 12 ∨ d.width = 12 :=
sorry

end NUMINAMATH_CALUDE_second_smallest_dimension_is_twelve_l3336_333624


namespace NUMINAMATH_CALUDE_vector_problem_l3336_333688

/-- Given two vectors a and b in ℝ², proves that if a is collinear with b and their dot product is -10, then b is equal to (-4, 2) -/
theorem vector_problem (a b : ℝ × ℝ) : 
  a = (2, -1) → 
  (∃ k : ℝ, b = k • a) → 
  a.1 * b.1 + a.2 * b.2 = -10 → 
  b = (-4, 2) := by
sorry

end NUMINAMATH_CALUDE_vector_problem_l3336_333688


namespace NUMINAMATH_CALUDE_expression_evaluation_l3336_333631

theorem expression_evaluation : 
  (Real.sqrt 3 - 4 * Real.sin (20 * π / 180) + 8 * (Real.sin (20 * π / 180))^3) / 
  (2 * Real.sin (20 * π / 180) * Real.sin (480 * π / 180)) = 
  2 * Real.sqrt 3 / 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3336_333631


namespace NUMINAMATH_CALUDE_root_exists_in_interval_l3336_333622

-- Define the function f(x) = x³ + x - 3
def f (x : ℝ) : ℝ := x^3 + x - 3

-- State the theorem
theorem root_exists_in_interval :
  ∃ x ∈ Set.Ioo 1 2, f x = 0 :=
by
  -- The proof would go here, but we're using sorry as instructed
  sorry

end NUMINAMATH_CALUDE_root_exists_in_interval_l3336_333622


namespace NUMINAMATH_CALUDE_modulus_of_complex_l3336_333609

theorem modulus_of_complex : Complex.abs (7/4 - 3*I) = (Real.sqrt 193)/4 := by sorry

end NUMINAMATH_CALUDE_modulus_of_complex_l3336_333609


namespace NUMINAMATH_CALUDE_december_sales_multiple_l3336_333684

/-- Represents the sales data for a department store --/
structure SalesData where
  /-- Average monthly sales from January to November --/
  avg_sales : ℝ
  /-- Multiple of average sales for December --/
  dec_multiple : ℝ

/-- Theorem stating the conditions and the result to be proved --/
theorem december_sales_multiple (data : SalesData) :
  (data.dec_multiple * data.avg_sales) / (11 * data.avg_sales + data.dec_multiple * data.avg_sales) = 0.35294117647058826 →
  data.dec_multiple = 6 := by
  sorry

end NUMINAMATH_CALUDE_december_sales_multiple_l3336_333684
