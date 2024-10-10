import Mathlib

namespace complex_modulus_l3601_360124

theorem complex_modulus (z : ℂ) (h : z * Complex.I = -2 - Complex.I) : Complex.abs z = Real.sqrt 5 := by
  sorry

end complex_modulus_l3601_360124


namespace set_operations_l3601_360188

def A : Set ℕ := {x | x > 0 ∧ x < 9}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

theorem set_operations :
  (A ∩ B = {1, 2, 3}) ∧
  (A ∩ C = {3, 4, 5, 6}) ∧
  (A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6}) ∧
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8}) := by
sorry

end set_operations_l3601_360188


namespace min_value_expression_l3601_360127

theorem min_value_expression (x y : ℝ) 
  (h : 4 - 16 * x^2 - 8 * x * y - y^2 > 0) : 
  (13 * x^2 + 24 * x * y + 13 * y^2 - 14 * x - 16 * y + 61) / 
  (4 - 16 * x^2 - 8 * x * y - y^2)^(7/2) ≥ 7/16 := by
  sorry

end min_value_expression_l3601_360127


namespace systematic_sampling_fifth_segment_l3601_360177

theorem systematic_sampling_fifth_segment 
  (total_students : ℕ) 
  (selected_students : ℕ) 
  (second_segment_student : ℕ) 
  (h1 : total_students = 700) 
  (h2 : selected_students = 50) 
  (h3 : second_segment_student = 20) :
  let interval := total_students / selected_students
  let first_student := second_segment_student - interval
  let fifth_segment_student := first_student + 4 * interval
  fifth_segment_student = 62 := by
sorry


end systematic_sampling_fifth_segment_l3601_360177


namespace disjoint_sets_imply_a_values_l3601_360168

-- Define the sets A and B
def A (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) = a + 1}

def B (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (a^2 - 1) * p.1 + (a - 1) * p.2 = 15}

-- State the theorem
theorem disjoint_sets_imply_a_values (a : ℝ) :
  A a ∩ B a = ∅ → a = 1 ∨ a = -1 ∨ a = 5/2 ∨ a = -4 :=
by sorry

end disjoint_sets_imply_a_values_l3601_360168


namespace correct_matching_probability_l3601_360125

def num_celebrities : ℕ := 3
def num_baby_photos : ℕ := 3

theorem correct_matching_probability :
  let total_arrangements := Nat.factorial num_celebrities
  let correct_arrangements := 1
  (correct_arrangements : ℚ) / total_arrangements = 1 / 6 := by
  sorry

end correct_matching_probability_l3601_360125


namespace max_volume_cross_section_area_l3601_360137

/-- Sphere with radius 2 -/
def Sphere : Type := Unit

/-- Points on the surface of the sphere -/
def Point : Type := Unit

/-- Angle between two points and the center of the sphere -/
def angle (p q : Point) : ℝ := sorry

/-- Volume of the triangular pyramid formed by three points and the center of the sphere -/
def pyramidVolume (a b c : Point) : ℝ := sorry

/-- Area of the circular cross-section formed by a plane through three points on the sphere -/
def crossSectionArea (a b c : Point) : ℝ := sorry

/-- The theorem statement -/
theorem max_volume_cross_section_area (o : Sphere) (a b c : Point) :
  (∀ (p q : Point), angle p q = angle a b) →
  (∀ (p q r : Point), pyramidVolume p q r ≤ pyramidVolume a b c) →
  crossSectionArea a b c = 8 * Real.pi / 3 := by sorry

end max_volume_cross_section_area_l3601_360137


namespace georgie_initial_avocados_l3601_360108

/-- The number of avocados needed per serving of guacamole -/
def avocados_per_serving : ℕ := 3

/-- The number of avocados Georgie's sister buys -/
def sister_bought : ℕ := 4

/-- The number of servings Georgie can make -/
def servings : ℕ := 3

/-- Georgie's initial number of avocados -/
def initial_avocados : ℕ := servings * avocados_per_serving - sister_bought

theorem georgie_initial_avocados : initial_avocados = 5 := by
  sorry

end georgie_initial_avocados_l3601_360108


namespace min_value_of_expression_l3601_360111

theorem min_value_of_expression (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  let A := (a^2 + b^2)^4 / (c*d)^4 + (b^2 + c^2)^4 / (a*d)^4 + (c^2 + d^2)^4 / (a*b)^4 + (d^2 + a^2)^4 / (b*c)^4
  A ≥ 64 ∧ (A = 64 ↔ a = b ∧ b = c ∧ c = d) :=
by sorry

end min_value_of_expression_l3601_360111


namespace condition_property_l3601_360131

theorem condition_property (a b : ℝ) :
  (∀ a b, a > b ∧ b > 0 → a + a^2 > b + b^2) ∧
  (∃ a b, a + a^2 > b + b^2 ∧ ¬(a > b ∧ b > 0)) :=
by sorry

end condition_property_l3601_360131


namespace stock_value_change_l3601_360190

theorem stock_value_change (initial_value : ℝ) (h : initial_value > 0) :
  let day1_value := initial_value * (1 - 0.25)
  let day2_value := day1_value * (1 + 0.40)
  day2_value = initial_value * 1.05 := by
    sorry

end stock_value_change_l3601_360190


namespace base_eight_47_equals_39_l3601_360183

/-- Converts a two-digit base-eight number to base-ten -/
def base_eight_to_ten (a b : Nat) : Nat :=
  a * 8 + b

/-- The base-eight number 47 is equal to the base-ten number 39 -/
theorem base_eight_47_equals_39 : base_eight_to_ten 4 7 = 39 := by
  sorry

end base_eight_47_equals_39_l3601_360183


namespace detergent_amount_in_new_solution_l3601_360102

/-- Represents a solution with bleach, detergent, and water -/
structure Solution where
  bleach : ℝ
  detergent : ℝ
  water : ℝ

/-- The original ratio of the solution -/
def original_ratio : Solution :=
  { bleach := 2, detergent := 40, water := 100 }

/-- The new ratio after adjustments -/
def new_ratio (s : Solution) : Solution :=
  { bleach := 3 * s.bleach,
    detergent := s.detergent,
    water := 2 * s.water }

/-- The theorem stating the amount of detergent in the new solution -/
theorem detergent_amount_in_new_solution :
  let s := new_ratio original_ratio
  let water_amount := 300
  let detergent_amount := (s.detergent / s.water) * water_amount
  detergent_amount = 120 := by sorry

end detergent_amount_in_new_solution_l3601_360102


namespace bus_driver_compensation_l3601_360149

/-- Calculates the total compensation for a bus driver given their work hours and pay rates. -/
def calculate_compensation (regular_rate : ℚ) (overtime_multiplier : ℚ) (total_hours : ℕ) (regular_hours : ℕ) : ℚ :=
  let overtime_rate := regular_rate * (1 + overtime_multiplier)
  let regular_pay := regular_rate * regular_hours
  let overtime_pay := overtime_rate * (total_hours - regular_hours)
  regular_pay + overtime_pay

/-- Theorem stating that the bus driver's compensation for 60 hours of work is $1200. -/
theorem bus_driver_compensation :
  calculate_compensation 16 0.75 60 40 = 1200 := by
  sorry

end bus_driver_compensation_l3601_360149


namespace problem_1_problem_2_l3601_360182

-- Problem 1
theorem problem_1 : -3 + (-2) * 5 - (-3) = -10 := by sorry

-- Problem 2
theorem problem_2 : -1^4 + ((-5)^2 - 3) / |(-2)| = 10 := by sorry

end problem_1_problem_2_l3601_360182


namespace greatest_common_divisor_of_180_and_n_l3601_360135

theorem greatest_common_divisor_of_180_and_n (n : ℕ) : 
  (∃ (d₁ d₂ d₃ : ℕ), d₁ < d₂ ∧ d₂ < d₃ ∧ 
   {d : ℕ | d ∣ 180 ∧ d ∣ n} = {d₁, d₂, d₃}) →
  (Nat.gcd 180 n = 9) :=
by sorry

end greatest_common_divisor_of_180_and_n_l3601_360135


namespace fraction_simplification_l3601_360164

theorem fraction_simplification (a b : ℝ) (h : b ≠ 0) :
  (20 * a^4 * b) / (120 * a^3 * b^2) = a / (6 * b) ∧
  (20 * 2^4 * 3) / (120 * 2^3 * 3^2) = 1 / 9 := by
  sorry

#check fraction_simplification

end fraction_simplification_l3601_360164


namespace expression_evaluation_l3601_360115

theorem expression_evaluation :
  (2^(2+1) - 2*(2-1)^(2+1))^2 = 36 := by
  sorry

end expression_evaluation_l3601_360115


namespace fixed_stable_points_range_l3601_360172

/-- The function f(x) = a x^2 - 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - 1

/-- The set of fixed points of f -/
def fixedPoints (a : ℝ) : Set ℝ := {x | f a x = x}

/-- The set of stable points of f -/
def stablePoints (a : ℝ) : Set ℝ := {x | f a (f a x) = x}

/-- Theorem stating the range of a for which the fixed points and stable points are equal and non-empty -/
theorem fixed_stable_points_range (a : ℝ) :
  (fixedPoints a = stablePoints a ∧ (fixedPoints a).Nonempty) ↔ -1/4 ≤ a ∧ a ≤ 3/4 :=
sorry

end fixed_stable_points_range_l3601_360172


namespace cat_ratio_l3601_360134

theorem cat_ratio (jacob_cats : ℕ) (melanie_cats : ℕ) :
  jacob_cats = 90 →
  melanie_cats = 60 →
  ∃ (annie_cats : ℕ),
    annie_cats = jacob_cats / 3 ∧
    melanie_cats = annie_cats ∧
    melanie_cats / annie_cats = 2 :=
by sorry

end cat_ratio_l3601_360134


namespace minimum_jars_needed_spice_jar_problem_l3601_360157

theorem minimum_jars_needed 
  (medium_jar_capacity : ℕ) 
  (large_container_capacity : ℕ) 
  (potential_loss : ℕ) : ℕ :=
  let min_jars := (large_container_capacity + medium_jar_capacity - 1) / medium_jar_capacity
  min_jars + potential_loss

theorem spice_jar_problem : 
  minimum_jars_needed 50 825 1 = 18 := by
  sorry

end minimum_jars_needed_spice_jar_problem_l3601_360157


namespace number_difference_l3601_360133

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 17402)
  (b_div_10 : 10 ∣ b)
  (a_eq_b_div_10 : a = b / 10) : 
  b - a = 14238 := by sorry

end number_difference_l3601_360133


namespace lcm_gcd_relation_l3601_360146

theorem lcm_gcd_relation (a b : ℕ) : 
  (Nat.lcm a b = Nat.gcd a b + 19) ↔ 
  ((a = 1 ∧ b = 20) ∨ (a = 20 ∧ b = 1) ∨ 
   (a = 4 ∧ b = 5) ∨ (a = 5 ∧ b = 4) ∨ 
   (a = 19 ∧ b = 38) ∨ (a = 38 ∧ b = 19)) :=
by sorry

end lcm_gcd_relation_l3601_360146


namespace complex_equation_solution_l3601_360199

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_equation_solution :
  ∀ z : ℂ, (1 - i) * z = 1 + i → z = i := by
  sorry

end complex_equation_solution_l3601_360199


namespace game_ends_in_53_rounds_l3601_360150

/-- Represents the state of the game at any given round -/
structure GameState :=
  (A B C D : ℕ)

/-- The initial state of the game -/
def initial_state : GameState :=
  ⟨16, 15, 14, 13⟩

/-- Function to update the game state after one round -/
def update_state (state : GameState) : GameState :=
  sorry

/-- Predicate to check if the game has ended -/
def game_ended (state : GameState) : Prop :=
  state.A = 0 ∨ state.B = 0 ∨ state.C = 0 ∨ state.D = 0

/-- The number of rounds the game lasts -/
def game_duration : ℕ := 53

theorem game_ends_in_53_rounds :
  ∃ (final_state : GameState),
    (game_duration.iterate update_state initial_state = final_state) ∧
    game_ended final_state ∧
    ∀ (n : ℕ), n < game_duration →
      ¬game_ended (n.iterate update_state initial_state) :=
  sorry

end game_ends_in_53_rounds_l3601_360150


namespace fraction_ordering_l3601_360118

theorem fraction_ordering : (12 : ℚ) / 35 < 10 / 29 ∧ 10 / 29 < 6 / 17 := by
  sorry

end fraction_ordering_l3601_360118


namespace lcm_of_fractions_l3601_360184

theorem lcm_of_fractions (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  lcm (1 / x) (lcm (1 / (x * y)) (1 / (x * y * z))) = 1 / (x * y * z) := by
  sorry

end lcm_of_fractions_l3601_360184


namespace rectangle_quadrilateral_area_l3601_360100

/-- Given a rectangle with sides 5 cm and 48 cm, where the longer side is divided into three equal parts
    and the midpoint of the shorter side is connected to the first division point on the longer side,
    the area of the resulting smaller quadrilateral is 90 cm². -/
theorem rectangle_quadrilateral_area :
  let short_side : ℝ := 5
  let long_side : ℝ := 48
  let division_point : ℝ := long_side / 3
  let midpoint : ℝ := short_side / 2
  let total_area : ℝ := short_side * long_side
  let part_area : ℝ := short_side * division_point
  let quadrilateral_area : ℝ := part_area + (part_area / 2)
  quadrilateral_area = 90
  := by sorry

end rectangle_quadrilateral_area_l3601_360100


namespace small_bottle_price_theorem_l3601_360180

/-- The price of a small bottle that results in the given average price -/
def price_small_bottle (large_quantity : ℕ) (small_quantity : ℕ) (large_price : ℚ) (average_price : ℚ) : ℚ :=
  ((large_quantity + small_quantity : ℚ) * average_price - large_quantity * large_price) / small_quantity

theorem small_bottle_price_theorem (large_quantity small_quantity : ℕ) (large_price average_price : ℚ) :
  large_quantity = 1365 →
  small_quantity = 720 →
  large_price = 189/100 →
  average_price = 173/100 →
  ∃ ε > 0, |price_small_bottle large_quantity small_quantity large_price average_price - 142/100| < ε :=
by sorry


end small_bottle_price_theorem_l3601_360180


namespace vector_sum_magnitude_l3601_360141

theorem vector_sum_magnitude (a b : ℝ × ℝ) :
  let angle := 60 * π / 180
  let mag_a := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let mag_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  mag_a = 1 ∧ mag_b = 2 ∧
  a.1 * b.1 + a.2 * b.2 = mag_a * mag_b * Real.cos angle →
  Real.sqrt (((a.1 + b.1) ^ 2) + ((a.2 + b.2) ^ 2)) = Real.sqrt 7 :=
by sorry

end vector_sum_magnitude_l3601_360141


namespace cone_sin_theta_l3601_360145

/-- Theorem: For a cone with base radius 5 and lateral area 65π, 
    if θ is the angle between the slant height and the height of the cone, 
    then sinθ = 5/13 -/
theorem cone_sin_theta (r : ℝ) (lat_area : ℝ) (θ : ℝ) 
    (h1 : r = 5) 
    (h2 : lat_area = 65 * Real.pi) 
    (h3 : θ = Real.arcsin (r / (lat_area / (2 * Real.pi * r)))) : 
  Real.sin θ = 5 / 13 := by
  sorry

end cone_sin_theta_l3601_360145


namespace area_enclosed_by_trajectory_l3601_360112

def f (x : ℝ) : ℝ := x^2 + 1

theorem area_enclosed_by_trajectory (a b : ℝ) (h1 : a < b) 
  (h2 : Set.range f = Set.Icc 1 5) 
  (h3 : Set.Icc a b = f⁻¹' (Set.Icc 1 5)) : 
  (b - a) * 1 = 4 := by sorry

end area_enclosed_by_trajectory_l3601_360112


namespace cost_per_serving_soup_l3601_360193

/-- Calculates the cost per serving of soup given ingredient quantities and prices -/
theorem cost_per_serving_soup (beef_quantity beef_price chicken_quantity chicken_price
                               carrot_quantity carrot_price potato_quantity potato_price
                               onion_quantity onion_price servings : ℚ) :
  beef_quantity = 4 →
  beef_price = 6 →
  chicken_quantity = 3 →
  chicken_price = 4 →
  carrot_quantity = 2 →
  carrot_price = (3/2) →
  potato_quantity = 3 →
  potato_price = 2 →
  onion_quantity = 1 →
  onion_price = 3 →
  servings = 12 →
  (beef_quantity * beef_price +
   chicken_quantity * chicken_price +
   carrot_quantity * carrot_price +
   potato_quantity * potato_price +
   onion_quantity * onion_price) / servings = 4 :=
by sorry

end cost_per_serving_soup_l3601_360193


namespace arithmetic_sequence_max_sum_l3601_360181

/-- An arithmetic sequence with its properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  is_arithmetic : ∀ n m : ℕ+, a (n + 1) - a n = a (m + 1) - a m

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

theorem arithmetic_sequence_max_sum (seq : ArithmeticSequence) :
  (seq.a 1 + seq.a 4 + seq.a 7 = 99) →
  (seq.a 2 + seq.a 5 + seq.a 8 = 93) →
  (∀ n : ℕ+, S seq n ≤ S seq 20) →
  (∀ k : ℕ+, (∀ n : ℕ+, S seq n ≤ S seq k) → k = 20) :=
by sorry

end arithmetic_sequence_max_sum_l3601_360181


namespace rehabilitation_centers_count_rehabilitation_centers_count_proof_l3601_360114

theorem rehabilitation_centers_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun lisa jude han jane =>
    lisa = 6 ∧
    jude = lisa / 2 ∧
    han = 2 * jude - 2 ∧
    jane = 2 * han + 6 →
    lisa + jude + han + jane = 27

#check rehabilitation_centers_count

-- The proof is omitted
theorem rehabilitation_centers_count_proof :
  ∃ (lisa jude han jane : ℕ),
    rehabilitation_centers_count lisa jude han jane :=
sorry

end rehabilitation_centers_count_rehabilitation_centers_count_proof_l3601_360114


namespace total_dolls_count_l3601_360153

/-- The number of dolls in a big box -/
def dolls_per_big_box : ℕ := 7

/-- The number of dolls in a small box -/
def dolls_per_small_box : ℕ := 4

/-- The number of big boxes -/
def num_big_boxes : ℕ := 5

/-- The number of small boxes -/
def num_small_boxes : ℕ := 9

/-- The total number of dolls in all boxes -/
def total_dolls : ℕ := dolls_per_big_box * num_big_boxes + dolls_per_small_box * num_small_boxes

theorem total_dolls_count : total_dolls = 71 := by
  sorry

end total_dolls_count_l3601_360153


namespace longest_tape_measure_l3601_360126

theorem longest_tape_measure (a b c : ℕ) 
  (ha : a = 600) (hb : b = 500) (hc : c = 1200) : 
  Nat.gcd a (Nat.gcd b c) = 100 := by
  sorry

end longest_tape_measure_l3601_360126


namespace odd_function_and_monotonicity_l3601_360194

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x^2 - 1) / (x + a)

theorem odd_function_and_monotonicity (a : ℝ) :
  (∀ x, f a x = -f a (-x)) →
  (a = 0 ∧ ∀ x y, 0 < x → x < y → f a x < f a y) := by
  sorry

end odd_function_and_monotonicity_l3601_360194


namespace sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3601_360148

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁ + x₂ = -b / a :=
by sorry

theorem sum_of_roots_specific_quadratic :
  let x₁ := (-(-10) + Real.sqrt ((-10)^2 - 4*1*36)) / (2*1)
  let x₂ := (-(-10) - Real.sqrt ((-10)^2 - 4*1*36)) / (2*1)
  x₁ + x₂ = 10 :=
by sorry

end sum_of_roots_quadratic_sum_of_roots_specific_quadratic_l3601_360148


namespace china_internet_users_scientific_notation_l3601_360120

/-- Represents the number of internet users in China in billions -/
def china_internet_users : ℝ := 1.067

/-- The scientific notation representation of the number of internet users -/
def scientific_notation : ℝ := 1.067 * (10 ^ 9)

theorem china_internet_users_scientific_notation :
  china_internet_users * (10 ^ 9) = scientific_notation := by sorry

end china_internet_users_scientific_notation_l3601_360120


namespace f_positive_implies_a_bounded_l3601_360136

/-- The function f(x) defined as x^2 - ax + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - a*x + 2

/-- Theorem stating that if f(x) > 0 for all x > 2, then a ≤ 3 -/
theorem f_positive_implies_a_bounded (a : ℝ) : 
  (∀ x > 2, f a x > 0) → a ≤ 3 := by
  sorry

end f_positive_implies_a_bounded_l3601_360136


namespace remainder_481207_div_8_l3601_360147

theorem remainder_481207_div_8 :
  ∃ q : ℕ, 481207 = 8 * q + 7 :=
by
  sorry

end remainder_481207_div_8_l3601_360147


namespace opposite_abs_equal_l3601_360169

theorem opposite_abs_equal (x : ℝ) : |x| = |-x| := by sorry

end opposite_abs_equal_l3601_360169


namespace function_range_l3601_360101

def f (x : ℝ) : ℝ := -x^2 + 3*x + 1

theorem function_range :
  ∃ (a b : ℝ), a = -3 ∧ b = 13/4 ∧
  (∀ x, x ∈ Set.Icc (-1) 2 → f x ∈ Set.Icc a b) ∧
  (∀ y ∈ Set.Icc a b, ∃ x ∈ Set.Icc (-1) 2, f x = y) :=
by sorry

end function_range_l3601_360101


namespace lizette_minerva_stamp_difference_l3601_360162

theorem lizette_minerva_stamp_difference :
  let lizette_stamps : ℕ := 813
  let minerva_stamps : ℕ := 688
  lizette_stamps > minerva_stamps →
  lizette_stamps - minerva_stamps = 125 := by
sorry

end lizette_minerva_stamp_difference_l3601_360162


namespace exponential_function_problem_l3601_360165

theorem exponential_function_problem (a : ℝ) (f : ℝ → ℝ) :
  a > 0 ∧ a ≠ 1 ∧ (∀ x, f x = a^x) ∧ f 3 = 8 →
  f (-1) = (1/2) := by
sorry

end exponential_function_problem_l3601_360165


namespace refrigerator_discount_proof_l3601_360138

/-- The original price of the refrigerator -/
def original_price : ℝ := 250.00

/-- The first discount rate -/
def first_discount : ℝ := 0.20

/-- The second discount rate -/
def second_discount : ℝ := 0.15

/-- The final price as a percentage of the original price -/
def final_percentage : ℝ := 0.68

theorem refrigerator_discount_proof :
  original_price * (1 - first_discount) * (1 - second_discount) = original_price * final_percentage :=
by sorry

end refrigerator_discount_proof_l3601_360138


namespace total_onions_is_fifteen_l3601_360143

/-- The number of onions grown by Nancy -/
def nancy_onions : ℕ := 2

/-- The number of onions grown by Dan -/
def dan_onions : ℕ := 9

/-- The number of onions grown by Mike -/
def mike_onions : ℕ := 4

/-- The number of days they worked on the farm -/
def days_worked : ℕ := 6

/-- The total number of onions grown by Nancy, Dan, and Mike -/
def total_onions : ℕ := nancy_onions + dan_onions + mike_onions

theorem total_onions_is_fifteen : total_onions = 15 := by sorry

end total_onions_is_fifteen_l3601_360143


namespace non_negative_for_all_non_negative_exists_l3601_360173

-- Define the function f
def f (x m : ℝ) : ℝ := x^2 - 2*x + m

-- Theorem for part (1)
theorem non_negative_for_all (m : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f x m ≥ 0) ↔ m ≥ 1 :=
sorry

-- Theorem for part (2)
theorem non_negative_exists (m : ℝ) :
  (∃ x ∈ Set.Icc 0 3, f x m ≥ 0) ↔ m ≥ -3 :=
sorry

end non_negative_for_all_non_negative_exists_l3601_360173


namespace box_volume_conversion_l3601_360170

/-- Proves that a box with a volume of 216 cubic feet has a volume of 8 cubic yards. -/
theorem box_volume_conversion (box_volume_cubic_feet : ℝ) 
  (h1 : box_volume_cubic_feet = 216) : 
  box_volume_cubic_feet / 27 = 8 := by
  sorry

end box_volume_conversion_l3601_360170


namespace locus_of_M_l3601_360129

-- Define the points A, B, and M
def A : ℝ × ℝ := (-1, 0)
def B : ℝ × ℝ := (2, 0)

-- Define the angle function
noncomputable def angle (p q r : ℝ × ℝ) : ℝ := sorry

-- Define the condition for point M
def satisfies_angle_condition (M : ℝ × ℝ) : Prop :=
  angle M B A = 2 * angle M A B

-- Define the locus conditions
def on_hyperbola (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  3 * x^2 - y^2 = 3 ∧ x > -1

def on_segment (M : ℝ × ℝ) : Prop :=
  let (x, y) := M
  y = 0 ∧ -1 < x ∧ x < 2

-- State the theorem
theorem locus_of_M (M : ℝ × ℝ) :
  satisfies_angle_condition M ↔ (on_hyperbola M ∨ on_segment M) :=
sorry

end locus_of_M_l3601_360129


namespace pencils_in_pack_judys_pencil_pack_l3601_360175

/-- Calculates the number of pencils in a pack given Judy's pencil usage and spending habits. -/
theorem pencils_in_pack (pencils_per_week : ℕ) (days_per_week : ℕ) (cost_per_pack : ℕ) 
  (total_spent : ℕ) (total_days : ℕ) : ℕ :=
  let pencils_used := pencils_per_week * (total_days / days_per_week)
  let packs_bought := total_spent / cost_per_pack
  pencils_used / packs_bought

/-- Proves that there are 30 pencils in a pack based on Judy's usage and spending. -/
theorem judys_pencil_pack : pencils_in_pack 10 5 4 12 45 = 30 := by
  sorry

end pencils_in_pack_judys_pencil_pack_l3601_360175


namespace expression_evaluation_l3601_360197

theorem expression_evaluation :
  let a : ℝ := Real.sqrt 3 - 3
  (3 - a) / (2 * a - 4) / (a + 2 - 5 / (a - 2)) = -Real.sqrt 3 / 6 := by
  sorry

end expression_evaluation_l3601_360197


namespace product_plus_one_square_l3601_360104

theorem product_plus_one_square (n : ℕ) :
  n * (n + 1) * (n + 2) * (n + 3) + 1 = (n^2 + 3*n + 1)^2 := by
  sorry

end product_plus_one_square_l3601_360104


namespace library_book_purchase_ratio_l3601_360179

theorem library_book_purchase_ratio :
  ∀ (initial_books last_year_purchase current_total : ℕ),
  initial_books = 100 →
  last_year_purchase = 50 →
  current_total = 300 →
  ∃ (this_year_purchase : ℕ),
    this_year_purchase = 3 * last_year_purchase ∧
    current_total = initial_books + last_year_purchase + this_year_purchase :=
by
  sorry

end library_book_purchase_ratio_l3601_360179


namespace fraction_subtraction_equality_l3601_360119

theorem fraction_subtraction_equality : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end fraction_subtraction_equality_l3601_360119


namespace circle_properties_l3601_360107

/-- The parabola to which the circle is tangent -/
def parabola (x y : ℝ) : Prop := y^2 = 5*x + 9

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := 2*x^2 - 10*x*y - 31*y^2 + 175*x - 6*y + 297 = 0

/-- Points through which the circle passes -/
def point_P : ℝ × ℝ := (0, 3)
def point_Q : ℝ × ℝ := (-1, -2)
def point_A : ℝ × ℝ := (-2, 1)

theorem circle_properties :
  (∀ x y : ℝ, circle_equation x y → ∃ r : ℝ, (x - 0)^2 + (y - 0)^2 = r^2) ∧
  (parabola (point_P.1) (point_P.2)) ∧
  (parabola (point_Q.1) (point_Q.2)) ∧
  (circle_equation (point_P.1) (point_P.2)) ∧
  (circle_equation (point_Q.1) (point_Q.2)) ∧
  (circle_equation (point_A.1) (point_A.2)) :=
sorry

end circle_properties_l3601_360107


namespace point_P_location_l3601_360151

-- Define the points on a line
structure Point :=
  (x : ℝ)

-- Define the distances
def OA (a : ℝ) : ℝ := a
def OB (b : ℝ) : ℝ := b
def OC (c : ℝ) : ℝ := c
def OE (e : ℝ) : ℝ := e

-- Define the condition for P being between B and C
def between (B C P : Point) : Prop :=
  B.x ≤ P.x ∧ P.x ≤ C.x

-- Define the ratio condition
def ratio_condition (A B C E P : Point) : Prop :=
  (A.x - P.x) * (P.x - C.x) = (B.x - P.x) * (P.x - E.x)

-- Theorem statement
theorem point_P_location 
  (O A B C E P : Point) 
  (a b c e : ℝ) 
  (h1 : O.x = 0) 
  (h2 : A.x = a) 
  (h3 : B.x = b) 
  (h4 : C.x = c) 
  (h5 : E.x = e) 
  (h6 : between B C P) 
  (h7 : ratio_condition A B C E P) : 
  P.x = (b * e - a * c) / (a - b + e - c) :=
sorry

end point_P_location_l3601_360151


namespace arrangement_exists_for_P_23_l3601_360122

/-- Fibonacci-like sequence defined by F_0 = 0, F_1 = 1, F_i = 3F_{i-1} - F_{i-2} for i ≥ 2 -/
def F : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | (n + 2) => 3 * F (n + 1) - F n

/-- Theorem stating the existence of an arrangement satisfying the given conditions for P = 23 -/
theorem arrangement_exists_for_P_23 : F 12 % 23 = 0 := by sorry

end arrangement_exists_for_P_23_l3601_360122


namespace ab_not_always_negative_l3601_360159

theorem ab_not_always_negative (a b : ℚ) 
  (h1 : (a - b)^2 + (b - a) * |a - b| = a * b) 
  (h2 : a * b ≠ 0) : 
  ¬(∀ a b : ℚ, (a - b)^2 + (b - a) * |a - b| = a * b → a * b < 0) := by
sorry

end ab_not_always_negative_l3601_360159


namespace smallest_positive_solution_tan_equation_l3601_360196

theorem smallest_positive_solution_tan_equation :
  let x : ℝ := π / 26
  (∀ y : ℝ, y > 0 ∧ y < x → ¬(Real.tan (4 * y) + Real.tan (3 * y) = 1 / Real.cos (3 * y))) ∧
  (Real.tan (4 * x) + Real.tan (3 * x) = 1 / Real.cos (3 * x)) := by
  sorry

end smallest_positive_solution_tan_equation_l3601_360196


namespace power_of_three_mod_eight_l3601_360186

theorem power_of_three_mod_eight : 3^2010 % 8 = 1 := by
  sorry

end power_of_three_mod_eight_l3601_360186


namespace tenth_term_of_specific_geometric_sequence_l3601_360178

/-- A geometric sequence with first term a and common ratio r -/
def geometric_sequence (a : ℝ) (r : ℝ) : ℕ → ℝ :=
  λ n => a * r^(n - 1)

theorem tenth_term_of_specific_geometric_sequence :
  let a := 10
  let second_term := -30
  let r := second_term / a
  let seq := geometric_sequence a r
  seq 10 = -196830 := by
  sorry

end tenth_term_of_specific_geometric_sequence_l3601_360178


namespace decimal_comparisons_l3601_360128

theorem decimal_comparisons : 
  (0.839 < 0.9) ∧ (6.7 > 6.07) ∧ (5.45 = 5.450) := by
  sorry

end decimal_comparisons_l3601_360128


namespace min_c_value_l3601_360187

theorem min_c_value (a b c : ℕ) (h1 : a < b) (h2 : b < c)
  (h3 : ∃! (x y : ℝ), 2*x + y = 2033 ∧ y = |x - a| + |x - b| + |x - c|) :
  c ≥ 1017 :=
sorry

end min_c_value_l3601_360187


namespace chess_team_arrangements_l3601_360189

/-- Represents the number of boys in the chess team -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the chess team -/
def num_girls : ℕ := 3

/-- Represents the total number of students in the chess team -/
def total_students : ℕ := num_boys + num_girls

/-- Represents the number of ways to arrange the ends of the row -/
def end_arrangements : ℕ := 2 * num_boys * num_girls

/-- Represents the number of ways to arrange the middle of the row -/
def middle_arrangements : ℕ := 2 * 2

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := end_arrangements * middle_arrangements

theorem chess_team_arrangements :
  total_arrangements = 72 :=
sorry

end chess_team_arrangements_l3601_360189


namespace shaded_area_rectangle_with_quarter_circles_l3601_360161

/-- The area of the shaded region in a rectangle with quarter circles in each corner -/
theorem shaded_area_rectangle_with_quarter_circles
  (length : ℝ) (width : ℝ) (radius : ℝ)
  (h_length : length = 12)
  (h_width : width = 8)
  (h_radius : radius = 4) :
  length * width - π * radius^2 = 96 - 16 * π :=
by sorry

end shaded_area_rectangle_with_quarter_circles_l3601_360161


namespace smallest_square_containing_circle_l3601_360130

theorem smallest_square_containing_circle (r : ℝ) (h : r = 7) :
  (2 * r) ^ 2 = 196 := by
  sorry

end smallest_square_containing_circle_l3601_360130


namespace complex_equation_sum_l3601_360167

theorem complex_equation_sum (x y : ℝ) : 
  (x : ℂ) + (y - 2) * Complex.I = 2 / (1 + Complex.I) → x + y = 2 := by
  sorry

end complex_equation_sum_l3601_360167


namespace basketball_team_min_score_l3601_360154

theorem basketball_team_min_score (n : ℕ) (min_score max_score : ℕ) 
  (h1 : n = 12) 
  (h2 : min_score = 7) 
  (h3 : max_score = 23) 
  (h4 : ∀ player_score, min_score ≤ player_score ∧ player_score ≤ max_score) : 
  n * min_score + (max_score - min_score) = 100 := by
sorry

end basketball_team_min_score_l3601_360154


namespace partial_fraction_decomposition_l3601_360198

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 9 ∧ x ≠ -7 →
  (5 * x - 3) / (x^2 - 2*x - 63) = (21/8) / (x - 9) + (19/8) / (x + 7) :=
by
  sorry

end partial_fraction_decomposition_l3601_360198


namespace binomial_10_choose_3_l3601_360123

theorem binomial_10_choose_3 : Nat.choose 10 3 = 120 := by
  sorry

end binomial_10_choose_3_l3601_360123


namespace inequality_proof_equality_condition_l3601_360105

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x + y + z = 9 * x * y * z) :
  x / Real.sqrt (x^2 + 2*y*z + 2) + y / Real.sqrt (y^2 + 2*z*x + 2) + z / Real.sqrt (z^2 + 2*x*y + 2) ≥ 1 :=
by sorry

theorem equality_condition (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : x + y + z = 9 * x * y * z) :
  x / Real.sqrt (x^2 + 2*y*z + 2) + y / Real.sqrt (y^2 + 2*z*x + 2) + z / Real.sqrt (z^2 + 2*x*y + 2) = 1 ↔
  x = y ∧ y = z ∧ x = Real.sqrt 3 / 3 :=
by sorry

end inequality_proof_equality_condition_l3601_360105


namespace points_form_parabola_l3601_360163

-- Define the set of points (x, y) parametrized by t
def S : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p.1 = Real.cos t ^ 2 ∧ p.2 = Real.sin t * Real.cos t}

-- Define the parabola
def P : Set (ℝ × ℝ) := {p | p.2 ^ 2 = p.1 * (1 - p.1)}

-- Theorem stating that S is equal to P
theorem points_form_parabola : S = P := by sorry

end points_form_parabola_l3601_360163


namespace product_first_fifth_l3601_360121

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n : ℕ, a (n + 1) - a n = a 1 - a 0
  third_term : a 3 = 3
  sum_reciprocals : 1 / a 1 + 1 / a 5 = 6 / 5

/-- The product of the first and fifth terms of the arithmetic sequence is 5 -/
theorem product_first_fifth (seq : ArithmeticSequence) : seq.a 1 * seq.a 5 = 5 := by
  sorry

end product_first_fifth_l3601_360121


namespace triangle_centroid_coordinates_l3601_360155

/-- The centroid of a triangle with vertices (2, 8), (6, 2), and (0, 4) has coordinates (8/3, 14/3). -/
theorem triangle_centroid_coordinates :
  let A : ℝ × ℝ := (2, 8)
  let B : ℝ × ℝ := (6, 2)
  let C : ℝ × ℝ := (0, 4)
  let centroid : ℝ × ℝ := ((A.1 + B.1 + C.1) / 3, (A.2 + B.2 + C.2) / 3)
  centroid = (8/3, 14/3) := by
sorry

end triangle_centroid_coordinates_l3601_360155


namespace route_down_length_is_18_l3601_360144

/-- A hiking trip up and down a mountain -/
structure HikingTrip where
  rate_up : ℝ
  time_up : ℝ
  rate_down_factor : ℝ
  time_down : ℝ

/-- The length of the route down the mountain -/
def route_down_length (trip : HikingTrip) : ℝ :=
  trip.rate_up * trip.rate_down_factor * trip.time_down

/-- Theorem stating the length of the route down the mountain -/
theorem route_down_length_is_18 (trip : HikingTrip) 
  (h1 : trip.time_up = trip.time_down)
  (h2 : trip.rate_down_factor = 1.5)
  (h3 : trip.rate_up = 6)
  (h4 : trip.time_up = 2) : 
  route_down_length trip = 18 := by
  sorry

#eval route_down_length ⟨6, 2, 1.5, 2⟩

end route_down_length_is_18_l3601_360144


namespace greatest_integer_radius_for_circle_l3601_360139

theorem greatest_integer_radius_for_circle (r : ℕ) : r * r ≤ 49 → r ≤ 7 ∧ ∃ (s : ℕ), s = 7 ∧ s * s ≤ 49 := by
  sorry

end greatest_integer_radius_for_circle_l3601_360139


namespace fish_per_person_l3601_360191

/-- Represents the number of fish eyes Oomyapeck eats in a day -/
def eyes_eaten : ℕ := 22

/-- Represents the number of fish eyes Oomyapeck gives to his dog -/
def eyes_to_dog : ℕ := 2

/-- Represents the number of eyes each fish has -/
def eyes_per_fish : ℕ := 2

/-- Represents the number of family members -/
def family_members : ℕ := 3

theorem fish_per_person (eyes_eaten : ℕ) (eyes_to_dog : ℕ) (eyes_per_fish : ℕ) (family_members : ℕ) :
  eyes_eaten = 22 →
  eyes_to_dog = 2 →
  eyes_per_fish = 2 →
  family_members = 3 →
  (eyes_eaten - eyes_to_dog) / eyes_per_fish = 10 :=
by sorry

end fish_per_person_l3601_360191


namespace vector_angle_theorem_l3601_360117

/-- Given two vectors in 2D space, if the angle between them is 5π/6 and the magnitude of one vector
    equals the magnitude of their sum, then the angle between that vector and their sum is 2π/3. -/
theorem vector_angle_theorem (a b : ℝ × ℝ) :
  let angle_between := Real.arccos ((a.1 * b.1 + a.2 * b.2) / (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2)))
  let magnitude (v : ℝ × ℝ) := Real.sqrt (v.1^2 + v.2^2)
  angle_between = 5 * Real.pi / 6 ∧ magnitude a = magnitude (a.1 + b.1, a.2 + b.2) →
  Real.arccos ((a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2)) /
    (magnitude a * magnitude (a.1 + b.1, a.2 + b.2))) = 2 * Real.pi / 3 := by
  sorry

end vector_angle_theorem_l3601_360117


namespace event_C_subset_event_B_l3601_360106

-- Define the sample space for tossing 3 coins
def SampleSpace := List Bool

-- Define the events A, B, and C
def event_A (outcome : SampleSpace) : Prop := outcome.contains true
def event_B (outcome : SampleSpace) : Prop := outcome.count true ≤ 2
def event_C (outcome : SampleSpace) : Prop := outcome.count true = 0

-- Theorem statement
theorem event_C_subset_event_B : 
  ∀ (outcome : SampleSpace), event_C outcome → event_B outcome :=
by
  sorry


end event_C_subset_event_B_l3601_360106


namespace one_correct_judgment_l3601_360152

theorem one_correct_judgment :
  let judgment1 := ∀ a b : ℝ, a + b ≠ 6 → a ≠ 3 ∨ b ≠ 3
  let judgment2 := ∀ p q : Prop, (p ∨ q) → (p ∧ q)
  let judgment3 := (¬ ∀ a b : ℝ, a^2 + b^2 ≥ 2*(a - b - 1)) ↔ (∃ a b : ℝ, a^2 + b^2 ≤ 2*(a - b - 1))
  let judgment4 := (judgment1 ∧ ¬judgment2 ∧ ¬judgment3)
  judgment4 := by sorry

end one_correct_judgment_l3601_360152


namespace book_pages_calculation_l3601_360113

/-- The number of pages Sally reads on weekdays -/
def weekday_pages : ℕ := 10

/-- The number of pages Sally reads on weekends -/
def weekend_pages : ℕ := 20

/-- The number of weeks it takes Sally to finish the book -/
def weeks_to_finish : ℕ := 2

/-- The number of weekdays in a week -/
def weekdays_per_week : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days_per_week : ℕ := 2

/-- The total number of pages in the book -/
def total_pages : ℕ := 180

theorem book_pages_calculation :
  total_pages = 
    weeks_to_finish * (weekdays_per_week * weekday_pages + weekend_days_per_week * weekend_pages) :=
by sorry

end book_pages_calculation_l3601_360113


namespace ellipse_triangle_perimeter_l3601_360160

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 25 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define a chord passing through F1
def Chord (A B : ℝ × ℝ) : Prop := sorry

-- Define the perimeter of a triangle
def TrianglePerimeter (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter 
  (A B : ℝ × ℝ) 
  (h_ellipse : Ellipse A.1 A.2 ∧ Ellipse B.1 B.2)
  (h_chord : Chord A B) :
  TrianglePerimeter A B F2 = 20 := by
  sorry

end ellipse_triangle_perimeter_l3601_360160


namespace triangle_side_length_l3601_360185

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π / 2 →
  C = 4 * A →
  a = 21 →
  c = 54 →
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧ 8 * x^2 - 12 * x - 4.5714 = 0 ∧
    b = 21 * (16 * x^2 - 20 * x + 5) :=
by sorry

end triangle_side_length_l3601_360185


namespace not_monotonic_iff_a_in_range_l3601_360156

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - a*x^2

theorem not_monotonic_iff_a_in_range (a : ℝ) :
  (∃ x y, 2 ≤ x ∧ x < y ∧ y ≤ 4 ∧ (f a x < f a y ∧ f a y < f a x)) ↔ 3 < a ∧ a < 6 := by
  sorry

end not_monotonic_iff_a_in_range_l3601_360156


namespace angle_identities_l3601_360110

/-- Given that α is an angle in the second quadrant and cos(α + π) = 3/13,
    prove that tan α = -4√10/3 and sin(α - π/2) * sin(-α - π) = -12√10/169 -/
theorem angle_identities (α : Real) 
    (h1 : π/2 < α ∧ α < π)  -- α is in the second quadrant
    (h2 : Real.cos (α + π) = 3/13) :
    Real.tan α = -4 * Real.sqrt 10 / 3 ∧ 
    Real.sin (α - π/2) * Real.sin (-α - π) = -12 * Real.sqrt 10 / 169 := by
  sorry

end angle_identities_l3601_360110


namespace sphere_surface_area_l3601_360140

theorem sphere_surface_area (v : ℝ) (h : v = 72 * Real.pi) :
  ∃ (r : ℝ), v = (4 / 3) * Real.pi * r^3 ∧ 4 * Real.pi * r^2 = 4 * Real.pi * (2916 ^ (1/3)) := by
  sorry

end sphere_surface_area_l3601_360140


namespace geometric_sequence_product_l3601_360109

/-- A positive geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1)^2 - 10 * (a 1) + 16 = 0 →
  (a 19)^2 - 10 * (a 19) + 16 = 0 →
  a 8 * a 12 = 16 := by
  sorry

end geometric_sequence_product_l3601_360109


namespace expression_values_l3601_360132

theorem expression_values (a b c d : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hd : d > 0) :
  let expr := a / |a| + b / |b| + c / |c| + (a * b * c) / |a * b * c| + d / |d|
  expr = 5 ∨ expr = 1 ∨ expr = -1 := by
  sorry

end expression_values_l3601_360132


namespace inequality_solution_l3601_360166

theorem inequality_solution (x : ℝ) :
  x ≥ 0 →
  (2021 * (x^2020)^(1/202) - 1 ≥ 2020 * x) ↔ x = 1 := by
  sorry

end inequality_solution_l3601_360166


namespace stratified_sample_intermediate_count_l3601_360195

/-- Represents the composition of teachers in a school -/
structure TeacherPopulation where
  total : Nat
  intermediate : Nat
  
/-- Represents a stratified sample of teachers -/
structure StratifiedSample where
  sampleSize : Nat
  intermediateSample : Nat

/-- Calculates the expected number of teachers with intermediate titles in a stratified sample -/
def expectedIntermediateSample (pop : TeacherPopulation) (sample : StratifiedSample) : Rat :=
  (pop.intermediate : Rat) * sample.sampleSize / pop.total

/-- Theorem stating that the number of teachers with intermediate titles in the sample is 7 -/
theorem stratified_sample_intermediate_count 
  (pop : TeacherPopulation) 
  (sample : StratifiedSample) : 
  pop.total = 160 → 
  pop.intermediate = 56 → 
  sample.sampleSize = 20 → 
  expectedIntermediateSample pop sample = 7 := by
  sorry

#check stratified_sample_intermediate_count

end stratified_sample_intermediate_count_l3601_360195


namespace polynomial_value_l3601_360174

theorem polynomial_value (a : ℝ) (h : a^2 + 3*a = 2) : 2*a^2 + 6*a - 10 = -6 := by
  sorry

end polynomial_value_l3601_360174


namespace decagon_diagonals_l3601_360176

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A regular decagon has 35 diagonals -/
theorem decagon_diagonals : num_diagonals 10 = 35 := by sorry

end decagon_diagonals_l3601_360176


namespace conic_sections_identification_l3601_360103

/-- The equation y^4 - 9x^4 = 3y^2 - 3 represents the union of a hyperbola and an ellipse -/
theorem conic_sections_identification (x y : ℝ) : 
  (y^4 - 9*x^4 = 3*y^2 - 3) ↔ 
  ((y^2 - 3*x^2 = 3/2) ∨ (y^2 + 3*x^2 = 3/2)) :=
sorry

end conic_sections_identification_l3601_360103


namespace compressor_stations_theorem_l3601_360171

/-- Represents the configuration of three compressor stations -/
structure CompressorStations where
  x : ℝ  -- Distance between first and second stations
  y : ℝ  -- Distance between second and third stations
  z : ℝ  -- Distance between first and third stations
  a : ℝ  -- Additional parameter

/-- Conditions for the compressor stations configuration -/
def valid_configuration (c : CompressorStations) : Prop :=
  c.x + c.y = 3 * c.z ∧
  c.z + c.y = c.x + c.a ∧
  c.x + c.z = 60 ∧
  c.x > 0 ∧ c.y > 0 ∧ c.z > 0

/-- Theorem stating the valid range for parameter a and specific values when a = 42 -/
theorem compressor_stations_theorem :
  ∀ c : CompressorStations,
    valid_configuration c →
    (0 < c.a ∧ c.a < 60) ∧
    (c.a = 42 → c.x = 33 ∧ c.y = 48 ∧ c.z = 27) :=
by sorry

end compressor_stations_theorem_l3601_360171


namespace choose_officers_specific_club_l3601_360158

/-- Represents a club with boys and girls -/
structure Club where
  total_members : ℕ
  boys : ℕ
  girls : ℕ

/-- Calculates the number of ways to choose officers in a club -/
def choose_officers (c : Club) : ℕ :=
  c.total_members * (c.boys - 1 + c.girls - 1) * (c.total_members - 2)

/-- Theorem: The number of ways to choose officers in a specific club configuration -/
theorem choose_officers_specific_club :
  let c : Club := { total_members := 30, boys := 15, girls := 15 }
  choose_officers c = 11760 := by
  sorry

#eval choose_officers { total_members := 30, boys := 15, girls := 15 }

end choose_officers_specific_club_l3601_360158


namespace min_purchase_price_l3601_360116

/-- Represents the coin denominations available on the Moon -/
def moon_coins : List Nat := [1, 15, 50]

/-- Theorem stating the minimum possible price of a purchase on the Moon -/
theorem min_purchase_price :
  ∀ (payment : List Nat) (change : List Nat),
    (∀ c ∈ payment, c ∈ moon_coins) →
    (∀ c ∈ change, c ∈ moon_coins) →
    (change.length = payment.length + 1) →
    (payment.sum - change.sum ≥ 6) →
    ∃ (p : List Nat) (c : List Nat),
      (∀ x ∈ p, x ∈ moon_coins) ∧
      (∀ x ∈ c, x ∈ moon_coins) ∧
      (c.length = p.length + 1) ∧
      (p.sum - c.sum = 6) :=
by sorry

end min_purchase_price_l3601_360116


namespace parabola_equation_l3601_360142

/-- The standard equation of a parabola with focus (3, 0) and vertex (0, 0) is y² = 12x -/
theorem parabola_equation (x y : ℝ) :
  let focus : ℝ × ℝ := (3, 0)
  let vertex : ℝ × ℝ := (0, 0)
  (x - vertex.1) ^ 2 + (y - vertex.2) ^ 2 = (x - focus.1) ^ 2 + (y - focus.2) ^ 2 →
  y ^ 2 = 12 * x := by
  sorry

end parabola_equation_l3601_360142


namespace roots_quadratic_equation_l3601_360192

theorem roots_quadratic_equation (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) :
  (∀ x, x^2 + p*x + q = 0 ↔ x = p ∨ x = q) →
  q = -2*p →
  p = 1 ∧ q = -2 := by
sorry

end roots_quadratic_equation_l3601_360192
