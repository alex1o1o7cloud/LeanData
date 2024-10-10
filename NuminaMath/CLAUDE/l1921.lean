import Mathlib

namespace school_pet_ownership_l1921_192181

theorem school_pet_ownership (total_students : ℕ) (cat_owners : ℕ) (rabbit_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 80)
  (h3 : rabbit_owners = 120) :
  (cat_owners : ℚ) / total_students * 100 = 16 ∧
  (rabbit_owners : ℚ) / total_students * 100 = 24 := by
  sorry

end school_pet_ownership_l1921_192181


namespace vector_square_difference_l1921_192141

theorem vector_square_difference (a b : ℝ × ℝ) 
  (h1 : a + b = (-3, 6)) 
  (h2 : a - b = (-3, 2)) 
  (h3 : a ≠ (0, 0)) 
  (h4 : b ≠ (0, 0)) : 
  (a.1^2 + a.2^2) - (b.1^2 + b.2^2) = 21 := by
  sorry

end vector_square_difference_l1921_192141


namespace song_circle_l1921_192195

theorem song_circle (S : Finset Nat) (covers : Finset Nat → Finset Nat)
  (h_card : S.card = 12)
  (h_cover_10 : ∀ T ⊆ S, T.card = 10 → (covers T).card = 20)
  (h_cover_8 : ∀ T ⊆ S, T.card = 8 → (covers T).card = 16) :
  (covers S).card = 24 := by
  sorry

end song_circle_l1921_192195


namespace half_and_neg_third_are_like_terms_l1921_192150

/-- Definition of like terms -/
def are_like_terms (a b : ℚ) : Prop :=
  (∀ x, a.num * x = 0 ↔ b.num * x = 0) ∧ (a ≠ 0 ∨ b ≠ 0)

/-- Theorem: 1/2 and -1/3 are like terms -/
theorem half_and_neg_third_are_like_terms :
  are_like_terms (1/2 : ℚ) (-1/3 : ℚ) := by
  sorry

end half_and_neg_third_are_like_terms_l1921_192150


namespace carpet_breadth_l1921_192111

/-- The breadth of the first carpet in meters -/
def b : ℝ := 6

/-- The length of the first carpet in meters -/
def l : ℝ := 1.44 * b

/-- The length of the second carpet in meters -/
def l2 : ℝ := 1.4 * l

/-- The breadth of the second carpet in meters -/
def b2 : ℝ := 1.25 * b

/-- The cost of the second carpet in rupees -/
def cost : ℝ := 4082.4

/-- The rate of the carpet in rupees per square meter -/
def rate : ℝ := 45

theorem carpet_breadth :
  b = 6 ∧
  l = 1.44 * b ∧
  l2 = 1.4 * l ∧
  b2 = 1.25 * b ∧
  cost = rate * l2 * b2 :=
by sorry

end carpet_breadth_l1921_192111


namespace first_grade_boys_count_l1921_192142

theorem first_grade_boys_count (num_classrooms : ℕ) (num_girls : ℕ) (students_per_classroom : ℕ) :
  num_classrooms = 4 →
  num_girls = 44 →
  students_per_classroom = 25 →
  (∀ classroom, classroom ≤ num_classrooms →
    (num_girls / num_classrooms = students_per_classroom / 2)) →
  num_girls = 44 :=
by
  sorry

end first_grade_boys_count_l1921_192142


namespace double_root_equation_example_double_root_equation_condition_double_root_equation_m_value_l1921_192186

/-- Definition of a double root equation -/
def is_double_root_equation (a b c : ℝ) : Prop :=
  ∃ x y : ℝ, x ≠ y ∧ (a * x^2 + b * x + c = 0) ∧ (a * y^2 + b * y + c = 0) ∧ (y = 2*x ∨ x = 2*y)

/-- Theorem 1: x^2 - 3x + 2 = 0 is a double root equation -/
theorem double_root_equation_example : is_double_root_equation 1 (-3) 2 := sorry

/-- Theorem 2: For (x-2)(x-m) = 0 to be a double root equation, m^2 + 2m + 2 = 26 or 5 -/
theorem double_root_equation_condition (m : ℝ) :
  is_double_root_equation 1 (-(2+m)) (2*m) →
  m^2 + 2*m + 2 = 26 ∨ m^2 + 2*m + 2 = 5 := sorry

/-- Theorem 3: For x^2 - (m-1)x + 32 = 0 to be a double root equation, m = 13 or -11 -/
theorem double_root_equation_m_value (m : ℝ) :
  is_double_root_equation 1 (-(m-1)) 32 →
  m = 13 ∨ m = -11 := sorry

end double_root_equation_example_double_root_equation_condition_double_root_equation_m_value_l1921_192186


namespace barbara_paper_count_l1921_192184

/-- The number of sheets in a bundle of colored paper -/
def sheets_per_bundle : ℕ := 2

/-- The number of sheets in a bunch of white paper -/
def sheets_per_bunch : ℕ := 4

/-- The number of sheets in a heap of scrap paper -/
def sheets_per_heap : ℕ := 20

/-- The number of bundles of colored paper -/
def colored_bundles : ℕ := 3

/-- The number of bunches of white paper -/
def white_bunches : ℕ := 2

/-- The number of heaps of scrap paper -/
def scrap_heaps : ℕ := 5

/-- The total number of sheets Barbara removed from the chest of drawers -/
def total_sheets : ℕ := colored_bundles * sheets_per_bundle + 
                         white_bunches * sheets_per_bunch + 
                         scrap_heaps * sheets_per_heap

theorem barbara_paper_count : total_sheets = 114 := by
  sorry

end barbara_paper_count_l1921_192184


namespace function_inequality_implies_lower_bound_on_a_l1921_192158

open Real

theorem function_inequality_implies_lower_bound_on_a :
  ∀ a : ℝ,
  (∀ x : ℝ, x > 0 → (log x - a ≤ x * exp x - x)) →
  a ≥ -1 :=
by sorry

end function_inequality_implies_lower_bound_on_a_l1921_192158


namespace uncle_bob_parking_probability_l1921_192182

def total_spaces : ℕ := 20
def parked_cars : ℕ := 14
def required_spaces : ℕ := 3

theorem uncle_bob_parking_probability :
  let total_configurations := Nat.choose total_spaces parked_cars
  let unfavorable_configurations := Nat.choose (parked_cars - required_spaces + 2) (parked_cars - required_spaces + 2 - parked_cars)
  (total_configurations - unfavorable_configurations) / total_configurations = 19275 / 19380 := by
  sorry

end uncle_bob_parking_probability_l1921_192182


namespace ratio_difference_increases_dependence_l1921_192196

/-- Represents a 2x2 contingency table -/
structure ContingencyTable where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Calculates the chi-square statistic for a 2x2 contingency table -/
def chi_square (table : ContingencyTable) : ℝ :=
  sorry

/-- Represents the probability of dependence between two variables -/
def dependence_probability (chi_square_value : ℝ) : ℝ :=
  sorry

/-- Theorem: As the difference between ratios increases, the probability of dependence increases -/
theorem ratio_difference_increases_dependence (table : ContingencyTable) :
  let ratio1 := table.a / (table.a + table.b)
  let ratio2 := table.c / (table.c + table.d)
  let diff := |ratio1 - ratio2|
  ∀ ε > 0, ∃ δ > 0,
    ∀ table' : ContingencyTable,
      let ratio1' := table'.a / (table'.a + table'.b)
      let ratio2' := table'.c / (table'.c + table'.d)
      let diff' := |ratio1' - ratio2'|
      diff' > diff + δ →
        dependence_probability (chi_square table') > dependence_probability (chi_square table) + ε :=
by
  sorry

end ratio_difference_increases_dependence_l1921_192196


namespace m_minus_n_squared_l1921_192131

theorem m_minus_n_squared (m n : ℝ) (h1 : m + n = 6) (h2 : m^2 + n^2 = 26) : 
  (m - n)^2 = 16 := by
sorry

end m_minus_n_squared_l1921_192131


namespace function_shift_l1921_192179

/-- Given a function f with the specified properties, prove that g can be obtained
    by shifting f to the left by π/8 units. -/
theorem function_shift (ω : ℝ) (h1 : ω > 0) : 
  let f : ℝ → ℝ := λ x ↦ Real.sin (ω * x + π / 4)
  let g : ℝ → ℝ := λ x ↦ Real.cos (ω * x)
  (∀ x, f (x + π / ω) = f x) →  -- minimum positive period is π
  ∀ x, g x = f (x + π / 8) := by
sorry

end function_shift_l1921_192179


namespace arithmetic_sequence_max_sum_l1921_192105

/-- An arithmetic sequence -/
def arithmetic_sequence : ℕ → ℝ := sorry

/-- Sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) : ℝ := sorry

/-- The value of n that maximizes S n -/
def n_max : ℕ := sorry

theorem arithmetic_sequence_max_sum :
  (S 16 > 0) → (S 17 < 0) → n_max = 8 := by sorry

end arithmetic_sequence_max_sum_l1921_192105


namespace complement_intersection_theorem_l1921_192101

def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 3, 5}
def N : Set ℕ := {4, 5, 6}

theorem complement_intersection_theorem :
  (U \ M) ∩ N = {4, 6} := by sorry

end complement_intersection_theorem_l1921_192101


namespace bridget_bakery_profit_l1921_192198

/-- Bridget's bakery problem -/
theorem bridget_bakery_profit :
  let total_loaves : ℕ := 60
  let morning_price : ℚ := 3
  let afternoon_discount : ℚ := 1
  let late_afternoon_price : ℚ := 3/2
  let production_cost : ℚ := 4/5
  let morning_sales : ℕ := total_loaves / 2
  let afternoon_sales : ℕ := ((total_loaves - morning_sales) * 3 + 2) / 4 -- Rounding up
  let late_afternoon_sales : ℕ := total_loaves - morning_sales - afternoon_sales
  let total_revenue : ℚ := 
    morning_sales * morning_price + 
    afternoon_sales * (morning_price - afternoon_discount) + 
    late_afternoon_sales * late_afternoon_price
  let total_cost : ℚ := total_loaves * production_cost
  let profit : ℚ := total_revenue - total_cost
  profit = 197/2 := by sorry

end bridget_bakery_profit_l1921_192198


namespace tan_double_angle_special_case_l1921_192123

theorem tan_double_angle_special_case (θ : ℝ) :
  3 * Real.cos (π / 2 - θ) + Real.cos (π + θ) = 0 →
  Real.tan (2 * θ) = 3 / 4 := by
  sorry

end tan_double_angle_special_case_l1921_192123


namespace michael_crayons_worth_l1921_192126

/-- Calculates the total worth of crayons after a purchase --/
def total_worth_after_purchase (initial_packs : ℕ) (additional_packs : ℕ) (cost_per_pack : ℚ) : ℚ :=
  (initial_packs + additional_packs) * cost_per_pack

/-- Proves that the total worth of crayons after Michael's purchase is $15 --/
theorem michael_crayons_worth :
  let initial_packs := 4
  let additional_packs := 2
  let cost_per_pack := 5/2
  total_worth_after_purchase initial_packs additional_packs cost_per_pack = 15 := by
  sorry

end michael_crayons_worth_l1921_192126


namespace intersection_A_complement_B_when_k_is_1_A_intersect_B_nonempty_iff_k_geq_neg_1_l1921_192144

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 3}
def B (k : ℝ) : Set ℝ := {x : ℝ | x - k ≤ 0}

-- Define the complement of B in the universal set U (which is ℝ in this case)
def C_U_B (k : ℝ) : Set ℝ := {x : ℝ | x - k > 0}

theorem intersection_A_complement_B_when_k_is_1 :
  A ∩ C_U_B 1 = {x : ℝ | 1 < x ∧ x < 3} := by sorry

theorem A_intersect_B_nonempty_iff_k_geq_neg_1 :
  ∀ k : ℝ, (A ∩ B k).Nonempty ↔ k ≥ -1 := by sorry

end intersection_A_complement_B_when_k_is_1_A_intersect_B_nonempty_iff_k_geq_neg_1_l1921_192144


namespace log_inequality_l1921_192143

theorem log_inequality (a b : ℝ) (ha : a = Real.log 0.3 / Real.log 0.2) (hb : b = Real.log 0.3 / Real.log 2) :
  a * b < a + b ∧ a + b < 0 := by
  sorry

end log_inequality_l1921_192143


namespace complex_magnitude_equals_five_l1921_192128

theorem complex_magnitude_equals_five (m : ℝ) (hm : m > 0) :
  Complex.abs (Complex.mk (-1) (2 * m)) = 5 ↔ m = Real.sqrt 6 := by
  sorry

end complex_magnitude_equals_five_l1921_192128


namespace trig_identity_proof_l1921_192149

theorem trig_identity_proof : 
  Real.sin (410 * π / 180) * Real.sin (550 * π / 180) - 
  Real.sin (680 * π / 180) * Real.cos (370 * π / 180) = 1/2 := by
  sorry

end trig_identity_proof_l1921_192149


namespace sixty_percent_of_three_fifths_of_hundred_l1921_192156

theorem sixty_percent_of_three_fifths_of_hundred (n : ℝ) : n = 100 → (0.6 * (3/5 * n)) = 36 := by
  sorry

end sixty_percent_of_three_fifths_of_hundred_l1921_192156


namespace zeros_imply_b_and_c_b_in_interval_l1921_192117

-- Define the quadratic function f(x)
def f (b c x : ℝ) : ℝ := x^2 + 2*b*x + c

-- Part 1: Prove that if -1 and 1 are zeros of f(x), then b = 0 and c = -1
theorem zeros_imply_b_and_c (b c : ℝ) :
  f b c (-1) = 0 ∧ f b c 1 = 0 → b = 0 ∧ c = -1 := by sorry

-- Part 2: Prove that given the conditions, b is in the interval (1/5, 5/7)
theorem b_in_interval (b c : ℝ) :
  f b c 1 = 0 ∧ 
  (∃ x₁ x₂, -3 < x₁ ∧ x₁ < -2 ∧ 0 < x₂ ∧ x₂ < 1 ∧ 
    f b c x₁ + x₁ + b = 0 ∧ f b c x₂ + x₂ + b = 0) →
  1/5 < b ∧ b < 5/7 := by sorry

end zeros_imply_b_and_c_b_in_interval_l1921_192117


namespace perfect_square_condition_l1921_192125

theorem perfect_square_condition (n : ℕ+) (p : ℕ) :
  (Nat.Prime p) → (∃ k : ℕ, p^2 + 7^n.val = k^2) ↔ (n = 1 ∧ p = 3) := by
  sorry

end perfect_square_condition_l1921_192125


namespace president_vice_president_selection_l1921_192173

def num_candidates : ℕ := 4

theorem president_vice_president_selection :
  (num_candidates * (num_candidates - 1) = 12) := by
  sorry

end president_vice_president_selection_l1921_192173


namespace function_properties_l1921_192194

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_odd_function (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

theorem function_properties (f g : ℝ → ℝ) 
  (h_even : is_even_function f) 
  (h_odd : is_odd_function g) 
  (h_diff : ∀ x, f x - g x = x^3 + x^2 + 1) : 
  (f 1 + g 1 = 1) ∧ (∀ x, f x = x^2 + 1) := by
  sorry

end function_properties_l1921_192194


namespace principal_exists_l1921_192132

/-- The principal amount that satisfies the given conditions -/
def find_principal : ℝ → Prop := fun P =>
  let first_year_rate : ℝ := 0.10
  let second_year_rate : ℝ := 0.12
  let semi_annual_rate1 : ℝ := first_year_rate / 2
  let semi_annual_rate2 : ℝ := second_year_rate / 2
  let compound_factor : ℝ := (1 + semi_annual_rate1)^2 * (1 + semi_annual_rate2)^2
  let simple_interest_factor : ℝ := first_year_rate + second_year_rate
  P * (compound_factor - 1 - simple_interest_factor) = 15

/-- Theorem stating the existence of a principal amount satisfying the given conditions -/
theorem principal_exists : ∃ P : ℝ, find_principal P := by
  sorry

end principal_exists_l1921_192132


namespace parallelogram_sum_xy_l1921_192106

/-- A parallelogram with sides measuring 10, 4x+2, 12y-2, and 10 units consecutively has x + y = 3 -/
theorem parallelogram_sum_xy (x y : ℝ) : 
  (10 : ℝ) = 4*x + 2 ∧ (10 : ℝ) = 12*y - 2 → x + y = 3 := by
  sorry

end parallelogram_sum_xy_l1921_192106


namespace flag_width_calculation_l1921_192157

theorem flag_width_calculation (height : ℝ) (paint_cost : ℝ) (paint_coverage : ℝ) 
  (total_spent : ℝ) (h1 : height = 4) (h2 : paint_cost = 2) (h3 : paint_coverage = 4) 
  (h4 : total_spent = 20) : ∃ (width : ℝ), width = 5 := by
  sorry

end flag_width_calculation_l1921_192157


namespace max_value_implies_a_equals_one_l1921_192169

/-- Given a function f(x) = x^2 - ax - a with maximum value 1 on [0, 2], prove a = 1 -/
theorem max_value_implies_a_equals_one (a : ℝ) :
  (∃ (f : ℝ → ℝ), (∀ x, f x = x^2 - a*x - a) ∧
   (∀ x ∈ Set.Icc 0 2, f x ≤ 1) ∧
   (∃ x ∈ Set.Icc 0 2, f x = 1)) →
  a = 1 := by
sorry

end max_value_implies_a_equals_one_l1921_192169


namespace november_savings_l1921_192160

def september_savings : ℕ := 50
def october_savings : ℕ := 37
def mom_gift : ℕ := 25
def video_game_cost : ℕ := 87
def money_left : ℕ := 36

theorem november_savings :
  ∃ (november_savings : ℕ),
    september_savings + october_savings + november_savings + mom_gift - video_game_cost = money_left ∧
    november_savings = 11 :=
sorry

end november_savings_l1921_192160


namespace grant_school_students_l1921_192191

theorem grant_school_students (S : ℕ) : 
  (S / 3 : ℚ) / 4 = 15 → S = 180 := by
  sorry

end grant_school_students_l1921_192191


namespace inequality_solution_l1921_192154

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4*x - 5) / (x^2 + 3*x + 2) < 0 ↔ x ∈ Set.Ioo (-2 : ℝ) (-1) ∪ Set.Ioo (-1 : ℝ) 5 :=
by sorry

end inequality_solution_l1921_192154


namespace complex_coordinate_of_reciprocal_i_cubed_l1921_192164

theorem complex_coordinate_of_reciprocal_i_cubed :
  let z : ℂ := (Complex.I ^ 3)⁻¹
  z = Complex.I :=
by sorry

end complex_coordinate_of_reciprocal_i_cubed_l1921_192164


namespace range_of_a_l1921_192137

theorem range_of_a (a : ℝ) : (2 * a - 1) ^ 0 = 1 → a ≠ 1/2 :=
by sorry

end range_of_a_l1921_192137


namespace sequence_properties_l1921_192161

theorem sequence_properties (a : Fin 4 → ℝ) 
  (h_decreasing : ∀ i j : Fin 4, i < j → a i > a j)
  (h_nonneg : a 3 ≥ 0)
  (h_closed : ∀ i j : Fin 4, i ≤ j → ∃ k : Fin 4, a i - a j = a k) :
  (∃ d : ℝ, ∀ i : Fin 4, i.val < 3 → a i.succ = a i - d) ∧ 
  (∃ i j : Fin 4, i < j ∧ (i.val + 1) * a i = (j.val + 1) * a j) ∧
  (∃ i : Fin 4, a i = 0) := by
  sorry

end sequence_properties_l1921_192161


namespace find_divisor_l1921_192152

theorem find_divisor (dividend quotient remainder : ℕ) (h : dividend = quotient * 4 + remainder) :
  ∃ (divisor : ℕ), dividend = quotient * divisor + remainder ∧ divisor = 4 :=
by
  sorry

end find_divisor_l1921_192152


namespace correct_reassembly_probability_l1921_192189

/-- Represents the number of subcubes in each dimension of the larger cube -/
def cubeDimension : ℕ := 3

/-- Represents the total number of subcubes in the larger cube -/
def totalSubcubes : ℕ := cubeDimension ^ 3

/-- Represents the number of corner subcubes -/
def cornerCubes : ℕ := 8

/-- Represents the number of edge subcubes -/
def edgeCubes : ℕ := 12

/-- Represents the number of face subcubes -/
def faceCubes : ℕ := 6

/-- Represents the number of center subcubes -/
def centerCubes : ℕ := 1

/-- Represents the number of possible orientations for a corner subcube -/
def cornerOrientations : ℕ := 3

/-- Represents the number of possible orientations for an edge subcube -/
def edgeOrientations : ℕ := 2

/-- Represents the number of possible orientations for a face subcube -/
def faceOrientations : ℕ := 4

/-- Represents the total number of possible orientations for any subcube -/
def totalOrientations : ℕ := 24

/-- Calculates the number of correct reassemblings -/
def correctReassemblings : ℕ :=
  (cornerOrientations ^ cornerCubes) * (cornerCubes.factorial) *
  (edgeOrientations ^ edgeCubes) * (edgeCubes.factorial) *
  (faceOrientations ^ faceCubes) * (faceCubes.factorial) *
  (centerCubes.factorial)

/-- Calculates the total number of possible reassemblings -/
def totalReassemblings : ℕ :=
  (totalOrientations ^ totalSubcubes) * (totalSubcubes.factorial)

/-- Theorem: The probability of correctly reassembling the cube is equal to
    the ratio of correct reassemblings to total possible reassemblings -/
theorem correct_reassembly_probability :
  (correctReassemblings : ℚ) / totalReassemblings =
  (correctReassemblings : ℚ) / totalReassemblings :=
by
  sorry

end correct_reassembly_probability_l1921_192189


namespace zoo_trip_average_bus_capacity_l1921_192118

theorem zoo_trip_average_bus_capacity (total_students : ℕ) (num_buses : ℕ) 
  (car1_capacity car2_capacity car3_capacity car4_capacity : ℕ) : 
  total_students = 396 →
  num_buses = 7 →
  car1_capacity = 5 →
  car2_capacity = 4 →
  car3_capacity = 3 →
  car4_capacity = 6 →
  (total_students - (car1_capacity + car2_capacity + car3_capacity + car4_capacity)) / num_buses = 54 := by
  sorry

end zoo_trip_average_bus_capacity_l1921_192118


namespace spying_arrangement_odd_l1921_192135

/-- A function representing the spying arrangement in a circular group -/
def spyingArrangement (n : ℕ) : ℕ → ℕ :=
  fun i => (i % n) + 1

/-- The theorem stating that the number of people in the spying arrangement must be odd -/
theorem spying_arrangement_odd (n : ℕ) (h : n > 0) :
  (∀ i : ℕ, i < n → spyingArrangement n (spyingArrangement n i) = (i + 2) % n + 1) →
  Odd n :=
sorry

end spying_arrangement_odd_l1921_192135


namespace power_fraction_equals_two_l1921_192167

theorem power_fraction_equals_two : (2^4 - 2) / (2^3 - 1) = 2 := by
  sorry

end power_fraction_equals_two_l1921_192167


namespace four_digit_diff_divisible_iff_middle_digits_same_l1921_192116

/-- Represents a four-digit number -/
structure FourDigitNumber where
  a : ℕ
  b : ℕ
  c : ℕ
  d : ℕ
  h1 : a ≥ 1 ∧ a ≤ 9
  h2 : b ≥ 0 ∧ b ≤ 9
  h3 : c ≥ 0 ∧ c ≤ 9
  h4 : d ≥ 0 ∧ d ≤ 9

/-- Calculates the value of a four-digit number -/
def fourDigitValue (n : FourDigitNumber) : ℕ :=
  1000 * n.a + 100 * n.b + 10 * n.c + n.d

/-- Calculates the value of the reversed four-digit number -/
def reversedValue (n : FourDigitNumber) : ℕ :=
  1000 * n.d + 100 * n.c + 10 * n.b + n.a

/-- Theorem: For a four-digit number, the difference between the number and its reverse
    is divisible by 37 if and only if the two middle digits are the same -/
theorem four_digit_diff_divisible_iff_middle_digits_same (n : FourDigitNumber) :
  (fourDigitValue n - reversedValue n) % 37 = 0 ↔ n.b = n.c := by
  sorry

end four_digit_diff_divisible_iff_middle_digits_same_l1921_192116


namespace angle_D_measure_l1921_192165

-- Define the hexagon and its properties
def ConvexHexagon (A B C D E F : ℝ) : Prop :=
  -- Angles A, B, and C are congruent
  A = B ∧ B = C
  -- Angles D, E, and F are congruent
  ∧ D = E ∧ E = F
  -- The measure of angle A is 50° less than the measure of angle D
  ∧ A + 50 = D
  -- Sum of interior angles of a hexagon is 720°
  ∧ A + B + C + D + E + F = 720

-- Theorem statement
theorem angle_D_measure (A B C D E F : ℝ) 
  (h : ConvexHexagon A B C D E F) : D = 145 := by
  sorry

end angle_D_measure_l1921_192165


namespace function_properties_l1921_192109

noncomputable def f (x φ : ℝ) : ℝ := Real.sin x * Real.cos φ + Real.cos x * Real.sin φ

theorem function_properties (φ : ℝ) (h1 : 0 < φ) (h2 : φ < π) :
  -- The smallest positive period of f is 2π
  (∃ (T : ℝ), T > 0 ∧ T = 2 * π ∧ ∀ (x : ℝ), f x φ = f (x + T) φ) ∧
  -- If the graph of y = f(2x + π/4) is symmetric about x = π/6, then φ = 11π/12
  (∀ (x : ℝ), f (2 * (π/6 - x) + π/4) φ = f (2 * (π/6 + x) + π/4) φ → φ = 11 * π / 12) ∧
  -- If f(α - 2π/3) = √2/4, then sin 2α = -3/4
  (∀ (α : ℝ), f (α - 2 * π / 3) φ = Real.sqrt 2 / 4 → Real.sin (2 * α) = -3 / 4) :=
by sorry

end function_properties_l1921_192109


namespace polynomial_sum_zero_l1921_192192

theorem polynomial_sum_zero (a b c d : ℝ) :
  (∀ x : ℝ, (1 + x)^2 * (1 - x) = a + b*x + c*x^2 + d*x^3) →
  a + b + c + d = 0 := by
sorry

end polynomial_sum_zero_l1921_192192


namespace solution_set_is_open_ray_l1921_192133

def solution_set (f : ℝ → ℝ) : Set ℝ :=
  {x | f x > 2 * x + 4}

theorem solution_set_is_open_ray
  (f : ℝ → ℝ)
  (h1 : Differentiable ℝ f)
  (h2 : ∀ x, deriv f x > 2)
  (h3 : f (-1) = 2) :
  solution_set f = Set.Ioi (-1) := by
  sorry

end solution_set_is_open_ray_l1921_192133


namespace parallel_vectors_x_value_l1921_192136

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), a.1 * b.2 = k * a.2 * b.1

theorem parallel_vectors_x_value :
  let a : ℝ × ℝ := (1, 2)
  let b : ℝ × ℝ := (2*x, -3)
  are_parallel a b → x = -3/4 := by
  sorry

end parallel_vectors_x_value_l1921_192136


namespace toms_remaining_candy_l1921_192199

theorem toms_remaining_candy (initial_boxes : ℕ) (boxes_given_away : ℕ) (pieces_per_box : ℕ) : 
  initial_boxes = 14 → boxes_given_away = 8 → pieces_per_box = 3 →
  (initial_boxes - boxes_given_away) * pieces_per_box = 18 := by
  sorry

end toms_remaining_candy_l1921_192199


namespace sum_of_coefficients_equals_one_l1921_192147

theorem sum_of_coefficients_equals_one 
  (a a₁ a₂ a₃ a₄ : ℝ) 
  (h : ∀ x : ℝ, (2*x - 3)^4 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4) : 
  a + a₁ + a₂ + a₃ + a₄ = 1 := by
sorry

end sum_of_coefficients_equals_one_l1921_192147


namespace seashells_remaining_l1921_192151

theorem seashells_remaining (initial_seashells : ℕ) (given_seashells : ℕ) 
  (h1 : initial_seashells = 70) 
  (h2 : given_seashells = 43) : 
  initial_seashells - given_seashells = 27 := by
  sorry

end seashells_remaining_l1921_192151


namespace goats_gifted_count_l1921_192114

/-- Represents the number of goats gifted by Jeremy to Fred -/
def goats_gifted (initial_horses initial_sheep initial_chickens : ℕ) 
  (male_animals : ℕ) : ℕ :=
  let initial_total := initial_horses + initial_sheep + initial_chickens
  let after_brian_sale := initial_total - initial_total / 2
  let final_total := male_animals * 2
  final_total - after_brian_sale

/-- Theorem stating the number of goats gifted by Jeremy -/
theorem goats_gifted_count : 
  goats_gifted 100 29 9 53 = 37 := by
  sorry

#eval goats_gifted 100 29 9 53

end goats_gifted_count_l1921_192114


namespace sum_of_roots_quadratic_l1921_192172

theorem sum_of_roots_quadratic (x : ℝ) : 
  (x^2 = 10*x - 14) → (∃ y : ℝ, y^2 = 10*y - 14 ∧ x + y = 10) :=
by sorry

end sum_of_roots_quadratic_l1921_192172


namespace rope_length_comparison_l1921_192177

theorem rope_length_comparison (L : ℝ) (h : L > 0) : 
  ¬ (∀ L, L - (1/3) = L - (L/3)) :=
sorry

end rope_length_comparison_l1921_192177


namespace population_reaches_max_capacity_in_140_years_l1921_192148

def initial_year : ℕ := 1998
def initial_population : ℕ := 200
def total_land : ℕ := 32000
def space_per_person : ℕ := 2
def growth_period : ℕ := 20

def max_capacity : ℕ := total_land / space_per_person

def population_after_years (years : ℕ) : ℕ :=
  initial_population * 2^(years / growth_period)

theorem population_reaches_max_capacity_in_140_years :
  ∃ (y : ℕ), y = 140 ∧ 
  population_after_years y ≥ max_capacity ∧
  population_after_years (y - growth_period) < max_capacity :=
sorry

end population_reaches_max_capacity_in_140_years_l1921_192148


namespace longest_segment_in_cylinder_l1921_192174

/-- The longest segment in a cylinder with radius 5 and height 12 is 2√61 -/
theorem longest_segment_in_cylinder : 
  let r : ℝ := 5
  let h : ℝ := 12
  let longest_segment := Real.sqrt ((2 * r) ^ 2 + h ^ 2)
  longest_segment = 2 * Real.sqrt 61 := by
  sorry

end longest_segment_in_cylinder_l1921_192174


namespace max_b_value_max_b_value_achieved_l1921_192183

theorem max_b_value (b : ℕ+) (x : ℤ) (h : x^2 + b * x = -20) : b ≤ 21 :=
sorry

theorem max_b_value_achieved : ∃ (b : ℕ+) (x : ℤ), x^2 + b * x = -20 ∧ b = 21 :=
sorry

end max_b_value_max_b_value_achieved_l1921_192183


namespace arccos_zero_equals_pi_half_l1921_192153

theorem arccos_zero_equals_pi_half : Real.arccos 0 = π / 2 := by
  sorry

end arccos_zero_equals_pi_half_l1921_192153


namespace cloth_sale_problem_l1921_192162

/-- Prove that the number of meters of cloth sold is 85, given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sale_problem (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
  (h1 : total_selling_price = 8925)
  (h2 : profit_per_meter = 35)
  (h3 : cost_price_per_meter = 70) :
  (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 85 := by
  sorry

end cloth_sale_problem_l1921_192162


namespace triangle_abc_is_right_triangle_l1921_192140

theorem triangle_abc_is_right_triangle (AB AC BC : ℝ) 
  (h1 : AB = 1) (h2 : AC = 2) (h3 : BC = Real.sqrt 5) : 
  AB ^ 2 + AC ^ 2 = BC ^ 2 := by
  sorry

end triangle_abc_is_right_triangle_l1921_192140


namespace gear_revolution_theorem_l1921_192112

/-- The number of revolutions per minute for gear p -/
def p_rpm : ℝ := 10

/-- The duration in minutes -/
def duration : ℝ := 0.5

/-- The difference in revolutions between gear q and gear p after the given duration -/
def revolution_difference : ℝ := 15

/-- The number of revolutions per minute for gear q -/
def q_rpm : ℝ := 40

theorem gear_revolution_theorem :
  q_rpm = 2 * (p_rpm * duration + revolution_difference) := by
  sorry

end gear_revolution_theorem_l1921_192112


namespace value_of_a_l1921_192197

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 + 8
def g (x : ℝ) : ℝ := x^2 - 4

-- State the theorem
theorem value_of_a (a : ℝ) (ha : a > 0) (h : f (g a) = 8) : a = 2 := by
  sorry

end value_of_a_l1921_192197


namespace student_boat_problem_l1921_192180

theorem student_boat_problem (students boats : ℕ) : 
  (7 * boats + 5 = students) → 
  (8 * boats = students + 2) → 
  (students = 54 ∧ boats = 7) :=
by sorry

end student_boat_problem_l1921_192180


namespace solve_system_l1921_192115

theorem solve_system (c d : ℝ) 
  (eq1 : 5 + c = 6 - d) 
  (eq2 : 6 + d = 9 + c) : 
  5 - c = 6 := by
sorry

end solve_system_l1921_192115


namespace geometric_series_first_term_l1921_192124

theorem geometric_series_first_term 
  (a r : ℝ) 
  (h1 : a / (1 - r) = 30) 
  (h2 : a^2 / (1 - r^2) = 150) : 
  a = 60 / 7 := by
sorry

end geometric_series_first_term_l1921_192124


namespace yield_difference_l1921_192190

-- Define the initial yields and growth rates
def tomatoes_initial : ℝ := 2073
def corn_initial : ℝ := 4112
def onions_initial : ℝ := 985
def carrots_initial : ℝ := 6250

def tomatoes_growth_rate : ℝ := 0.12
def corn_growth_rate : ℝ := 0.15
def onions_growth_rate : ℝ := 0.08
def carrots_growth_rate : ℝ := 0.10

-- Calculate the yields after growth
def tomatoes_yield : ℝ := tomatoes_initial * (1 + tomatoes_growth_rate)
def corn_yield : ℝ := corn_initial * (1 + corn_growth_rate)
def onions_yield : ℝ := onions_initial * (1 + onions_growth_rate)
def carrots_yield : ℝ := carrots_initial * (1 + carrots_growth_rate)

-- Define the theorem
theorem yield_difference : 
  (max tomatoes_yield (max corn_yield (max onions_yield carrots_yield))) - 
  (min tomatoes_yield (min corn_yield (min onions_yield carrots_yield))) = 5811.2 := by
  sorry

end yield_difference_l1921_192190


namespace M_union_N_eq_M_l1921_192175

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | |p.1 * p.2| = 1 ∧ p.1 > 0}
def N : Set (ℝ × ℝ) := {p | Real.arctan p.1 + Real.arctan p.2 = Real.pi}

-- Theorem statement
theorem M_union_N_eq_M : M ∪ N = M := by
  sorry

end M_union_N_eq_M_l1921_192175


namespace min_correct_answers_for_target_score_l1921_192159

/-- Represents the scoring system and conditions of the AMC 12 problem -/
structure AMC12Scoring where
  total_problems : Nat
  attempted_problems : Nat
  correct_points : Int
  incorrect_points : Int
  unanswered_points : Int
  target_score : Int

/-- Calculates the score based on the number of correct answers -/
def calculate_score (s : AMC12Scoring) (correct_answers : Nat) : Int :=
  let incorrect_answers := s.attempted_problems - correct_answers
  let unanswered := s.total_problems - s.attempted_problems
  correct_answers * s.correct_points + 
  incorrect_answers * s.incorrect_points + 
  unanswered * s.unanswered_points

/-- Theorem stating the minimum number of correct answers needed to reach the target score -/
theorem min_correct_answers_for_target_score (s : AMC12Scoring) 
  (h1 : s.total_problems = 30)
  (h2 : s.attempted_problems = 26)
  (h3 : s.correct_points = 7)
  (h4 : s.incorrect_points = -1)
  (h5 : s.unanswered_points = 1)
  (h6 : s.target_score = 150) :
  ∃ n : Nat, (∀ m : Nat, m < n → calculate_score s m < s.target_score) ∧ 
             calculate_score s n ≥ s.target_score ∧
             n = 22 := by
  sorry

end min_correct_answers_for_target_score_l1921_192159


namespace vaccine_effectiveness_theorem_l1921_192120

/-- Represents the data for a vaccine experiment -/
structure VaccineExperiment where
  total_participants : ℕ
  vaccinated_infected : ℕ
  placebo_infected : ℕ

/-- Calculates the vaccine effectiveness -/
def vaccine_effectiveness (exp : VaccineExperiment) : ℚ :=
  let p := exp.vaccinated_infected / (exp.total_participants / 2 : ℚ)
  let q := exp.placebo_infected / (exp.total_participants / 2 : ℚ)
  1 - p / q

/-- The main theorem about vaccine effectiveness -/
theorem vaccine_effectiveness_theorem (exp_A exp_B : VaccineExperiment) :
  exp_A.total_participants = 30000 →
  exp_A.vaccinated_infected = 50 →
  exp_A.placebo_infected = 500 →
  vaccine_effectiveness exp_A = 9/10 ∧
  ∃ (exp_B : VaccineExperiment),
    vaccine_effectiveness exp_B > 9/10 ∧
    exp_B.vaccinated_infected ≥ exp_A.vaccinated_infected :=
by sorry

end vaccine_effectiveness_theorem_l1921_192120


namespace cos_seven_pi_six_plus_x_l1921_192130

theorem cos_seven_pi_six_plus_x (x : Real) (h : Real.sin (2 * Real.pi / 3 + x) = 3 / 5) :
  Real.cos (7 * Real.pi / 6 + x) = -3 / 5 := by
  sorry

end cos_seven_pi_six_plus_x_l1921_192130


namespace symmetric_points_sum_l1921_192163

/-- Given two points M and N that are symmetric with respect to the x-axis,
    prove that the sum of their x and y coordinates is -3. -/
theorem symmetric_points_sum (b a : ℝ) : 
  (∃ (M N : ℝ × ℝ), 
    M = (-2, b) ∧ 
    N = (a, 1) ∧ 
    (M.1 = N.1 ∧ M.2 = -N.2)) → 
  a + b = -3 :=
by sorry

end symmetric_points_sum_l1921_192163


namespace rectangle_perimeter_l1921_192176

theorem rectangle_perimeter (a b c w : ℝ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (h4 : w = 6) : 
  let triangle_area := (1/2) * a * b
  let rectangle_length := triangle_area / w
  2 * (rectangle_length + w) = 30 := by
  sorry

end rectangle_perimeter_l1921_192176


namespace preimage_of_three_one_l1921_192155

/-- The mapping f from ℝ² to ℝ² -/
def f (p : ℝ × ℝ) : ℝ × ℝ := (p.1 + p.2, p.1 - p.2)

/-- Theorem stating that (2, 1) is the preimage of (3, 1) under f -/
theorem preimage_of_three_one :
  ∃! p : ℝ × ℝ, f p = (3, 1) ∧ p = (2, 1) := by
  sorry

end preimage_of_three_one_l1921_192155


namespace polynomial_equation_l1921_192102

theorem polynomial_equation (a : ℝ) (A : ℝ → ℝ) :
  (∀ x, A x * (x + 1) = x^2 - 1) → A = fun x ↦ x - 1 := by
  sorry

end polynomial_equation_l1921_192102


namespace katies_first_stopover_l1921_192108

/-- Calculates the distance to the first stopover given the total distance,
    distance to the second stopover, and additional distance to the final destination -/
def distance_to_first_stopover (total_distance : ℕ) (second_stopover : ℕ) (additional_distance : ℕ) : ℕ :=
  second_stopover - (total_distance - second_stopover - additional_distance)

/-- Proves that given the specific distances in Katie's trip,
    the distance to the first stopover is 104 miles -/
theorem katies_first_stopover :
  distance_to_first_stopover 436 236 68 = 104 := by
  sorry

#eval distance_to_first_stopover 436 236 68

end katies_first_stopover_l1921_192108


namespace weekend_grass_cutting_time_l1921_192127

/-- Calculates the total time Jason spends cutting grass over the weekend -/
def total_weekend_time (small_time medium_time large_time break_time weather_delay : ℕ)
  (saturday_small saturday_medium saturday_large : ℕ)
  (sunday_medium sunday_large : ℕ) : ℕ :=
  let saturday_time := 
    saturday_small * small_time + 
    saturday_medium * medium_time + 
    saturday_large * large_time + 
    (saturday_small + saturday_medium + saturday_large - 1) * break_time
  let sunday_time := 
    sunday_medium * (medium_time + weather_delay) + 
    sunday_large * (large_time + weather_delay) + 
    (sunday_medium + sunday_large - 1) * break_time
  saturday_time + sunday_time

theorem weekend_grass_cutting_time :
  total_weekend_time 25 30 40 5 10 2 4 2 6 2 = 11 * 60 := by
  sorry

end weekend_grass_cutting_time_l1921_192127


namespace grouping_schemes_l1921_192119

theorem grouping_schemes (drivers : Finset α) (ticket_sellers : Finset β) :
  (drivers.card = 4) → (ticket_sellers.card = 4) →
  (Finset.product drivers ticket_sellers).card = 24 := by
sorry

end grouping_schemes_l1921_192119


namespace joan_spent_four_half_dollars_on_wednesday_l1921_192121

/-- The number of half-dollars Joan spent on Wednesday -/
def wednesday_half_dollars : ℕ := sorry

/-- The number of half-dollars Joan spent on Thursday -/
def thursday_half_dollars : ℕ := 14

/-- The value of a half-dollar in dollars -/
def half_dollar_value : ℚ := 1/2

/-- The total amount Joan spent in dollars -/
def total_spent : ℚ := 9

/-- Theorem: Joan spent 4 half-dollars on Wednesday -/
theorem joan_spent_four_half_dollars_on_wednesday :
  wednesday_half_dollars = 4 :=
by
  have h1 : (wednesday_half_dollars : ℚ) * half_dollar_value + 
            (thursday_half_dollars : ℚ) * half_dollar_value = total_spent :=
    sorry
  sorry

end joan_spent_four_half_dollars_on_wednesday_l1921_192121


namespace square_division_l1921_192146

theorem square_division (a n : ℕ) : 
  a > 0 → 
  n > 1 → 
  a^2 = 88 + n^2 → 
  (a = 13 ∧ n = 9) ∨ (a = 23 ∧ n = 21) :=
by sorry

end square_division_l1921_192146


namespace triangle_side_length_l1921_192178

/-- In a triangle ABC, if a = 1, c = 2, and B = 60°, then b = √3 -/
theorem triangle_side_length (a c b : ℝ) (B : ℝ) : 
  a = 1 → c = 2 → B = π / 3 → b^2 = a^2 + c^2 - 2*a*c*(Real.cos B) → b = Real.sqrt 3 := by
  sorry

end triangle_side_length_l1921_192178


namespace granger_age_difference_l1921_192104

theorem granger_age_difference : 
  let granger_age : ℕ := 42
  let son_age : ℕ := 16
  granger_age - 2 * son_age = 10 := by sorry

end granger_age_difference_l1921_192104


namespace fifth_pile_magazines_l1921_192171

def magazine_sequence : ℕ → ℕ
  | 0 => 3
  | 1 => 4
  | 2 => 6
  | 3 => 9
  | n + 4 => magazine_sequence n + (n + 1)

theorem fifth_pile_magazines : magazine_sequence 4 = 13 := by
  sorry

end fifth_pile_magazines_l1921_192171


namespace male_kittens_count_l1921_192166

/-- Given an initial number of cats, number of female kittens, and total number of cats after kittens are born, 
    calculate the number of male kittens. -/
def male_kittens (initial_cats female_kittens total_cats : ℕ) : ℕ :=
  total_cats - initial_cats - female_kittens

/-- Theorem stating that given the problem conditions, the number of male kittens is 2. -/
theorem male_kittens_count : male_kittens 2 3 7 = 2 := by
  sorry

end male_kittens_count_l1921_192166


namespace product_of_distinct_roots_l1921_192138

theorem product_of_distinct_roots (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hxy : x ≠ y) 
  (h : x + 3 / x = y + 3 / y) : x * y = 3 := by
  sorry

end product_of_distinct_roots_l1921_192138


namespace arithmetic_geometric_sequence_ratio_l1921_192188

theorem arithmetic_geometric_sequence_ratio (d : ℝ) (q : ℚ) (a b : ℕ → ℝ) :
  d ≠ 0 →
  0 < q →
  q < 1 →
  (∀ n, a (n + 1) = a n + d) →
  (∀ n, b (n + 1) = q * b n) →
  a 1 = d →
  b 1 = d^2 →
  ∃ m : ℕ+, (a 1^2 + a 2^2 + a 3^2) / (b 1 + b 2 + b 3) = m →
  q = 1/2 := by
sorry

end arithmetic_geometric_sequence_ratio_l1921_192188


namespace equal_selection_probabilities_l1921_192107

/-- Represents a sampling method -/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Calculate the probability of selection for a given sampling method -/
def probability_of_selection (method : SamplingMethod) (population_size : ℕ) (sample_size : ℕ) : ℚ :=
  match method with
  | SamplingMethod.SimpleRandom => sample_size / population_size
  | SamplingMethod.Systematic => sample_size / population_size
  | SamplingMethod.Stratified => sample_size / population_size

theorem equal_selection_probabilities (population_size : ℕ) (sample_size : ℕ)
    (h1 : population_size = 50)
    (h2 : sample_size = 10) :
    (probability_of_selection SamplingMethod.SimpleRandom population_size sample_size =
     probability_of_selection SamplingMethod.Systematic population_size sample_size) ∧
    (probability_of_selection SamplingMethod.SimpleRandom population_size sample_size =
     probability_of_selection SamplingMethod.Stratified population_size sample_size) ∧
    (probability_of_selection SamplingMethod.SimpleRandom population_size sample_size = 1/5) :=
  sorry

end equal_selection_probabilities_l1921_192107


namespace journey_theorem_l1921_192103

/-- Represents the journey to Koschei's kingdom -/
structure Journey where
  total_distance : ℝ
  first_day_distance : ℝ
  second_day_distance : ℝ
  third_day_distance : ℝ
  fourth_day_distance : ℝ

/-- The conditions of Leshy's journey -/
def leshy_journey (j : Journey) : Prop :=
  j.first_day_distance = j.total_distance / 3 ∧
  j.second_day_distance = j.first_day_distance / 2 ∧
  j.third_day_distance = j.first_day_distance ∧
  j.fourth_day_distance = 100 ∧
  j.total_distance = j.first_day_distance + j.second_day_distance + j.third_day_distance + j.fourth_day_distance

theorem journey_theorem (j : Journey) (h : leshy_journey j) :
  j.total_distance = 600 ∧ j.fourth_day_distance = 100 := by
  sorry

#check journey_theorem

end journey_theorem_l1921_192103


namespace jimmy_needs_four_packs_l1921_192145

/-- The number of packs of bread Jimmy needs to buy for his sandwiches -/
def breadPacksNeeded (sandwiches : ℕ) (slicesPerSandwich : ℕ) (slicesPerPack : ℕ) : ℕ :=
  (sandwiches * slicesPerSandwich + slicesPerPack - 1) / slicesPerPack

/-- Proof that Jimmy needs to buy 4 packs of bread -/
theorem jimmy_needs_four_packs :
  breadPacksNeeded 8 2 4 = 4 := by
  sorry

end jimmy_needs_four_packs_l1921_192145


namespace one_cow_one_bag_days_l1921_192139

def num_cows : ℕ := 45
def num_bags : ℕ := 90
def num_days : ℕ := 60

theorem one_cow_one_bag_days : 
  (num_days * num_cows) / num_bags = 30 := by
  sorry

end one_cow_one_bag_days_l1921_192139


namespace basketball_team_score_l1921_192122

theorem basketball_team_score (two_pointers three_pointers free_throws : ℕ) : 
  (3 * three_pointers = 2 * (2 * two_pointers)) →
  (free_throws = 2 * two_pointers) →
  (2 * two_pointers + 3 * three_pointers + free_throws = 65) →
  free_throws = 18 := by
sorry

end basketball_team_score_l1921_192122


namespace final_distance_after_two_hours_l1921_192187

/-- The distance between Jay and Paul after walking for a given time -/
def distance_after_time (initial_distance : ℝ) (jay_speed : ℝ) (paul_speed : ℝ) (time : ℝ) : ℝ :=
  initial_distance + jay_speed * time + paul_speed * time

/-- Theorem stating the final distance between Jay and Paul after 2 hours -/
theorem final_distance_after_two_hours :
  let initial_distance : ℝ := 3
  let jay_speed : ℝ := 1 / (20 / 60) -- miles per hour
  let paul_speed : ℝ := 3 / (40 / 60) -- miles per hour
  let time : ℝ := 2 -- hours
  distance_after_time initial_distance jay_speed paul_speed time = 18 := by
  sorry


end final_distance_after_two_hours_l1921_192187


namespace textbook_cost_calculation_l1921_192168

theorem textbook_cost_calculation : 
  let sale_price : ℝ := 15 * (1 - 0.2)
  let sale_books : ℝ := 5
  let friend_books_cost : ℝ := 12 + 2 * 15
  let online_books_cost : ℝ := 45 * (1 - 0.1)
  let bookstore_books_cost : ℝ := 3 * 45
  let tax_rate : ℝ := 0.08
  
  sale_price * sale_books + friend_books_cost + online_books_cost + bookstore_books_cost + 
  ((sale_price * sale_books + friend_books_cost + online_books_cost + bookstore_books_cost) * tax_rate) = 299.70 := by
sorry


end textbook_cost_calculation_l1921_192168


namespace boys_present_age_boys_present_age_proof_l1921_192170

theorem boys_present_age : ℕ → Prop :=
  fun x => (x + 4 = 2 * (x - 6)) → x = 16

-- The proof is omitted
theorem boys_present_age_proof : ∃ x : ℕ, boys_present_age x :=
  sorry

end boys_present_age_boys_present_age_proof_l1921_192170


namespace perpendicular_vectors_imply_m_l1921_192185

/-- Given points A, B, and C in a 2D plane, prove that if AB is perpendicular to BC,
    then the x-coordinate of C is 8/3. -/
theorem perpendicular_vectors_imply_m (A B C : ℝ × ℝ) :
  A = (-1, 3) →
  B = (2, 1) →
  C.2 = 2 →
  (B.1 - A.1, B.2 - A.2) • (C.1 - B.1, C.2 - B.2) = 0 →
  C.1 = 8/3 := by
  sorry

#check perpendicular_vectors_imply_m

end perpendicular_vectors_imply_m_l1921_192185


namespace inequality_proof_l1921_192129

theorem inequality_proof (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0) 
  (h_ineq : 2 * (a + b + c + d) ≥ a * b * c * d) : 
  a^2 + b^2 + c^2 + d^2 ≥ a * b * c * d := by
sorry

end inequality_proof_l1921_192129


namespace opposite_signs_abs_sum_less_abs_diff_l1921_192110

theorem opposite_signs_abs_sum_less_abs_diff (x y : ℝ) (h : x * y < 0) :
  |x + y| < |x - y| := by
  sorry

end opposite_signs_abs_sum_less_abs_diff_l1921_192110


namespace linear_function_implies_m_equals_negative_one_l1921_192193

theorem linear_function_implies_m_equals_negative_one (m : ℝ) :
  (∃ a b : ℝ, ∀ x y : ℝ, y = (m^2 - m) * x / (m^2 + 1) ↔ y = a * x + b) →
  m = -1 := by
  sorry

end linear_function_implies_m_equals_negative_one_l1921_192193


namespace m_equals_two_iff_parallel_l1921_192134

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := 2 * x - m * y - 1 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := (m - 1) * x - y + 1 = 0

-- Define parallel lines
def parallel (m : ℝ) : Prop := ∀ (x y : ℝ), l₁ m x y ↔ l₂ m x y

-- Theorem statement
theorem m_equals_two_iff_parallel :
  ∀ m : ℝ, m = 2 ↔ parallel m := by sorry

end m_equals_two_iff_parallel_l1921_192134


namespace consecutive_integers_sum_l1921_192113

theorem consecutive_integers_sum (x : ℤ) : 
  x + 1 < 20 → 
  x * (x + 1) + x + (x + 1) = 156 → 
  x + (x + 1) = 23 := by
sorry

end consecutive_integers_sum_l1921_192113


namespace sunglasses_cost_theorem_l1921_192100

/-- The cost of sunglasses for a vendor -/
theorem sunglasses_cost_theorem 
  (selling_price : ℝ) 
  (pairs_sold : ℕ) 
  (sign_cost : ℝ) 
  (h1 : selling_price = 30)
  (h2 : pairs_sold = 10)
  (h3 : sign_cost = 20)
  (h4 : (pairs_sold : ℝ) * (selling_price - cost_per_pair) / 2 = sign_cost) :
  cost_per_pair = 26 := by
  sorry

#check sunglasses_cost_theorem

end sunglasses_cost_theorem_l1921_192100
