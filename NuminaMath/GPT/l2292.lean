import Mathlib

namespace min_value_frac_l2292_229270

theorem min_value_frac (x y : ℝ) (h1 : x > -1) (h2 : y > 0) (h3 : x + 2 * y = 2) : 
  ∃ L, (L = 3) ∧ (∀ (x y : ℝ), x > -1 → y > 0 → x + 2*y = 2 → 
  (∃ L, (L = 3) ∧ (∀ (x y : ℝ), x > -1 → y > 0 → x + 2*y = 2 → 
  ∀ (f : ℝ), f = (1 / (x + 1) + 2 / y) → f ≥ L))) :=
sorry

end min_value_frac_l2292_229270


namespace circle_radii_order_l2292_229215

theorem circle_radii_order (r_A r_B r_C : ℝ) 
  (h1 : r_A = Real.sqrt 10) 
  (h2 : 2 * Real.pi * r_B = 10 * Real.pi)
  (h3 : Real.pi * r_C^2 = 16 * Real.pi) : 
  r_C < r_A ∧ r_A < r_B := 
  sorry

end circle_radii_order_l2292_229215


namespace bella_started_with_136_candies_l2292_229226

/-
Theorem:
Bella started with 136 candies.
-/

-- define the initial number of candies
variable (x : ℝ)

-- define the conditions
def condition1 : Prop := (x / 2 - 3 / 4) - 5 = 9
def condition2 : Prop := x = 136

-- structure the proof statement 
theorem bella_started_with_136_candies : condition1 x -> condition2 x :=
by
  sorry

end bella_started_with_136_candies_l2292_229226


namespace cages_used_l2292_229277

theorem cages_used (total_puppies sold_puppies puppies_per_cage remaining_puppies needed_cages additional_cage total_cages: ℕ) 
  (h1 : total_puppies = 36) 
  (h2 : sold_puppies = 7) 
  (h3 : puppies_per_cage = 4) 
  (h4 : remaining_puppies = total_puppies - sold_puppies) 
  (h5 : needed_cages = remaining_puppies / puppies_per_cage) 
  (h6 : additional_cage = if (remaining_puppies % puppies_per_cage = 0) then 0 else 1) 
  (h7 : total_cages = needed_cages + additional_cage) : 
  total_cages = 8 := 
by 
  sorry

end cages_used_l2292_229277


namespace original_solution_is_10_percent_l2292_229221

def sugar_percentage_original_solution (x : ℕ) :=
  (3 / 4 : ℚ) * x + (1 / 4 : ℚ) * 42 = 18

theorem original_solution_is_10_percent : sugar_percentage_original_solution 10 :=
by
  unfold sugar_percentage_original_solution
  norm_num

end original_solution_is_10_percent_l2292_229221


namespace reptile_house_animal_multiple_l2292_229268

theorem reptile_house_animal_multiple (R F x : ℕ) (hR : R = 16) (hF : F = 7) (hCond : R = x * F - 5) : x = 3 := by
  sorry

end reptile_house_animal_multiple_l2292_229268


namespace present_age_of_eldest_is_45_l2292_229282

theorem present_age_of_eldest_is_45 (x : ℕ) 
  (h1 : (5 * x - 10) + (7 * x - 10) + (8 * x - 10) + (9 * x - 10) = 107) :
  9 * x = 45 :=
sorry

end present_age_of_eldest_is_45_l2292_229282


namespace correct_operation_l2292_229283

theorem correct_operation (x y : ℝ) : (x^3 * y^2 - y^2 * x^3 = 0) :=
by sorry

end correct_operation_l2292_229283


namespace total_hiking_distance_l2292_229295

def saturday_distance : ℝ := 8.2
def sunday_distance : ℝ := 1.6
def total_distance (saturday_distance sunday_distance : ℝ) : ℝ := saturday_distance + sunday_distance

theorem total_hiking_distance :
  total_distance saturday_distance sunday_distance = 9.8 :=
by
  -- The proof is omitted
  sorry

end total_hiking_distance_l2292_229295


namespace bigger_number_l2292_229274

theorem bigger_number (yoongi : ℕ) (jungkook : ℕ) (h1 : yoongi = 4) (h2 : jungkook = 6 + 3) : jungkook > yoongi :=
by
  sorry

end bigger_number_l2292_229274


namespace inequality_condition_sufficient_l2292_229200

theorem inequality_condition_sufficient (A B C : ℝ) (x y z : ℝ) 
  (hA : 0 ≤ A) 
  (hB : 0 ≤ B) 
  (hC : 0 ≤ C) 
  (hABC : A^2 + B^2 + C^2 ≤ 2 * (A * B + A * C + B * C)) :
  A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0 :=
sorry

end inequality_condition_sufficient_l2292_229200


namespace function_even_l2292_229260

theorem function_even (n : ℤ) (h : 30 ∣ n)
    (h_prop: (1 : ℝ)^n^2 + (-1: ℝ)^n^2 = 2 * ((1: ℝ)^n + (-1: ℝ)^n - 1)) :
    ∀ x : ℝ, (x^n = (-x)^n) :=
by
    sorry

end function_even_l2292_229260


namespace maximum_additional_payment_expected_value_difference_l2292_229253

-- Add the conditions as definitions
def a1 : ℕ := 1298
def a2 : ℕ := 1347
def a3 : ℕ := 1337
def b1 : ℕ := 1402
def b2 : ℕ := 1310
def b3 : ℕ := 1298

-- Prices in rubles per kilowatt-hour
def peak_price : ℝ := 4.03
def night_price : ℝ := 1.01
def semi_peak_price : ℝ := 3.39

-- Actual consumptions in kilowatt-hour
def ΔP : ℝ := 104
def ΔN : ℝ := 37
def ΔSP : ℝ := 39

-- Correct payment calculated by the company
def correct_payment : ℝ := 660.72

-- Statements to prove
theorem maximum_additional_payment : 397.34 = (104 * 4.03 + 39 * 3.39 + 37 * 1.01 - 660.72) :=
by
  sorry

theorem expected_value_difference : 19.3 = ((5 * 1402 + 3 * 1347 + 1337 - 1298 - 3 * 1270 - 5 * 1214) / 15 * 8.43 - 660.72) :=
by
  sorry

end maximum_additional_payment_expected_value_difference_l2292_229253


namespace min_C2_minus_D2_is_36_l2292_229285

noncomputable def find_min_C2_minus_D2 (x y z : ℝ) : ℝ :=
  (Real.sqrt (x + 3) + Real.sqrt (y + 6) + Real.sqrt (z + 11))^2 -
  (Real.sqrt (x + 2) + Real.sqrt (y + 2) + Real.sqrt (z + 2))^2

theorem min_C2_minus_D2_is_36 : ∀ (x y z : ℝ), 0 ≤ x → 0 ≤ y → 0 ≤ z → 
  find_min_C2_minus_D2 x y z ≥ 36 :=
by
  intros x y z hx hy hz
  sorry

end min_C2_minus_D2_is_36_l2292_229285


namespace largest_c_such_that_neg5_in_range_l2292_229264

theorem largest_c_such_that_neg5_in_range :
  ∃ (c : ℝ), (∀ x : ℝ, x^2 + 5 * x + c = -5) → c = 5 / 4 :=
sorry

end largest_c_such_that_neg5_in_range_l2292_229264


namespace trapezoid_area_l2292_229293

theorem trapezoid_area (EF GH EG FH : ℝ) (h : ℝ)
  (h_EF : EF = 60) (h_GH : GH = 30) (h_EG : EG = 25) (h_FH : FH = 18) (h_alt : h = 15) :
  (1 / 2 * (EF + GH) * h) = 675 :=
by
  rw [h_EF, h_GH, h_alt]
  sorry

end trapezoid_area_l2292_229293


namespace length_of_train_l2292_229206

variable (L V : ℝ)

def platform_crossing (L V : ℝ) := L + 350 = V * 39
def post_crossing (L V : ℝ) := L = V * 18

theorem length_of_train (h1 : platform_crossing L V) (h2 : post_crossing L V) : L = 300 :=
by
  sorry

end length_of_train_l2292_229206


namespace range_of_a_l2292_229214

theorem range_of_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 1 < 0) ↔ (a < -2 ∨ a > 2) := 
by
  sorry

end range_of_a_l2292_229214


namespace julia_more_kids_on_monday_l2292_229276

-- Definition of the problem statement
def playedWithOnMonday : ℕ := 6
def playedWithOnTuesday : ℕ := 5
def difference := playedWithOnMonday - playedWithOnTuesday

theorem julia_more_kids_on_monday : difference = 1 :=
by
  -- Proof can be filled out here.
  sorry

end julia_more_kids_on_monday_l2292_229276


namespace largest_interesting_number_l2292_229211

def is_interesting_number (x : ℝ) : Prop :=
  ∃ y z : ℝ, (0 ≤ y ∧ y < 1) ∧ (0 ≤ z ∧ z < 1) ∧ x = 0 + y * 10⁻¹ + z ∧ 2 * (0 + y * 10⁻¹ + z) = 0 + z

theorem largest_interesting_number : ∀ x, is_interesting_number x → x ≤ 0.375 :=
by
  sorry

end largest_interesting_number_l2292_229211


namespace six_hundred_sixes_not_square_l2292_229256

theorem six_hundred_sixes_not_square : 
  ∀ (n : ℕ), (n = 66666666666666666666666666666666666666666666666666666666666 -- continued 600 times
  ∨ n = 66666666666666666666666666666666666666666666666666666666666 -- continued with some zeros
  ) → ¬ (∃ k : ℕ, k * k = n) := 
by
  sorry

end six_hundred_sixes_not_square_l2292_229256


namespace length_AB_of_parabola_l2292_229289

theorem length_AB_of_parabola (x1 x2 : ℝ)
  (h : x1 + x2 = 6) :
  abs (x1 + x2 + 2) = 8 := by
  sorry

end length_AB_of_parabola_l2292_229289


namespace correct_adjacent_book_left_l2292_229202

-- Define the parameters
variable (prices : ℕ → ℕ)
variable (n : ℕ)
variable (step : ℕ)

-- Given conditions
axiom h1 : n = 31
axiom h2 : step = 2
axiom h3 : ∀ k : ℕ, 0 ≤ k ∧ k < n - 1 → prices (k + 1) = prices k + step
axiom h4 : prices 30 = prices 15 + prices 14

-- We need to show that the adjacent book referred to is at the left of the middle book.
theorem correct_adjacent_book_left (h : n = 31) (prices_step : ∀ k : ℕ, 0 ≤ k ∧ k < n - 1 → prices (k + 1) = prices k + step) : prices 30 = prices 15 + prices 14 := by
  sorry

end correct_adjacent_book_left_l2292_229202


namespace expr_simplified_l2292_229244

theorem expr_simplified : |2 - Real.sqrt 2| - Real.sqrt (1 / 12) * Real.sqrt 27 + Real.sqrt 12 / Real.sqrt 6 = 1 / 2 := 
by 
  sorry

end expr_simplified_l2292_229244


namespace green_function_solution_l2292_229275

noncomputable def G (x ξ : ℝ) (α : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ ξ then α + Real.log ξ else if ξ ≤ x ∧ x ≤ 1 then α + Real.log x else 0

theorem green_function_solution (x ξ α : ℝ) (hα : α ≠ 0) (hx_bound : 0 < x ∧ x ≤ 1) :
  ( G x ξ α = if 0 < x ∧ x ≤ ξ then α + Real.log ξ else if ξ ≤ x ∧ x ≤ 1 then α + Real.log x else 0 ) :=
sorry

end green_function_solution_l2292_229275


namespace calc_result_l2292_229261

theorem calc_result : 
  let a := 82 + 3/5
  let b := 1/15
  let c := 3
  let d := 42 + 7/10
  (a / b) * c - d = 3674.3 :=
by
  sorry

end calc_result_l2292_229261


namespace factor_polynomial_l2292_229232

noncomputable def polynomial (x y n : ℤ) : ℤ := x^2 + 4 * x * y + 2 * x + n * y - n

theorem factor_polynomial (n : ℤ) :
  (∃ A B C D E F : ℤ, polynomial A B C = (A * x + B * y + C) * (D * x + E * y + F)) ↔ n = 0 :=
sorry

end factor_polynomial_l2292_229232


namespace M_subset_N_iff_l2292_229233

section
variables {a x : ℝ}

-- Definitions based on conditions in the problem
def M (a : ℝ) : Set ℝ := { x | x^2 - a * x - x < 0 }
def N : Set ℝ := { x | x^2 - 2 * x - 3 ≤ 0 }

theorem M_subset_N_iff (a : ℝ) : M a ⊆ N ↔ -2 ≤ a ∧ a ≤ 2 :=
by
  sorry
end

end M_subset_N_iff_l2292_229233


namespace little_john_remaining_money_l2292_229222

noncomputable def initial_amount: ℝ := 8.50
noncomputable def spent_on_sweets: ℝ := 1.25
noncomputable def given_to_each_friend: ℝ := 1.20
noncomputable def number_of_friends: ℝ := 2

theorem little_john_remaining_money : 
  initial_amount - (spent_on_sweets + given_to_each_friend * number_of_friends) = 4.85 :=
by
  sorry

end little_john_remaining_money_l2292_229222


namespace shaded_region_area_l2292_229290

/-- A rectangle measuring 12cm by 8cm has four semicircles drawn with their diameters as the sides
of the rectangle. Prove that the area of the shaded region inside the rectangle but outside
the semicircles is equal to 96 - 52π (cm²). --/
theorem shaded_region_area (A : ℝ) (π : ℝ) (hA : A = 96 - 52 * π) : 
  ∀ (length width r1 r2 : ℝ) (hl : length = 12) (hw : width = 8) 
  (hr1 : r1 = length / 2) (hr2 : r2 = width / 2),
  (length * width) - (2 * (1/2 * π * r1^2 + 1/2 * π * r2^2)) = A := 
by 
  sorry

end shaded_region_area_l2292_229290


namespace find_g_of_2_l2292_229267

open Real

noncomputable def g : ℝ → ℝ := sorry

theorem find_g_of_2
  (H: ∀ x : ℝ, g (2 ^ x) + x * g (2 ^ (-x)) + x = 1) : g 2 = -1 :=
by
  sorry

end find_g_of_2_l2292_229267


namespace trajectory_eq_l2292_229237

theorem trajectory_eq {x y : ℝ} (h₁ : (x-2)^2 + y^2 = 1) (h₂ : ∃ r, (x+1)^2 = (x-2)^2 + y^2 - r^2) :
  y^2 = 6 * x - 3 :=
by
  sorry

end trajectory_eq_l2292_229237


namespace new_paint_intensity_l2292_229227

def red_paint_intensity (initial_intensity replacement_intensity : ℝ) (replacement_fraction : ℝ) : ℝ :=
  (1 - replacement_fraction) * initial_intensity + replacement_fraction * replacement_intensity

theorem new_paint_intensity :
  red_paint_intensity 0.1 0.2 0.5 = 0.15 :=
by sorry

end new_paint_intensity_l2292_229227


namespace difference_of_two_smallest_integers_l2292_229225

/--
The difference between the two smallest integers greater than 1 which, when divided by any integer 
\( k \) in the range from \( 3 \leq k \leq 13 \), leave a remainder of \( 2 \), is \( 360360 \).
-/
theorem difference_of_two_smallest_integers (n m : ℕ) (h_n : ∀ k : ℕ, 3 ≤ k ∧ k ≤ 13 → n % k = 2) (h_m : ∀ k : ℕ, 3 ≤ k ∧ k ≤ 13 → m % k = 2) (h_smallest : m > n) :
  m - n = 360360 :=
sorry

end difference_of_two_smallest_integers_l2292_229225


namespace real_m_of_complex_product_l2292_229296

-- Define the conditions that m is a real number and (m^2 + i)(1 - mi) is a real number
def is_real (z : ℂ) : Prop := z.im = 0
def cplx_eq (m : ℝ) : ℂ := (⟨m^2, 1⟩ : ℂ) * (⟨1, -m⟩ : ℂ)

theorem real_m_of_complex_product (m : ℝ) : is_real (cplx_eq m) ↔ m = 1 :=
by
  sorry

end real_m_of_complex_product_l2292_229296


namespace minimal_length_AX_XB_l2292_229238

theorem minimal_length_AX_XB 
  (AA' BB' : ℕ) (A'B' : ℕ) 
  (h1 : AA' = 680) (h2 : BB' = 2000) (h3 : A'B' = 2010) 
  : ∃ X : ℕ, AX + XB = 3350 := 
sorry

end minimal_length_AX_XB_l2292_229238


namespace jake_weight_l2292_229241

variable (J K : ℕ)

-- Conditions given in the problem
axiom h1 : J - 8 = 2 * K
axiom h2 : J + K = 293

-- Statement to prove
theorem jake_weight : J = 198 :=
by
  sorry

end jake_weight_l2292_229241


namespace boat_speed_in_still_water_l2292_229228

-- Problem Definitions
def V_s : ℕ := 16
def t : ℕ := sorry -- t is arbitrary positive value
def V_b : ℕ := 48

-- Conditions
def upstream_time := 2 * t
def downstream_time := t
def upstream_distance := (V_b - V_s) * upstream_time
def downstream_distance := (V_b + V_s) * downstream_time

-- Proof Problem
theorem boat_speed_in_still_water :
  upstream_distance = downstream_distance → V_b = 48 :=
by sorry

end boat_speed_in_still_water_l2292_229228


namespace minimum_value_expression_l2292_229250

theorem minimum_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (4 * z / (2 * x + y)) + (4 * x / (y + 2 * z)) + (y / (x + z)) ≥ 3 :=
by 
  sorry

end minimum_value_expression_l2292_229250


namespace intersection_A_B_eq_B_l2292_229212

-- Define set A
def setA : Set ℝ := { x : ℝ | x > -3 }

-- Define set B
def setB : Set ℝ := { x : ℝ | x ≥ 2 }

-- Theorem statement of proving the intersection of setA and setB is setB itself
theorem intersection_A_B_eq_B : setA ∩ setB = setB :=
by
  -- proof skipped
  sorry

end intersection_A_B_eq_B_l2292_229212


namespace area_of_smaller_circle_l2292_229224

theorem area_of_smaller_circle
  (PA AB : ℝ)
  (r s : ℝ)
  (tangent_at_T : true) -- placeholder; represents the tangency condition
  (common_tangents : true) -- placeholder; represents the external tangents condition
  (PA_eq_AB : PA = AB) :
  PA = 5 →
  AB = 5 →
  r = 2 * s →
  ∃ (s : ℝ) (area : ℝ), s = 5 / (2 * (Real.sqrt 2)) ∧ area = (Real.pi * s^2) ∧ area = (25 * Real.pi) / 8 := by
  intros hPA hAB h_r_s
  use 5 / (2 * (Real.sqrt 2))
  use (Real.pi * (5 / (2 * (Real.sqrt 2)))^2)
  simp [←hPA,←hAB]
  sorry

end area_of_smaller_circle_l2292_229224


namespace simplify_and_evaluate_expression_l2292_229297

variable (x y : ℝ)
variable (h1 : x = 1)
variable (h2 : y = Real.sqrt 2)

theorem simplify_and_evaluate_expression : 
  (x + 2 * y) ^ 2 - x * (x + 4 * y) + (1 - y) * (1 + y) = 7 := by
  sorry

end simplify_and_evaluate_expression_l2292_229297


namespace negation_of_proposition_p_l2292_229280

def has_real_root (m : ℝ) : Prop := ∃ x : ℝ, x^2 + m * x + 1 = 0

def negation_of_p : Prop := ∀ m : ℝ, ¬ has_real_root m

theorem negation_of_proposition_p : negation_of_p :=
by sorry

end negation_of_proposition_p_l2292_229280


namespace train_speed_l2292_229231

variable (length : ℕ) (time : ℕ)
variable (h_length : length = 120)
variable (h_time : time = 6)

theorem train_speed (length time : ℕ) (h_length : length = 120) (h_time : time = 6) :
  length / time = 20 := by
  sorry

end train_speed_l2292_229231


namespace equivalence_l2292_229286

theorem equivalence (a b c : ℝ) (h : a + c = 2 * b) : a^2 + 8 * b * c = (2 * b + c)^2 := 
by 
  sorry

end equivalence_l2292_229286


namespace intersection_A_B_subset_A_B_l2292_229248

-- Definition of sets A and B
def set_A (a : ℝ) : Set ℝ := {x | 0 < 2 * x + a ∧ 2 * x + a ≤ 3}
def set_B : Set ℝ := {x | -1 / 2 < x ∧ x < 2}

-- Problem 1: Prove A ∩ B when a = -1
theorem intersection_A_B (a : ℝ) (h : a = -1) : set_A a ∩ set_B = {x | 1 / 2 < x ∧ x < 2} :=
sorry

-- Problem 2: Find the range of a such that A ⊆ B
theorem subset_A_B (a : ℝ) : (-1 < a ∧ a ≤ 1) ↔ (set_A a ⊆ set_B) :=
sorry

end intersection_A_B_subset_A_B_l2292_229248


namespace smallest_of_seven_even_numbers_l2292_229278

theorem smallest_of_seven_even_numbers (a b c d e f g : ℕ) 
  (h1 : a % 2 = 0) 
  (h2 : b = a + 2) 
  (h3 : c = a + 4) 
  (h4 : d = a + 6) 
  (h5 : e = a + 8) 
  (h6 : f = a + 10) 
  (h7 : g = a + 12) 
  (h_sum : a + b + c + d + e + f + g = 700) : 
  a = 94 :=
by sorry

end smallest_of_seven_even_numbers_l2292_229278


namespace star_comm_star_distrib_over_add_star_special_case_star_no_identity_star_not_assoc_l2292_229287

def star (x y : ℤ) := (x + 2) * (y + 2) - 2

-- Statement A: commutativity
theorem star_comm : ∀ x y : ℤ, star x y = star y x := 
by sorry

-- Statement B: distributivity over addition
theorem star_distrib_over_add : ¬(∀ x y z : ℤ, star x (y + z) = star x y + star x z) :=
by sorry

-- Statement C: special case
theorem star_special_case : ¬(∀ x : ℤ, star (x - 2) (x + 2) = star x x - 2) :=
by sorry

-- Statement D: identity element
theorem star_no_identity : ¬(∃ e : ℤ, ∀ x : ℤ, star x e = x ∧ star e x = x) :=
by sorry

-- Statement E: associativity
theorem star_not_assoc : ¬(∀ x y z : ℤ, star (star x y) z = star x (star y z)) :=
by sorry

end star_comm_star_distrib_over_add_star_special_case_star_no_identity_star_not_assoc_l2292_229287


namespace no_such_pairs_exist_l2292_229229

theorem no_such_pairs_exist : ¬ ∃ (n m : ℕ), n > 1 ∧ (∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n) ∧ 
                                    (∀ d : ℕ, d ≠ n → d ∣ n → d + 1 ∣ m ∧ d + 1 ≠ m ∧ d + 1 ≠ 1) :=
by
  sorry

end no_such_pairs_exist_l2292_229229


namespace heather_bicycling_time_l2292_229291

theorem heather_bicycling_time (distance speed : ℕ) (h1 : distance = 96) (h2 : speed = 6) : 
(distance / speed) = 16 := by
  sorry

end heather_bicycling_time_l2292_229291


namespace rainy_days_l2292_229201

namespace Mo

def drinks (R NR n : ℕ) :=
  -- Condition 3: Total number of days in the week equation
  R + NR = 7 ∧
  -- Condition 1-2: Total cups of drinks equation
  n * R + 3 * NR = 26 ∧
  -- Condition 4: Difference in cups of tea and hot chocolate equation
  3 * NR - n * R = 10

theorem rainy_days (R NR n : ℕ) (h: drinks R NR n) : 
  R = 1 := sorry

end Mo

end rainy_days_l2292_229201


namespace maxwell_distance_when_meeting_l2292_229252

variable (total_distance : ℝ := 50)
variable (maxwell_speed : ℝ := 4)
variable (brad_speed : ℝ := 6)
variable (t : ℝ := total_distance / (maxwell_speed + brad_speed))

theorem maxwell_distance_when_meeting :
  (maxwell_speed * t = 20) :=
by
  sorry

end maxwell_distance_when_meeting_l2292_229252


namespace john_paint_area_l2292_229259

noncomputable def area_to_paint (length width height openings : ℝ) : ℝ :=
  let wall_area := 2 * (length * height) + 2 * (width * height)
  let ceiling_area := length * width
  let total_area := wall_area + ceiling_area
  total_area - openings

theorem john_paint_area :
  let length := 15
  let width := 12
  let height := 10
  let openings := 70
  let bedrooms := 2
  2 * (area_to_paint length width height openings) = 1300 :=
by
  let length := 15
  let width := 12
  let height := 10
  let openings := 70
  let bedrooms := 2
  sorry

end john_paint_area_l2292_229259


namespace exponent_multiplication_l2292_229234

theorem exponent_multiplication :
  (10 ^ 10000) * (10 ^ 8000) = 10 ^ 18000 :=
by
  sorry

end exponent_multiplication_l2292_229234


namespace total_investment_is_correct_l2292_229207

def Raghu_investment : ℕ := 2300
def Trishul_investment (Raghu_investment : ℕ) : ℕ := Raghu_investment - (Raghu_investment / 10)
def Vishal_investment (Trishul_investment : ℕ) : ℕ := Trishul_investment + (Trishul_investment / 10)

theorem total_investment_is_correct :
    let Raghu_inv := Raghu_investment;
    let Trishul_inv := Trishul_investment Raghu_inv;
    let Vishal_inv := Vishal_investment Trishul_inv;
    Raghu_inv + Trishul_inv + Vishal_inv = 6647 :=
by
    sorry

end total_investment_is_correct_l2292_229207


namespace cos_double_angle_l2292_229284

theorem cos_double_angle 
  (α β : ℝ) 
  (h1 : Real.sin (α - β) = 1/3) 
  (h2 : Real.cos α * Real.sin β = 1/6) :
  Real.cos (2 * α + 2 * β) = 1/9 :=
sorry

end cos_double_angle_l2292_229284


namespace sum_of_three_numbers_l2292_229204

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a + b = 35) 
  (h2 : b + c = 57) 
  (h3 : c + a = 62) : 
  a + b + c = 77 :=
by
  sorry

end sum_of_three_numbers_l2292_229204


namespace sheena_sewing_weeks_l2292_229294

theorem sheena_sewing_weeks (sew_time : ℕ) (bridesmaids : ℕ) (sewing_per_week : ℕ) 
    (h_sew_time : sew_time = 12) (h_bridesmaids : bridesmaids = 5) (h_sewing_per_week : sewing_per_week = 4) : 
    (bridesmaids * sew_time) / sewing_per_week = 15 := 
  by sorry

end sheena_sewing_weeks_l2292_229294


namespace unique_ordered_pairs_satisfying_equation_l2292_229281

theorem unique_ordered_pairs_satisfying_equation :
  ∃ x y : ℕ, 0 < x ∧ 0 < y ∧ x^6 * y^6 - 19 * x^3 * y^3 + 18 = 0 ↔ (x, y) = (1, 1) ∧
  (∀ x y : ℕ, 0 < x ∧ 0 < y ∧ x^6 * y^6 - 19 * x^3 * y^3 + 18 = 0 → (x, y) = (1, 1)) :=
by
  sorry

end unique_ordered_pairs_satisfying_equation_l2292_229281


namespace clock_hand_positions_l2292_229239

theorem clock_hand_positions : ∃ n : ℕ, n = 143 ∧ 
  (∀ t : ℝ, let hour_pos := t / 12
            let min_pos := t
            let switched_hour_pos := t
            let switched_min_pos := t / 12
            hour_pos = switched_min_pos ∧ min_pos = switched_hour_pos ↔
            ∃ k : ℤ, t = k / 11) :=
by sorry

end clock_hand_positions_l2292_229239


namespace geometric_sequence_increasing_condition_l2292_229240

noncomputable def is_geometric (a : ℕ → ℝ) : Prop :=
∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_increasing_condition (a : ℕ → ℝ) (h_geo : is_geometric a) (h_cond : a 0 < a 1 ∧ a 1 < a 2) :
  ¬(∀ n : ℕ, a n < a (n + 1)) → (a 0 < a 1 ∧ a 1 < a 2) :=
sorry

end geometric_sequence_increasing_condition_l2292_229240


namespace unfair_coin_probability_l2292_229279

theorem unfair_coin_probability (P : ℕ → ℝ) :
  let heads := 3/4
  let initial_condition := P 0 = 1
  let recurrence_relation := ∀n, P (n + 1) = 3 / 4 * (1 - P n) + 1 / 4 * P n
  recurrence_relation →
  initial_condition →
  P 40 = 1 / 2 * (1 + (1 / 2) ^ 40) :=
by
  sorry

end unfair_coin_probability_l2292_229279


namespace elizabeth_spendings_elizabeth_savings_l2292_229208

section WeddingGift

def steak_knife_set_cost : ℝ := 80
def steak_knife_sets : ℕ := 2
def dinnerware_set_cost : ℝ := 200
def fancy_napkins_sets : ℕ := 3
def fancy_napkins_total_cost : ℝ := 45
def wine_glasses_cost : ℝ := 100
def discount_steak_dinnerware : ℝ := 0.10
def discount_napkins : ℝ := 0.20
def sales_tax : ℝ := 0.05

def total_cost_before_discounts : ℝ :=
  (steak_knife_sets * steak_knife_set_cost) + dinnerware_set_cost + fancy_napkins_total_cost + wine_glasses_cost

def total_discount : ℝ :=
  ((steak_knife_sets * steak_knife_set_cost) * discount_steak_dinnerware) + (dinnerware_set_cost * discount_steak_dinnerware) + (fancy_napkins_total_cost * discount_napkins)

def total_cost_after_discounts : ℝ :=
  total_cost_before_discounts - total_discount

def total_cost_with_tax : ℝ :=
  total_cost_after_discounts + (total_cost_after_discounts * sales_tax)

def savings : ℝ :=
  total_cost_before_discounts - total_cost_after_discounts

theorem elizabeth_spendings :
  total_cost_with_tax = 558.60 :=
by sorry

theorem elizabeth_savings :
  savings = 63 :=
by sorry

end WeddingGift

end elizabeth_spendings_elizabeth_savings_l2292_229208


namespace find_picture_area_l2292_229217

variable (x y : ℕ)
    (h1 : x > 1)
    (h2 : y > 1)
    (h3 : (3 * x + 2) * (y + 4) - x * y = 62)

theorem find_picture_area : x * y = 10 :=
by
  sorry

end find_picture_area_l2292_229217


namespace number_of_workers_l2292_229219

-- Definitions corresponding to problem conditions
def total_contribution := 300000
def extra_total_contribution := 325000
def extra_amount := 50

-- Main statement to prove the number of workers
theorem number_of_workers : ∃ W C : ℕ, W * C = total_contribution ∧ W * (C + extra_amount) = extra_total_contribution ∧ W = 500 := by
  sorry

end number_of_workers_l2292_229219


namespace intersection_condition_sufficient_but_not_necessary_l2292_229298

theorem intersection_condition_sufficient_but_not_necessary (k : ℝ) :
  (-Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3) →
  ((∃ x : ℝ, (k^2 + 1) * x^2 + (2 * k^2 - 2) * x + k^2 = 0) ∧ 
   ¬ (∃ k, (∃ x : ℝ, (k^2 + 1) * x^2 + (2 * k^2 - 2) * x + k^2 = 0) → 
   (¬ (-Real.sqrt 3 / 3 < k ∧ k < Real.sqrt 3 / 3)))) :=
sorry

end intersection_condition_sufficient_but_not_necessary_l2292_229298


namespace arithmetic_value_l2292_229258

theorem arithmetic_value : (8 * 4) + 3 = 35 := by
  sorry

end arithmetic_value_l2292_229258


namespace partner_profit_share_correct_l2292_229216

-- Definitions based on conditions
def total_profit : ℝ := 280000
def profit_share_shekhar : ℝ := 0.28
def profit_share_rajeev : ℝ := 0.22
def profit_share_jatin : ℝ := 0.20
def profit_share_simran : ℝ := 0.18
def profit_share_ramesh : ℝ := 0.12

-- Each partner's share in the profit
def shekhar_share : ℝ := profit_share_shekhar * total_profit
def rajeev_share : ℝ := profit_share_rajeev * total_profit
def jatin_share : ℝ := profit_share_jatin * total_profit
def simran_share : ℝ := profit_share_simran * total_profit
def ramesh_share : ℝ := profit_share_ramesh * total_profit

-- Statement to be proved
theorem partner_profit_share_correct :
    shekhar_share = 78400 ∧ 
    rajeev_share = 61600 ∧ 
    jatin_share = 56000 ∧ 
    simran_share = 50400 ∧ 
    ramesh_share = 33600 ∧ 
    (shekhar_share + rajeev_share + jatin_share + simran_share + ramesh_share = total_profit) :=
by sorry

end partner_profit_share_correct_l2292_229216


namespace daily_chicken_loss_l2292_229236

/--
A small poultry farm has initially 300 chickens, 200 turkeys, and 80 guinea fowls. Every day, the farm loses some chickens, 8 turkeys, and 5 guinea fowls. After one week (7 days), there are 349 birds left in the farm. Prove the number of chickens the farmer loses daily.
-/
theorem daily_chicken_loss (initial_chickens initial_turkeys initial_guinea_fowls : ℕ)
  (daily_turkey_loss daily_guinea_fowl_loss days total_birds_left : ℕ)
  (h1 : initial_chickens = 300)
  (h2 : initial_turkeys = 200)
  (h3 : initial_guinea_fowls = 80)
  (h4 : daily_turkey_loss = 8)
  (h5 : daily_guinea_fowl_loss = 5)
  (h6 : days = 7)
  (h7 : total_birds_left = 349)
  (h8 : initial_chickens + initial_turkeys + initial_guinea_fowls
       - (daily_turkey_loss * days + daily_guinea_fowl_loss * days + (initial_chickens - total_birds_left)) = total_birds_left) :
  initial_chickens - (total_birds_left + daily_turkey_loss * days + daily_guinea_fowl_loss * days) / days = 20 :=
by {
    -- Proof goes here
    sorry
}

end daily_chicken_loss_l2292_229236


namespace ratio_of_remaining_areas_of_squares_l2292_229245

/--
  Given:
  - Square C has a side length of 48 cm.
  - Square D has a side length of 60 cm.
  - A smaller square of side length 12 cm is cut out from both squares.

  Show that:
  - The ratio of the remaining area of square C to the remaining area of square D is 5/8.
-/
theorem ratio_of_remaining_areas_of_squares : 
  let sideC := 48
  let sideD := 60
  let sideSmall := 12
  let areaC := sideC * sideC
  let areaD := sideD * sideD
  let areaSmall := sideSmall * sideSmall
  let remainingC := areaC - areaSmall
  let remainingD := areaD - areaSmall
  (remainingC : ℚ) / remainingD = 5 / 8 :=
by
  sorry

end ratio_of_remaining_areas_of_squares_l2292_229245


namespace olivia_bags_count_l2292_229254

def cans_per_bag : ℕ := 5
def total_cans : ℕ := 20

theorem olivia_bags_count : total_cans / cans_per_bag = 4 := by
  sorry

end olivia_bags_count_l2292_229254


namespace ella_dog_food_ratio_l2292_229251

variable (ella_food_per_day : ℕ) (total_food_10days : ℕ) (x : ℕ)

theorem ella_dog_food_ratio
  (h1 : ella_food_per_day = 20)
  (h2 : total_food_10days = 1000) :
  (x : ℕ) = 4 :=
by
  sorry

end ella_dog_food_ratio_l2292_229251


namespace dot_product_a_b_l2292_229257

def a : ℝ × ℝ := (1, 2)
def b : ℝ × ℝ := (-3, 1)

theorem dot_product_a_b : (a.1 * b.1 + a.2 * b.2) = -1 := by
  sorry

end dot_product_a_b_l2292_229257


namespace fx_fixed_point_l2292_229213

theorem fx_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∃ x y, (x = -1) ∧ (y = 3) ∧ (a * (x + 1) + 2 = y) :=
by
  sorry

end fx_fixed_point_l2292_229213


namespace solution_of_inequality_system_l2292_229243

theorem solution_of_inequality_system (x : ℝ) : 
  (x - 1 < 0) ∧ (x + 1 > 0) ↔ (-1 < x) ∧ (x < 1) := 
by sorry

end solution_of_inequality_system_l2292_229243


namespace cubic_polynomial_roots_l2292_229230

noncomputable def cubic_polynomial (a_3 a_2 a_1 a_0 x : ℝ) : ℝ :=
  a_3 * x^3 + a_2 * x^2 + a_1 * x + a_0

theorem cubic_polynomial_roots (a_3 a_2 a_1 a_0 : ℝ) 
    (h_nonzero_a3 : a_3 ≠ 0)
    (r1 r2 r3 : ℝ)
    (h_roots : cubic_polynomial a_3 a_2 a_1 a_0 r1 = 0 ∧
               cubic_polynomial a_3 a_2 a_1 a_0 r2 = 0 ∧
               cubic_polynomial a_3 a_2 a_1 a_0 r3 = 0)
    (h_condition : (cubic_polynomial a_3 a_2 a_1 a_0 (1/2) 
                    + cubic_polynomial a_3 a_2 a_1 a_0 (-1/2)) 
                    / (cubic_polynomial a_3 a_2 a_1 a_0 0) = 1003) :
  (1 / (r1 * r2) + 1 / (r2 * r3) + 1 / (r3 * r1)) = 2002 :=
sorry

end cubic_polynomial_roots_l2292_229230


namespace total_pears_picked_l2292_229292

def pears_Alyssa : ℕ := 42
def pears_Nancy : ℕ := 17

theorem total_pears_picked : pears_Alyssa + pears_Nancy = 59 :=
by sorry

end total_pears_picked_l2292_229292


namespace ratio_volume_sphere_to_hemisphere_l2292_229262

noncomputable def volume_sphere (r : ℝ) : ℝ :=
  (4/3) * Real.pi * r^3

noncomputable def volume_hemisphere (r : ℝ) : ℝ :=
  (1/2) * volume_sphere r

theorem ratio_volume_sphere_to_hemisphere (p : ℝ) (hp : 0 < p) :
  (volume_sphere p) / (volume_hemisphere (2 * p)) = 1 / 4 :=
by
  sorry

end ratio_volume_sphere_to_hemisphere_l2292_229262


namespace trapezoid_area_is_correct_l2292_229265

noncomputable def trapezoid_area (base_short : ℝ) (angle_adj : ℝ) (angle_diag : ℝ) : ℝ :=
  let width := 2 * base_short -- calculated width from angle_adj
  let height := base_short / Real.tan (angle_adj / 2 * Real.pi / 180)
  (base_short + base_short + width) * height / 2

theorem trapezoid_area_is_correct :
  trapezoid_area 2 135 150 = 2 :=
by
  sorry

end trapezoid_area_is_correct_l2292_229265


namespace zoo_revenue_l2292_229249

def num_children_mon : ℕ := 7
def num_adults_mon : ℕ := 5
def num_children_tue : ℕ := 4
def num_adults_tue : ℕ := 2
def cost_child : ℕ := 3
def cost_adult : ℕ := 4

theorem zoo_revenue : 
  (num_children_mon * cost_child + num_adults_mon * cost_adult) + 
  (num_children_tue * cost_child + num_adults_tue * cost_adult) 
  = 61 := 
by
  sorry

end zoo_revenue_l2292_229249


namespace rake_yard_alone_time_l2292_229272

-- Definitions for the conditions
def brother_time := 45 -- Brother takes 45 minutes
def together_time := 18 -- Together it takes 18 minutes

-- Define and prove the time it takes you to rake the yard alone based on given conditions
theorem rake_yard_alone_time : 
  ∃ (x : ℕ), (1 / (x : ℚ) + 1 / (brother_time : ℚ) = 1 / (together_time : ℚ)) ∧ x = 30 :=
by
  sorry

end rake_yard_alone_time_l2292_229272


namespace icosagon_diagonals_l2292_229235

-- Definitions for the number of sides and the diagonal formula
def sides_icosagon : ℕ := 20

def diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

-- Statement:
theorem icosagon_diagonals : diagonals sides_icosagon = 170 := by
  apply sorry

end icosagon_diagonals_l2292_229235


namespace reciprocal_of_neg_eight_l2292_229210

theorem reciprocal_of_neg_eight : -8 * (-1/8) = 1 := 
by
  sorry

end reciprocal_of_neg_eight_l2292_229210


namespace find_total_cows_l2292_229203

-- Define the conditions given in the problem
def ducks_legs (D : ℕ) : ℕ := 2 * D
def cows_legs (C : ℕ) : ℕ := 4 * C
def total_legs (D C : ℕ) : ℕ := ducks_legs D + cows_legs C
def total_heads (D C : ℕ) : ℕ := D + C

-- State the problem in Lean 4
theorem find_total_cows (D C : ℕ) (h : total_legs D C = 2 * total_heads D C + 32) : C = 16 :=
sorry

end find_total_cows_l2292_229203


namespace annie_weeks_off_sick_l2292_229271

-- Define the conditions and the question
def weekly_hours_chess : ℕ := 2
def weekly_hours_drama : ℕ := 8
def weekly_hours_glee : ℕ := 3
def semester_weeks : ℕ := 12
def total_hours_before_midterms : ℕ := 52

-- Define the proof problem
theorem annie_weeks_off_sick :
  let total_weekly_hours := weekly_hours_chess + weekly_hours_drama + weekly_hours_glee
  let attended_weeks := total_hours_before_midterms / total_weekly_hours
  semester_weeks - attended_weeks = 8 :=
by
  -- Automatically prove by computation of above assumptions.
  sorry

end annie_weeks_off_sick_l2292_229271


namespace expr1_eval_expr2_eval_l2292_229218

theorem expr1_eval : (3 * Real.sqrt 27 - 2 * Real.sqrt 12) * (2 * Real.sqrt (16 / 3) + 3 * Real.sqrt (25 / 3)) = 115 := 
by
  -- Sorry serves as a placeholder for the proof.
  sorry

theorem expr2_eval : (5 * Real.sqrt 21 - 3 * Real.sqrt 15) / (5 * Real.sqrt (8 / 3) - 3 * Real.sqrt (5 / 3)) = 3 := 
by
  -- Sorry serves as a placeholder for the proof.
  sorry

end expr1_eval_expr2_eval_l2292_229218


namespace prove_x_minus_y_squared_l2292_229246

variable (x y : ℝ)
variable (h1 : (x + y)^2 = 64)
variable (h2 : x * y = 12)

theorem prove_x_minus_y_squared : (x - y)^2 = 16 :=
by
  sorry

end prove_x_minus_y_squared_l2292_229246


namespace find_cost_of_jersey_l2292_229266

def cost_of_jersey (J : ℝ) : Prop := 
  let shorts_cost := 15.20
  let socks_cost := 6.80
  let total_players := 16
  let total_cost := 752
  total_players * (J + shorts_cost + socks_cost) = total_cost

theorem find_cost_of_jersey : cost_of_jersey 25 :=
  sorry

end find_cost_of_jersey_l2292_229266


namespace machines_used_l2292_229273

variable (R S : ℕ)

/-- 
  A company has two types of machines, type R and type S. 
  Operating at a constant rate, a machine of type R does a certain job in 36 hours, 
  and a machine of type S does the job in 9 hours. 
  If the company used the same number of each type of machine to do the job in 12 hours, 
  then the company used 15 machines of type R.
-/
theorem machines_used (hR : ∀ ⦃n⦄, n * (1 / 36) + n * (1 / 9) = (1 / 12)) :
  R = 15 := 
by 
  sorry

end machines_used_l2292_229273


namespace cos_beta_value_l2292_229247

variable (α β : ℝ)
variable (h₁ : 0 < α ∧ α < π)
variable (h₂ : 0 < β ∧ β < π)
variable (h₃ : Real.sin (α + β) = 5 / 13)
variable (h₄ : Real.tan (α / 2) = 1 / 2)

theorem cos_beta_value : Real.cos β = -16 / 65 := by
  sorry

end cos_beta_value_l2292_229247


namespace gcd_lcm_product_24_36_l2292_229299

-- Definitions for gcd, lcm, and product for given numbers, skipping proof with sorry
theorem gcd_lcm_product_24_36 : Nat.gcd 24 36 * Nat.lcm 24 36 = 864 := by
  -- Sorry used to skip proof
  sorry

end gcd_lcm_product_24_36_l2292_229299


namespace hotel_R_greater_than_G_l2292_229269

variables (R G P : ℝ)

def hotel_charges_conditions :=
  P = 0.50 * R ∧ P = 0.80 * G

theorem hotel_R_greater_than_G :
  hotel_charges_conditions R G P → R = 1.60 * G :=
by
  sorry

end hotel_R_greater_than_G_l2292_229269


namespace Andy_late_minutes_l2292_229205

theorem Andy_late_minutes 
  (school_start : Nat := 8*60) -- 8:00 AM in minutes since midnight
  (normal_travel_time : Nat := 30) -- 30 minutes
  (red_light_stops : Nat := 3 * 4) -- 3 minutes each at 4 lights
  (construction_wait : Nat := 10) -- 10 minutes
  (detour_time : Nat := 7) -- 7 minutes
  (store_stop_time : Nat := 5) -- 5 minutes
  (traffic_delay : Nat := 15) -- 15 minutes
  (departure_time : Nat := 7*60 + 15) -- 7:15 AM in minutes since midnight
  : 34 = departure_time + normal_travel_time + red_light_stops + construction_wait + detour_time + store_stop_time + traffic_delay - school_start := 
by sorry

end Andy_late_minutes_l2292_229205


namespace divisible_by_4_l2292_229209

theorem divisible_by_4 (n m : ℕ) (h1 : n > 0) (h2 : m > 0) (h3 : n^3 + (n + 1)^3 + (n + 2)^3 = m^3) : 4 ∣ n + 1 :=
sorry

end divisible_by_4_l2292_229209


namespace product_of_two_numbers_l2292_229223

theorem product_of_two_numbers (x y : ℝ) (h1 : x - y = 9) (h2 : x^2 + y^2 = 153) : x * y = 36 :=
by
  sorry

end product_of_two_numbers_l2292_229223


namespace cos_180_eq_neg1_sin_180_eq_0_l2292_229288

theorem cos_180_eq_neg1 : Real.cos (180 * Real.pi / 180) = -1 := sorry
theorem sin_180_eq_0 : Real.sin (180 * Real.pi / 180) = 0 := sorry

end cos_180_eq_neg1_sin_180_eq_0_l2292_229288


namespace value_of_y_l2292_229255

theorem value_of_y (y : ℝ) (h : (y / 5) / 3 = 5 / (y / 3)) : y = 15 ∨ y = -15 :=
by
  sorry

end value_of_y_l2292_229255


namespace find_prime_pairs_l2292_229263

def is_prime (n : ℕ) := n ≥ 2 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def has_prime_root (m n : ℕ) : Prop :=
  ∃ (p: ℕ), is_prime p ∧ (p * p - m * p - n = 0)

theorem find_prime_pairs :
  ∀ (m n : ℕ), (is_prime m ∧ is_prime n) → has_prime_root m n → (m, n) = (2, 3) :=
by sorry

end find_prime_pairs_l2292_229263


namespace sum_of_grid_numbers_l2292_229242

theorem sum_of_grid_numbers (A E: ℕ) (S: ℕ) 
    (hA: A = 2) 
    (hE: E = 3)
    (h1: ∃ B : ℕ, 2 + B = S ∧ 3 + B = S)
    (h2: ∃ D : ℕ, 2 + D = S ∧ D + 3 = S)
    (h3: ∃ F : ℕ, 3 + F = S ∧ F + 3 = S)
    (h4: ∃ G H I: ℕ, 
         2 + G = S ∧ G + H = S ∧ H + C = S ∧ 
         3 + H = S ∧ E + I = S ∧ H + I = S):
  A + B + C + D + E + F + G + H + I = 22 := 
by 
  sorry

end sum_of_grid_numbers_l2292_229242


namespace approx_sum_l2292_229220

-- Definitions of the costs
def cost_bicycle : ℕ := 389
def cost_fan : ℕ := 189

-- Definition of the approximations
def approx_bicycle : ℕ := 400
def approx_fan : ℕ := 200

-- The statement to prove
theorem approx_sum (h₁ : cost_bicycle = 389) (h₂ : cost_fan = 189) : 
  approx_bicycle + approx_fan = 600 := 
by 
  sorry

end approx_sum_l2292_229220
