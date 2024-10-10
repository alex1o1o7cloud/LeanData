import Mathlib

namespace red_flower_area_is_54_total_area_red_yellow_equal_red_yellow_half_total_l1297_129786

/-- Represents a rectangular plot with flowers and grass -/
structure FlowerPlot where
  length : ℝ
  width : ℝ
  red_flower_area : ℝ
  yellow_flower_area : ℝ
  grass_area : ℝ

/-- The properties of the flower plot as described in the problem -/
def school_plot : FlowerPlot where
  length := 18
  width := 12
  red_flower_area := 54
  yellow_flower_area := 54
  grass_area := 108

/-- Theorem stating that the area of red flowers in the school plot is 54 square meters -/
theorem red_flower_area_is_54 (plot : FlowerPlot) (h1 : plot = school_plot) :
  plot.red_flower_area = 54 := by
  sorry

/-- Theorem stating that the total area of the plot is length * width -/
theorem total_area (plot : FlowerPlot) : 
  plot.length * plot.width = plot.red_flower_area + plot.yellow_flower_area + plot.grass_area := by
  sorry

/-- Theorem stating that red and yellow flower areas are equal -/
theorem red_yellow_equal (plot : FlowerPlot) :
  plot.red_flower_area = plot.yellow_flower_area := by
  sorry

/-- Theorem stating that red and yellow flowers together occupy half the total area -/
theorem red_yellow_half_total (plot : FlowerPlot) :
  plot.red_flower_area + plot.yellow_flower_area = (plot.length * plot.width) / 2 := by
  sorry

end red_flower_area_is_54_total_area_red_yellow_equal_red_yellow_half_total_l1297_129786


namespace k_is_negative_l1297_129741

/-- A linear function y = x + k passes through a quadrant if there exists a point (x, y) in that quadrant satisfying the equation. -/
def passes_through_quadrant (k : ℝ) (quadrant : ℕ) : Prop :=
  match quadrant with
  | 1 => ∃ x > 0, x + k > 0
  | 3 => ∃ x < 0, x + k < 0
  | 4 => ∃ x > 0, x + k < 0
  | _ => False

/-- If the graph of y = x + k passes through the first, third, and fourth quadrants, then k < 0. -/
theorem k_is_negative (k : ℝ) 
  (h1 : passes_through_quadrant k 1)
  (h3 : passes_through_quadrant k 3)
  (h4 : passes_through_quadrant k 4) : 
  k < 0 := by
  sorry


end k_is_negative_l1297_129741


namespace complement_union_M_N_l1297_129759

-- Define the universe U
def U : Set (ℝ × ℝ) := Set.univ

-- Define set M
def M : Set (ℝ × ℝ) := {p : ℝ × ℝ | (p.2 - 3) / (p.1 - 2) ≠ 1}

-- Define set N
def N : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 ≠ p.1 + 1}

-- Theorem statement
theorem complement_union_M_N : 
  (U \ (M ∪ N)) = {(2, 3)} := by sorry

end complement_union_M_N_l1297_129759


namespace bills_ratio_l1297_129798

/-- Proves that the ratio of bills Geric had to bills Kyla had at the beginning is 2:1 --/
theorem bills_ratio (jessa_bills_after geric_bills kyla_bills : ℕ) : 
  jessa_bills_after = 7 →
  geric_bills = 16 →
  kyla_bills = (jessa_bills_after + 3) - 2 →
  (geric_bills : ℚ) / kyla_bills = 2 := by
  sorry

end bills_ratio_l1297_129798


namespace abs_sum_greater_than_one_necessary_not_sufficient_l1297_129743

theorem abs_sum_greater_than_one_necessary_not_sufficient (a b : ℝ) :
  (∀ b, b < -1 → ∀ a, |a| + |b| > 1) ∧
  (∃ a b, |a| + |b| > 1 ∧ b ≥ -1) := by
  sorry

end abs_sum_greater_than_one_necessary_not_sufficient_l1297_129743


namespace extra_crayons_l1297_129755

theorem extra_crayons (num_packs : ℕ) (crayons_per_pack : ℕ) (total_crayons : ℕ) : 
  num_packs = 4 →
  crayons_per_pack = 10 →
  total_crayons = 40 →
  total_crayons - (num_packs * crayons_per_pack) = 0 := by
  sorry

end extra_crayons_l1297_129755


namespace product_mod_seven_l1297_129737

theorem product_mod_seven : (2023 * 2024 * 2025 * 2026 * 2027) % 7 = 0 := by
  sorry

end product_mod_seven_l1297_129737


namespace cost_of_pens_calculation_l1297_129797

/-- The cost of the box of pens Linda bought -/
def cost_of_pens : ℝ := 1.70

/-- The number of notebooks Linda bought -/
def num_notebooks : ℕ := 3

/-- The cost of each notebook -/
def cost_per_notebook : ℝ := 1.20

/-- The cost of the box of pencils -/
def cost_of_pencils : ℝ := 1.50

/-- The total amount Linda spent -/
def total_spent : ℝ := 6.80

theorem cost_of_pens_calculation :
  cost_of_pens = total_spent - (↑num_notebooks * cost_per_notebook + cost_of_pencils) :=
by sorry

end cost_of_pens_calculation_l1297_129797


namespace hackathon_ends_at_noon_l1297_129787

-- Define the start time of the hackathon
def hackathon_start : Nat := 12 * 60  -- noon in minutes since midnight

-- Define the duration of the hackathon
def hackathon_duration : Nat := 1440  -- duration in minutes

-- Define a function to calculate the end time of the hackathon
def hackathon_end (start : Nat) (duration : Nat) : Nat :=
  (start + duration) % (24 * 60)

-- Theorem to prove
theorem hackathon_ends_at_noon :
  hackathon_end hackathon_start hackathon_duration = hackathon_start :=
by sorry

end hackathon_ends_at_noon_l1297_129787


namespace gcd_18_30_l1297_129776

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end gcd_18_30_l1297_129776


namespace sqrt_eight_plus_abs_sqrt_two_minus_two_plus_neg_half_inv_eq_sqrt_two_l1297_129763

theorem sqrt_eight_plus_abs_sqrt_two_minus_two_plus_neg_half_inv_eq_sqrt_two :
  Real.sqrt 8 + |Real.sqrt 2 - 2| + (-1/2)⁻¹ = Real.sqrt 2 := by
  sorry

end sqrt_eight_plus_abs_sqrt_two_minus_two_plus_neg_half_inv_eq_sqrt_two_l1297_129763


namespace prism_surface_area_l1297_129731

/-- A rectangular prism with prime edge lengths and volume 627 has surface area 598 -/
theorem prism_surface_area : ∀ a b c : ℕ,
  Prime a → Prime b → Prime c →
  a * b * c = 627 →
  2 * (a * b + b * c + c * a) = 598 := by
sorry

end prism_surface_area_l1297_129731


namespace toy_store_revenue_ratio_l1297_129757

/-- Given a toy store's revenue data for three months, prove that January's revenue is 1/5 of November's revenue. -/
theorem toy_store_revenue_ratio :
  ∀ (nov dec jan : ℝ),
  nov > 0 →
  nov = (2/5) * dec →
  dec = (25/6) * ((nov + jan) / 2) →
  jan = (1/5) * nov :=
by sorry

end toy_store_revenue_ratio_l1297_129757


namespace population_growth_proof_l1297_129713

theorem population_growth_proof (growth_rate_1 : ℝ) (growth_rate_2 : ℝ) : 
  growth_rate_1 = 0.2 →
  growth_rate_2 = growth_rate_1 + 0.3 * growth_rate_1 →
  (1 + growth_rate_1) * (1 + growth_rate_2) - 1 = 0.512 :=
by
  sorry

#check population_growth_proof

end population_growth_proof_l1297_129713


namespace f_2017_negative_two_equals_three_fifths_l1297_129779

def f (x : ℚ) : ℚ := (x - 1) / (3 * x + 1)

def iterate_f (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_f n x)

theorem f_2017_negative_two_equals_three_fifths :
  iterate_f 2017 (-2 : ℚ) = 3/5 := by
  sorry

end f_2017_negative_two_equals_three_fifths_l1297_129779


namespace variance_of_letters_l1297_129756

def letters : List ℕ := [10, 6, 8, 5, 6]

def mean (xs : List ℕ) : ℚ :=
  (xs.sum : ℚ) / xs.length

def variance (xs : List ℕ) : ℚ :=
  let m := mean xs
  (xs.map (fun x => ((x : ℚ) - m) ^ 2)).sum / xs.length

theorem variance_of_letters : variance letters = 16/5 := by
  sorry

end variance_of_letters_l1297_129756


namespace perpendicular_line_exists_l1297_129764

-- Define the concept of a line
def Line : Type := sorry

-- Define the concept of a plane
def Plane : Type := sorry

-- Define what it means for a line to be within a plane
def within_plane (l : Line) (p : Plane) : Prop := sorry

-- Define what it means for two lines to be perpendicular
def perpendicular (l1 l2 : Line) : Prop := sorry

-- State the theorem
theorem perpendicular_line_exists (l : Line) (α : Plane) :
  ∃ m : Line, within_plane m α ∧ perpendicular m l := by
  sorry

end perpendicular_line_exists_l1297_129764


namespace other_person_age_is_six_l1297_129717

-- Define Noah's current age
def noah_current_age : ℕ := 22 - 10

-- Define the relationship between Noah's age and the other person's age
def other_person_age : ℕ := noah_current_age / 2

-- Theorem to prove
theorem other_person_age_is_six : other_person_age = 6 := by
  sorry

end other_person_age_is_six_l1297_129717


namespace digit_sum_equation_l1297_129770

-- Define the digits as natural numbers
def X : ℕ := sorry
def Y : ℕ := sorry
def M : ℕ := sorry
def Z : ℕ := sorry
def F : ℕ := sorry

-- Define the two-digit numbers
def XY : ℕ := 10 * X + Y
def MZ : ℕ := 10 * M + Z

-- Define the three-digit number FFF
def FFF : ℕ := 100 * F + 10 * F + F

-- Theorem statement
theorem digit_sum_equation : 
  (X ≠ 0) ∧ (Y ≠ 0) ∧ (M ≠ 0) ∧ (Z ≠ 0) ∧ (F ≠ 0) ∧  -- non-zero digits
  (X ≠ Y) ∧ (X ≠ M) ∧ (X ≠ Z) ∧ (X ≠ F) ∧
  (Y ≠ M) ∧ (Y ≠ Z) ∧ (Y ≠ F) ∧
  (M ≠ Z) ∧ (M ≠ F) ∧
  (Z ≠ F) ∧  -- unique digits
  (X < 10) ∧ (Y < 10) ∧ (M < 10) ∧ (Z < 10) ∧ (F < 10) ∧  -- single digits
  (XY * MZ = FFF) →  -- equation condition
  X + Y + M + Z + F = 28 := by
sorry

end digit_sum_equation_l1297_129770


namespace x_squared_mod_25_l1297_129710

theorem x_squared_mod_25 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 25]) 
  (h2 : 4 * x ≡ 20 [ZMOD 25]) : 
  x^2 ≡ 4 [ZMOD 25] := by
sorry

end x_squared_mod_25_l1297_129710


namespace sum_of_reciprocals_squared_l1297_129726

theorem sum_of_reciprocals_squared (a b c d : ℝ) : 
  a = Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2 →
  b = -Real.sqrt 5 + Real.sqrt 7 + Real.sqrt 35 + 2 →
  c = Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2 →
  d = -Real.sqrt 5 - Real.sqrt 7 + Real.sqrt 35 + 2 →
  (1/a + 1/b + 1/c + 1/d)^2 = 39/140 := by
sorry

end sum_of_reciprocals_squared_l1297_129726


namespace distribute_and_combine_l1297_129782

theorem distribute_and_combine (a b : ℝ) : 2 * (a - b) + 3 * b = 2 * a + b := by
  sorry

end distribute_and_combine_l1297_129782


namespace reflection_of_circle_center_l1297_129785

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.2, -p.1)

theorem reflection_of_circle_center :
  let original_center : ℝ × ℝ := (8, -3)
  let reflected_center := reflect_about_neg_x original_center
  reflected_center = (3, -8) := by
sorry

end reflection_of_circle_center_l1297_129785


namespace expense_calculation_correct_l1297_129794

/-- Calculates the total out-of-pocket expense for James' purchases and transactions --/
def total_expense (initial_purchase : ℝ) (discount_rate : ℝ) (tax_rate : ℝ)
  (tv_cost : ℝ) (bike_cost : ℝ) (usd_to_eur_initial : ℝ) (usd_to_eur_refund : ℝ)
  (usd_to_gbp : ℝ) (other_bike_markup : ℝ) (other_bike_sale_rate : ℝ)
  (toaster_cost_eur : ℝ) (microwave_cost_eur : ℝ)
  (subscription_cost_gbp : ℝ) (subscription_discount : ℝ) (subscription_months : ℕ) : ℝ :=
  sorry

/-- The total out-of-pocket expense matches the calculated value --/
theorem expense_calculation_correct :
  total_expense 5000 0.1 0.05 1000 700 0.85 0.87 0.77 0.2 0.85 100 150 80 0.3 12 = 2291.63 :=
  sorry

end expense_calculation_correct_l1297_129794


namespace abc_book_cost_l1297_129738

/-- The cost of the best-selling book "TOP" -/
def top_cost : ℝ := 8

/-- The number of "TOP" books sold -/
def top_sold : ℕ := 13

/-- The number of "ABC" books sold -/
def abc_sold : ℕ := 4

/-- The difference in earnings between "TOP" and "ABC" books -/
def earnings_difference : ℝ := 12

/-- The cost of the "ABC" book -/
def abc_cost : ℝ := 23

theorem abc_book_cost :
  top_cost * top_sold - abc_cost * abc_sold = earnings_difference :=
sorry

end abc_book_cost_l1297_129738


namespace vector_operation_l1297_129784

/-- Given planar vectors a and b, prove that 1/2a - 3/2b equals (-1,2) -/
theorem vector_operation (a b : ℝ × ℝ) :
  a = (1, 1) →
  b = (1, -1) →
  (1/2 : ℝ) • a - (3/2 : ℝ) • b = (-1, 2) := by
sorry

end vector_operation_l1297_129784


namespace purely_imaginary_z_l1297_129760

theorem purely_imaginary_z (b : ℝ) :
  let z : ℂ := Complex.I * (1 + b * Complex.I) + 2 + 3 * b * Complex.I
  (z.re = 0) → z = 7 * Complex.I := by
  sorry

end purely_imaginary_z_l1297_129760


namespace sum_of_cubes_of_roots_l1297_129750

theorem sum_of_cubes_of_roots (p q r : ℝ) : 
  p^3 - 2*p^2 + 3*p - 4 = 0 →
  q^3 - 2*q^2 + 3*q - 4 = 0 →
  r^3 - 2*r^2 + 3*r - 4 = 0 →
  p^3 + q^3 + r^3 = 2 := by
sorry

end sum_of_cubes_of_roots_l1297_129750


namespace solution_of_equation_l1297_129703

theorem solution_of_equation (x : ℝ) : 
  (6 * x) / (x + 2) - 4 / (x + 2) = 2 / (x + 2) ↔ x = 1 := by
sorry

end solution_of_equation_l1297_129703


namespace expression_simplification_redundant_condition_l1297_129769

theorem expression_simplification (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  (2 * x * (x^2 * y - x * y^2) + x * y * (2 * x * y - x^2)) / (x^2 * y) = x :=
by sorry

theorem redundant_condition (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) :
  ∃ (z : ℝ), z ≠ y ∧
  (2 * x * (x^2 * y - x * y^2) + x * y * (2 * x * y - x^2)) / (x^2 * y) =
  (2 * x * (x^2 * z - x * z^2) + x * z * (2 * x * z - x^2)) / (x^2 * z) :=
by sorry

end expression_simplification_redundant_condition_l1297_129769


namespace total_people_in_groups_l1297_129709

theorem total_people_in_groups (art_group : ℕ) (dance_group_ratio : ℚ) : 
  art_group = 25 → dance_group_ratio = 1.4 → 
  art_group + (↑art_group * dance_group_ratio) = 55 := by
  sorry

end total_people_in_groups_l1297_129709


namespace catchup_time_correct_l1297_129789

/-- The time (in hours) it takes for the second car to catch up with the first car -/
def catchup_time : ℝ := 1.5

/-- The speed of the first car in km/h -/
def speed1 : ℝ := 60

/-- The speed of the second car in km/h -/
def speed2 : ℝ := 80

/-- The head start time of the first car in hours -/
def head_start : ℝ := 0.5

/-- Theorem stating that the catchup time is correct given the conditions -/
theorem catchup_time_correct : 
  speed1 * (catchup_time + head_start) = speed2 * catchup_time := by
  sorry

#check catchup_time_correct

end catchup_time_correct_l1297_129789


namespace largest_y_satisfies_equation_forms_triangle_largest_y_forms_triangle_l1297_129732

def largest_y : ℝ := 23

theorem largest_y_satisfies_equation :
  |largest_y - 8| = 15 ∧
  ∀ y : ℝ, |y - 8| = 15 → y ≤ largest_y :=
sorry

theorem forms_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem largest_y_forms_triangle :
  forms_triangle largest_y 20 9 :=
sorry

end largest_y_satisfies_equation_forms_triangle_largest_y_forms_triangle_l1297_129732


namespace min_f_gt_min_g_l1297_129725

open Set

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- Define the condition given in the problem
variable (h : ∀ x : ℝ, ∃ x₀ : ℝ, f x > g x₀)

-- State the theorem to be proved
theorem min_f_gt_min_g : (⨅ x, f x) > (⨅ x, g x) := by sorry

end min_f_gt_min_g_l1297_129725


namespace exists_permutation_with_unique_sums_l1297_129704

/-- A permutation of numbers 1 to 10 -/
def Permutation := Fin 10 → Fin 10

/-- Function to check if a permutation results in unique adjacent sums when arranged in a circle -/
def has_unique_adjacent_sums (p : Permutation) : Prop :=
  ∀ i j : Fin 10, i ≠ j → 
    (p i + p ((i + 1) % 10) : ℕ) ≠ (p j + p ((j + 1) % 10) : ℕ)

/-- Theorem stating that there exists a permutation with unique adjacent sums -/
theorem exists_permutation_with_unique_sums : 
  ∃ p : Permutation, Function.Bijective p ∧ has_unique_adjacent_sums p :=
sorry

end exists_permutation_with_unique_sums_l1297_129704


namespace log_10_7_exists_function_l1297_129735

-- Define the variables and conditions
variable (r s : ℝ)
variable (h1 : Real.log 3 / Real.log 4 = r)
variable (h2 : Real.log 5 / Real.log 7 = s)

-- State the theorem
theorem log_10_7_exists_function (r s : ℝ) (h1 : Real.log 3 / Real.log 4 = r) (h2 : Real.log 5 / Real.log 7 = s) :
  ∃ f : ℝ → ℝ → ℝ, Real.log 7 / Real.log 10 = f r s := by
  sorry

end log_10_7_exists_function_l1297_129735


namespace three_function_properties_l1297_129714

theorem three_function_properties :
  (∃ f : ℝ → ℝ, ∀ x : ℝ, f x - (deriv f) x = f (-x) - (deriv f) (-x)) ∧
  (∃ f : ℝ → ℝ, (∀ x : ℝ, (deriv f) x ≠ 0) ∧ (∀ x : ℝ, f x = (deriv f) x)) ∧
  (∃ f : ℝ → ℝ, (∀ x : ℝ, (deriv f) x ≠ 0) ∧ (∀ x : ℝ, f x = -(deriv f) x)) :=
by sorry

end three_function_properties_l1297_129714


namespace bobby_candy_problem_l1297_129772

theorem bobby_candy_problem (initial : ℕ) :
  initial + 17 = 43 → initial = 26 := by
  sorry

end bobby_candy_problem_l1297_129772


namespace square_root_of_25_l1297_129727

theorem square_root_of_25 : 
  {x : ℝ | x^2 = 25} = {5, -5} := by sorry

end square_root_of_25_l1297_129727


namespace sum_equals_twelve_l1297_129792

theorem sum_equals_twelve (a b c : ℕ) (h : 28 * a + 30 * b + 31 * c = 365) : a + b + c = 12 := by
  sorry

end sum_equals_twelve_l1297_129792


namespace parabola_translation_l1297_129795

/-- Represents a parabola in the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Translates a parabola horizontally and vertically -/
def translate (p : Parabola) (h : ℝ) (v : ℝ) : Parabola :=
  { a := p.a
  , b := 2 * p.a * h + p.b
  , c := p.a * h^2 + p.b * h + p.c - v }

theorem parabola_translation (p : Parabola) :
  p.a = 1/2 ∧ p.b = 0 ∧ p.c = 1 →
  let p' := translate p 1 3
  p'.a = 1/2 ∧ p'.b = 1 ∧ p'.c = -3/2 := by sorry

end parabola_translation_l1297_129795


namespace variance_of_scores_l1297_129733

def scores : List ℝ := [9, 10, 9, 7, 10]

theorem variance_of_scores : 
  let n : ℕ := scores.length
  let mean : ℝ := (scores.sum) / n
  let variance : ℝ := (scores.map (λ x => (x - mean)^2)).sum / n
  variance = 6/5 := by sorry

end variance_of_scores_l1297_129733


namespace local_minimum_at_two_l1297_129748

def f (x : ℝ) := x^3 - 12*x

theorem local_minimum_at_two :
  ∃ δ > 0, ∀ x : ℝ, |x - 2| < δ → f x ≥ f 2 :=
sorry

end local_minimum_at_two_l1297_129748


namespace A_inter_B_l1297_129771

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -2 ≤ x ∧ x < 2}

theorem A_inter_B : A ∩ B = {-1, 0, 1} := by sorry

end A_inter_B_l1297_129771


namespace xy_plus_x_plus_y_odd_l1297_129762

def S : Set ℕ := {1, 3, 5, 7, 9, 11, 13, 15, 17, 19}

theorem xy_plus_x_plus_y_odd (x y : ℕ) (hx : x ∈ S) (hy : y ∈ S) (hxy : x ≠ y) :
  ¬Even (x * y + x + y) :=
by sorry

end xy_plus_x_plus_y_odd_l1297_129762


namespace only_solutions_are_72_and_88_l1297_129718

/-- The product of digits of a positive integer -/
def product_of_digits (k : ℕ+) : ℕ :=
  sorry

/-- The main theorem stating that 72 and 88 are the only solutions -/
theorem only_solutions_are_72_and_88 :
  ∀ k : ℕ+, (product_of_digits k = (25 * k : ℚ) / 8 - 211) ↔ (k = 72 ∨ k = 88) :=
by sorry

end only_solutions_are_72_and_88_l1297_129718


namespace tan_value_fourth_quadrant_l1297_129783

theorem tan_value_fourth_quadrant (α : Real) 
  (h1 : α ∈ Set.Icc (3 * Real.pi / 2) (2 * Real.pi)) 
  (h2 : Real.sin α + Real.cos α = 1/5) : 
  Real.tan α = -3/4 := by
sorry

end tan_value_fourth_quadrant_l1297_129783


namespace initial_amount_equals_sum_l1297_129701

/-- The amount of money Agatha initially had to spend on the bike. -/
def initial_amount : ℕ := 60

/-- The amount Agatha spent on the frame. -/
def frame_cost : ℕ := 15

/-- The amount Agatha spent on the front wheel. -/
def front_wheel_cost : ℕ := 25

/-- The amount Agatha has left for the seat and handlebar tape. -/
def remaining_amount : ℕ := 20

/-- Theorem stating that the initial amount equals the sum of all expenses and remaining amount. -/
theorem initial_amount_equals_sum :
  initial_amount = frame_cost + front_wheel_cost + remaining_amount :=
by sorry

end initial_amount_equals_sum_l1297_129701


namespace square_difference_fourth_power_l1297_129768

theorem square_difference_fourth_power : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end square_difference_fourth_power_l1297_129768


namespace evaporation_problem_l1297_129740

theorem evaporation_problem (x : ℚ) : 
  (1 - x) * (1 - 1/4) = 1/6 → x = 7/9 := by
sorry

end evaporation_problem_l1297_129740


namespace teacher_grading_problem_l1297_129720

def remaining_problems (total_worksheets : ℕ) (problems_per_worksheet : ℕ) (graded_worksheets : ℕ) : ℕ :=
  (total_worksheets - graded_worksheets) * problems_per_worksheet

theorem teacher_grading_problem :
  remaining_problems 14 2 7 = 14 := by
  sorry

end teacher_grading_problem_l1297_129720


namespace teacher_distribution_l1297_129708

/-- The number of ways to distribute n distinct objects among k groups, 
    with each group receiving at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

theorem teacher_distribution : distribute 4 3 = 36 := by sorry

end teacher_distribution_l1297_129708


namespace reciprocal_of_opposite_of_negative_l1297_129712

theorem reciprocal_of_opposite_of_negative : 
  (1 / -(- -3)) = -1/3 := by sorry

end reciprocal_of_opposite_of_negative_l1297_129712


namespace sqrt_equation_solution_l1297_129766

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (9 + 3 * z) = 12 := by
  sorry

end sqrt_equation_solution_l1297_129766


namespace first_year_after_2010_sum_15_is_correct_l1297_129745

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- Check if a year is after 2010 -/
def is_after_2010 (year : ℕ) : Prop :=
  year > 2010

/-- First year after 2010 with sum of digits equal to 15 -/
def first_year_after_2010_sum_15 : ℕ :=
  2039

theorem first_year_after_2010_sum_15_is_correct :
  (is_after_2010 first_year_after_2010_sum_15) ∧ 
  (sum_of_digits first_year_after_2010_sum_15 = 15) ∧
  (∀ y : ℕ, is_after_2010 y ∧ y < first_year_after_2010_sum_15 → sum_of_digits y ≠ 15) :=
by sorry

end first_year_after_2010_sum_15_is_correct_l1297_129745


namespace water_depth_multiple_of_height_l1297_129765

theorem water_depth_multiple_of_height (ron_height : ℕ) (water_depth : ℕ) :
  ron_height = 13 →
  water_depth = 208 →
  ∃ k : ℕ, water_depth = k * ron_height →
  water_depth / ron_height = 16 := by
  sorry

end water_depth_multiple_of_height_l1297_129765


namespace twelve_percent_greater_than_80_l1297_129774

theorem twelve_percent_greater_than_80 (x : ℝ) : 
  x = 80 * (1 + 12 / 100) → x = 89.6 := by
sorry

end twelve_percent_greater_than_80_l1297_129774


namespace circle_center_from_equation_l1297_129751

/-- A circle in the 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle in standard form --/
def CircleEquation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_center_from_equation :
  ∃ (c : Circle), (∀ x y : ℝ, CircleEquation c x y ↔ (x - 1)^2 + (y - 2)^2 = 5) ∧ c.center = (1, 2) :=
sorry

end circle_center_from_equation_l1297_129751


namespace max_list_length_l1297_129761

def is_valid_list (D : List Nat) : Prop :=
  ∀ x ∈ D, 1 ≤ x ∧ x ≤ 10

def count_occurrences (x : Nat) (L : List Nat) : Nat :=
  L.filter (· = x) |>.length

def generate_M (D : List Nat) : List Nat :=
  D.map (λ x => count_occurrences x D)

theorem max_list_length :
  ∃ (D : List Nat),
    is_valid_list D ∧
    D.length = 10 ∧
    generate_M D = D.reverse ∧
    ∀ (D' : List Nat),
      is_valid_list D' →
      generate_M D' = D'.reverse →
      D'.length ≤ 10 :=
by sorry

end max_list_length_l1297_129761


namespace cody_purchase_tax_rate_l1297_129758

/-- Proves that the tax rate is 5% given the conditions of Cody's purchase --/
theorem cody_purchase_tax_rate 
  (initial_purchase : ℝ)
  (post_tax_discount : ℝ)
  (cody_payment : ℝ)
  (h1 : initial_purchase = 40)
  (h2 : post_tax_discount = 8)
  (h3 : cody_payment = 17)
  : ∃ (tax_rate : ℝ), 
    tax_rate = 0.05 ∧ 
    (initial_purchase + initial_purchase * tax_rate - post_tax_discount) / 2 = cody_payment :=
by sorry

end cody_purchase_tax_rate_l1297_129758


namespace problem_statement_l1297_129721

theorem problem_statement (x y : ℝ) (hx : x = 7) (hy : y = 3) :
  (x - y)^2 * (x + y)^2 = 1600 := by
  sorry

end problem_statement_l1297_129721


namespace chads_cat_food_packages_l1297_129742

/-- Chad's pet food purchase problem -/
theorem chads_cat_food_packages :
  ∀ (c : ℕ), -- c represents the number of packages of cat food
  (9 * c = 2 * 3 + 48) → -- Equation representing the difference in cans
  c = 6 := by
sorry

end chads_cat_food_packages_l1297_129742


namespace digit_count_theorem_l1297_129754

/-- The number of n-digit positive integers -/
def nDigitNumbers (n : ℕ) : ℕ := 9 * 10^(n-1)

/-- The total number of digits needed to write all natural numbers from 1 to 10^n (not including 10^n) -/
def totalDigits (n : ℕ) : ℚ := n * 10^n - (10^n - 1) / 9

theorem digit_count_theorem (n : ℕ) (h : n > 0) :
  (∀ k : ℕ, k > 0 → k ≤ n → nDigitNumbers k = 9 * 10^(k-1)) ∧
  totalDigits n = n * 10^n - (10^n - 1) / 9 :=
sorry

end digit_count_theorem_l1297_129754


namespace polynomial_divisibility_l1297_129796

theorem polynomial_divisibility (p q : ℤ) : 
  (∀ x : ℤ, ∃ k : ℤ, x^3 + p*x + q = 3*k) ↔ 
  (∃ m : ℤ, p = 3*m + 2) ∧ (∃ n : ℤ, q = 3*n) := by
sorry

end polynomial_divisibility_l1297_129796


namespace conic_is_ellipse_l1297_129767

/-- The equation of the conic section -/
def conic_equation (x y : ℝ) : Prop :=
  x^2 - 72*y^2 - 12*x + 144 = 0

/-- Definition of an ellipse in standard form -/
def is_ellipse (a b h k : ℝ) (x y : ℝ) : Prop :=
  ((x - h)^2 / a^2) + ((y - k)^2 / b^2) = 1

/-- Theorem stating that the given equation represents an ellipse -/
theorem conic_is_ellipse :
  ∃ a b h k : ℝ, a > 0 ∧ b > 0 ∧
  ∀ x y : ℝ, conic_equation x y ↔ is_ellipse a b h k x y :=
sorry

end conic_is_ellipse_l1297_129767


namespace roots_fourth_power_sum_lower_bound_l1297_129702

theorem roots_fourth_power_sum_lower_bound (p : ℝ) (hp : p ≠ 0) :
  let x₁ := (-p + Real.sqrt (p^2 + 2/p^2)) / 2
  let x₂ := (-p - Real.sqrt (p^2 + 2/p^2)) / 2
  x₁^4 + x₂^4 ≥ 2 + Real.sqrt 2 := by
sorry

end roots_fourth_power_sum_lower_bound_l1297_129702


namespace fermat_fourth_power_l1297_129799

theorem fermat_fourth_power (x y z : ℤ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) :
  x^4 + y^4 ≠ z^4 := by
  sorry

end fermat_fourth_power_l1297_129799


namespace product_division_equality_l1297_129739

theorem product_division_equality : ∃ x : ℝ, (400 * 7000 : ℝ) = 28000 * x^1 ∧ x = 100 := by
  sorry

end product_division_equality_l1297_129739


namespace min_value_of_f_l1297_129723

def f (x a b : ℝ) : ℝ := (x + a + b) * (x + a - b) * (x - a + b) * (x - a - b)

theorem min_value_of_f (a b : ℝ) : 
  ∃ (m : ℝ), ∀ (x : ℝ), f x a b ≥ m ∧ ∃ (x₀ : ℝ), f x₀ a b = m ∧ m = -4 * a^2 * b^2 := by
  sorry

end min_value_of_f_l1297_129723


namespace octagon_dual_reflection_area_l1297_129724

/-- The area of the region bounded by 8 arcs created by dual reflection over consecutive sides of a regular octagon inscribed in a circle -/
theorem octagon_dual_reflection_area (s : ℝ) (h : s = 2) :
  let r := 1 / Real.sin (22.5 * π / 180)
  let sector_area := π * r^2 / 8
  let dual_reflected_sector_area := 2 * sector_area
  8 * dual_reflected_sector_area = 2 * (1 / Real.sin (22.5 * π / 180))^2 * π :=
by sorry

end octagon_dual_reflection_area_l1297_129724


namespace reservoir_water_ratio_l1297_129773

/-- Proof of the ratio of water in a reservoir --/
theorem reservoir_water_ratio :
  ∀ (total_capacity current_amount normal_level : ℝ),
  current_amount = 14000000 →
  current_amount = 0.7 * total_capacity →
  normal_level = total_capacity - 10000000 →
  current_amount / normal_level = 1.4 := by
  sorry

end reservoir_water_ratio_l1297_129773


namespace expand_expression_l1297_129780

theorem expand_expression (x : ℝ) : (2*x - 3) * (2*x + 3) * (4*x^2 + 9) = 4*x^4 - 81 := by
  sorry

end expand_expression_l1297_129780


namespace gasoline_price_change_l1297_129788

/-- Represents the price change of gasoline over two months -/
theorem gasoline_price_change 
  (initial_price : ℝ) 
  (final_price : ℝ) 
  (x : ℝ) 
  (h1 : initial_price = 7.5)
  (h2 : final_price = 8.4)
  (h3 : x ≥ 0) -- Assuming non-negative growth rate
  : initial_price * (1 + x)^2 = final_price :=
by sorry

end gasoline_price_change_l1297_129788


namespace equation_proof_l1297_129730

theorem equation_proof : ((12 : ℝ)^2 * (6 : ℝ)^4 / 432)^(1/2) = 4 * 3 * (3 : ℝ)^(1/2) := by
  sorry

end equation_proof_l1297_129730


namespace consecutive_integers_problem_l1297_129791

theorem consecutive_integers_problem (x y z : ℤ) : 
  x = y + 1 → 
  y = z + 1 → 
  x > y → 
  y > z → 
  2*x + 3*y + 3*z = 5*y + 8 → 
  z = 2 := by
sorry

end consecutive_integers_problem_l1297_129791


namespace triangle_area_l1297_129716

/-- The area of a triangle with base 12 cm and height 7 cm is 42 square centimeters. -/
theorem triangle_area : 
  let base : ℝ := 12
  let height : ℝ := 7
  (1 / 2 : ℝ) * base * height = 42 := by sorry

end triangle_area_l1297_129716


namespace arithmetic_sequence_problem_l1297_129719

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a where a₁₃ + a₅ = 32,
    prove that a₉ = 16 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_sum : a 13 + a 5 = 32) : 
  a 9 = 16 := by
sorry

end arithmetic_sequence_problem_l1297_129719


namespace cyclist_return_speed_l1297_129777

/-- Calculates the average speed for the return trip of a cyclist -/
theorem cyclist_return_speed (total_distance : ℝ) (first_half_distance : ℝ) (first_speed : ℝ) (second_speed : ℝ) (total_time : ℝ)
  (h1 : total_distance = 32)
  (h2 : first_half_distance = 16)
  (h3 : first_speed = 8)
  (h4 : second_speed = 10)
  (h5 : total_time = 6.8)
  : (total_distance / (total_time - (first_half_distance / first_speed + (total_distance - first_half_distance) / second_speed))) = 10 := by
  sorry

#check cyclist_return_speed

end cyclist_return_speed_l1297_129777


namespace log_xy_value_l1297_129707

theorem log_xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h1 : Real.log (x^3 * y^5) = 2)
  (h2 : Real.log (x^4 * y^2) = 2)
  (h3 : Real.log (x^2 * y^7) = 3) :
  Real.log (x * y) = 4/7 := by
sorry

end log_xy_value_l1297_129707


namespace permutation_sum_theorem_combination_sum_theorem_l1297_129722

-- Define the permutation function
def A (n : ℕ) (k : ℕ) : ℕ := Nat.factorial n / Nat.factorial (n - k)

-- Define the combination function
def C (n : ℕ) (k : ℕ) : ℕ := Nat.choose n k

theorem permutation_sum_theorem :
  A 5 1 + A 5 2 + A 5 3 + A 5 4 + A 5 5 = 325 := by sorry

theorem combination_sum_theorem (m : ℕ) (h1 : m > 1) (h2 : C 5 m = C 5 (2*m - 1)) :
  C 6 m + C 6 (m+1) + C 7 (m+2) + C 8 (m+3) = 126 := by sorry

end permutation_sum_theorem_combination_sum_theorem_l1297_129722


namespace power_sum_simplification_l1297_129706

theorem power_sum_simplification (n : ℕ) : (-3)^n + 2*(-3)^(n-1) = -(-3)^(n-1) := by
  sorry

end power_sum_simplification_l1297_129706


namespace solution_to_system_of_equations_l1297_129700

theorem solution_to_system_of_equations :
  let system (x y z : ℝ) : Prop :=
    x^2 - y^2 + z = 64 / (x * y) ∧
    y^2 - z^2 + x = 64 / (y * z) ∧
    z^2 - x^2 + y = 64 / (x * z)
  ∀ x y z : ℝ, system x y z →
    ((x = 4 ∧ y = 4 ∧ z = 4) ∨
     (x = -4 ∧ y = -4 ∧ z = 4) ∨
     (x = -4 ∧ y = 4 ∧ z = -4) ∨
     (x = 4 ∧ y = -4 ∧ z = -4)) :=
by
  sorry

#check solution_to_system_of_equations

end solution_to_system_of_equations_l1297_129700


namespace marble_arrangement_theorem_l1297_129736

def num_marbles : ℕ := 4

def num_arrangements (n : ℕ) : ℕ := n.factorial

def num_adjacent_arrangements (n : ℕ) : ℕ := 2 * ((n - 1).factorial)

theorem marble_arrangement_theorem :
  num_arrangements num_marbles - num_adjacent_arrangements num_marbles = 12 :=
by sorry

end marble_arrangement_theorem_l1297_129736


namespace race_time_calculation_l1297_129728

theorem race_time_calculation (prejean_speed rickey_speed rickey_time prejean_time : ℝ) : 
  prejean_speed = (3 / 4) * rickey_speed →
  rickey_time + prejean_time = 70 →
  prejean_time = (4 / 3) * rickey_time →
  rickey_time = 40 :=
by
  sorry

end race_time_calculation_l1297_129728


namespace factor_2x_squared_minus_8_l1297_129749

theorem factor_2x_squared_minus_8 (x : ℝ) : 2 * x^2 - 8 = 2 * (x + 2) * (x - 2) := by
  sorry

end factor_2x_squared_minus_8_l1297_129749


namespace alcohol_dilution_l1297_129753

theorem alcohol_dilution (initial_volume : ℝ) (initial_percentage : ℝ) (added_water : ℝ) :
  initial_volume = 15 →
  initial_percentage = 26 →
  added_water = 5 →
  let alcohol_volume := initial_volume * (initial_percentage / 100)
  let total_volume := initial_volume + added_water
  let final_percentage := (alcohol_volume / total_volume) * 100
  final_percentage = 19.5 := by
    sorry

end alcohol_dilution_l1297_129753


namespace painted_faces_difference_l1297_129747

/-- Represents a 3D cube structure --/
structure CubeStructure where
  length : Nat
  width : Nat
  height : Nat

/-- Counts cubes with exactly n painted faces in the structure --/
def countPaintedFaces (cs : CubeStructure) (n : Nat) : Nat :=
  sorry

/-- The main theorem to be proved --/
theorem painted_faces_difference (cs : CubeStructure) :
  cs.length = 7 → cs.width = 7 → cs.height = 3 →
  countPaintedFaces cs 3 - countPaintedFaces cs 2 = 12 := by
  sorry


end painted_faces_difference_l1297_129747


namespace triangle_angle_measure_l1297_129705

/-- Given a triangle DEF where the measure of angle D is 75 degrees,
    and the measure of angle E is 18 degrees more than four times the measure of angle F,
    prove that the measure of angle F is 17.4 degrees. -/
theorem triangle_angle_measure (D E F : ℝ) : 
  D = 75 →
  E = 4 * F + 18 →
  D + E + F = 180 →
  F = 17.4 := by
sorry

end triangle_angle_measure_l1297_129705


namespace proposition_evaluation_l1297_129729

-- Define propositions p and q
def p : Prop := ∀ x y : ℝ, x > y → -x < -y
def q : Prop := ∀ x y : ℝ, x > y → x^2 > y^2

-- State the theorem
theorem proposition_evaluation :
  (p ∧ q = False) ∧
  (p ∨ q = True) ∧
  (p ∧ (¬q) = True) ∧
  ((¬p) ∨ q = False) :=
by sorry

end proposition_evaluation_l1297_129729


namespace second_offer_more_advantageous_l1297_129734

/-- Represents the total cost of four items -/
def S : ℕ := 1000

/-- Represents the minimum cost of any item -/
def X : ℕ := 99

/-- Represents the prices of four items -/
structure Prices where
  s₁ : ℕ
  s₂ : ℕ
  s₃ : ℕ
  s₄ : ℕ
  sum_eq_S : s₁ + s₂ + s₃ + s₄ = S
  ordered : s₁ ≥ s₂ ∧ s₂ ≥ s₃ ∧ s₃ ≥ s₄
  min_price : s₄ ≥ X

/-- The maximum N for which the second offer is more advantageous -/
def maxN : ℕ := 504

theorem second_offer_more_advantageous (prices : Prices) :
  ∀ N : ℕ, N ≤ maxN →
  (0.2 * prices.s₁ + 0.8 * S : ℚ) < (S - prices.s₄ : ℚ) ∧
  ¬∃ M : ℕ, M > maxN ∧ (0.2 * prices.s₁ + 0.8 * S : ℚ) < (S - prices.s₄ : ℚ) :=
sorry

end second_offer_more_advantageous_l1297_129734


namespace broker_investment_l1297_129752

theorem broker_investment (P : ℝ) (x : ℝ) (h : P > 0) :
  (P + x / 100 * P) * (1 - 30 / 100) = P * (1 + 26 / 100) →
  x = 80 := by
sorry

end broker_investment_l1297_129752


namespace theater_empty_seats_l1297_129711

/-- Given a theater with total seats and occupied seats, calculate the number of empty seats. -/
def empty_seats (total_seats occupied_seats : ℕ) : ℕ :=
  total_seats - occupied_seats

/-- Theorem: In a theater with 750 seats and 532 people watching, there are 218 empty seats. -/
theorem theater_empty_seats :
  empty_seats 750 532 = 218 := by
  sorry

end theater_empty_seats_l1297_129711


namespace least_addition_for_divisibility_l1297_129715

theorem least_addition_for_divisibility (n : ℕ) : 
  let x := Nat.minFac (9 - n % 9)
  x > 0 ∧ (4499 + x) % 9 = 0 ∧ ∀ y : ℕ, y < x → (4499 + y) % 9 ≠ 0 :=
by sorry

end least_addition_for_divisibility_l1297_129715


namespace exactly_two_numbers_satisfy_l1297_129778

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, 1 < m → m < n → ¬(n % m = 0)

def satisfies_condition (n : ℕ) : Prop :=
  n < 500 ∧ n = 7 * sum_of_digits n ∧ is_prime (sum_of_digits n)

theorem exactly_two_numbers_satisfy :
  ∃! (s : Finset ℕ), (∀ n ∈ s, satisfies_condition n) ∧ s.card = 2 :=
sorry

end exactly_two_numbers_satisfy_l1297_129778


namespace balloon_permutations_l1297_129790

def balloon_letters : Nat := 7
def balloon_l_count : Nat := 2
def balloon_o_count : Nat := 2

theorem balloon_permutations :
  (balloon_letters.factorial) / (balloon_l_count.factorial * balloon_o_count.factorial) = 1260 := by
  sorry

end balloon_permutations_l1297_129790


namespace complex_magnitude_l1297_129781

theorem complex_magnitude (z : ℂ) (h : Complex.I * z + 2 = z - 2 * Complex.I) : Complex.abs z = 2 := by
  sorry

end complex_magnitude_l1297_129781


namespace inequality_solution_range_l1297_129746

theorem inequality_solution_range (a : ℝ) : 
  (∀ x : ℝ, ((a - 8) * x > a - 8) ↔ (x < 1)) → (a < 8) :=
by sorry

end inequality_solution_range_l1297_129746


namespace max_product_on_curve_l1297_129793

theorem max_product_on_curve (x y : ℝ) :
  0 ≤ x ∧ x ≤ 12 ∧ 0 ≤ y ∧ y ≤ 12 →
  x * y = (12 - x)^2 * (12 - y)^2 →
  x * y ≤ 81 :=
by sorry

end max_product_on_curve_l1297_129793


namespace circle_with_rational_center_multiple_lattice_points_l1297_129744

/-- A point in the 2D plane with rational coordinates -/
structure RationalPoint where
  x : ℚ
  y : ℚ

/-- A circle in the 2D plane -/
structure Circle where
  center : RationalPoint
  radius : ℝ

/-- A lattice point in the 2D plane -/
structure LatticePoint where
  x : ℤ
  y : ℤ

/-- Check if a lattice point is on the circumference of a circle -/
def isOnCircumference (c : Circle) (p : LatticePoint) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- The main theorem statement -/
theorem circle_with_rational_center_multiple_lattice_points
  (K : RationalPoint) (c : Circle) (p : LatticePoint) :
  c.center = K → isOnCircumference c p →
  ∃ q : LatticePoint, q ≠ p ∧ isOnCircumference c q :=
by sorry

end circle_with_rational_center_multiple_lattice_points_l1297_129744


namespace emilee_earnings_l1297_129775

/-- Given the total earnings and individual earnings of Jermaine and Terrence, 
    calculate Emilee's earnings. -/
theorem emilee_earnings 
  (total : ℕ) 
  (terrence_earnings : ℕ) 
  (jermaine_terrence_diff : ℕ) 
  (h1 : total = 90) 
  (h2 : terrence_earnings = 30) 
  (h3 : jermaine_terrence_diff = 5) : 
  total - (terrence_earnings + (terrence_earnings + jermaine_terrence_diff)) = 25 :=
by
  sorry

#check emilee_earnings

end emilee_earnings_l1297_129775
