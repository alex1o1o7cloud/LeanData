import Mathlib

namespace NUMINAMATH_GPT_line_intersection_l1026_102636

noncomputable def line1 (t : ℚ) : ℚ × ℚ := (1 - 2 * t, 4 + 3 * t)
noncomputable def line2 (u : ℚ) : ℚ × ℚ := (5 + u, 2 + 6 * u)

theorem line_intersection :
  ∃ t u : ℚ, line1 t = (21 / 5, -4 / 5) ∧ line2 u = (21 / 5, -4 / 5) :=
sorry

end NUMINAMATH_GPT_line_intersection_l1026_102636


namespace NUMINAMATH_GPT_exist_positive_int_for_arithmetic_mean_of_divisors_l1026_102654

theorem exist_positive_int_for_arithmetic_mean_of_divisors
  (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (h_distinct : p ≠ q) :
  ∃ a b : ℕ, 0 < a ∧ 0 < b ∧ 
  (∃ k : ℕ, k * (a + 1) * (b + 1) = (p^(a+1) - 1) / (p - 1) * (q^(b+1) - 1) / (q - 1)) :=
sorry

end NUMINAMATH_GPT_exist_positive_int_for_arithmetic_mean_of_divisors_l1026_102654


namespace NUMINAMATH_GPT_total_amount_divided_l1026_102601

theorem total_amount_divided (P1 : ℝ) (r1 : ℝ) (r2 : ℝ) (interest : ℝ) (T : ℝ) :
  P1 = 1550 →
  r1 = 0.03 →
  r2 = 0.05 →
  interest = 144 →
  (P1 * r1 + (T - P1) * r2 = interest) → T = 3500 :=
by
  intros hP1 hr1 hr2 hint htotal
  sorry

end NUMINAMATH_GPT_total_amount_divided_l1026_102601


namespace NUMINAMATH_GPT_exists_integers_x_l1026_102648

theorem exists_integers_x (a1 a2 a3 : ℤ) (h : 0 < a1 ∧ a1 < a2 ∧ a2 < a3) :
  ∃ (x1 x2 x3 : ℤ), (|x1| + |x2| + |x3| > 0) ∧ (a1 * x1 + a2 * x2 + a3 * x3 = 0) ∧ (max (max (|x1|) (|x2|)) (|x3|) < (2 / Real.sqrt 3 * Real.sqrt a3) + 1) := 
sorry

end NUMINAMATH_GPT_exists_integers_x_l1026_102648


namespace NUMINAMATH_GPT_fourth_month_sale_is_7200_l1026_102624

-- Define the sales amounts for each month
def sale_first_month : ℕ := 6400
def sale_second_month : ℕ := 7000
def sale_third_month : ℕ := 6800
def sale_fifth_month : ℕ := 6500
def sale_sixth_month : ℕ := 5100
def average_sale : ℕ := 6500

-- Total requirements for the six months
def total_required_sales : ℕ := 6 * average_sale

-- Known sales for five months
def total_known_sales : ℕ := sale_first_month + sale_second_month + sale_third_month + sale_fifth_month + sale_sixth_month

-- Sale in the fourth month
def sale_fourth_month : ℕ := total_required_sales - total_known_sales

-- The theorem to prove
theorem fourth_month_sale_is_7200 : sale_fourth_month = 7200 :=
by
  sorry

end NUMINAMATH_GPT_fourth_month_sale_is_7200_l1026_102624


namespace NUMINAMATH_GPT_largest_fraction_l1026_102651

theorem largest_fraction :
  let A := (5 : ℚ) / 11
  let B := (6 : ℚ) / 13
  let C := (18 : ℚ) / 37
  let D := (101 : ℚ) / 202
  let E := (200 : ℚ) / 399
  E > A ∧ E > B ∧ E > C ∧ E > D := by
  sorry

end NUMINAMATH_GPT_largest_fraction_l1026_102651


namespace NUMINAMATH_GPT_ice_cream_orders_l1026_102643

variables (V C S M O T : ℕ)

theorem ice_cream_orders :
  (V = 56) ∧ (C = 28) ∧ (S = 70) ∧ (M = 42) ∧ (O = 84) ↔
  (V = 2 * C) ∧
  (S = 25 * T / 100) ∧
  (M = 15 * T / 100) ∧
  (T = 280) ∧
  (V = 20 * T / 100) ∧
  (V + C + S + M + O = T) :=
by
  sorry

end NUMINAMATH_GPT_ice_cream_orders_l1026_102643


namespace NUMINAMATH_GPT_sum_of_remainders_l1026_102650

theorem sum_of_remainders (n : ℤ) (h : n % 20 = 13) : (n % 4) + (n % 5) = 4 := 
by {
  -- proof omitted
  sorry
}

end NUMINAMATH_GPT_sum_of_remainders_l1026_102650


namespace NUMINAMATH_GPT_quadratic_unbounded_above_l1026_102693

theorem quadratic_unbounded_above : ∀ (x y : ℝ), ∃ M : ℝ, ∀ z : ℝ, M < (2 * x^2 + 4 * x * y + 5 * y^2 + 8 * x - 6 * y + z) :=
by
  intro x y
  use 1000 -- Example to denote that for any point greater than 1000
  intro z
  have h1 : 2 * x^2 + 4 * x * y + 5 * y^2 + 8 * x - 6 * y + z ≥ 2 * 0^2 + 4 * 0 * y + 5 * y^2 + 8 * 0 - 6 * y + z := by sorry
  sorry

end NUMINAMATH_GPT_quadratic_unbounded_above_l1026_102693


namespace NUMINAMATH_GPT_divide_plane_into_regions_l1026_102606

theorem divide_plane_into_regions :
  (∀ (x y : ℝ), y = 3 * x ∨ y = x / 3) →
  ∃ (regions : ℕ), regions = 4 :=
by
  sorry

end NUMINAMATH_GPT_divide_plane_into_regions_l1026_102606


namespace NUMINAMATH_GPT_sheepdog_catches_sheep_l1026_102629

-- Define the speeds and the time taken
def v_s : ℝ := 12 -- speed of the sheep in feet/second
def v_d : ℝ := 20 -- speed of the sheepdog in feet/second
def t : ℝ := 20 -- time in seconds

-- Define the initial distance between the sheep and the sheepdog
def initial_distance (v_s v_d t : ℝ) : ℝ :=
  v_d * t - v_s * t

theorem sheepdog_catches_sheep :
  initial_distance v_s v_d t = 160 :=
by
  -- The formal proof would go here, but for now we replace it with sorry
  sorry

end NUMINAMATH_GPT_sheepdog_catches_sheep_l1026_102629


namespace NUMINAMATH_GPT_total_rods_required_l1026_102699

-- Define the number of rods needed per unit for each type
def rods_per_sheet_A : ℕ := 10
def rods_per_sheet_B : ℕ := 8
def rods_per_sheet_C : ℕ := 12
def rods_per_beam_A : ℕ := 6
def rods_per_beam_B : ℕ := 4
def rods_per_beam_C : ℕ := 5

-- Define the composition per panel
def sheets_A_per_panel : ℕ := 2
def sheets_B_per_panel : ℕ := 1
def beams_C_per_panel : ℕ := 2

-- Define the number of panels
def num_panels : ℕ := 10

-- Prove the total number of metal rods required for the entire fence
theorem total_rods_required : 
  (sheets_A_per_panel * rods_per_sheet_A + 
   sheets_B_per_panel * rods_per_sheet_B +
   beams_C_per_panel * rods_per_beam_C) * num_panels = 380 :=
by 
  sorry

end NUMINAMATH_GPT_total_rods_required_l1026_102699


namespace NUMINAMATH_GPT_sum_cube_eq_l1026_102635

theorem sum_cube_eq (a b c : ℝ) (h : a + b + c = 0) : a^3 + b^3 + c^3 = 3 * a * b * c :=
by 
  sorry

end NUMINAMATH_GPT_sum_cube_eq_l1026_102635


namespace NUMINAMATH_GPT_sequence_property_l1026_102673

theorem sequence_property {m : ℤ} (h_m : |m| ≥ 2) (a : ℕ → ℤ)
  (h_nonzero : ¬(a 1 = 0 ∧ a 2 = 0))
  (h_rec : ∀ n : ℕ, a (n+2) = a (n+1) - m * a n)
  (r s : ℕ) (h_r_s : r > s ∧ s ≥ 2) (h_eq : a r = a s ∧ a s = a 1) :
  r - s ≥ |m| := sorry

end NUMINAMATH_GPT_sequence_property_l1026_102673


namespace NUMINAMATH_GPT_cylinder_ratio_l1026_102625

theorem cylinder_ratio (m r : ℝ) (h1 : m + 2 * r = Real.sqrt (m^2 + (r * Real.pi)^2)) :
  m / (2 * r) = (Real.pi^2 - 4) / 8 := by
  sorry

end NUMINAMATH_GPT_cylinder_ratio_l1026_102625


namespace NUMINAMATH_GPT_tangent_line_l1026_102667

variable (a b x₀ y₀ x y : ℝ)
variable (h_ab : a > b)
variable (h_b0 : b > 0)

def ellipse (a b x y : ℝ) : Prop :=
  (x^2 / a^2) + (y^2 / b^2) = 1

theorem tangent_line (h_el : ellipse a b x₀ y₀) : 
  (x₀ * x / a^2) + (y₀ * y / b^2) = 1 :=
sorry

end NUMINAMATH_GPT_tangent_line_l1026_102667


namespace NUMINAMATH_GPT_sum_of_cubes_form_l1026_102623

theorem sum_of_cubes_form (a b : ℤ) (x1 y1 x2 y2 : ℤ)
  (h1 : a = x1^2 + 3 * y1^2) (h2 : b = x2^2 + 3 * y2^2) :
  ∃ x y : ℤ, a^3 + b^3 = x^2 + 3 * y^2 := sorry

end NUMINAMATH_GPT_sum_of_cubes_form_l1026_102623


namespace NUMINAMATH_GPT_number_of_married_men_at_least_11_l1026_102687

-- Definitions based only on conditions from a)
def total_men := 100
def men_with_tv := 75
def men_with_radio := 85
def men_with_ac := 70
def married_with_tv_radio_ac := 11

-- Theorem that needs to be proven based on the conditions
theorem number_of_married_men_at_least_11 : total_men ≥ married_with_tv_radio_ac :=
by
  sorry

end NUMINAMATH_GPT_number_of_married_men_at_least_11_l1026_102687


namespace NUMINAMATH_GPT_cubic_sum_l1026_102664

theorem cubic_sum (x y : ℝ) (h1 : x + y = 8) (h2 : x * y = 12) : x^3 + y^3 = 224 :=
by
  sorry

end NUMINAMATH_GPT_cubic_sum_l1026_102664


namespace NUMINAMATH_GPT_book_selection_l1026_102615

theorem book_selection (total_books novels : ℕ) (choose_books : ℕ)
  (h_total : total_books = 15)
  (h_novels : novels = 5)
  (h_choose : choose_books = 3) :
  (Nat.choose 15 3 - Nat.choose 10 3) = 335 :=
by
  sorry

end NUMINAMATH_GPT_book_selection_l1026_102615


namespace NUMINAMATH_GPT_find_c_for_two_zeros_l1026_102695

noncomputable def f (x c : ℝ) : ℝ := x^3 - 3*x + c

theorem find_c_for_two_zeros (c : ℝ) : (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ f x1 c = 0 ∧ f x2 c = 0) ↔ c = -2 ∨ c = 2 :=
sorry

end NUMINAMATH_GPT_find_c_for_two_zeros_l1026_102695


namespace NUMINAMATH_GPT_hiker_final_distance_l1026_102672

theorem hiker_final_distance :
  let east := 24
  let north := 7
  let west := 15
  let south := 5
  let net_east := east - west
  let net_north := north - south
  net_east = 9 ∧ net_north = 2 →
  Real.sqrt ((net_east)^2 + (net_north)^2) = Real.sqrt 85 :=
by
  intros
  sorry

end NUMINAMATH_GPT_hiker_final_distance_l1026_102672


namespace NUMINAMATH_GPT_final_answer_is_d_l1026_102620

-- Definitions of the propositions p and q
def p : Prop := ∃ x : ℝ, Real.tan x > 1
def q : Prop := false  -- since the distance between focus and directrix is not 1/6 but 3/2

-- The statement to be proven
theorem final_answer_is_d : p ∧ ¬ q := by sorry

end NUMINAMATH_GPT_final_answer_is_d_l1026_102620


namespace NUMINAMATH_GPT_bucket_weight_full_l1026_102633

theorem bucket_weight_full (c d : ℝ) (x y : ℝ) 
  (h1 : x + (1 / 3) * y = c) 
  (h2 : x + (3 / 4) * y = d) : 
  x + y = (-3 * c + 8 * d) / 5 :=
sorry

end NUMINAMATH_GPT_bucket_weight_full_l1026_102633


namespace NUMINAMATH_GPT_erased_angle_is_97_l1026_102678

theorem erased_angle_is_97 (n : ℕ) (h1 : 3 ≤ n) (h2 : (n - 2) * 180 = 1703 + x) : 
  1800 - 1703 = 97 :=
by sorry

end NUMINAMATH_GPT_erased_angle_is_97_l1026_102678


namespace NUMINAMATH_GPT_polygon_sides_eq_nine_l1026_102669

theorem polygon_sides_eq_nine (n : ℕ) 
  (interior_sum : ℕ := (n - 2) * 180)
  (exterior_sum : ℕ := 360)
  (condition : interior_sum = 4 * exterior_sum - 180) : 
  n = 9 :=
by {
  sorry
}

end NUMINAMATH_GPT_polygon_sides_eq_nine_l1026_102669


namespace NUMINAMATH_GPT_solve_fractional_equation_l1026_102665

theorem solve_fractional_equation (x : ℝ) (h : (3 / (x + 1) - 2 / (x - 1)) = 0) : x = 5 :=
sorry

end NUMINAMATH_GPT_solve_fractional_equation_l1026_102665


namespace NUMINAMATH_GPT_no_perfect_square_m_in_range_l1026_102694

theorem no_perfect_square_m_in_range : 
  ∀ m : ℕ, 4 ≤ m ∧ m ≤ 12 → ¬(∃ k : ℕ, 2 * m^2 + 3 * m + 2 = k^2) := by
sorry

end NUMINAMATH_GPT_no_perfect_square_m_in_range_l1026_102694


namespace NUMINAMATH_GPT_gcd_of_three_numbers_l1026_102646

theorem gcd_of_three_numbers : 
  let a := 4560
  let b := 6080
  let c := 16560
  gcd (gcd a b) c = 80 := 
by {
  -- placeholder for the proof
  sorry
}

end NUMINAMATH_GPT_gcd_of_three_numbers_l1026_102646


namespace NUMINAMATH_GPT_probability_last_passenger_own_seat_is_half_l1026_102682

open Classical

-- Define the behavior and probability question:

noncomputable def probability_last_passenger_own_seat (n : ℕ) : ℚ :=
  if n = 0 then 0 else 1 / 2

-- The main theorem stating the probability for an arbitrary number of passengers n
-- The theorem that needs to be proved:
theorem probability_last_passenger_own_seat_is_half (n : ℕ) (h : n > 0) : 
  probability_last_passenger_own_seat n = 1 / 2 :=
by sorry

end NUMINAMATH_GPT_probability_last_passenger_own_seat_is_half_l1026_102682


namespace NUMINAMATH_GPT_quadratic_no_real_roots_l1026_102610

theorem quadratic_no_real_roots (m : ℝ) (h : ∀ x : ℝ, x^2 - m * x + 1 ≠ 0) : m = 0 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_no_real_roots_l1026_102610


namespace NUMINAMATH_GPT_h_f_equals_h_g_l1026_102617

def f (x : ℝ) := x^2 - x + 1

def g (x : ℝ) := -x^2 + x + 1

def h (x : ℝ) := (x - 1)^2

theorem h_f_equals_h_g : ∀ x : ℝ, h (f x) = h (g x) :=
by
  intro x
  unfold f g h
  sorry

end NUMINAMATH_GPT_h_f_equals_h_g_l1026_102617


namespace NUMINAMATH_GPT_problem_solution_l1026_102616

def positive (n : ℕ) : Prop := n > 0
def pairwise_coprime (a b c : ℕ) : Prop :=
  Nat.gcd a b = 1 ∧ Nat.gcd a c = 1 ∧ Nat.gcd b c = 1
def divides (m n : ℕ) : Prop := ∃ k, n = k * m

theorem problem_solution (a b c : ℕ) :
  positive a → positive b → positive c →
  pairwise_coprime a b c →
  divides (a^2) (b^3 + c^3) →
  divides (b^2) (a^3 + c^3) →
  divides (c^2) (a^3 + b^3) →
  (a = 1 ∧ b = 1 ∧ c = 1) ∨ 
  (a = 3 ∧ b = 2 ∧ c = 1) ∨ 
  (a = 3 ∧ b = 1 ∧ c = 2) ∨ 
  (a = 2 ∧ b = 3 ∧ c = 1) ∨ 
  (a = 2 ∧ b = 1 ∧ c = 3) ∨ 
  (a = 1 ∧ b = 3 ∧ c = 2) ∨ 
  (a = 1 ∧ b = 2 ∧ c = 3) := by
  sorry

end NUMINAMATH_GPT_problem_solution_l1026_102616


namespace NUMINAMATH_GPT_friction_coefficient_example_l1026_102614

variable (α : ℝ) (mg : ℝ) (μ : ℝ)

theorem friction_coefficient_example
    (hα : α = 85 * Real.pi / 180) -- converting degrees to radians
    (hN : ∀ (N : ℝ), N = 6 * mg) -- Normal force in the vertical position
    (F : ℝ) -- Force applied horizontally by boy
    (hvert : F * Real.sin α - mg + (6 * mg) * Real.cos α = 0) -- vertical equilibrium
    (hhor : F * Real.cos α - μ * (6 * mg) - (6 * mg) * Real.sin α = 0) -- horizontal equilibrium
    : μ = 0.08 :=
by
  sorry

end NUMINAMATH_GPT_friction_coefficient_example_l1026_102614


namespace NUMINAMATH_GPT_monotonic_increasing_interval_l1026_102630

noncomputable def f (x : ℝ) : ℝ := Real.logb 2 (x^2 - 4)

theorem monotonic_increasing_interval :
  ∀ x : ℝ, 2 < x → (f x < f (x + 1)) :=
by
  intros x h
  sorry

end NUMINAMATH_GPT_monotonic_increasing_interval_l1026_102630


namespace NUMINAMATH_GPT_line_passes_through_circle_center_l1026_102649

theorem line_passes_through_circle_center (a : ℝ) :
  (∃ x y : ℝ, (x^2 + y^2 + 2*x - 4*y = 0) ∧ (3*x + y + a = 0)) → a = 1 :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_circle_center_l1026_102649


namespace NUMINAMATH_GPT_negate_proposition_p_l1026_102698

theorem negate_proposition_p (f : ℝ → ℝ) :
  (¬ ∀ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) >= 0) ↔ ∃ (x₁ x₂ : ℝ), (f x₂ - f x₁) * (x₂ - x₁) < 0 :=
sorry

end NUMINAMATH_GPT_negate_proposition_p_l1026_102698


namespace NUMINAMATH_GPT_max_area_rectangle_l1026_102655

theorem max_area_rectangle (l w : ℝ) 
  (h1 : 2 * l + 2 * w = 60) 
  (h2 : l - w = 10) : 
  l * w = 200 := 
by
  sorry

end NUMINAMATH_GPT_max_area_rectangle_l1026_102655


namespace NUMINAMATH_GPT_value_of_expression_l1026_102680

theorem value_of_expression : (7^2 - 3^2)^4 = 2560000 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1026_102680


namespace NUMINAMATH_GPT_ellipse_equation_l1026_102626

theorem ellipse_equation (a b : ℝ) (h1 : 0 < b) (h2 : b < a) 
  (h3 : ∃ (P : ℝ × ℝ), P = (0, -1) ∧ P.2^2 = b^2) 
  (h4 : ∃ (C2 : ℝ → ℝ → Prop), (∀ x y : ℝ, C2 x y ↔ x^2 + y^2 = 4) ∧ 2 * a = 4) :
  (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 ↔ (x^2 / 4) + y^2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_ellipse_equation_l1026_102626


namespace NUMINAMATH_GPT_simplify_fraction_l1026_102631

theorem simplify_fraction (x y : ℕ) (h₁ : x = 3) (h₂ : y = 4) : 
  (12 * x * y^3) / (9 * x^2 * y^2) = 16 / 9 :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1026_102631


namespace NUMINAMATH_GPT_differential_savings_l1026_102627

def annual_income_before_tax : ℝ := 42400
def initial_tax_rate : ℝ := 0.42
def new_tax_rate : ℝ := 0.32

theorem differential_savings :
  annual_income_before_tax * initial_tax_rate - annual_income_before_tax * new_tax_rate = 4240 :=
by
  sorry

end NUMINAMATH_GPT_differential_savings_l1026_102627


namespace NUMINAMATH_GPT_average_words_per_page_l1026_102628

theorem average_words_per_page
  (sheets_to_pages : ℕ := 16)
  (total_sheets : ℕ := 12)
  (total_word_count : ℕ := 240000) :
  (total_word_count / (total_sheets * sheets_to_pages)) = 1250 :=
by
  sorry

end NUMINAMATH_GPT_average_words_per_page_l1026_102628


namespace NUMINAMATH_GPT_car_speed_l1026_102608

variable (fuel_efficiency : ℝ) (fuel_decrease_gallons : ℝ) (time_hours : ℝ) 
          (gallons_to_liters : ℝ) (kilometers_to_miles : ℝ)
          (car_speed_mph : ℝ)

-- Conditions given in the problem
def fuelEfficiency : ℝ := 40 -- km per liter
def fuelDecreaseGallons : ℝ := 3.9 -- gallons
def timeHours : ℝ := 5.7 -- hours
def gallonsToLiters : ℝ := 3.8 -- liters per gallon
def kilometersToMiles : ℝ := 1.6 -- km per mile

theorem car_speed (fuel_efficiency fuelDecreaseGallons timeHours gallonsToLiters kilometersToMiles : ℝ) : 
  let fuelDecreaseLiters := fuelDecreaseGallons * gallonsToLiters
  let distanceKm := fuelDecreaseLiters * fuel_efficiency
  let distanceMiles := distanceKm / kilometersToMiles
  let averageSpeed := distanceMiles / timeHours
  averageSpeed = 65 := sorry

end NUMINAMATH_GPT_car_speed_l1026_102608


namespace NUMINAMATH_GPT_triangle_area_is_120_l1026_102670

-- Define the triangle sides
def a : ℕ := 10
def b : ℕ := 24
def c : ℕ := 26

-- Define a function to calculate the area of a right-angled triangle
noncomputable def right_triangle_area (a b : ℕ) : ℕ := (a * b) / 2

-- Statement to prove the area of the triangle
theorem triangle_area_is_120 : right_triangle_area 10 24 = 120 :=
by
  sorry

end NUMINAMATH_GPT_triangle_area_is_120_l1026_102670


namespace NUMINAMATH_GPT_smaller_circle_circumference_l1026_102681

theorem smaller_circle_circumference (r r2 : ℝ) : 
  (60:ℝ) / 360 * 2 * Real.pi * r = 8 →
  r = 24 / Real.pi →
  1 / 4 * (24 / Real.pi)^2 = (24 / Real.pi - 2 * r2) * (24 / Real.pi) →
  2 * Real.pi * r2 = 36 :=
  by
    intros h1 h2 h3
    sorry

end NUMINAMATH_GPT_smaller_circle_circumference_l1026_102681


namespace NUMINAMATH_GPT_negated_roots_quadratic_reciprocals_roots_quadratic_l1026_102603

-- For (1)
theorem negated_roots_quadratic (x y : ℝ) : 
    (x^2 + 3 * x - 2 = 0) ↔ (y^2 - 3 * y - 2 = 0) :=
sorry

-- For (2)
theorem reciprocals_roots_quadratic (a b c x y : ℝ) (h : a ≠ 0) :
    (a * x^2 - b * x + c = 0) ↔ (c * y^2 - b * y + a = 0) :=
sorry

end NUMINAMATH_GPT_negated_roots_quadratic_reciprocals_roots_quadratic_l1026_102603


namespace NUMINAMATH_GPT_range_of_a_l1026_102671

noncomputable def f (x a : ℝ) := Real.exp (-x) - 2 * x - a

def curve (x : ℝ) := x ^ 3 + x

def y_in_range (x : ℝ) := x >= -2 ∧ x <= 2

theorem range_of_a : ∀ (a : ℝ), (∃ x, y_in_range (curve x) ∧ f (curve x) a = curve x) ↔ a ∈ Set.Icc (Real.exp (-2) - 6) (Real.exp 2 + 6) := by
  sorry

end NUMINAMATH_GPT_range_of_a_l1026_102671


namespace NUMINAMATH_GPT_difference_of_roots_l1026_102652

theorem difference_of_roots :
  ∀ (x : ℝ), (x^2 - 5*x + 6 = 0) → (∃ r1 r2 : ℝ, r1 > 2 ∧ r2 < r1 ∧ r1 - r2 = 1) :=
by
  sorry

end NUMINAMATH_GPT_difference_of_roots_l1026_102652


namespace NUMINAMATH_GPT_harry_pencils_lost_l1026_102600

-- Define the conditions
def anna_pencils : ℕ := 50
def harry_initial_pencils : ℕ := 2 * anna_pencils
def harry_current_pencils : ℕ := 81

-- Define the proof statement
theorem harry_pencils_lost :
  harry_initial_pencils - harry_current_pencils = 19 :=
by
  -- The proof is to be filled in
  sorry

end NUMINAMATH_GPT_harry_pencils_lost_l1026_102600


namespace NUMINAMATH_GPT_find_original_number_l1026_102685

theorem find_original_number (c : ℝ) (h₁ : c / 12.75 = 16) (h₂ : 2.04 / 1.275 = 1.6) : c = 204 :=
by
  sorry

end NUMINAMATH_GPT_find_original_number_l1026_102685


namespace NUMINAMATH_GPT_total_output_equal_at_20_l1026_102686

noncomputable def total_output_A (x : ℕ) : ℕ :=
  200 + 20 * x

noncomputable def total_output_B (x : ℕ) : ℕ :=
  30 * x

theorem total_output_equal_at_20 :
  total_output_A 20 = total_output_B 20 :=
by
  sorry

end NUMINAMATH_GPT_total_output_equal_at_20_l1026_102686


namespace NUMINAMATH_GPT_initial_average_l1026_102674

variable (A : ℝ)
variables (nums : Fin 5 → ℝ)
variables (h_sum : 5 * A = nums 0 + nums 1 + nums 2 + nums 3 + nums 4)
variables (h_num : nums 0 = 12)
variables (h_new_avg : (5 * A + 12) / 5 = 9.2)

theorem initial_average :
  A = 6.8 :=
sorry

end NUMINAMATH_GPT_initial_average_l1026_102674


namespace NUMINAMATH_GPT_hyperbola_slope_product_l1026_102683

open Real

theorem hyperbola_slope_product
  (a b : ℝ) (a_pos : a > 0) (b_pos : b > 0)
  (h : ∀ {x y : ℝ}, x ≠ 0 → (x^2 / a^2 - y^2 / b^2 = 1) → 
    ∀ {k1 k2 : ℝ}, (x = 0 ∨ y = 0) → (k1 * k2 = ((b^2) / (a^2)))) :
  (b^2 / a^2 = 3) :=
by 
  sorry

end NUMINAMATH_GPT_hyperbola_slope_product_l1026_102683


namespace NUMINAMATH_GPT_correct_operation_l1026_102634

theorem correct_operation (a b : ℝ) : ((-3 * a^2 * b)^2 = 9 * a^4 * b^2) := sorry

end NUMINAMATH_GPT_correct_operation_l1026_102634


namespace NUMINAMATH_GPT_initial_marbles_l1026_102644

theorem initial_marbles (M : ℝ) (h0 : 0.2 * M + 0.35 * (0.8 * M) + 130 = M) : M = 250 :=
by
  sorry

end NUMINAMATH_GPT_initial_marbles_l1026_102644


namespace NUMINAMATH_GPT_universal_proposition_l1026_102647

def is_multiple_of_two (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2 * k

def is_even (x : ℕ) : Prop :=
  ∃ k : ℕ, x = 2 * k

theorem universal_proposition : 
  (∀ x : ℕ, is_multiple_of_two x → is_even x) :=
by
  sorry

end NUMINAMATH_GPT_universal_proposition_l1026_102647


namespace NUMINAMATH_GPT_angle_of_inclination_l1026_102666

theorem angle_of_inclination (x y : ℝ) (θ : ℝ) :
  (x - y - 1 = 0) → θ = 45 :=
by
  sorry

end NUMINAMATH_GPT_angle_of_inclination_l1026_102666


namespace NUMINAMATH_GPT_students_math_inequality_l1026_102604

variables {n x a b c : ℕ}

theorem students_math_inequality (h1 : x + a ≥ 8 * n / 10) 
                                (h2 : x + b ≥ 8 * n / 10) 
                                (h3 : n ≥ a + b + c + x) : 
                                x * 5 ≥ 4 * (x + c) :=
by
  sorry

end NUMINAMATH_GPT_students_math_inequality_l1026_102604


namespace NUMINAMATH_GPT_coupons_used_l1026_102658

theorem coupons_used
  (initial_books : ℝ)
  (sold_books : ℝ)
  (coupons_per_book : ℝ)
  (remaining_books := initial_books - sold_books)
  (total_coupons := remaining_books * coupons_per_book) :
  initial_books = 40.0 →
  sold_books = 20.0 →
  coupons_per_book = 4.0 →
  total_coupons = 80.0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_coupons_used_l1026_102658


namespace NUMINAMATH_GPT_class3_total_score_l1026_102661

theorem class3_total_score 
  (total_points : ℕ)
  (class1_score class2_score class3_score : ℕ)
  (class1_places class2_places class3_places : ℕ)
  (total_places : ℕ)
  (points_1st  points_2nd  points_3rd : ℕ)
  (h1 : total_points = 27)
  (h2 : class1_score = class2_score)
  (h3 : 2 * class1_places = class2_places)
  (h4 : class1_places + class2_places + class3_places = total_places)
  (h5 : 3 * points_1st + 3 * points_2nd + 3 * points_3rd = total_points)
  (h6 : total_places = 9)
  (h7 : points_1st = 5)
  (h8 : points_2nd = 3)
  (h9 : points_3rd = 1) :
  class3_score = 7 :=
sorry

end NUMINAMATH_GPT_class3_total_score_l1026_102661


namespace NUMINAMATH_GPT_simplify_power_multiplication_l1026_102637

theorem simplify_power_multiplication (x : ℝ) : (-x) ^ 3 * (-x) ^ 2 = -x ^ 5 :=
by sorry

end NUMINAMATH_GPT_simplify_power_multiplication_l1026_102637


namespace NUMINAMATH_GPT_real_y_iff_x_ranges_l1026_102690

-- Definitions for conditions
variable (x y : ℝ)

-- Condition for the equation
def equation := 9 * y^2 - 6 * x * y + 2 * x + 7 = 0

-- Theorem statement
theorem real_y_iff_x_ranges :
  (∃ y : ℝ, equation x y) ↔ (x ≤ -2 ∨ x ≥ 7) :=
sorry

end NUMINAMATH_GPT_real_y_iff_x_ranges_l1026_102690


namespace NUMINAMATH_GPT_thomas_total_blocks_l1026_102602

-- Definitions according to the conditions
def a1 : Nat := 7
def a2 : Nat := a1 + 3
def a3 : Nat := a2 - 6
def a4 : Nat := a3 + 10
def a5 : Nat := 2 * a2

-- The total number of blocks
def total_blocks : Nat := a1 + a2 + a3 + a4 + a5

-- The proof statement
theorem thomas_total_blocks :
  total_blocks = 55 := 
sorry

end NUMINAMATH_GPT_thomas_total_blocks_l1026_102602


namespace NUMINAMATH_GPT_batsman_average_after_17th_inning_l1026_102689

theorem batsman_average_after_17th_inning 
    (A : ℕ) 
    (hA : A = 15) 
    (runs_17th_inning : ℕ)
    (increase_in_average : ℕ) 
    (hscores : runs_17th_inning = 100)
    (hincrease : increase_in_average = 5) :
    (A + increase_in_average = 20) :=
by
  sorry

end NUMINAMATH_GPT_batsman_average_after_17th_inning_l1026_102689


namespace NUMINAMATH_GPT_polygon_D_has_largest_area_l1026_102642

noncomputable def area_A := 4 * 1 + 2 * (1 / 2) -- 5
noncomputable def area_B := 2 * 1 + 2 * (1 / 2) + Real.pi / 4 -- ≈ 3.785
noncomputable def area_C := 3 * 1 + 3 * (1 / 2) -- 4.5
noncomputable def area_D := 3 * 1 + 1 * (1 / 2) + 2 * (Real.pi / 4) -- ≈ 5.07
noncomputable def area_E := 1 * 1 + 3 * (1 / 2) + 3 * (Real.pi / 4) -- ≈ 4.855

theorem polygon_D_has_largest_area :
  area_D > area_A ∧
  area_D > area_B ∧
  area_D > area_C ∧
  area_D > area_E :=
by
  sorry

end NUMINAMATH_GPT_polygon_D_has_largest_area_l1026_102642


namespace NUMINAMATH_GPT_vinnie_tips_l1026_102677

variable (Paul Vinnie : ℕ)

def tips_paul := 14
def more_vinnie_than_paul := 16

theorem vinnie_tips :
  Vinnie = tips_paul + more_vinnie_than_paul :=
by
  unfold tips_paul more_vinnie_than_paul -- unfolding defined values
  exact sorry

end NUMINAMATH_GPT_vinnie_tips_l1026_102677


namespace NUMINAMATH_GPT_investment_value_change_l1026_102692

theorem investment_value_change (k m : ℝ) : 
  let increaseFactor := 1 + k / 100
  let decreaseFactor := 1 - m / 100 
  let overallFactor := increaseFactor * decreaseFactor 
  let changeFactor := overallFactor - 1
  let percentageChange := changeFactor * 100 
  percentageChange = k - m - (k * m) / 100 := 
by 
  sorry

end NUMINAMATH_GPT_investment_value_change_l1026_102692


namespace NUMINAMATH_GPT_vector_computation_l1026_102675

def c : ℝ × ℝ × ℝ := (-3, 5, 2)
def d : ℝ × ℝ × ℝ := (5, -1, 3)

theorem vector_computation : 2 • c - 5 • d + c = (-34, 20, -9) := by
  sorry

end NUMINAMATH_GPT_vector_computation_l1026_102675


namespace NUMINAMATH_GPT_households_used_both_brands_l1026_102632

theorem households_used_both_brands 
  (total_households : ℕ)
  (neither_AB : ℕ)
  (only_A : ℕ)
  (h3 : ∀ (both : ℕ), ∃ (only_B : ℕ), only_B = 3 * both)
  (h_sum : ∀ (both : ℕ), neither_AB + only_A + both + (3 * both) = total_households) :
  ∃ (both : ℕ), both = 10 :=
by 
  sorry

end NUMINAMATH_GPT_households_used_both_brands_l1026_102632


namespace NUMINAMATH_GPT_variance_of_data_set_l1026_102613

theorem variance_of_data_set (m : ℝ) (h_mean : (6 + 7 + 8 + 9 + m) / 5 = 8) :
    (1/5) * ((6-8)^2 + (7-8)^2 + (8-8)^2 + (9-8)^2 + (m-8)^2) = 2 := 
sorry

end NUMINAMATH_GPT_variance_of_data_set_l1026_102613


namespace NUMINAMATH_GPT_find_d_value_l1026_102609

/-- Let d be an odd prime number. If 89 - (d+3)^2 is the square of an integer, then d = 5. -/
theorem find_d_value (d : ℕ) (h₁ : Nat.Prime d) (h₂ : Odd d) (h₃ : ∃ m : ℤ, 89 - (d + 3)^2 = m^2) : d = 5 := 
by
  sorry

end NUMINAMATH_GPT_find_d_value_l1026_102609


namespace NUMINAMATH_GPT_plywood_perimeter_difference_l1026_102612

theorem plywood_perimeter_difference :
  ∀ (length width : ℕ) (n : ℕ), 
    length = 6 ∧ width = 9 ∧ n = 6 → 
    ∃ (max_perimeter min_perimeter : ℕ), 
      (max_perimeter - min_perimeter = 10) ∧
      max_perimeter = 20 ∧ 
      min_perimeter = 10 :=
by
  sorry

end NUMINAMATH_GPT_plywood_perimeter_difference_l1026_102612


namespace NUMINAMATH_GPT_angle_bisectors_geq_nine_times_inradius_l1026_102622

theorem angle_bisectors_geq_nine_times_inradius 
  (r : ℝ) (f_a f_b f_c : ℝ) 
  (h_triangle : ∃ a b c : ℝ, a > 0 ∧ b > 0 ∧ c > 0 ∧ r = (1 / 2) * (a + b + c) * r ∧ 
      f_a ≥ (2 * a * b / (a + b) + 2 * a * c / (a + c)) / 2 ∧ 
      f_b ≥ (2 * b * a / (b + a) + 2 * b * c / (b + c)) / 2 ∧ 
      f_c ≥ (2 * c * a / (c + a) + 2 * c * b / (c + b)) / 2)
  : f_a + f_b + f_c ≥ 9 * r :=
sorry

end NUMINAMATH_GPT_angle_bisectors_geq_nine_times_inradius_l1026_102622


namespace NUMINAMATH_GPT_carla_catches_up_in_three_hours_l1026_102641

-- Definitions as lean statements based on conditions
def john_speed : ℝ := 30
def carla_speed : ℝ := 35
def john_start_time : ℝ := 0
def carla_start_time : ℝ := 0.5

-- Lean problem statement to prove the catch-up time
theorem carla_catches_up_in_three_hours : 
  ∃ t : ℝ, 35 * t = 30 * (t + 0.5) ∧ t = 3 :=
by
  sorry

end NUMINAMATH_GPT_carla_catches_up_in_three_hours_l1026_102641


namespace NUMINAMATH_GPT_equal_product_groups_exist_l1026_102684

def numbers : List ℕ := [21, 22, 34, 39, 44, 45, 65, 76, 133, 153]

theorem equal_product_groups_exist :
  ∃ (g1 g2 : List ℕ), 
    g1.length = 5 ∧ g2.length = 5 ∧ 
    g1.prod = g2.prod ∧ g1.prod = 349188840 ∧ 
    (g1 ++ g2 = numbers ∨ g1 ++ g2 = numbers.reverse) :=
by
  sorry

end NUMINAMATH_GPT_equal_product_groups_exist_l1026_102684


namespace NUMINAMATH_GPT_sequence_property_exists_l1026_102676

theorem sequence_property_exists :
  ∃ a₁ a₂ a₃ a₄ : ℝ, 
  a₂ - a₁ = a₃ - a₂ ∧ a₃ - a₂ = a₄ - a₃ ∧
  (a₃ / a₁ = a₄ / a₃) ∧ ∃ r : ℝ, r ≠ 0 ∧ a₁ = -4 * r ∧ a₂ = -3 * r ∧ a₃ = -2 * r ∧ a₄ = -r :=
by
  sorry

end NUMINAMATH_GPT_sequence_property_exists_l1026_102676


namespace NUMINAMATH_GPT_ratio_of_two_numbers_l1026_102688

theorem ratio_of_two_numbers (a b : ℝ) (h1 : a + b = 7 * (a - b)) (h2 : a > b) (h3 : a > 0) (h4 : b > 0) :
  a / b = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_two_numbers_l1026_102688


namespace NUMINAMATH_GPT_difference_sweaters_Monday_Tuesday_l1026_102697

-- Define conditions
def sweaters_knit_on_Monday : ℕ := 8
def sweaters_knit_on_Tuesday (T : ℕ) : Prop := T > 8
def sweaters_knit_on_Wednesday (T : ℕ) : ℕ := T - 4
def sweaters_knit_on_Thursday (T : ℕ) : ℕ := T - 4
def sweaters_knit_on_Friday : ℕ := 4

-- Define total sweaters knit in the week
def total_sweaters_knit (T : ℕ) : ℕ :=
  sweaters_knit_on_Monday + T + sweaters_knit_on_Wednesday T + sweaters_knit_on_Thursday T + sweaters_knit_on_Friday

-- Lean Theorem Statement
theorem difference_sweaters_Monday_Tuesday : ∀ T : ℕ, sweaters_knit_on_Tuesday T → total_sweaters_knit T = 34 → T - sweaters_knit_on_Monday = 2 :=
by
  intros T hT_total
  sorry

end NUMINAMATH_GPT_difference_sweaters_Monday_Tuesday_l1026_102697


namespace NUMINAMATH_GPT_average_gas_mileage_round_trip_l1026_102696

theorem average_gas_mileage_round_trip
  (d : ℝ) (ms mr : ℝ)
  (h1 : d = 150)
  (h2 : ms = 35)
  (h3 : mr = 15) :
  (2 * d) / ((d / ms) + (d / mr)) = 21 :=
by
  sorry

end NUMINAMATH_GPT_average_gas_mileage_round_trip_l1026_102696


namespace NUMINAMATH_GPT_relationship_between_a_b_c_l1026_102679

-- Define the given parabola function
def parabola (x : ℝ) (k : ℝ) : ℝ := -(x - 2)^2 + k

-- Define the points A, B, C with their respective coordinates and expressions on the parabola
variables {a b c k : ℝ}

-- Conditions: Points lie on the parabola
theorem relationship_between_a_b_c (hA : a = parabola (-2) k)
                                  (hB : b = parabola (-1) k)
                                  (hC : c = parabola 3 k) :
  a < b ∧ b < c :=
by
  sorry

end NUMINAMATH_GPT_relationship_between_a_b_c_l1026_102679


namespace NUMINAMATH_GPT_students_owning_both_pets_l1026_102656

theorem students_owning_both_pets:
  ∀ (students total students_dog students_cat : ℕ),
    total = 45 →
    students_dog = 28 →
    students_cat = 38 →
    -- Each student owning at least one pet means 
    -- total = students_dog ∪ students_cat
    total = students_dog + students_cat - students →
    students = 21 :=
by
  intros students total students_dog students_cat h_total h_dog h_cat h_union
  sorry

end NUMINAMATH_GPT_students_owning_both_pets_l1026_102656


namespace NUMINAMATH_GPT_number_of_red_balls_eq_47_l1026_102611

theorem number_of_red_balls_eq_47
  (T : ℕ) (white green yellow purple : ℕ)
  (neither_red_nor_purple_prob : ℚ)
  (hT : T = 100)
  (hWhite : white = 10)
  (hGreen : green = 30)
  (hYellow : yellow = 10)
  (hPurple : purple = 3)
  (hProb : neither_red_nor_purple_prob = 0.5)
  : T - (white + green + yellow + purple) = 47 :=
by
  -- Sorry is used to skip the actual proof
  sorry

end NUMINAMATH_GPT_number_of_red_balls_eq_47_l1026_102611


namespace NUMINAMATH_GPT_perpendicular_vectors_l1026_102605

-- Definitions based on the conditions
def vector_a (x : ℝ) := (x, 3)
def vector_b := (3, 1)

-- Statement to prove
theorem perpendicular_vectors (x : ℝ) :
  (vector_a x).1 * (vector_b).1 + (vector_a x).2 * (vector_b).2 = 0 → x = -1 := by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_perpendicular_vectors_l1026_102605


namespace NUMINAMATH_GPT_perpendicular_condition_l1026_102619

theorem perpendicular_condition (m : ℝ) :
  (∀ x y : ℝ, 2 * x - y - 1 = 0 → (m * x + y + 1 = 0 → (2 * m - 1 = 0))) ↔ (m = 1/2) :=
by sorry

end NUMINAMATH_GPT_perpendicular_condition_l1026_102619


namespace NUMINAMATH_GPT_find_number_l1026_102691

theorem find_number
  (n : ℕ)
  (h : 80641 * n = 806006795) :
  n = 9995 :=
by 
  sorry

end NUMINAMATH_GPT_find_number_l1026_102691


namespace NUMINAMATH_GPT_find_constant_l1026_102659

theorem find_constant (x1 x2 : ℝ) (C : ℝ) :
  x1 - x2 = 5.5 ∧
  x1 + x2 = -5 / 2 ∧
  x1 * x2 = C / 2 →
  C = -12 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_constant_l1026_102659


namespace NUMINAMATH_GPT_second_reduction_percentage_l1026_102662

variable (P : ℝ) -- Original price
variable (x : ℝ) -- Second reduction percentage

-- Condition 1: After a 25% reduction
def first_reduction (P : ℝ) : ℝ := 0.75 * P

-- Condition 3: Combined reduction equivalent to 47.5%
def combined_reduction (P : ℝ) : ℝ := 0.525 * P

-- Question: Given the conditions, prove that the second reduction is 0.3
theorem second_reduction_percentage (P : ℝ) (x : ℝ) :
  (1 - x) * first_reduction P = combined_reduction P → x = 0.3 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_second_reduction_percentage_l1026_102662


namespace NUMINAMATH_GPT_fraction_area_of_shaded_square_in_larger_square_is_one_eighth_l1026_102668

theorem fraction_area_of_shaded_square_in_larger_square_is_one_eighth :
  let side_larger_square := 4
  let area_larger_square := side_larger_square^2
  let side_shaded_square := Real.sqrt (1^2 + 1^2)
  let area_shaded_square := side_shaded_square^2
  area_shaded_square / area_larger_square = 1 / 8 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_area_of_shaded_square_in_larger_square_is_one_eighth_l1026_102668


namespace NUMINAMATH_GPT_correct_bio_experiment_technique_l1026_102660

-- Let's define our conditions as hypotheses.
def yeast_count_method := "sampling_inspection"
def small_animal_group_method := "sampler_sampling"
def mitosis_rinsing_purpose := "wash_away_dissociation_solution"
def fat_identification_solution := "alcohol"

-- The question translated into a statement is to show that the method for counting yeast is the sampling inspection method.
theorem correct_bio_experiment_technique :
  yeast_count_method = "sampling_inspection" ∧
  small_animal_group_method ≠ "mark-recapture" ∧
  mitosis_rinsing_purpose ≠ "wash_away_dye" ∧
  fat_identification_solution ≠ "50%_hydrochloric_acid" :=
sorry

end NUMINAMATH_GPT_correct_bio_experiment_technique_l1026_102660


namespace NUMINAMATH_GPT_martha_initial_blocks_l1026_102638

theorem martha_initial_blocks (final_blocks : ℕ) (found_blocks : ℕ) (initial_blocks : ℕ) : 
  final_blocks = initial_blocks + found_blocks → 
  final_blocks = 84 →
  found_blocks = 80 → 
  initial_blocks = 4 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_martha_initial_blocks_l1026_102638


namespace NUMINAMATH_GPT_number_of_educated_employees_l1026_102645

-- Define the context and input values
variable (T: ℕ) (I: ℕ := 20) (decrease_illiterate: ℕ := 15) (total_decrease_illiterate: ℕ := I * decrease_illiterate) (average_salary_decrease: ℕ := 10)

-- The theorem statement
theorem number_of_educated_employees (h1: total_decrease_illiterate / T = average_salary_decrease) (h2: T = I + 10): L = 10 := by
  sorry

end NUMINAMATH_GPT_number_of_educated_employees_l1026_102645


namespace NUMINAMATH_GPT_colby_mango_sales_l1026_102657

theorem colby_mango_sales
  (total_kg : ℕ)
  (mangoes_per_kg : ℕ)
  (remaining_mangoes : ℕ)
  (half_sold_to_market : ℕ) :
  total_kg = 60 →
  mangoes_per_kg = 8 →
  remaining_mangoes = 160 →
  half_sold_to_market = 20 := by
    sorry

end NUMINAMATH_GPT_colby_mango_sales_l1026_102657


namespace NUMINAMATH_GPT_incorrect_operation_l1026_102640

theorem incorrect_operation 
    (x y : ℝ) :
    (x - y) / (x + y) = (y - x) / (y + x) ↔ False := 
by 
  sorry

end NUMINAMATH_GPT_incorrect_operation_l1026_102640


namespace NUMINAMATH_GPT_tim_interest_rate_l1026_102618

theorem tim_interest_rate
  (r : ℝ)
  (h1 : ∀ n, (600 * (1 + r)^2 - 600) = (1000 * (1.05)^(n) - 1000))
  (h2 : ∀ n, (600 * (1 + r)^2 - 600) = (1000 * (1.05)^(n) - 1000) + 23.5) : 
  r = 0.1 :=
by
  sorry

end NUMINAMATH_GPT_tim_interest_rate_l1026_102618


namespace NUMINAMATH_GPT_find_K_find_t_l1026_102621

-- Proof Problem for G9.2
theorem find_K (x : ℚ) (K : ℚ) (h1 : x = 1.9898989) (h2 : x - 1 = K / 99) : K = 98 :=
sorry

-- Proof Problem for G9.3
theorem find_t (p q r t : ℚ)
  (h_avg1 : (p + q + r) / 3 = 18)
  (h_avg2 : ((p + 1) + (q - 2) + (r + 3) + t) / 4 = 19) : t = 20 :=
sorry

end NUMINAMATH_GPT_find_K_find_t_l1026_102621


namespace NUMINAMATH_GPT_range_of_a_l1026_102663

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem range_of_a (a : ℝ) : 
  (∃ x ∈ Set.Ioo a (a + 1), ∃ f' : ℝ → ℝ, ∀ x, f' x = (x * Real.exp x) * (x + 2) ∧ f' x = 0) ↔ 
  a ∈ Set.Ioo (-3 : ℝ) (-2) ∪ Set.Ioo (-1) (0) := 
sorry

end NUMINAMATH_GPT_range_of_a_l1026_102663


namespace NUMINAMATH_GPT_trip_drop_probability_l1026_102607

-- Definitions
def P_Trip : ℝ := 0.4
def P_Drop_not : ℝ := 0.9

-- Main theorem
theorem trip_drop_probability : ∀ (P_Trip P_Drop_not : ℝ), P_Trip = 0.4 → P_Drop_not = 0.9 → 1 - P_Drop_not = 0.1 :=
by
  intros P_Trip P_Drop_not h1 h2
  rw [h2]
  norm_num

end NUMINAMATH_GPT_trip_drop_probability_l1026_102607


namespace NUMINAMATH_GPT_first_shaded_square_ensuring_all_columns_l1026_102639

def triangular_number (n : ℕ) : ℕ :=
  n * (n + 1) / 2

def shaded_squares_in_columns (k : ℕ) : Prop :=
  ∀ j : ℕ, j < 7 → ∃ n : ℕ, triangular_number n % 7 = j ∧ triangular_number n ≤ k

theorem first_shaded_square_ensuring_all_columns:
  shaded_squares_in_columns 55 :=
by
  sorry

end NUMINAMATH_GPT_first_shaded_square_ensuring_all_columns_l1026_102639


namespace NUMINAMATH_GPT_shaded_percentage_l1026_102653

noncomputable def percent_shaded (side_len : ℕ) : ℝ :=
  let total_area := (side_len : ℝ) * side_len
  let shaded_area := (2 * 2) + (2 * 5) + (1 * 7)
  100 * (shaded_area / total_area)

theorem shaded_percentage (PQRS_side : ℕ) (hPQRS : PQRS_side = 7) :
  percent_shaded PQRS_side = 42.857 :=
  by
  rw [hPQRS]
  sorry

end NUMINAMATH_GPT_shaded_percentage_l1026_102653
