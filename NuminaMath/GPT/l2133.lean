import Mathlib

namespace NUMINAMATH_GPT_triangle_is_isosceles_if_median_bisects_perimeter_l2133_213327

-- Defining the sides of the triangle
variables {a b c : ℝ}

-- Defining the median condition
def median_bisects_perimeter (a b c : ℝ) : Prop :=
  a + b + c = 2 * (a/2 + b)

-- The main theorem stating that the triangle is isosceles if the median bisects the perimeter
theorem triangle_is_isosceles_if_median_bisects_perimeter (a b c : ℝ) 
  (h : median_bisects_perimeter a b c) : b = c :=
by
  sorry

end NUMINAMATH_GPT_triangle_is_isosceles_if_median_bisects_perimeter_l2133_213327


namespace NUMINAMATH_GPT_sum_of_midpoints_l2133_213303

variable (a b c : ℝ)

def sum_of_vertices := a + b + c

theorem sum_of_midpoints (h : sum_of_vertices a b c = 15) :
  (a + b)/2 + (a + c)/2 + (b + c)/2 = 15 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_midpoints_l2133_213303


namespace NUMINAMATH_GPT_mike_total_hours_worked_l2133_213337

-- Define the conditions
def time_to_wash_car := 10
def time_to_change_oil := 15
def time_to_change_tires := 30

def number_of_cars_washed := 9
def number_of_oil_changes := 6
def number_of_tire_changes := 2

-- Define the conversion factor
def minutes_per_hour := 60

-- Prove that the total time worked equals 4 hours
theorem mike_total_hours_worked : 
  (number_of_cars_washed * time_to_wash_car + 
   number_of_oil_changes * time_to_change_oil + 
   number_of_tire_changes * time_to_change_tires) / minutes_per_hour = 4 := by
  sorry

end NUMINAMATH_GPT_mike_total_hours_worked_l2133_213337


namespace NUMINAMATH_GPT_markup_correct_l2133_213354

theorem markup_correct (purchase_price : ℝ) (overhead_percentage : ℝ) (net_profit : ℝ) :
  purchase_price = 48 → overhead_percentage = 0.15 → net_profit = 12 →
  (purchase_price * (1 + overhead_percentage) + net_profit - purchase_price) = 19.2 :=
by
  intros
  sorry

end NUMINAMATH_GPT_markup_correct_l2133_213354


namespace NUMINAMATH_GPT_smallest_positive_x_l2133_213335

theorem smallest_positive_x (x : ℝ) (h : ⌊x^2⌋ - x * ⌊x⌋ = 8) : x = 89 / 9 :=
sorry

end NUMINAMATH_GPT_smallest_positive_x_l2133_213335


namespace NUMINAMATH_GPT_find_cost_price_l2133_213325

variable (CP : ℝ) -- cost price
variable (SP_loss SP_gain : ℝ) -- selling prices

-- Conditions
def loss_condition := SP_loss = 0.9 * CP
def gain_condition := SP_gain = 1.04 * CP
def difference_condition := SP_gain - SP_loss = 190

-- Theorem to prove
theorem find_cost_price (h_loss : loss_condition CP SP_loss)
                        (h_gain : gain_condition CP SP_gain)
                        (h_diff : difference_condition SP_loss SP_gain) :
  CP = 1357.14 := 
sorry

end NUMINAMATH_GPT_find_cost_price_l2133_213325


namespace NUMINAMATH_GPT_prod_sum_reciprocal_bounds_l2133_213368

-- Define the product of the sum of three positive numbers and the sum of their reciprocals.
theorem prod_sum_reciprocal_bounds (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  9 ≤ (a + b + c) * (1 / a + 1 / b + 1 / c) :=
by
  sorry

end NUMINAMATH_GPT_prod_sum_reciprocal_bounds_l2133_213368


namespace NUMINAMATH_GPT_train_speed_is_117_l2133_213377

noncomputable def train_speed (train_length : ℝ) (crossing_time : ℝ) (man_speed_kmph : ℝ) : ℝ :=
  let man_speed_mps := man_speed_kmph * 1000 / 3600
  let relative_speed := train_length / crossing_time
  (relative_speed - man_speed_mps) * 3.6

theorem train_speed_is_117 :
  train_speed 300 9 3 = 117 :=
by
  -- We leave the proof as sorry since only the statement is needed
  sorry

end NUMINAMATH_GPT_train_speed_is_117_l2133_213377


namespace NUMINAMATH_GPT_train_length_l2133_213300

theorem train_length :
  ∀ (t : ℝ) (v_man : ℝ) (v_train : ℝ),
  t = 41.9966402687785 →
  v_man = 3 →
  v_train = 63 →
  (v_train - v_man) * (5 / 18) * t = 699.94400447975 :=
by
  intros t v_man v_train ht hv_man hv_train
  -- Use the given conditions as definitions
  rw [ht, hv_man, hv_train]
  sorry

end NUMINAMATH_GPT_train_length_l2133_213300


namespace NUMINAMATH_GPT_mushrooms_safe_to_eat_l2133_213381

theorem mushrooms_safe_to_eat (S : ℕ) (Total_mushrooms Poisonous_mushrooms Uncertain_mushrooms : ℕ)
  (h1: Total_mushrooms = 32)
  (h2: Poisonous_mushrooms = 2 * S)
  (h3: Uncertain_mushrooms = 5)
  (h4: S + Poisonous_mushrooms + Uncertain_mushrooms = Total_mushrooms) :
  S = 9 :=
sorry

end NUMINAMATH_GPT_mushrooms_safe_to_eat_l2133_213381


namespace NUMINAMATH_GPT_perimeter_of_regular_polygon_l2133_213347

/-- 
Given a regular polygon with a central angle of 45 degrees and a side length of 5,
the perimeter of the polygon is 40.
-/
theorem perimeter_of_regular_polygon 
  (central_angle : ℝ) (side_length : ℝ) (h1 : central_angle = 45)
  (h2 : side_length = 5) :
  ∃ P, P = 40 :=
by
  sorry

end NUMINAMATH_GPT_perimeter_of_regular_polygon_l2133_213347


namespace NUMINAMATH_GPT_gecko_third_day_crickets_l2133_213320

def total_crickets : ℕ := 70
def first_day_percentage : ℝ := 0.30
def first_day_crickets : ℝ := first_day_percentage * total_crickets
def second_day_crickets : ℝ := first_day_crickets - 6
def third_day_crickets : ℝ := total_crickets - (first_day_crickets + second_day_crickets)

theorem gecko_third_day_crickets :
  third_day_crickets = 34 :=
by
  sorry

end NUMINAMATH_GPT_gecko_third_day_crickets_l2133_213320


namespace NUMINAMATH_GPT_playgroup_count_l2133_213322

-- Definitions based on the conditions
def total_people (girls boys parents : ℕ) := girls + boys + parents
def playgroups (total size_per_group : ℕ) := total / size_per_group

-- Statement of the problem
theorem playgroup_count (girls boys parents size_per_group : ℕ)
  (h_girls : girls = 14)
  (h_boys : boys = 11)
  (h_parents : parents = 50)
  (h_size_per_group : size_per_group = 25) :
  playgroups (total_people girls boys parents) size_per_group = 3 :=
by {
  -- This is just the statement, the proof is skipped with sorry
  sorry
}

end NUMINAMATH_GPT_playgroup_count_l2133_213322


namespace NUMINAMATH_GPT_alma_score_l2133_213329

variables (A M S : ℕ)

-- Given conditions
axiom h1 : M = 60
axiom h2 : M = 3 * A
axiom h3 : A + M = 2 * S

theorem alma_score : S = 40 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_alma_score_l2133_213329


namespace NUMINAMATH_GPT_Carlton_button_up_shirts_l2133_213395

/-- 
Given that the number of sweater vests V is twice the number of button-up shirts S, 
and the total number of unique outfits (each combination of a sweater vest and a button-up shirt) is 18, 
prove that the number of button-up shirts S is 3. 
-/
theorem Carlton_button_up_shirts (V S : ℕ) (h1 : V = 2 * S) (h2 : V * S = 18) : S = 3 := by
  sorry

end NUMINAMATH_GPT_Carlton_button_up_shirts_l2133_213395


namespace NUMINAMATH_GPT_pay_for_notebook_with_change_l2133_213317

theorem pay_for_notebook_with_change : ∃ (a b : ℤ), 16 * a - 27 * b = 1 :=
by
  sorry

end NUMINAMATH_GPT_pay_for_notebook_with_change_l2133_213317


namespace NUMINAMATH_GPT_cost_per_book_eq_three_l2133_213372

-- Let T be the total amount spent, B be the number of books, and C be the cost per book
variables (T B C : ℕ)
-- Conditions: Edward spent $6 (T = 6) to buy 2 books (B = 2)
-- Each book costs the same amount (C = T / B)
axiom total_amount : T = 6
axiom number_of_books : B = 2

-- We need to prove that each book cost $3
theorem cost_per_book_eq_three (h1 : T = 6) (h2 : B = 2) : (T / B) = 3 := by
  sorry

end NUMINAMATH_GPT_cost_per_book_eq_three_l2133_213372


namespace NUMINAMATH_GPT_birds_on_fence_total_l2133_213359

variable (initial_birds : ℕ) (additional_birds : ℕ)

theorem birds_on_fence_total {initial_birds additional_birds : ℕ} (h1 : initial_birds = 4) (h2 : additional_birds = 6) :
    initial_birds + additional_birds = 10 :=
  by
  sorry

end NUMINAMATH_GPT_birds_on_fence_total_l2133_213359


namespace NUMINAMATH_GPT_first_hour_rain_l2133_213348

variable (x : ℝ)
variable (rain_1st_hour : ℝ) (rain_2nd_hour : ℝ)
variable (total_rain : ℝ)

-- Define the conditions
def condition_1 (x rain_2nd_hour : ℝ) : Prop :=
  rain_2nd_hour = 2 * x + 7

def condition_2 (x rain_2nd_hour total_rain : ℝ) : Prop :=
  x + rain_2nd_hour = total_rain

-- Prove the amount of rain in the first hour
theorem first_hour_rain (h1 : condition_1 x rain_2nd_hour)
                         (h2 : condition_2 x rain_2nd_hour total_rain)
                         (total_rain_is_22 : total_rain = 22) :
  x = 5 :=
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_first_hour_rain_l2133_213348


namespace NUMINAMATH_GPT_decreased_value_of_expression_l2133_213386

theorem decreased_value_of_expression (x y z : ℝ) :
  let x' := 0.6 * x
  let y' := 0.6 * y
  let z' := 0.6 * z
  (x' * y' * z'^2) = 0.1296 * (x * y * z^2) :=
by
  sorry

end NUMINAMATH_GPT_decreased_value_of_expression_l2133_213386


namespace NUMINAMATH_GPT_corrected_mean_35_25_l2133_213378

theorem corrected_mean_35_25 (n : ℕ) (mean : ℚ) (x_wrong x_correct : ℚ) :
  n = 20 → mean = 36 → x_wrong = 40 → x_correct = 25 → 
  ( (mean * n - x_wrong + x_correct) / n = 35.25) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_corrected_mean_35_25_l2133_213378


namespace NUMINAMATH_GPT_total_pennies_l2133_213314

theorem total_pennies (rachelle_pennies : ℕ) (gretchen_pennies : ℕ) (rocky_pennies : ℕ)
  (h1 : rachelle_pennies = 180)
  (h2 : gretchen_pennies = rachelle_pennies / 2)
  (h3 : rocky_pennies = gretchen_pennies / 3) :
  rachelle_pennies + gretchen_pennies + rocky_pennies = 300 :=
by
  sorry

end NUMINAMATH_GPT_total_pennies_l2133_213314


namespace NUMINAMATH_GPT_imaginary_part_of_complex_l2133_213363

theorem imaginary_part_of_complex (z : ℂ) (h : (1 - I) * z = I) : z.im = 1 / 2 :=
sorry

end NUMINAMATH_GPT_imaginary_part_of_complex_l2133_213363


namespace NUMINAMATH_GPT_cups_of_flour_per_pound_of_pasta_l2133_213308

-- Definitions from conditions
def pounds_of_pasta_per_rack : ℕ := 3
def racks_owned : ℕ := 3
def additional_rack_needed : ℕ := 1
def cups_per_bag : ℕ := 8
def bags_used : ℕ := 3

-- Derived definitions from above conditions
def total_cups_of_flour : ℕ := bags_used * cups_per_bag  -- 24 cups
def total_racks_needed : ℕ := racks_owned + additional_rack_needed  -- 4 racks
def total_pounds_of_pasta : ℕ := total_racks_needed * pounds_of_pasta_per_rack  -- 12 pounds

theorem cups_of_flour_per_pound_of_pasta (x : ℕ) :
  (total_cups_of_flour / total_pounds_of_pasta) = x → x = 2 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cups_of_flour_per_pound_of_pasta_l2133_213308


namespace NUMINAMATH_GPT_sheila_hourly_wage_is_correct_l2133_213385

-- Definitions based on conditions
def works_hours_per_day_mwf : ℕ := 8
def works_days_mwf : ℕ := 3
def works_hours_per_day_tt : ℕ := 6
def works_days_tt : ℕ := 2
def weekly_earnings : ℕ := 216

-- Total calculated hours based on the problem conditions
def total_weekly_hours : ℕ := (works_hours_per_day_mwf * works_days_mwf) + (works_hours_per_day_tt * works_days_tt)

-- Target wage per hour
def wage_per_hour : ℕ := weekly_earnings / total_weekly_hours

-- The theorem stating the proof problem
theorem sheila_hourly_wage_is_correct : wage_per_hour = 6 := by
  sorry

end NUMINAMATH_GPT_sheila_hourly_wage_is_correct_l2133_213385


namespace NUMINAMATH_GPT_oxen_b_is_12_l2133_213358

variable (oxen_b : ℕ)

def share (oxen months : ℕ) : ℕ := oxen * months

def total_share (oxen_a oxen_b oxen_c months_a months_b months_c : ℕ) : ℕ :=
  share oxen_a months_a + share oxen_b months_b + share oxen_c months_c

def proportion (rent_c rent total_share_c total_share : ℕ) : Prop :=
  rent_c * total_share = rent * total_share_c

theorem oxen_b_is_12 : oxen_b = 12 := by
  let oxen_a := 10
  let oxen_c := 15
  let months_a := 7
  let months_b := 5
  let months_c := 3
  let rent := 210
  let rent_c := 54
  let share_a := share oxen_a months_a
  let share_c := share oxen_c months_c
  let total_share_val := total_share oxen_a oxen_b oxen_c months_a months_b months_c
  let total_rent := share_a + 5 * oxen_b + share_c
  have h1 : proportion rent_c rent share_c total_rent := by sorry
  rw [proportion] at h1
  sorry

end NUMINAMATH_GPT_oxen_b_is_12_l2133_213358


namespace NUMINAMATH_GPT_ways_to_draw_at_least_two_defective_l2133_213333

-- Definitions based on the conditions of the problem
def total_products : ℕ := 100
def defective_products : ℕ := 3
def selected_products : ℕ := 5

-- Binomial coefficient function
def C (n k : ℕ) : ℕ := Nat.choose n k

-- The theorem to prove
theorem ways_to_draw_at_least_two_defective :
  C defective_products 2 * C (total_products - defective_products) 3 + C defective_products 3 * C (total_products - defective_products) 2 =
  (C total_products selected_products - C defective_products 1 * C (total_products - defective_products) 4) :=
sorry

end NUMINAMATH_GPT_ways_to_draw_at_least_two_defective_l2133_213333


namespace NUMINAMATH_GPT_price_of_horse_and_cow_l2133_213352

theorem price_of_horse_and_cow (x y : ℝ) (h1 : 4 * x + 6 * y = 48) (h2 : 3 * x + 5 * y = 38) :
  (4 * x + 6 * y = 48) ∧ (3 * x + 5 * y = 38) := 
by
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_price_of_horse_and_cow_l2133_213352


namespace NUMINAMATH_GPT_composite_solid_volume_l2133_213382

theorem composite_solid_volume :
  let V_prism := 2 * 2 * 1
  let V_cylinder := Real.pi * 1^2 * 3
  let V_overlap := Real.pi / 2
  V_prism + V_cylinder - V_overlap = 4 + 5 * Real.pi / 2 :=
by
  sorry

end NUMINAMATH_GPT_composite_solid_volume_l2133_213382


namespace NUMINAMATH_GPT_fraction_of_sum_l2133_213394

theorem fraction_of_sum (n S : ℕ) 
  (h1 : S = (n-1) * ((n:ℚ) / 3))
  (h2 : n > 0) : 
  (n:ℚ) / (S + n) = 3 / (n + 2) := 
by 
  sorry

end NUMINAMATH_GPT_fraction_of_sum_l2133_213394


namespace NUMINAMATH_GPT_geometric_sequence_problem_l2133_213396

theorem geometric_sequence_problem (a : ℕ → ℤ)
  (q : ℤ)
  (h1 : a 2 * a 5 = -32)
  (h2 : a 3 + a 4 = 4)
  (hq : ∃ (k : ℤ), q = k) :
  a 9 = -256 := 
sorry

end NUMINAMATH_GPT_geometric_sequence_problem_l2133_213396


namespace NUMINAMATH_GPT_trapezoid_area_l2133_213365

theorem trapezoid_area (x : ℝ) (y : ℝ) :
  (∀ x, y = x + 1) →
  (∀ y, y = 12) →
  (∀ y, y = 7) →
  (∀ x, x = 0) →
  ∃ area,
  area = (1/2) * (6 + 11) * 5 ∧ area = 42.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_trapezoid_area_l2133_213365


namespace NUMINAMATH_GPT_value_of_y_l2133_213334

theorem value_of_y (x y : ℤ) (h1 : x^2 = y - 2) (h2 : x = -6) : y = 38 :=
by
  sorry

end NUMINAMATH_GPT_value_of_y_l2133_213334


namespace NUMINAMATH_GPT_int_to_fourth_power_l2133_213374

theorem int_to_fourth_power:
  3^4 * 9^8 = 243^4 :=
by 
  sorry

end NUMINAMATH_GPT_int_to_fourth_power_l2133_213374


namespace NUMINAMATH_GPT_soda_preference_l2133_213332

theorem soda_preference (total_surveyed : ℕ) (angle_soda_sector : ℕ) (h_total_surveyed : total_surveyed = 540) (h_angle_soda_sector : angle_soda_sector = 270) :
  let fraction_soda := angle_soda_sector / 360
  let people_soda := fraction_soda * total_surveyed
  people_soda = 405 :=
by
  sorry

end NUMINAMATH_GPT_soda_preference_l2133_213332


namespace NUMINAMATH_GPT_root_equation_identity_l2133_213313

theorem root_equation_identity {a b c p q : ℝ} 
  (h1 : a^2 + p*a + 1 = 0)
  (h2 : b^2 + p*b + 1 = 0)
  (h3 : b^2 + q*b + 2 = 0)
  (h4 : c^2 + q*c + 2 = 0) 
  : (b - a) * (b - c) = p*q - 6 := 
sorry

end NUMINAMATH_GPT_root_equation_identity_l2133_213313


namespace NUMINAMATH_GPT_monotonicity_and_max_of_f_g_range_of_a_l2133_213360

noncomputable def f (x : ℝ) : ℝ := 2 * Real.log x - x^2

noncomputable def g (x a : ℝ) : ℝ := x * Real.exp x - (a - 1) * x^2 - x - 2 * Real.log x

theorem monotonicity_and_max_of_f : 
  (∀ x, 0 < x → x < 1 → f x > f (x + 1)) ∧ 
  (∀ x, x > 1 → f x < f (x - 1)) ∧ 
  (f 1 = -1) := 
by
  sorry

theorem g_range_of_a (a : ℝ) : 
  (∀ x, x > 0 → f x + g x a ≥ 0) → (a ≤ 1) := 
by
  sorry

end NUMINAMATH_GPT_monotonicity_and_max_of_f_g_range_of_a_l2133_213360


namespace NUMINAMATH_GPT_digit_difference_one_l2133_213312

theorem digit_difference_one (p q : ℕ) (h_pq : p < 10 ∧ q < 10) (h_diff : (10 * p + q) - (10 * q + p) = 9) :
  p - q = 1 :=
by
  sorry

end NUMINAMATH_GPT_digit_difference_one_l2133_213312


namespace NUMINAMATH_GPT_ratio_of_ducks_l2133_213328

theorem ratio_of_ducks (lily_ducks lily_geese rayden_geese rayden_ducks : ℕ) 
  (h1 : lily_ducks = 20) 
  (h2 : lily_geese = 10) 
  (h3 : rayden_geese = 4 * lily_geese) 
  (h4 : rayden_ducks + rayden_geese = lily_ducks + lily_geese + 70) : 
  rayden_ducks / lily_ducks = 3 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_ducks_l2133_213328


namespace NUMINAMATH_GPT_sum_of_first_41_terms_is_94_l2133_213387

def equal_product_sequence (a : ℕ → ℕ) (k : ℕ) : Prop := 
∀ (n : ℕ), a (n+1) * a (n+2) * a (n+3) = k

theorem sum_of_first_41_terms_is_94
  (a : ℕ → ℕ)
  (h1 : equal_product_sequence a 8)
  (h2 : a 1 = 1)
  (h3 : a 2 = 2) :
  (Finset.range 41).sum a = 94 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_41_terms_is_94_l2133_213387


namespace NUMINAMATH_GPT_sqrt_c_is_202_l2133_213321

theorem sqrt_c_is_202 (a b c : ℝ) (h1 : a + b = -2020) (h2 : a * b = c) (h3 : a / b + b / a = 98) : 
  Real.sqrt c = 202 :=
by
  sorry

end NUMINAMATH_GPT_sqrt_c_is_202_l2133_213321


namespace NUMINAMATH_GPT_probability_space_diagonal_l2133_213388

theorem probability_space_diagonal : 
  let vertices := 8
  let space_diagonals := 4
  let total_pairs := Nat.choose vertices 2
  4 / total_pairs = 1 / 7 :=
by
  sorry

end NUMINAMATH_GPT_probability_space_diagonal_l2133_213388


namespace NUMINAMATH_GPT_find_value_l2133_213370

variables (a b c d : ℝ)

theorem find_value
  (h1 : a - b = 3)
  (h2 : c + d = 2) :
  (a + c) - (b - d) = 5 :=
by sorry

end NUMINAMATH_GPT_find_value_l2133_213370


namespace NUMINAMATH_GPT_find_m_l2133_213371

theorem find_m (C D m : ℤ) (h1 : C = D + m) (h2 : C - 1 = 6 * (D - 1)) (h3 : C = D^3) : m = 0 :=
by sorry

end NUMINAMATH_GPT_find_m_l2133_213371


namespace NUMINAMATH_GPT_artist_paints_total_exposed_surface_area_l2133_213367

def num_cubes : Nat := 18
def edge_length : Nat := 1

-- Define the configuration of cubes
def bottom_layer_grid : Nat := 9 -- Number of cubes in the 3x3 grid (bottom layer)
def top_layer_cross : Nat := 9 -- Number of cubes in the cross shape (top layer)

-- Exposed surfaces in bottom layer
def bottom_layer_exposed_surfaces : Nat :=
  let top_surfaces := 9 -- 9 top surfaces for 9 cubes
  let corner_cube_sides := 4 * 3 -- 4 corners, 3 exposed sides each
  let edge_cube_sides := 4 * 2 -- 4 edge (non-corner) cubes, 2 exposed sides each
  top_surfaces + corner_cube_sides + edge_cube_sides

-- Exposed surfaces in top layer
def top_layer_exposed_surfaces : Nat :=
  let top_surfaces := 5 -- 5 top surfaces for 5 cubes in the cross
  let side_surfaces_of_cross_arms := 4 * 3 -- 4 arms, 3 exposed sides each
  top_surfaces + side_surfaces_of_cross_arms

-- Total exposed surface area
def total_exposed_surface_area : Nat :=
  bottom_layer_exposed_surfaces + top_layer_exposed_surfaces

-- Problem statement
theorem artist_paints_total_exposed_surface_area :
  total_exposed_surface_area = 46 := by
    sorry

end NUMINAMATH_GPT_artist_paints_total_exposed_surface_area_l2133_213367


namespace NUMINAMATH_GPT_sum_of_reciprocals_roots_transformed_eq_neg11_div_4_l2133_213357

theorem sum_of_reciprocals_roots_transformed_eq_neg11_div_4 :
  (∃ a b c : ℝ, (a^3 - a - 2 = 0) ∧ (b^3 - b - 2 = 0) ∧ (c^3 - c - 2 = 0)) → 
  ( ∃ a b c : ℝ, a^3 - a - 2 = 0 ∧ b^3 - b - 2 = 0 ∧ c^3 - c - 2 = 0 ∧ 
  (1 / (a - 2) + 1 / (b - 2) + 1 / (c - 2) = - 11 / 4)) :=
by
  sorry

end NUMINAMATH_GPT_sum_of_reciprocals_roots_transformed_eq_neg11_div_4_l2133_213357


namespace NUMINAMATH_GPT_probability_of_non_defective_product_l2133_213340

-- Define the probability of producing a grade B product
def P_B : ℝ := 0.03

-- Define the probability of producing a grade C product
def P_C : ℝ := 0.01

-- Define the probability of producing a non-defective product (grade A)
def P_A : ℝ := 1 - P_B - P_C

-- The theorem to prove: The probability of producing a non-defective product is 0.96
theorem probability_of_non_defective_product : P_A = 0.96 := by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_probability_of_non_defective_product_l2133_213340


namespace NUMINAMATH_GPT_volume_of_cuboid_l2133_213343

theorem volume_of_cuboid (l w h : ℝ) (hl_pos : 0 < l) (hw_pos : 0 < w) (hh_pos : 0 < h) 
  (h1 : l * w = 120) (h2 : w * h = 72) (h3 : h * l = 60) : l * w * h = 4320 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_cuboid_l2133_213343


namespace NUMINAMATH_GPT_conic_sections_hyperbola_and_ellipse_l2133_213339

theorem conic_sections_hyperbola_and_ellipse
  (x y : ℝ) (h : y^4 - 9 * x^4 = 3 * y^2 - 3) :
  (∃ a b c : ℝ, a * y^2 - b * x^2 = c ∧ a = b ∧ c ≠ 0) ∨ (∃ a b c : ℝ, a * y^2 + b * x^2 = c ∧ a ≠ b ∧ c ≠ 0) :=
by
  sorry

end NUMINAMATH_GPT_conic_sections_hyperbola_and_ellipse_l2133_213339


namespace NUMINAMATH_GPT_quadrilateral_diagonal_length_l2133_213391

theorem quadrilateral_diagonal_length (D A₁ A₂ : ℝ) (hA₁ : A₁ = 9) (hA₂ : A₂ = 6) (Area : ℝ) (hArea : Area = 165) :
  (1/2) * D * (A₁ + A₂) = Area → D = 22 :=
by
  -- Use the given conditions and solve to obtain D = 22
  intros
  sorry

end NUMINAMATH_GPT_quadrilateral_diagonal_length_l2133_213391


namespace NUMINAMATH_GPT_no_mult_of_5_end_in_2_l2133_213326

theorem no_mult_of_5_end_in_2 (n : ℕ) : n < 500 → ∃ k, n = 5 * k → (n % 10 = 2) = false :=
by
  sorry

end NUMINAMATH_GPT_no_mult_of_5_end_in_2_l2133_213326


namespace NUMINAMATH_GPT_classroom_students_count_l2133_213392

-- Definitions of given conditions
def total_students : ℕ := 1260

def aud_students : ℕ := (7 * total_students) / 18

def non_aud_students : ℕ := total_students - aud_students

def classroom_students : ℕ := (6 * non_aud_students) / 11

-- Theorem statement
theorem classroom_students_count : classroom_students = 420 := by
  sorry

end NUMINAMATH_GPT_classroom_students_count_l2133_213392


namespace NUMINAMATH_GPT_quadratic_complete_square_l2133_213373

open Real

theorem quadratic_complete_square (d e : ℝ) :
  (∀ x, x^2 - 24 * x + 50 = (x + d)^2 + e) → d + e = -106 :=
by
  intros h
  have h_eq := h 12
  sorry

end NUMINAMATH_GPT_quadratic_complete_square_l2133_213373


namespace NUMINAMATH_GPT_pythagorean_triple_transformation_l2133_213393

theorem pythagorean_triple_transformation
  (a b c α β γ s p q r : ℝ)
  (h₁ : a^2 + b^2 = c^2)
  (h₂ : α^2 + β^2 - γ^2 = 2)
  (h₃ : s = a * α + b * β - c * γ)
  (h₄ : p = a - α * s)
  (h₅ : q = b - β * s)
  (h₆ : r = c - γ * s) :
  p^2 + q^2 = r^2 :=
by
  sorry

end NUMINAMATH_GPT_pythagorean_triple_transformation_l2133_213393


namespace NUMINAMATH_GPT_exists_real_root_in_interval_l2133_213305

noncomputable def f (x : ℝ) : ℝ := x^3 + x - 3

theorem exists_real_root_in_interval (f : ℝ → ℝ)
  (h_mono : ∀ x y, x < y → f x < f y)
  (h1 : f 1 < 0)
  (h2 : f 2 > 0) : 
  ∃ x, 1 < x ∧ x < 2 ∧ f x = 0 := 
sorry

end NUMINAMATH_GPT_exists_real_root_in_interval_l2133_213305


namespace NUMINAMATH_GPT_general_formula_sum_formula_l2133_213355

-- Define the geometric sequence
def geoseq (n : ℕ) : ℕ := 2^n

-- Define the sum of the first n terms of the geometric sequence
def sum_first_n_terms (n : ℕ) : ℕ := 2^(n+1) - 2

-- Given conditions
def a1 : ℕ := 2
def a4 : ℕ := 16

-- Theorem statements
theorem general_formula (n : ℕ) : 
  (geoseq 1 = a1) → (geoseq 4 = a4) → geoseq n = 2^n := sorry

theorem sum_formula (n : ℕ) : 
  (geoseq 1 = a1) → (geoseq 4 = a4) → sum_first_n_terms n = 2^(n+1) - 2 := sorry

end NUMINAMATH_GPT_general_formula_sum_formula_l2133_213355


namespace NUMINAMATH_GPT_log_relationships_l2133_213315

theorem log_relationships (c d y : ℝ) (hc : c > 0) (hd : d > 0) (hy : y > 0) :
  9 * (Real.log y / Real.log c)^2 + 5 * (Real.log y / Real.log d)^2 = 18 * (Real.log y)^2 / (Real.log c * Real.log d) →
  d = c^(1 / Real.sqrt 3) ∨ d = c^(Real.sqrt 3) ∨ d = c^(1 / Real.sqrt (6 / 10)) ∨ d = c^(Real.sqrt (6 / 10)) :=
sorry

end NUMINAMATH_GPT_log_relationships_l2133_213315


namespace NUMINAMATH_GPT_ratio_of_perimeters_of_squares_l2133_213356

theorem ratio_of_perimeters_of_squares (d1 d11 : ℝ) (s1 s11 : ℝ) (P1 P11 : ℝ) 
  (h1 : d11 = 11 * d1)
  (h2 : d1 = s1 * Real.sqrt 2)
  (h3 : d11 = s11 * Real.sqrt 2) :
  P11 / P1 = 11 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_perimeters_of_squares_l2133_213356


namespace NUMINAMATH_GPT_integer_solutions_of_quadratic_eq_l2133_213318

theorem integer_solutions_of_quadratic_eq (b : ℤ) :
  ∃ p q : ℤ, (p+9) * (q+9) = 81 ∧ p + q = -b ∧ p * q = 9*b :=
sorry

end NUMINAMATH_GPT_integer_solutions_of_quadratic_eq_l2133_213318


namespace NUMINAMATH_GPT_rachel_study_time_l2133_213384

-- Define the conditions
def pages_math := 2
def pages_reading := 3
def pages_biology := 10
def pages_history := 4
def pages_physics := 5
def pages_chemistry := 8

def total_pages := pages_math + pages_reading + pages_biology + pages_history + pages_physics + pages_chemistry

def percent_study_time_biology := 30
def percent_study_time_reading := 30

-- State the theorem
theorem rachel_study_time :
  percent_study_time_biology = 30 ∧ 
  percent_study_time_reading = 30 →
  (100 - (percent_study_time_biology + percent_study_time_reading)) = 40 :=
by
  sorry

end NUMINAMATH_GPT_rachel_study_time_l2133_213384


namespace NUMINAMATH_GPT_number_of_newborn_members_in_group_l2133_213362

noncomputable def N : ℝ :=
  let p_death := 1 / 10
  let p_survive := 1 - p_death
  let prob_survive_3_months := p_survive * p_survive * p_survive
  218.7 / prob_survive_3_months

theorem number_of_newborn_members_in_group : N = 300 := by
  sorry

end NUMINAMATH_GPT_number_of_newborn_members_in_group_l2133_213362


namespace NUMINAMATH_GPT_digging_project_depth_l2133_213369

theorem digging_project_depth : 
  ∀ (P : ℕ) (D : ℝ), 
  (12 * P) * (25 * 30 * D) / 12 = (12 * P) * (75 * 20 * 50) / 12 → 
  D = 100 :=
by
  intros P D h
  sorry

end NUMINAMATH_GPT_digging_project_depth_l2133_213369


namespace NUMINAMATH_GPT_sum_of_integers_l2133_213341

theorem sum_of_integers (x y : ℕ) (h1 : x - y = 8) (h2 : x * y = 288) : x + y = 35 :=
sorry

end NUMINAMATH_GPT_sum_of_integers_l2133_213341


namespace NUMINAMATH_GPT_weight_of_7th_person_l2133_213380

-- Defining the constants and conditions
def num_people_initial : ℕ := 6
def avg_weight_initial : ℝ := 152
def num_people_total : ℕ := 7
def avg_weight_total : ℝ := 151

-- Calculating the total weights from the given average weights
def total_weight_initial := num_people_initial * avg_weight_initial
def total_weight_total := num_people_total * avg_weight_total

-- Theorem stating the weight of the 7th person
theorem weight_of_7th_person : total_weight_total - total_weight_initial = 145 := 
sorry

end NUMINAMATH_GPT_weight_of_7th_person_l2133_213380


namespace NUMINAMATH_GPT_total_sections_after_admissions_l2133_213323

theorem total_sections_after_admissions (S : ℕ) (h1 : (S * 24 + 24 = (S + 3) * 21)) :
  (S + 3) = 16 :=
  sorry

end NUMINAMATH_GPT_total_sections_after_admissions_l2133_213323


namespace NUMINAMATH_GPT_roller_skate_wheels_l2133_213375

theorem roller_skate_wheels (number_of_people : ℕ)
  (feet_per_person : ℕ)
  (skates_per_foot : ℕ)
  (wheels_per_skate : ℕ)
  (h_people : number_of_people = 40)
  (h_feet : feet_per_person = 2)
  (h_skates : skates_per_foot = 1)
  (h_wheels : wheels_per_skate = 4)
  : (number_of_people * feet_per_person * skates_per_foot * wheels_per_skate) = 320 := 
by
  sorry

end NUMINAMATH_GPT_roller_skate_wheels_l2133_213375


namespace NUMINAMATH_GPT_diminished_value_160_l2133_213306

theorem diminished_value_160 (x : ℕ) (n : ℕ) : 
  (∀ m, m > 200 ∧ (∀ k, m = k * 180) → n = m) →
  (200 + x = n) →
  x = 160 :=
by
  sorry

end NUMINAMATH_GPT_diminished_value_160_l2133_213306


namespace NUMINAMATH_GPT_find_amount_l2133_213366

theorem find_amount (x : ℝ) (A : ℝ) (h1 : 0.65 * x = 0.20 * A) (h2 : x = 230) : A = 747.5 := by
  sorry

end NUMINAMATH_GPT_find_amount_l2133_213366


namespace NUMINAMATH_GPT_problem1_solution_problem2_solution_l2133_213345

-- Problem 1: Prove the solution set for the given inequality
theorem problem1_solution (x : ℝ) : (2 < x ∧ x ≤ (7 / 2)) ↔ ((x + 1) / (x - 2) ≥ 3) := 
sorry

-- Problem 2: Prove the solution set for the given inequality
theorem problem2_solution (x a : ℝ) : 
  (a = 0 ∧ x = 0) ∨ 
  (a > 0 ∧ -a ≤ x ∧ x ≤ 2 * a) ∨ 
  (a < 0 ∧ 2 * a ≤ x ∧ x ≤ -a) ↔ 
  x^2 - a * x - 2 * a^2 ≤ 0 := 
sorry

end NUMINAMATH_GPT_problem1_solution_problem2_solution_l2133_213345


namespace NUMINAMATH_GPT_longer_diagonal_length_l2133_213398

-- Conditions
def rhombus_side_length := 65
def shorter_diagonal_length := 72

-- Prove that the length of the longer diagonal is 108
theorem longer_diagonal_length : 
  (2 * (Real.sqrt ((rhombus_side_length: ℝ)^2 - (shorter_diagonal_length / 2)^2))) = 108 := 
by 
  sorry

end NUMINAMATH_GPT_longer_diagonal_length_l2133_213398


namespace NUMINAMATH_GPT_pet_preferences_l2133_213309

/-- A store has several types of pets: 20 puppies, 10 kittens, 8 hamsters, and 5 birds.
Alice, Bob, Charlie, and David each want a different kind of pet, with the following preferences:
- Alice does not want a bird.
- Bob does not want a hamster.
- Charlie does not want a kitten.
- David does not want a puppy.
Prove that the number of ways they can choose different types of pets satisfying
their preferences is 791440. -/
theorem pet_preferences :
  let P := 20    -- Number of puppies
  let K := 10    -- Number of kittens
  let H := 8     -- Number of hamsters
  let B := 5     -- Number of birds
  let Alice_options := P + K + H -- Alice does not want a bird
  let Bob_options := P + K + B   -- Bob does not want a hamster
  let Charlie_options := P + H + B -- Charlie does not want a kitten
  let David_options := K + H + B   -- David does not want a puppy
  let Alice_pick := Alice_options
  let Bob_pick := Bob_options - 1
  let Charlie_pick := Charlie_options - 2
  let David_pick := David_options - 3
  Alice_pick * Bob_pick * Charlie_pick * David_pick = 791440 :=
by
  sorry

end NUMINAMATH_GPT_pet_preferences_l2133_213309


namespace NUMINAMATH_GPT_best_fit_model_l2133_213336

-- Definition of the given R^2 values for different models
def R2_A : ℝ := 0.62
def R2_B : ℝ := 0.63
def R2_C : ℝ := 0.68
def R2_D : ℝ := 0.65

-- Theorem statement that model with R2_C has the best fitting effect
theorem best_fit_model : R2_C = max R2_A (max R2_B (max R2_C R2_D)) :=
by
  sorry -- Proof is not required

end NUMINAMATH_GPT_best_fit_model_l2133_213336


namespace NUMINAMATH_GPT_man_older_than_son_l2133_213316

theorem man_older_than_son (S M : ℕ) (h1 : S = 23) (h2 : M + 2 = 2 * (S + 2)) : M - S = 25 :=
by
  sorry

end NUMINAMATH_GPT_man_older_than_son_l2133_213316


namespace NUMINAMATH_GPT_range_of_t_l2133_213346

theorem range_of_t 
  (k t : ℝ)
  (tangent_condition : (t + 1)^2 = 1 + k^2)
  (intersect_condition : ∃ x y, y = k * x + t ∧ y = x^2 / 4) : 
  t > 0 ∨ t < -3 :=
sorry

end NUMINAMATH_GPT_range_of_t_l2133_213346


namespace NUMINAMATH_GPT_max_possible_percent_error_in_garden_area_l2133_213390

open Real

theorem max_possible_percent_error_in_garden_area :
  ∃ (error_max : ℝ), error_max = 21 :=
by
  -- Given conditions
  let accurate_diameter := 30
  let max_error_percent := 10

  -- Defining lower and upper bounds for the diameter
  let lower_diameter := accurate_diameter - accurate_diameter * (max_error_percent / 100)
  let upper_diameter := accurate_diameter + accurate_diameter * (max_error_percent / 100)

  -- Calculating the exact and potential extreme areas
  let exact_area := π * (accurate_diameter / 2) ^ 2
  let lower_area := π * (lower_diameter / 2) ^ 2
  let upper_area := π * (upper_diameter / 2) ^ 2

  -- Calculating the percent errors
  let lower_error_percent := ((exact_area - lower_area) / exact_area) * 100
  let upper_error_percent := ((upper_area - exact_area) / exact_area) * 100

  -- We need to show the maximum error is 21%
  use upper_error_percent -- which should be 21% according to the problem statement
  sorry -- proof goes here

end NUMINAMATH_GPT_max_possible_percent_error_in_garden_area_l2133_213390


namespace NUMINAMATH_GPT_find_f_neg3_l2133_213330

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

noncomputable def f (x : ℝ) : ℝ :=
  if h : x > 0 then x * (1 - x) else -x * (1 + x)

theorem find_f_neg3 :
  is_odd_function f →
  (∀ x, x > 0 → f x = x * (1 - x)) →
  f (-3) = 6 :=
by
  intros h_odd h_condition
  sorry

end NUMINAMATH_GPT_find_f_neg3_l2133_213330


namespace NUMINAMATH_GPT_lengths_available_total_cost_l2133_213301

def available_lengths := [1, 2, 3, 4, 5, 6]
def pipe_prices := [10, 15, 20, 25, 30, 35]

-- Given conditions
def purchased_pipes := [2, 5]
def target_perimeter_is_even := True

-- Prove: 
theorem lengths_available (x : ℕ) (hx : x ∈ available_lengths) : 
  3 < x ∧ x < 7 → x = 4 ∨ x = 5 ∨ x = 6 := by
  sorry

-- Prove: 
theorem total_cost (p : ℕ) (h : target_perimeter_is_even) : 
  p = 75 := by
  sorry

end NUMINAMATH_GPT_lengths_available_total_cost_l2133_213301


namespace NUMINAMATH_GPT_parabola_vertex_expression_l2133_213302

theorem parabola_vertex_expression (h k : ℝ) :
  (h = 2 ∧ k = 3) →
  ∃ (a : ℝ), (a ≠ 0) ∧
    (∀ x y : ℝ, y = a * (x - h)^2 + k ↔ y = -(x - 2)^2 + 3) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_expression_l2133_213302


namespace NUMINAMATH_GPT_snow_leopards_arrangement_l2133_213307

theorem snow_leopards_arrangement : 
  ∃ (perm : Fin 9 → Fin 9), 
    (∀ i, perm i ≠ perm j → i ≠ j) ∧ 
    (perm 0 < perm 1 ∧ perm 8 < perm 1 ∧ perm 0 < perm 8) ∧ 
    (∃ count_ways, count_ways = 4320) :=
sorry

end NUMINAMATH_GPT_snow_leopards_arrangement_l2133_213307


namespace NUMINAMATH_GPT_cosine_square_plus_alpha_sine_l2133_213376

variable (α : ℝ)

theorem cosine_square_plus_alpha_sine (h1 : 0 ≤ α) (h2 : α ≤ Real.pi / 2) : 
  Real.cos α * Real.cos α + α * Real.sin α ≥ 1 :=
sorry

end NUMINAMATH_GPT_cosine_square_plus_alpha_sine_l2133_213376


namespace NUMINAMATH_GPT_shopkeeper_packets_l2133_213349

noncomputable def milk_packets (oz_to_ml: ℝ) (ml_per_packet: ℝ) (total_milk_oz: ℝ) : ℝ :=
  (total_milk_oz * oz_to_ml) / ml_per_packet

theorem shopkeeper_packets (oz_to_ml: ℝ) (ml_per_packet: ℝ) (total_milk_oz: ℝ) :
  oz_to_ml = 30 → ml_per_packet = 250 → total_milk_oz = 1250 → milk_packets oz_to_ml ml_per_packet total_milk_oz = 150 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_shopkeeper_packets_l2133_213349


namespace NUMINAMATH_GPT_mean_score_l2133_213350

theorem mean_score (M SD : ℝ) (h₁ : 58 = M - 2 * SD) (h₂ : 98 = M + 3 * SD) : M = 74 :=
by
  sorry

end NUMINAMATH_GPT_mean_score_l2133_213350


namespace NUMINAMATH_GPT_work_completion_days_l2133_213383

theorem work_completion_days (A B : ℕ) (h1 : A = 2 * B) (h2 : 6 * (A + B) = 18) : B = 1 → 18 = 18 :=
by
  sorry

end NUMINAMATH_GPT_work_completion_days_l2133_213383


namespace NUMINAMATH_GPT_unoccupied_volume_proof_l2133_213331

-- Definitions based on conditions
def tank_length : ℕ := 12
def tank_width : ℕ := 10
def tank_height : ℕ := 8
def tank_volume : ℕ := tank_length * tank_width * tank_height

def oil_fill_ratio : ℚ := 2 / 3
def ice_cube_volume : ℕ := 1
def number_of_ice_cubes : ℕ := 15

-- Volume calculations
def oil_volume : ℚ := oil_fill_ratio * tank_volume
def total_ice_volume : ℚ := number_of_ice_cubes * ice_cube_volume
def occupied_volume : ℚ := oil_volume + total_ice_volume

-- The final question to be proved
theorem unoccupied_volume_proof : tank_volume - occupied_volume = 305 := by
  sorry

end NUMINAMATH_GPT_unoccupied_volume_proof_l2133_213331


namespace NUMINAMATH_GPT_min_distance_AB_tangent_line_circle_l2133_213389

theorem min_distance_AB_tangent_line_circle 
  (a b : ℝ) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (h_tangent : a^2 + b^2 = 1) :
  ∃ A B : ℝ × ℝ, (A = (0, 1/b) ∧ B = (2/a, 0)) ∧ dist A B = 3 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_AB_tangent_line_circle_l2133_213389


namespace NUMINAMATH_GPT_value_of_expression_l2133_213311

variable (p q r s : ℝ)

-- Given condition in a)
def polynomial_function (x : ℝ) := p * x^3 + q * x^2 + r * x + s
def passes_through_point := polynomial_function p q r s (-1) = 4

-- Proof statement in c)
theorem value_of_expression (h : passes_through_point p q r s) : 6 * p - 3 * q + r - 2 * s = -24 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l2133_213311


namespace NUMINAMATH_GPT_maximum_value_a1_l2133_213310

noncomputable def max_possible_value (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h3 : a 1 = a 10) : ℝ :=
  16

theorem maximum_value_a1 (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∀ n, (2 * a (n + 1) - a n) * (a (n + 1) * a n - 1) = 0)
  (h3 : a 1 = a 10) : a 1 ≤ max_possible_value a h1 h2 h3 :=
  sorry

end NUMINAMATH_GPT_maximum_value_a1_l2133_213310


namespace NUMINAMATH_GPT_simplify_expression_l2133_213379

theorem simplify_expression (x : ℝ) : (x + 2)^2 + x * (x - 4) = 2 * x^2 + 4 := by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2133_213379


namespace NUMINAMATH_GPT_inequality_solution_real_l2133_213397

theorem inequality_solution_real (x : ℝ) :
  (x + 1) * (2 - x) < 4 ↔ true :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_real_l2133_213397


namespace NUMINAMATH_GPT_no_integer_solutions_l2133_213324

theorem no_integer_solutions (x y z : ℤ) (h : 2 * x^4 + 2 * x^2 * y^2 + y^4 = z^2) (hx : x ≠ 0) : false :=
sorry

end NUMINAMATH_GPT_no_integer_solutions_l2133_213324


namespace NUMINAMATH_GPT_students_left_l2133_213361

theorem students_left (initial_students new_students final_students students_left : ℕ)
  (h1 : initial_students = 10)
  (h2 : new_students = 42)
  (h3 : final_students = 48)
  : initial_students + new_students - students_left = final_students → students_left = 4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_students_left_l2133_213361


namespace NUMINAMATH_GPT_A_subscribed_fraction_l2133_213319

theorem A_subscribed_fraction 
  (total_profit : ℝ) (A_share : ℝ) 
  (B_fraction : ℝ) (C_fraction : ℝ) 
  (A_fraction : ℝ) :
  total_profit = 2430 →
  A_share = 810 →
  B_fraction = 1/4 →
  C_fraction = 1/5 →
  A_fraction = A_share / total_profit →
  A_fraction = 1/3 :=
by
  intros h_total_profit h_A_share h_B_fraction h_C_fraction h_A_fraction
  sorry

end NUMINAMATH_GPT_A_subscribed_fraction_l2133_213319


namespace NUMINAMATH_GPT_warehouse_length_l2133_213351

theorem warehouse_length (L W : ℕ) (times supposed_times : ℕ) (total_distance : ℕ)
  (h1 : W = 400)
  (h2 : supposed_times = 10)
  (h3 : times = supposed_times - 2)
  (h4 : total_distance = times * (2 * L + 2 * W))
  (h5 : total_distance = 16000) :
  L = 600 := by
  sorry

end NUMINAMATH_GPT_warehouse_length_l2133_213351


namespace NUMINAMATH_GPT_gas_cost_l2133_213353

theorem gas_cost (x : ℝ) (h₁ : 5 * (x / 5 - 9) = 8 * (x / 8)) : x = 120 :=
by
  sorry

end NUMINAMATH_GPT_gas_cost_l2133_213353


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2133_213304

theorem solution_set_of_inequality (x : ℝ) (h : 3 * x + 2 > 5) : x > 1 :=
sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2133_213304


namespace NUMINAMATH_GPT_two_pow_65537_mod_19_l2133_213399

theorem two_pow_65537_mod_19 : (2 ^ 65537) % 19 = 2 := by
  -- We will use Fermat's Little Theorem and given conditions.
  sorry

end NUMINAMATH_GPT_two_pow_65537_mod_19_l2133_213399


namespace NUMINAMATH_GPT_matrix_determinant_zero_l2133_213364

theorem matrix_determinant_zero (a b c : ℝ) : 
  Matrix.det (Matrix.of ![![1, a + b, b + c], ![1, a + 2 * b, b + 2 * c], ![1, a + 3 * b, b + 3 * c]]) = 0 := 
by
  sorry

end NUMINAMATH_GPT_matrix_determinant_zero_l2133_213364


namespace NUMINAMATH_GPT_trajectory_of_P_l2133_213338

-- Define points P, A, and B in a 2D plane
variable {P A B : EuclideanSpace ℝ (Fin 2)}

-- Define the condition that the sum of the distances from P to A and P to B equals the distance between A and B
def sum_of_distances_condition (P A B : EuclideanSpace ℝ (Fin 2)) : Prop :=
  dist P A + dist P B = dist A B

-- Main theorem statement: If P satisfies the above condition, then P lies on the line segment AB
theorem trajectory_of_P (P A B : EuclideanSpace ℝ (Fin 2)) (h : sum_of_distances_condition P A B) :
    ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = t • A + (1 - t) • B :=
  sorry

end NUMINAMATH_GPT_trajectory_of_P_l2133_213338


namespace NUMINAMATH_GPT_pow_divisible_by_13_l2133_213344

theorem pow_divisible_by_13 (n : ℕ) (h : 0 < n) : (4^(2*n+1) + 3^(n+2)) % 13 = 0 :=
sorry

end NUMINAMATH_GPT_pow_divisible_by_13_l2133_213344


namespace NUMINAMATH_GPT_a_minus_two_sufficient_but_not_necessary_for_pure_imaginary_l2133_213342

def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0

def complex_from_a (a : ℝ) : ℂ :=
  (a^2 - 4 : ℝ) + (a + 1 : ℝ) * Complex.I

theorem a_minus_two_sufficient_but_not_necessary_for_pure_imaginary :
  (is_pure_imaginary (complex_from_a (-2))) ∧ ¬ (∀ (a : ℝ), is_pure_imaginary (complex_from_a a) → a = -2) :=
by
  sorry

end NUMINAMATH_GPT_a_minus_two_sufficient_but_not_necessary_for_pure_imaginary_l2133_213342
