import Mathlib

namespace trapezium_area_proof_l1572_157287

def trapeziumArea (a b h : ℕ) : ℕ :=
  (1 / 2) * (a + b) * h

theorem trapezium_area_proof :
  let a := 20
  let b := 18
  let h := 14
  trapeziumArea a b h = 266 := by
  sorry

end trapezium_area_proof_l1572_157287


namespace total_days_2001_2005_l1572_157209

theorem total_days_2001_2005 : 
  let is_leap_year (y : ℕ) := y % 4 = 0 ∧ (y % 100 ≠ 0 ∨ y % 400 = 0)
  let days_in_year (y : ℕ) := if is_leap_year y then 366 else 365 
  (days_in_year 2001) + (days_in_year 2002) + (days_in_year 2003) + (days_in_year 2004) + (days_in_year 2005) = 1461 :=
by
  sorry

end total_days_2001_2005_l1572_157209


namespace number_greater_than_neg_one_by_two_l1572_157205

/-- Theorem: The number that is greater than -1 by 2 is 1. -/
theorem number_greater_than_neg_one_by_two : -1 + 2 = 1 :=
by
  sorry

end number_greater_than_neg_one_by_two_l1572_157205


namespace gdp_scientific_notation_l1572_157206

theorem gdp_scientific_notation : 
  (33.5 * 10^12 = 3.35 * 10^13) := 
by
  sorry

end gdp_scientific_notation_l1572_157206


namespace find_even_odd_functions_l1572_157241

variable {X : Type} [AddGroup X]

def even_function (f : X → X) : Prop :=
∀ x, f (-x) = f x

def odd_function (f : X → X) : Prop :=
∀ x, f (-x) = -f x

theorem find_even_odd_functions
  (f g : X → X)
  (h_even : even_function f)
  (h_odd : odd_function g)
  (h_eq : ∀ x, f x + g x = 0) :
  (∀ x, f x = 0) ∧ (∀ x, g x = 0) :=
sorry

end find_even_odd_functions_l1572_157241


namespace domain_f_2x_minus_1_l1572_157297

theorem domain_f_2x_minus_1 (f : ℝ → ℝ) : 
  (∀ x, 1 ≤ x ∧ x ≤ 2 → 2 ≤ x + 1 ∧ x + 1 ≤ 3) → 
  (∀ z, 2 ≤ 2 * z - 1 ∧ 2 * z - 1 ≤ 3 → ∃ x, 3/2 ≤ x ∧ x ≤ 2 ∧ 2 * x - 1 = z) := 
sorry

end domain_f_2x_minus_1_l1572_157297


namespace tank_holds_21_liters_l1572_157216

def tank_capacity (S L : ℝ) : Prop :=
  (L = 2 * S + 3) ∧
  (L = 4) ∧
  (2 * S + 5 * L = 21)

theorem tank_holds_21_liters :
  ∃ S L : ℝ, tank_capacity S L :=
by
  use 1/2, 4
  unfold tank_capacity
  simp
  sorry

end tank_holds_21_liters_l1572_157216


namespace ricky_time_difference_l1572_157283

noncomputable def old_man_time_per_mile : ℚ := 300 / 8
noncomputable def young_man_time_per_mile : ℚ := 160 / 12
noncomputable def time_difference : ℚ := old_man_time_per_mile - young_man_time_per_mile

theorem ricky_time_difference :
  time_difference = 24 := by
sorry

end ricky_time_difference_l1572_157283


namespace value_of_x_abs_not_positive_l1572_157260

theorem value_of_x_abs_not_positive {x : ℝ} : |4 * x - 6| = 0 → x = 3 / 2 :=
by
  sorry

end value_of_x_abs_not_positive_l1572_157260


namespace minimize_total_cost_l1572_157259

noncomputable def event_probability_without_measures : ℚ := 0.3
noncomputable def loss_if_event_occurs : ℚ := 4000000
noncomputable def cost_measure_A : ℚ := 450000
noncomputable def prob_event_not_occurs_measure_A : ℚ := 0.9
noncomputable def cost_measure_B : ℚ := 300000
noncomputable def prob_event_not_occurs_measure_B : ℚ := 0.85

noncomputable def total_cost_no_measures : ℚ :=
  event_probability_without_measures * loss_if_event_occurs

noncomputable def total_cost_measure_A : ℚ :=
  cost_measure_A + (1 - prob_event_not_occurs_measure_A) * loss_if_event_occurs

noncomputable def total_cost_measure_B : ℚ :=
  cost_measure_B + (1 - prob_event_not_occurs_measure_B) * loss_if_event_occurs

noncomputable def total_cost_measures_A_and_B : ℚ :=
  cost_measure_A + cost_measure_B + (1 - prob_event_not_occurs_measure_A) * (1 - prob_event_not_occurs_measure_B) * loss_if_event_occurs

theorem minimize_total_cost :
  min (min total_cost_no_measures total_cost_measure_A) (min total_cost_measure_B total_cost_measures_A_and_B) = total_cost_measures_A_and_B :=
by sorry

end minimize_total_cost_l1572_157259


namespace cost_of_potatoes_l1572_157276

theorem cost_of_potatoes
  (per_person_potatoes : ℕ → ℕ → ℕ)
  (amount_of_people : ℕ)
  (bag_cost : ℕ)
  (bag_weight : ℕ)
  (people : ℕ)
  (cost : ℕ) :
  (per_person_potatoes people amount_of_people = 60) →
  (60 / bag_weight = 3) →
  (3 * bag_cost = cost) →
  cost = 15 :=
by
  sorry

end cost_of_potatoes_l1572_157276


namespace probability_of_drawing_red_ball_l1572_157289

theorem probability_of_drawing_red_ball (total_balls red_balls : ℕ) (h_total : total_balls = 10) (h_red : red_balls = 7) : (red_balls : ℚ) / total_balls = 7 / 10 :=
by
  sorry

end probability_of_drawing_red_ball_l1572_157289


namespace unit_place_3_pow_34_l1572_157243

theorem unit_place_3_pow_34 : Nat.mod (3^34) 10 = 9 :=
by
  sorry

end unit_place_3_pow_34_l1572_157243


namespace non_working_games_l1572_157295

def total_games : ℕ := 30
def working_games : ℕ := 17

theorem non_working_games :
  total_games - working_games = 13 := 
by 
  sorry

end non_working_games_l1572_157295


namespace midpoint_3d_l1572_157249

/-- Midpoint calculation in 3D space -/
theorem midpoint_3d (x1 y1 z1 x2 y2 z2 : ℝ) : 
  (x1, y1, z1) = (2, -3, 6) → 
  (x2, y2, z2) = (8, 5, -4) → 
  ((x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2) = (5, 1, 1) := 
by
  intros
  sorry

end midpoint_3d_l1572_157249


namespace second_group_people_l1572_157230

theorem second_group_people (x : ℕ) (K : ℕ) (hK : K > 0) :
  (96 - 16 = K * (x + 16) + 6) → (x = 58 ∨ x = 21) :=
by
  intro h
  sorry

end second_group_people_l1572_157230


namespace pet_store_cages_l1572_157211

theorem pet_store_cages (initial_puppies sold_puppies puppies_per_cage : ℕ) (h₁ : initial_puppies = 78)
(h₂ : sold_puppies = 30) (h₃ : puppies_per_cage = 8) : (initial_puppies - sold_puppies) / puppies_per_cage = 6 :=
by
  -- assumptions: initial_puppies = 78, sold_puppies = 30, puppies_per_cage = 8
  -- goal: (initial_puppies - sold_puppies) / puppies_per_cage = 6
  sorry

end pet_store_cages_l1572_157211


namespace root_shifted_is_root_of_quadratic_with_integer_coeffs_l1572_157261

theorem root_shifted_is_root_of_quadratic_with_integer_coeffs
  (a b c t : ℤ)
  (h : a ≠ 0)
  (h_root : a * t^2 + b * t + c = 0) :
  ∃ (x : ℤ), a * x^2 + (4 * a + b) * x + (4 * a + 2 * b + c) = 0 :=
by {
  sorry
}

end root_shifted_is_root_of_quadratic_with_integer_coeffs_l1572_157261


namespace sunflower_is_taller_l1572_157219

def sister_height_ft : Nat := 4
def sister_height_in : Nat := 3
def sunflower_height_ft : Nat := 6

def feet_to_inches (ft : Nat) : Nat := ft * 12

def sister_height := feet_to_inches sister_height_ft + sister_height_in
def sunflower_height := feet_to_inches sunflower_height_ft

def height_difference : Nat := sunflower_height - sister_height

theorem sunflower_is_taller : height_difference = 21 :=
by
  -- proof has to be provided:
  sorry

end sunflower_is_taller_l1572_157219


namespace new_mean_rent_l1572_157291

theorem new_mean_rent
  (num_friends : ℕ)
  (avg_rent : ℕ)
  (original_rent_increased : ℕ)
  (increase_percentage : ℝ)
  (new_mean_rent : ℕ) :
  num_friends = 4 →
  avg_rent = 800 →
  original_rent_increased = 1400 →
  increase_percentage = 0.2 →
  new_mean_rent = 870 :=
by
  intros h1 h2 h3 h4
  sorry

end new_mean_rent_l1572_157291


namespace herd_total_cows_l1572_157210

noncomputable def total_cows (n : ℕ) : Prop :=
  let fraction_first_son := 1 / 3
  let fraction_second_son := 1 / 5
  let fraction_third_son := 1 / 9
  let fraction_combined := fraction_first_son + fraction_second_son + fraction_third_son
  let fraction_fourth_son := 1 - fraction_combined
  let cows_fourth_son := 11
  fraction_fourth_son * n = cows_fourth_son

theorem herd_total_cows : ∃ n : ℕ, total_cows n ∧ n = 31 :=
by
  existsi 31
  sorry

end herd_total_cows_l1572_157210


namespace fewest_tiles_needed_to_cover_rectangle_l1572_157226

noncomputable def height_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (Real.sqrt 3 / 2) * side_length

noncomputable def area_of_equilateral_triangle (side_length : ℝ) : ℝ :=
  (1 / 2) * side_length * height_of_equilateral_triangle side_length

noncomputable def area_of_floor_in_square_inches (length_in_feet : ℝ) (width_in_feet : ℝ) : ℝ :=
  length_in_feet * width_in_feet * (12 * 12)

noncomputable def number_of_tiles_required (floor_area : ℝ) (tile_area : ℝ) : ℝ :=
  floor_area / tile_area

theorem fewest_tiles_needed_to_cover_rectangle :
  number_of_tiles_required (area_of_floor_in_square_inches 3 4) (area_of_equilateral_triangle 2) = 997 := 
by
  sorry

end fewest_tiles_needed_to_cover_rectangle_l1572_157226


namespace car_price_is_5_l1572_157225

variable (numCars : ℕ) (totalEarnings legoCost carCost : ℕ)

-- Conditions
axiom h1 : numCars = 3
axiom h2 : totalEarnings = 45
axiom h3 : legoCost = 30
axiom h4 : totalEarnings - legoCost = 15
axiom h5 : (totalEarnings - legoCost) / numCars = carCost

-- The proof problem statement
theorem car_price_is_5 : carCost = 5 :=
  by
    -- Here the proof steps would be filled in, but are not required for this task.
    sorry

end car_price_is_5_l1572_157225


namespace first_term_of_geometric_series_l1572_157270

theorem first_term_of_geometric_series (r : ℚ) (S : ℚ) (a : ℚ) (h1 : r = 1/5) (h2 : S = 100) (h3 : S = a / (1 - r)) : a = 80 := 
by
  sorry

end first_term_of_geometric_series_l1572_157270


namespace flooring_cost_correct_l1572_157237

noncomputable def cost_of_flooring (l w h_t b_t c : ℝ) : ℝ :=
  let area_rectangle := l * w
  let area_triangle := (b_t * h_t) / 2
  let area_to_be_floored := area_rectangle - area_triangle
  area_to_be_floored * c

theorem flooring_cost_correct :
  cost_of_flooring 10 7 3 4 900 = 57600 :=
by
  sorry

end flooring_cost_correct_l1572_157237


namespace completing_square_l1572_157256

-- Define the theorem statement
theorem completing_square (x : ℝ) : 
  x^2 - 2 * x = 2 -> (x - 1)^2 = 3 :=
by sorry

end completing_square_l1572_157256


namespace range_of_a_l1572_157246

theorem range_of_a (a : ℝ) 
  (h : ∀ x y, (a * x^2 - 3 * x + 2 = 0) ∧ (a * y^2 - 3 * y + 2 = 0) → x = y) :
  a = 0 ∨ a ≥ 9/8 :=
sorry

end range_of_a_l1572_157246


namespace find_S40_l1572_157271

noncomputable def geometric_sequence_sum (a r : ℝ) (n : ℕ) : ℝ := a * (r^n - 1) / (r - 1)

theorem find_S40 (a r : ℝ) (S : ℕ → ℝ)
  (h1 : ∀ n, S n = geometric_sequence_sum a r n)
  (h2 : S 10 = 10)
  (h3 : S 30 = 70) :
  S 40 = 150 ∨ S 40 = 110 := 
sorry

end find_S40_l1572_157271


namespace smallest_visible_sum_of_3x3x3_cube_is_90_l1572_157275

theorem smallest_visible_sum_of_3x3x3_cube_is_90 
: ∀ (dices: Fin 27 → Fin 6 → ℕ),
    (∀ i j k, dices (3*i+j) k = 7 - dices (3*i+j) (5-k)) → 
    (∃ s, s = 90 ∧
    s = (8 * (dices 0 0 + dices 0 1 + dices 0 2)) + 
        (12 * (dices 0 0 + dices 0 1)) +
        (6 * (dices 0 0))) := sorry

end smallest_visible_sum_of_3x3x3_cube_is_90_l1572_157275


namespace arithmetic_sequence_condition_l1572_157286

theorem arithmetic_sequence_condition {a : ℕ → ℤ} 
  (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (m p q : ℕ) (hpq_pos : 0 < p) (hq_pos : 0 < q) (hm_pos : 0 < m) : 
  (p + q = 2 * m) → (a p + a q = 2 * a m) ∧ ¬((a p + a q = 2 * a m) → (p + q = 2 * m)) :=
by 
  sorry

end arithmetic_sequence_condition_l1572_157286


namespace value_independent_of_b_value_for_d_zero_l1572_157229

theorem value_independent_of_b
  (c b d h : ℝ)
  (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ)
  (h1 : x1 = b - d - h)
  (h2 : x2 = b - d)
  (h3 : x3 = b + d)
  (h4 : x4 = b + d + h)
  (hy1 : y1 = c * x1^2)
  (hy2 : y2 = c * x2^2)
  (hy3 : y3 = c * x3^2)
  (hy4 : y4 = c * x4^2) :
  (y1 + y4 - y2 - y3) = 2 * c * h * (2 * d + h) :=
by
  sorry

theorem value_for_d_zero
  (c b h : ℝ)
  (x1 x2 x3 x4 y1 y2 y3 y4 : ℝ)
  (d : ℝ := 0)
  (h1 : x1 = b - h)
  (h2 : x2 = b)
  (h3 : x3 = b)
  (h4 : x4 = b + h)
  (hy1 : y1 = c * x1^2)
  (hy2 : y2 = c * x2^2)
  (hy3 : y3 = c * x3^2)
  (hy4 : y4 = c * x4^2) :
  (y1 + y4 - y2 - y3) = 2 * c * h^2 :=
by
  sorry

end value_independent_of_b_value_for_d_zero_l1572_157229


namespace inequality_proof_l1572_157236

variable (a b c d : ℝ)

theorem inequality_proof (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) (h5 : a + b + c + d = 1) : 
  (1 / (4 * a + 3 * b + c) + 1 / (3 * a + b + 4 * d) + 1 / (a + 4 * c + 3 * d) + 1 / (4 * b + 3 * c + d)) ≥ 2 :=
by
  sorry

end inequality_proof_l1572_157236


namespace factorization_divisibility_l1572_157232

theorem factorization_divisibility (n : ℤ) : 120 ∣ (n^5 - 5 * n^3 + 4 * n) :=
by
  sorry

end factorization_divisibility_l1572_157232


namespace find_value_of_expression_l1572_157263

-- Conditions translated to Lean 4 definitions
variable (a b : ℝ)
axiom h1 : (a^2 * b^3) / 5 = 1000
axiom h2 : a * b = 2

-- The theorem stating what we need to prove
theorem find_value_of_expression :
  (a^3 * b^2) / 3 = 2 / 705 :=
by
  sorry

end find_value_of_expression_l1572_157263


namespace polynomial_product_roots_l1572_157264

theorem polynomial_product_roots (a b c : ℝ) : 
  (∀ x, (x - (Real.sin (Real.pi / 6))) * (x - (Real.sin (Real.pi / 3))) * (x - (Real.sin (5 * Real.pi / 6))) = x^3 + a * x^2 + b * x + c) → 
  a * b * c = Real.sqrt 3 / 2 :=
by
  sorry

end polynomial_product_roots_l1572_157264


namespace regular_polygon_num_sides_l1572_157258

theorem regular_polygon_num_sides (angle : ℝ) (h : angle = 45) : 
  (∃ n : ℕ, n = 360 / angle ∧ n ≠ 0) → n = 8 :=
by
  sorry

end regular_polygon_num_sides_l1572_157258


namespace complete_the_square_l1572_157228

theorem complete_the_square (d e f : ℤ) (h1 : 0 < d)
    (h2 : ∀ x : ℝ, 100 * x^2 + 60 * x - 90 = 0 ↔ (d * x + e)^2 = f) :
  d + e + f = 112 := by
  sorry

end complete_the_square_l1572_157228


namespace determine_k_for_one_real_solution_l1572_157292

theorem determine_k_for_one_real_solution (k : ℝ):
  (∃ x : ℝ, 9 * x^2 + k * x + 49 = 0 ∧ (∀ y : ℝ, 9 * y^2 + k * y + 49 = 0 → y = x)) → k = 42 :=
sorry

end determine_k_for_one_real_solution_l1572_157292


namespace inequality_solution_sets_l1572_157269

noncomputable def solve_inequality (m : ℝ) : Set ℝ :=
  if m = 0 then Set.Iic (-2)
  else if m > 0 then Set.Iic (-2) ∪ Set.Ici (1 / m)
  else if (-(1/2) < m ∧ m < 0) then Set.Ioo (1 / m) (-2)
  else if m = -(1 / 2) then ∅
  else Set.Ioo (-2) (1 / m)

theorem inequality_solution_sets (m : ℝ) :
  solve_inequality m = 
    if m = 0 then Set.Iic (-2)
    else if m > 0 then Set.Iic (-2) ∪ Set.Ici (1 / m)
    else if (-(1/2) < m ∧ m < 0) then Set.Ioo (1 / m) (-2)
    else if m = -(1 / 2) then ∅
    else Set.Ioo (-2) (1 / m) :=
sorry

end inequality_solution_sets_l1572_157269


namespace total_words_read_l1572_157239

/-- Proof Problem Statement:
  Given the following conditions:
  - Henri has 8 hours to watch movies and read.
  - He watches one movie for 3.5 hours.
  - He watches another movie for 1.5 hours.
  - He watches two more movies with durations of 1.25 hours and 0.75 hours, respectively.
  - He reads for the remaining time after watching movies.
  - For the first 30 minutes of reading, he reads at a speed of 12 words per minute.
  - For the following 20 minutes, his reading speed decreases to 8 words per minute.
  - In the last remaining minutes, his reading speed increases to 15 words per minute.
  Prove that the total number of words Henri reads during his free time is 670.
--/
theorem total_words_read : 8 * 60 - (7 * 60) = 60 ∧
  (30 * 12) + (20 * 8) + ((60 - 30 - 20) * 15) = 670 :=
by
  sorry

end total_words_read_l1572_157239


namespace ratio_of_pq_l1572_157247

def is_pure_imaginary (z : Complex) : Prop :=
  z.re = 0

theorem ratio_of_pq (p q : ℝ) (hp : p ≠ 0) (hq : q ≠ 0)
  (H : is_pure_imaginary ((Complex.ofReal 3 - Complex.ofReal 4 * Complex.I) * (Complex.ofReal p + Complex.ofReal q * Complex.I))) :
  p / q = -4 / 3 :=
by
  sorry

end ratio_of_pq_l1572_157247


namespace triangle_first_side_l1572_157222

theorem triangle_first_side (x : ℕ) (h1 : 10 + 15 + x = 32) : x = 7 :=
by
  sorry

end triangle_first_side_l1572_157222


namespace geometric_series_first_term_l1572_157267

theorem geometric_series_first_term (r a S : ℝ) (hr : r = 1 / 8) (hS : S = 60) (hS_formula : S = a / (1 - r)) : 
  a = 105 / 2 := by
  rw [hr, hS] at hS_formula
  sorry

end geometric_series_first_term_l1572_157267


namespace bugs_eat_same_flowers_l1572_157240

theorem bugs_eat_same_flowers (num_bugs : ℕ) (total_flowers : ℕ) (flowers_per_bug : ℕ) 
  (h1 : num_bugs = 3) (h2 : total_flowers = 6) (h3 : flowers_per_bug = total_flowers / num_bugs) : 
  flowers_per_bug = 2 :=
by
  sorry

end bugs_eat_same_flowers_l1572_157240


namespace total_shoes_in_box_l1572_157252

theorem total_shoes_in_box (pairs : ℕ) (prob_matching : ℚ) (h1 : pairs = 7) (h2 : prob_matching = 1 / 13) : 
  ∃ (n : ℕ), n = 2 * pairs ∧ n = 14 :=
by 
  sorry

end total_shoes_in_box_l1572_157252


namespace angle_in_fourth_quadrant_l1572_157242

variable (α : ℝ)

def is_in_first_quadrant (α : ℝ) : Prop := 0 < α ∧ α < 90

def is_in_fourth_quadrant (θ : ℝ) : Prop := 270 < θ ∧ θ < 360

theorem angle_in_fourth_quadrant (h : is_in_first_quadrant α) : is_in_fourth_quadrant (360 - α) := sorry

end angle_in_fourth_quadrant_l1572_157242


namespace degree_of_angle_C_l1572_157202

theorem degree_of_angle_C 
  (A B C : ℝ) 
  (h1 : A = 4 * x) 
  (h2 : B = 4 * x) 
  (h3 : C = 7 * x) 
  (h_sum : A + B + C = 180) : 
  C = 84 := 
by 
  sorry

end degree_of_angle_C_l1572_157202


namespace geometric_seq_fraction_l1572_157280

theorem geometric_seq_fraction (a : ℕ → ℝ) (q : ℝ) (h1 : ∀ n, a (n + 1) = q * a n) 
  (h2 : (a 1 + 3 * a 3) / (a 2 + 3 * a 4) = 1 / 2) : 
  (a 4 * a 6 + a 6 * a 8) / (a 6 * a 8 + a 8 * a 10) = 1 / 16 :=
by
  sorry

end geometric_seq_fraction_l1572_157280


namespace martha_apples_l1572_157298

theorem martha_apples (initial_apples jane_apples extra_apples final_apples : ℕ)
  (h1 : initial_apples = 20)
  (h2 : jane_apples = 5)
  (h3 : extra_apples = 2)
  (h4 : final_apples = 4) :
  initial_apples - jane_apples - (jane_apples + extra_apples) - final_apples = final_apples := 
by
  sorry

end martha_apples_l1572_157298


namespace hexagon_interior_angle_Q_l1572_157227

theorem hexagon_interior_angle_Q 
  (A B C D E F : ℕ)
  (hA : A = 135) (hB : B = 150) (hC : C = 120) (hD : D = 130) (hE : E = 100)
  (hex_angle_sum : A + B + C + D + E + F = 720) :
  F = 85 :=
by
  rw [hA, hB, hC, hD, hE] at hex_angle_sum
  sorry

end hexagon_interior_angle_Q_l1572_157227


namespace MikeSalaryNow_l1572_157204

-- Definitions based on conditions
def FredSalary  := 1000   -- Fred's salary five months ago
def MikeSalaryFiveMonthsAgo := 10 * FredSalary  -- Mike's salary five months ago
def SalaryIncreasePercent := 40 / 100  -- 40 percent salary increase
def SalaryIncrease := SalaryIncreasePercent * MikeSalaryFiveMonthsAgo  -- Increase in Mike's salary

-- Statement to be proved
theorem MikeSalaryNow : MikeSalaryFiveMonthsAgo + SalaryIncrease = 14000 :=
by
  -- Proof is skipped
  sorry

end MikeSalaryNow_l1572_157204


namespace number_of_terms_l1572_157203

variable {α : Type} [LinearOrderedField α]

def sum_of_arithmetic_sequence (a₁ aₙ d : α) (n : ℕ) : α :=
  n * (a₁ + aₙ) / 2

theorem number_of_terms (a₁ aₙ : α) (d : α) (n : ℕ)
  (h₀ : 4 * (2 * a₁ + 3 * d) / 2 = 21)
  (h₁ : 4 * (2 * aₙ - 3 * d) / 2 = 67)
  (h₂ : sum_of_arithmetic_sequence a₁ aₙ d n = 286) :
  n = 26 :=
sorry

end number_of_terms_l1572_157203


namespace geometric_sequence_product_l1572_157233

variable {α : Type*} [LinearOrderedField α]
variable (a : ℕ → α)

theorem geometric_sequence_product (h : a 7 * a 12 = 5) : 
  a 8 * a 9 * a 10 * a 11 = 25 :=
by
  -- Proof goes here
  sorry

end geometric_sequence_product_l1572_157233


namespace power_sum_inequality_l1572_157294

theorem power_sum_inequality (k l m : ℕ) : 
  2 ^ (k + l) + 2 ^ (k + m) + 2 ^ (l + m) ≤ 2 ^ (k + l + m + 1) + 1 := 
by 
  sorry

end power_sum_inequality_l1572_157294


namespace total_bottles_remaining_is_14090_l1572_157221

-- Define the constants
def total_small_bottles : ℕ := 5000
def total_big_bottles : ℕ := 12000
def small_bottles_sold_percentage : ℕ := 15
def big_bottles_sold_percentage : ℕ := 18

-- Define the remaining bottles
def calc_remaining_bottles (total_bottles sold_percentage : ℕ) : ℕ :=
  total_bottles - (sold_percentage * total_bottles / 100)

-- Define the remaining small and big bottles
def remaining_small_bottles : ℕ := calc_remaining_bottles total_small_bottles small_bottles_sold_percentage
def remaining_big_bottles : ℕ := calc_remaining_bottles total_big_bottles big_bottles_sold_percentage

-- Define the total remaining bottles
def total_remaining_bottles : ℕ := remaining_small_bottles + remaining_big_bottles

-- State the theorem
theorem total_bottles_remaining_is_14090 : total_remaining_bottles = 14090 := by
  sorry

end total_bottles_remaining_is_14090_l1572_157221


namespace semicircle_radius_correct_l1572_157268

noncomputable def semicircle_radius (P : ℝ) : ℝ := P / (Real.pi + 2)

theorem semicircle_radius_correct (h :127 =113): semicircle_radius 113 = 113 / (Real.pi + 2) :=
by
  sorry

end semicircle_radius_correct_l1572_157268


namespace maria_high_school_students_l1572_157255

variable (M D : ℕ)

theorem maria_high_school_students (h1 : M = 4 * D) (h2 : M - D = 1800) : M = 2400 :=
by
  sorry

end maria_high_school_students_l1572_157255


namespace distance_between_A_and_B_is_45_kilometers_l1572_157290

variable (speedA speedB : ℝ)
variable (distanceAB : ℝ)

noncomputable def problem_conditions := 
  speedA = 1.2 * speedB ∧
  ∃ (distanceMalfunction : ℝ), distanceMalfunction = 5 ∧
  ∃ (timeFixingMalfunction : ℝ), timeFixingMalfunction = (distanceAB / 6) / speedB ∧
  ∃ (increasedSpeedB : ℝ), increasedSpeedB = 1.6 * speedB ∧
  ∃ (timeA timeB timeB_new : ℝ),
    timeA = (distanceAB / speedA) ∧
    timeB = (distanceMalfunction / speedB) + timeFixingMalfunction + (distanceAB - distanceMalfunction) / increasedSpeedB ∧
    timeA = timeB

theorem distance_between_A_and_B_is_45_kilometers
  (speedA speedB distanceAB : ℝ) 
  (cond : problem_conditions speedA speedB distanceAB) :
  distanceAB = 45 :=
sorry

end distance_between_A_and_B_is_45_kilometers_l1572_157290


namespace root_expr_calculation_l1572_157279

theorem root_expr_calculation : (3 + Real.sqrt 10) * (Real.sqrt 2 - Real.sqrt 5) = -2 * Real.sqrt 2 - Real.sqrt 5 := 
by 
  sorry

end root_expr_calculation_l1572_157279


namespace time_in_1867_minutes_correct_l1572_157235

def current_time := (3, 15) -- (hours, minutes)
def minutes_in_hour := 60
def total_minutes := 1867
def hours_after := total_minutes / minutes_in_hour
def remainder_minutes := total_minutes % minutes_in_hour
def result_time := ((current_time.1 + hours_after) % 24, current_time.2 + remainder_minutes)
def expected_time := (22, 22) -- 10:22 p.m. in 24-hour format

theorem time_in_1867_minutes_correct : result_time = expected_time := 
by
    -- No proof is required according to the instructions.
    sorry

end time_in_1867_minutes_correct_l1572_157235


namespace hose_filling_time_l1572_157250

theorem hose_filling_time :
  ∀ (P A B C : ℝ), 
  (P / 3 = A + B) →
  (P / 5 = A + C) →
  (P / 4 = B + C) →
  (P / (A + B + C) = 2.55) :=
by
  intros P A B C hAB hAC hBC
  sorry

end hose_filling_time_l1572_157250


namespace negation_proof_l1572_157214

-- Definitions based on conditions
def Line : Type := sorry  -- Define a type for lines (using sorry for now)
def Plane : Type := sorry  -- Define a type for planes (using sorry for now)

-- Condition definition
def is_perpendicular (l : Line) (α : Plane) : Prop := sorry  -- Define what it means for a plane to be perpendicular to a line (using sorry for now)

-- Given condition
axiom condition : ∀ (l : Line), ∃ (α : Plane), is_perpendicular l α

-- Statement to prove
theorem negation_proof : (∃ (l : Line), ∀ (α : Plane), ¬is_perpendicular l α) :=
sorry

end negation_proof_l1572_157214


namespace true_value_of_product_l1572_157200

theorem true_value_of_product (a b c : ℕ) (h1 : a > b) (h2 : b > c) (h3 : c > 0) :
  let product := (100 * a + 10 * b + c) * (100 * b + 10 * c + a) * (100 * c + 10 * a + b)
  product = 2342355286 → (product % 10 = 6) → product = 328245326 :=
by
  sorry

end true_value_of_product_l1572_157200


namespace sum_minimal_area_k_l1572_157281

def vertices_triangle_min_area (k : ℤ) : Prop :=
  let x1 := 1
  let y1 := 7
  let x2 := 13
  let y2 := 16
  let x3 := 5
  ((y1 - k) * (x2 - x1) ≠ (x1 - x3) * (y2 - y1))

def minimal_area_sum_k : ℤ :=
  9 + 11

theorem sum_minimal_area_k :
  ∃ k1 k2 : ℤ, vertices_triangle_min_area k1 ∧ vertices_triangle_min_area k2 ∧ k1 + k2 = 20 := 
sorry

end sum_minimal_area_k_l1572_157281


namespace fraction_equality_l1572_157284

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 :=
by
  sorry

end fraction_equality_l1572_157284


namespace find_d_square_plus_5d_l1572_157224

theorem find_d_square_plus_5d (a b c d : ℤ) (h₁: a^2 + 2 * a = 65) (h₂: b^2 + 3 * b = 125) (h₃: c^2 + 4 * c = 205) (h₄: d = 5 + 6) :
  d^2 + 5 * d = 176 :=
by
  rw [h₄]
  sorry

end find_d_square_plus_5d_l1572_157224


namespace rabbits_ate_three_potatoes_l1572_157272

variable (initial_potatoes remaining_potatoes eaten_potatoes : ℕ)

-- Definitions from the conditions
def mary_initial_potatoes : initial_potatoes = 8 := sorry
def mary_remaining_potatoes : remaining_potatoes = 5 := sorry

-- The goal to prove
theorem rabbits_ate_three_potatoes :
  initial_potatoes - remaining_potatoes = 3 := sorry

end rabbits_ate_three_potatoes_l1572_157272


namespace add_base_6_l1572_157212

theorem add_base_6 (a b c : ℕ) (h₀ : a = 3 * 6^3 + 4 * 6^2 + 2 * 6 + 1)
                    (h₁ : b = 4 * 6^3 + 5 * 6^2 + 2 * 6 + 5)
                    (h₂ : c = 1 * 6^4 + 2 * 6^3 + 3 * 6^2 + 5 * 6 + 0) : 
  a + b = c :=
by  
  sorry

end add_base_6_l1572_157212


namespace smallest_crate_side_l1572_157223

/-- 
A crate measures some feet by 8 feet by 12 feet on the inside. 
A stone pillar in the shape of a right circular cylinder must fit into the crate for shipping so that 
it rests upright when the crate sits on at least one of its six sides. 
The radius of the pillar is 7 feet. 
Prove that the length of the crate's smallest side is 8 feet.
-/
theorem smallest_crate_side (x : ℕ) (hx : x >= 14) : min (min x 8) 12 = 8 :=
by {
  sorry
}

end smallest_crate_side_l1572_157223


namespace total_pages_l1572_157282

def Johnny_word_count : ℕ := 195
def Madeline_word_count : ℕ := 2 * Johnny_word_count
def Timothy_word_count : ℕ := Madeline_word_count + 50
def Samantha_word_count : ℕ := 3 * Madeline_word_count
def Ryan_word_count : ℕ := Johnny_word_count + 100
def Words_per_page : ℕ := 235

def pages_needed (words : ℕ) : ℕ :=
  if words % Words_per_page = 0 then words / Words_per_page else words / Words_per_page + 1

theorem total_pages :
  pages_needed Johnny_word_count +
  pages_needed Madeline_word_count +
  pages_needed Timothy_word_count +
  pages_needed Samantha_word_count +
  pages_needed Ryan_word_count = 12 :=
  by sorry

end total_pages_l1572_157282


namespace eliot_account_balance_l1572_157207

variable (A E : ℝ)

theorem eliot_account_balance (h1 : A - E = (1/12) * (A + E)) (h2 : 1.10 * A = 1.20 * E + 20) : 
  E = 200 := 
by 
  sorry

end eliot_account_balance_l1572_157207


namespace does_not_round_to_72_56_l1572_157244

-- Definitions for the numbers in question
def numA := 72.558
def numB := 72.563
def numC := 72.55999
def numD := 72.564
def numE := 72.555

-- Function to round a number to the nearest hundredth
def round_nearest_hundredth (x : Float) : Float :=
  (Float.round (x * 100) / 100 : Float)

-- Lean statement for the equivalent proof problem
theorem does_not_round_to_72_56 :
  round_nearest_hundredth numA = 72.56 ∧
  round_nearest_hundredth numB = 72.56 ∧
  round_nearest_hundredth numC = 72.56 ∧
  round_nearest_hundredth numD = 72.56 ∧
  round_nearest_hundredth numE ≠ 72.56 :=
by
  sorry

end does_not_round_to_72_56_l1572_157244


namespace Doris_spent_6_l1572_157217

variable (D : ℝ)

theorem Doris_spent_6 (h0 : 24 - (D + D / 2) = 15) : D = 6 :=
by
  sorry

end Doris_spent_6_l1572_157217


namespace solve_for_x_l1572_157208

theorem solve_for_x:
  ∀ (x : ℝ), (x + 10) / (x - 4) = (x - 3) / (x + 6) → x = -(48 / 23) :=
by
  sorry

end solve_for_x_l1572_157208


namespace log_x_inequality_l1572_157265

noncomputable def log_x_over_x (x : ℝ) := (Real.log x) / x

theorem log_x_inequality {x : ℝ} (h1 : 1 < x) (h2 : x < 2) : 
  (log_x_over_x x) ^ 2 < log_x_over_x x ∧ log_x_over_x x < log_x_over_x (x * x) :=
by
  sorry

end log_x_inequality_l1572_157265


namespace area_region_inside_but_outside_l1572_157234

noncomputable def area_diff (side_large side_small : ℝ) : ℝ :=
  (side_large ^ 2) - (side_small ^ 2)

theorem area_region_inside_but_outside (h_large : 10 > 0) (h_small : 4 > 0) :
  area_diff 10 4 = 84 :=
by
  -- The proof steps would go here
  sorry

end area_region_inside_but_outside_l1572_157234


namespace frequency_count_l1572_157218

theorem frequency_count (n : ℕ) (f : ℝ) (h1 : n = 1000) (h2 : f = 0.4) : n * f = 400 := by
  sorry

end frequency_count_l1572_157218


namespace percentage_increase_area_rectangle_l1572_157254

theorem percentage_increase_area_rectangle (L W : ℝ) :
  let new_length := 1.20 * L
  let new_width := 1.20 * W
  let original_area := L * W
  let new_area := new_length * new_width
  let percentage_increase := ((new_area - original_area) / original_area) * 100
  percentage_increase = 44 := by
  sorry

end percentage_increase_area_rectangle_l1572_157254


namespace bucket_full_weight_l1572_157231

variable (p q : ℝ)

theorem bucket_full_weight (p q : ℝ) (x y: ℝ) (h1 : x + 3/4 * y = p) (h2 : x + 1/3 * y = q) :
  x + y = (8 * p - 7 * q) / 5 :=
by
  sorry

end bucket_full_weight_l1572_157231


namespace locus_of_points_where_tangents_are_adjoint_lines_l1572_157253

theorem locus_of_points_where_tangents_are_adjoint_lines 
  (p : ℝ) (y x : ℝ)
  (h_parabola : y^2 = 2 * p * x) :
  y^2 = - (p / 2) * x :=
sorry

end locus_of_points_where_tangents_are_adjoint_lines_l1572_157253


namespace total_boxes_l1572_157257

theorem total_boxes (r_cost y_cost : ℝ) (avg_cost : ℝ) (R Y : ℕ) (hc_r : r_cost = 1.30) (hc_y : y_cost = 2.00) 
                    (hc_avg : avg_cost = 1.72) (hc_R : R = 4) (hc_Y : Y = 4) : 
  R + Y = 8 :=
by
  sorry

end total_boxes_l1572_157257


namespace initial_amount_of_A_l1572_157245

variable (a b c : ℕ)

-- Conditions
axiom condition1 : a - b - c = 32
axiom condition2 : b + c = 48
axiom condition3 : a + b + c = 128

-- The goal is to prove that A had 80 cents initially.
theorem initial_amount_of_A : a = 80 :=
by
  -- We need to skip the proof here
  sorry

end initial_amount_of_A_l1572_157245


namespace exists_polynomial_h_l1572_157215

variable {R : Type} [CommRing R] [IsDomain R] [CharZero R]

noncomputable def f (x : R) : ℝ := sorry -- define the polynomial f(x) here
noncomputable def g (x : R) : ℝ := sorry -- define the polynomial g(x) here

theorem exists_polynomial_h (m n : ℕ) (a : ℕ → ℝ) (b : ℕ → ℝ) (h_mn : m + n > 0)
  (h_fg_squares : ∀ x : ℝ, (∃ k : ℤ, f x = k^2) ↔ (∃ l : ℤ, g x = l^2)) :
  ∃ h : ℝ → ℝ, ∀ x : ℝ, f x * g x = (h x)^2 :=
sorry

end exists_polynomial_h_l1572_157215


namespace minimum_value_x_plus_y_l1572_157213

theorem minimum_value_x_plus_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x + 2 * y = x * y) :
  x + y = 3 + 2 * Real.sqrt 2 :=
sorry

end minimum_value_x_plus_y_l1572_157213


namespace pattern_formula_l1572_157201

theorem pattern_formula (n : ℤ) : n * (n + 2) = (n + 1) ^ 2 - 1 := 
by sorry

end pattern_formula_l1572_157201


namespace equidistant_cyclist_l1572_157248

-- Definition of key parameters
def speed_car := 60  -- in km/h
def speed_cyclist := 18  -- in km/h
def speed_pedestrian := 6  -- in km/h
def distance_AC := 10  -- in km
def angle_ACB := 60  -- in degrees
def time_car_start := (7, 58)  -- 7:58 AM
def time_cyclist_start := (8, 0)  -- 8:00 AM
def time_pedestrian_start := (6, 44) -- 6:44 AM
def time_solution := (8, 6)  -- 8:06 AM

-- Time difference function
def time_diff (t1 t2 : Nat × Nat) : Nat :=
  (t2.1 - t1.1) * 60 + (t2.2 - t1.2)  -- time difference in minutes

-- Convert minutes to hours
noncomputable def minutes_to_hours (m : Nat) : ℝ :=
  m / 60.0

-- Distances traveled by car, cyclist, and pedestrian by the given time
noncomputable def distance_car (t1 t2 : Nat × Nat) : ℝ :=
  speed_car * (minutes_to_hours (time_diff t1 t2) + 2 / 60.0)

noncomputable def distance_cyclist (t1 t2 : Nat × Nat) : ℝ :=
  speed_cyclist * minutes_to_hours (time_diff t1 t2)

noncomputable def distance_pedestrian (t1 t2 : Nat × Nat) : ℝ :=
  speed_pedestrian * (minutes_to_hours (time_diff t1 t2) + 136 / 60.0)

-- Verification statement
theorem equidistant_cyclist :
  distance_car time_car_start time_solution = distance_pedestrian time_pedestrian_start time_solution → 
  distance_cyclist time_cyclist_start time_solution = 
  distance_car time_car_start time_solution ∧
  distance_cyclist time_cyclist_start time_solution = 
  distance_pedestrian time_pedestrian_start time_solution :=
by
  -- Given conditions and the correctness to be shown
  sorry

end equidistant_cyclist_l1572_157248


namespace product_of_primes_l1572_157266

theorem product_of_primes : 5 * 7 * 997 = 34895 :=
by
  sorry

end product_of_primes_l1572_157266


namespace find_multiple_l1572_157238

theorem find_multiple (a b m : ℤ) (h1 : b = 7) (h2 : b - a = 2) 
  (h3 : a * b = m * (a + b) + 11) : m = 2 :=
by {
  sorry
}

end find_multiple_l1572_157238


namespace Ella_jellybeans_l1572_157273

-- Definitions based on conditions from part (a)
def Dan_volume := 10
def Dan_jellybeans := 200
def scaling_factor := 3

-- Prove that Ella's box holds 5400 jellybeans
theorem Ella_jellybeans : scaling_factor^3 * Dan_jellybeans = 5400 := 
by
  sorry

end Ella_jellybeans_l1572_157273


namespace volume_of_cylindrical_block_l1572_157299

variable (h_cylindrical : ℕ) (combined_value : ℝ)

theorem volume_of_cylindrical_block (h_cylindrical : ℕ) (combined_value : ℝ):
  h_cylindrical = 3 → combined_value / 5 * h_cylindrical = 15.42 := by
suffices combined_value / 5 = 5.14 from sorry
suffices 5.14 * 3 = 15.42 from sorry
suffices h_cylindrical = 3 from sorry
suffices 25.7 = combined_value from sorry
sorry

end volume_of_cylindrical_block_l1572_157299


namespace number_of_points_l1572_157277

theorem number_of_points (a b : ℤ) : (|a| = 3 ∧ |b| = 2) → ∃! (P : ℤ × ℤ), P = (a, b) :=
by sorry

end number_of_points_l1572_157277


namespace ratio_of_part_to_whole_l1572_157220

theorem ratio_of_part_to_whole (N : ℝ) (h1 : (1/3) * (2/5) * N = 15) (h2 : (40/100) * N = 180) :
  (15 / N) = (1 / 7.5) :=
by
  sorry

end ratio_of_part_to_whole_l1572_157220


namespace question_1_question_2_l1572_157288

def f (x : ℝ) : ℝ := |x + 1| - |x - 4|

theorem question_1 (m : ℝ) :
  (∀ x : ℝ, f x ≤ -m^2 + 6 * m) ↔ (1 ≤ m ∧ m ≤ 5) :=
by
  sorry

theorem question_2 (a b c : ℝ) (h1 : 3 * a + 4 * b + 5 * c = 5) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a^2 + b^2 + c^2 ≥ 1 / 2 :=
by
  sorry

end question_1_question_2_l1572_157288


namespace equalize_rice_move_amount_l1572_157251

open Real

noncomputable def containerA_kg : Real := 12
noncomputable def containerA_g : Real := 400
noncomputable def containerB_g : Real := 7600

noncomputable def total_rice_in_A_g : Real := containerA_kg * 1000 + containerA_g
noncomputable def total_rice_in_A_and_B_g : Real := total_rice_in_A_g + containerB_g
noncomputable def equalized_rice_per_container_g : Real := total_rice_in_A_and_B_g / 2

noncomputable def amount_to_move_g : Real := total_rice_in_A_g - equalized_rice_per_container_g
noncomputable def amount_to_move_kg : Real := amount_to_move_g / 1000

theorem equalize_rice_move_amount :
  amount_to_move_kg = 2.4 :=
by
  sorry

end equalize_rice_move_amount_l1572_157251


namespace find_stiffnesses_l1572_157293

def stiffnesses (m g x1 x2 k1 k2 : ℝ) : Prop :=
  (m = 3) ∧ (g = 10) ∧ (x1 = 0.4) ∧ (x2 = 0.075) ∧
  (k1 * k2 / (k1 + k2) * x1 = m * g) ∧
  ((k1 + k2) * x2 = m * g)

theorem find_stiffnesses (k1 k2 : ℝ) :
  stiffnesses 3 10 0.4 0.075 k1 k2 → 
  k1 = 300 ∧ k2 = 100 := 
sorry

end find_stiffnesses_l1572_157293


namespace Terry_driving_speed_is_40_l1572_157296

-- Conditions
def distance_home_to_workplace : ℕ := 60
def total_time_driving : ℕ := 3

-- Computation for total distance
def total_distance := distance_home_to_workplace * 2

-- Desired speed computation
def driving_speed := total_distance / total_time_driving

-- Problem statement to prove
theorem Terry_driving_speed_is_40 : driving_speed = 40 :=
by 
  sorry -- proof not required as per instructions

end Terry_driving_speed_is_40_l1572_157296


namespace exists_three_distinct_nats_sum_prod_squares_l1572_157274

theorem exists_three_distinct_nats_sum_prod_squares :
  ∃ (a b c : ℕ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  (∃ (x : ℕ), a + b + c = x^2) ∧ 
  (∃ (y : ℕ), a * b * c = y^2) :=
sorry

end exists_three_distinct_nats_sum_prod_squares_l1572_157274


namespace neg_of_univ_prop_l1572_157278

theorem neg_of_univ_prop :
  (∃ (x₀ : ℝ), 0 ≤ x₀ ∧ x₀^3 + x₀ < 0) ↔ ¬ (∀ (x : ℝ), 0 ≤ x → x^3 + x ≥ 0) := by
sorry

end neg_of_univ_prop_l1572_157278


namespace budget_equality_year_l1572_157285

theorem budget_equality_year :
  ∃ n : ℕ, 540000 + 30000 * n = 780000 - 10000 * n ∧ 1990 + n = 1996 :=
by
  sorry

end budget_equality_year_l1572_157285


namespace count_valid_numbers_l1572_157262

def digits_set : List ℕ := [0, 2, 4, 7, 8, 9]

def divisible_by_4 (n : ℕ) : Prop :=
  n % 4 = 0

def divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def sum_digits (digits : List ℕ) : ℕ :=
  List.sum digits

def last_two_digits_divisibility (last_two_digits : ℕ) : Prop :=
  last_two_digits % 4 = 0

def number_is_valid (digits : List ℕ) : Prop :=
  sum_digits digits % 3 = 0

theorem count_valid_numbers :
  let possible_digits := [0, 2, 4, 7, 8, 9]
  let positions := 5
  let combinations := Nat.pow (List.length possible_digits) (positions - 1)
  let last_digit_choices := [0, 4, 8]
  3888 = 3 * combinations :=
sorry

end count_valid_numbers_l1572_157262
