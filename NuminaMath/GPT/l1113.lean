import Mathlib

namespace NUMINAMATH_GPT_smith_oldest_child_age_l1113_111314

theorem smith_oldest_child_age
  (avg_age : ℕ)
  (youngest : ℕ)
  (middle : ℕ)
  (oldest : ℕ)
  (h1 : avg_age = 9)
  (h2 : youngest = 6)
  (h3 : middle = 8)
  (h4 : (youngest + middle + oldest) / 3 = avg_age) :
  oldest = 13 :=
by
  sorry

end NUMINAMATH_GPT_smith_oldest_child_age_l1113_111314


namespace NUMINAMATH_GPT_range_of_a_l1113_111332

theorem range_of_a
    (a : ℝ)
    (h : ∀ x y : ℝ, (x - a) ^ 2 + (y - a) ^ 2 = 4 → x ^ 2 + y ^ 2 = 1) :
    a ∈ (Set.Ioo (-(3 * Real.sqrt 2 / 2)) (-(Real.sqrt 2 / 2)) ∪ Set.Ioo (Real.sqrt 2 / 2) (3 * Real.sqrt 2 / 2)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1113_111332


namespace NUMINAMATH_GPT_total_treats_is_237_l1113_111310

def num_children : ℕ := 3
def hours_out : ℕ := 4
def houses_visited (hour : ℕ) : ℕ :=
  match hour with
  | 1 => 4
  | 2 => 6
  | 3 => 5
  | 4 => 7
  | _ => 0

def treats_per_kid_per_house (hour : ℕ) : ℕ :=
  match hour with
  | 1 => 3
  | 3 => 3
  | 2 => 4
  | 4 => 4
  | _ => 0

def total_treats : ℕ :=
  (houses_visited 1 * treats_per_kid_per_house 1 * num_children) + 
  (houses_visited 2 * treats_per_kid_per_house 2 * num_children) +
  (houses_visited 3 * treats_per_kid_per_house 3 * num_children) +
  (houses_visited 4 * treats_per_kid_per_house 4 * num_children)

theorem total_treats_is_237 : total_treats = 237 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_total_treats_is_237_l1113_111310


namespace NUMINAMATH_GPT_exists_four_numbers_product_fourth_power_l1113_111304

theorem exists_four_numbers_product_fourth_power :
  ∃ (numbers : Fin 81 → ℕ),
    (∀ i, ∃ a b c : ℕ, numbers i = 2^a * 3^b * 5^c) ∧
    ∃ (i j k l : Fin 81), i ≠ j ∧ j ≠ k ∧ k ≠ l ∧ l ≠ i ∧
    ∃ m : ℕ, m^4 = numbers i * numbers j * numbers k * numbers l :=
by
  sorry

end NUMINAMATH_GPT_exists_four_numbers_product_fourth_power_l1113_111304


namespace NUMINAMATH_GPT_domain_of_f_l1113_111377

noncomputable def domain_f : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 2}

theorem domain_of_f : domain_f = {x : ℝ | -1 ≤ x ∧ x < 2} := by
  sorry

end NUMINAMATH_GPT_domain_of_f_l1113_111377


namespace NUMINAMATH_GPT_solve_for_m_l1113_111371

theorem solve_for_m (m : ℝ) : (∀ x : ℝ, 0 < x ∧ x < 2 → - (1/2) * x^2 + 2 * x > m * x) → m = 1 :=
by
  -- Skip the proof by using sorry
  sorry

end NUMINAMATH_GPT_solve_for_m_l1113_111371


namespace NUMINAMATH_GPT_perfect_square_trinomial_m_l1113_111338

theorem perfect_square_trinomial_m (m : ℤ) (x : ℤ) : (∃ a : ℤ, x^2 - mx + 16 = (x - a)^2) ↔ (m = 8 ∨ m = -8) :=
by sorry

end NUMINAMATH_GPT_perfect_square_trinomial_m_l1113_111338


namespace NUMINAMATH_GPT_value_range_for_inequality_solution_set_l1113_111326

-- Define the condition
def condition (a : ℝ) : Prop := a > 0

-- Define the inequality
def inequality (x a : ℝ) : Prop := |x - 4| + |x - 3| < a

-- State the theorem to be proven
theorem value_range_for_inequality_solution_set (a : ℝ) (h: condition a) : (a > 1) ↔ ∃ x : ℝ, inequality x a := 
sorry

end NUMINAMATH_GPT_value_range_for_inequality_solution_set_l1113_111326


namespace NUMINAMATH_GPT_Grace_minus_Lee_l1113_111327

-- Definitions for the conditions
def Grace_calculation : ℤ := 12 - (3 * 4 - 2)
def Lee_calculation : ℤ := (12 - 3) * 4 - 2

-- Statement of the problem to prove
theorem Grace_minus_Lee : Grace_calculation - Lee_calculation = -32 := by
  sorry

end NUMINAMATH_GPT_Grace_minus_Lee_l1113_111327


namespace NUMINAMATH_GPT_more_tails_than_heads_l1113_111365

def total_flips : ℕ := 211
def heads_flips : ℕ := 65
def tails_flips : ℕ := total_flips - heads_flips

theorem more_tails_than_heads : tails_flips - heads_flips = 81 := by
  -- proof is unnecessary according to the instructions
  sorry

end NUMINAMATH_GPT_more_tails_than_heads_l1113_111365


namespace NUMINAMATH_GPT_exponent_division_l1113_111396

theorem exponent_division (m n : ℕ) (h : m - n = 1) : 5 ^ m / 5 ^ n = 5 :=
by {
  sorry
}

end NUMINAMATH_GPT_exponent_division_l1113_111396


namespace NUMINAMATH_GPT_complement_of_union_l1113_111351

open Set

variable (U A B : Set ℕ)
variable (u_def : U = {0, 1, 2, 3, 4, 5, 6})
variable (a_def : A = {1, 3})
variable (b_def : B = {3, 5})

theorem complement_of_union :
  (U \ (A ∪ B)) = {0, 2, 4, 6} :=
by
  sorry

end NUMINAMATH_GPT_complement_of_union_l1113_111351


namespace NUMINAMATH_GPT_smallest_integer_solution_l1113_111300

theorem smallest_integer_solution (y : ℤ) (h : 7 - 3 * y < 25) : y ≥ -5 :=
by {
  sorry
}

end NUMINAMATH_GPT_smallest_integer_solution_l1113_111300


namespace NUMINAMATH_GPT_polynomial_g_l1113_111388

def f (x : ℝ) : ℝ := x^2

theorem polynomial_g (g : ℝ → ℝ) :
  (∀ x, f (g x) = 9 * x ^ 2 - 6 * x + 1) →
  (∀ x, g x = 3 * x - 1 ∨ g x = -3 * x + 1) :=
by
  sorry

end NUMINAMATH_GPT_polynomial_g_l1113_111388


namespace NUMINAMATH_GPT_boat_breadth_is_two_l1113_111362

noncomputable def breadth_of_boat (L h m g ρ : ℝ) : ℝ :=
  let W := m * g
  let V := W / (ρ * g)
  V / (L * h)

theorem boat_breadth_is_two :
  breadth_of_boat 7 0.01 140 9.81 1000 = 2 := 
by
  unfold breadth_of_boat
  simp
  sorry

end NUMINAMATH_GPT_boat_breadth_is_two_l1113_111362


namespace NUMINAMATH_GPT_greatest_possible_value_l1113_111307

theorem greatest_possible_value :
  ∃ (N P M : ℕ), (M < 10) ∧ (N < 10) ∧ (P < 10) ∧ (M * (111 * M) = N * 1000 + P * 100 + M * 10 + M)
                ∧ (N * 1000 + P * 100 + M * 10 + M = 3996) :=
by
  sorry

end NUMINAMATH_GPT_greatest_possible_value_l1113_111307


namespace NUMINAMATH_GPT_number_of_people_is_8_l1113_111378

noncomputable def find_number_of_people (avg_increase : ℝ) (old_weight : ℝ) (new_weight : ℝ) (weight_diff : ℝ) (n : ℕ) :=
  avg_increase = weight_diff / n ∧ old_weight = 70 ∧ new_weight = 90 ∧ weight_diff = new_weight - old_weight → n = 8

theorem number_of_people_is_8 :
  ∃ n : ℕ, find_number_of_people 2.5 70 90 20 n :=
by
  use 8
  sorry

end NUMINAMATH_GPT_number_of_people_is_8_l1113_111378


namespace NUMINAMATH_GPT_initial_fish_count_l1113_111397

variable (x : ℕ)

theorem initial_fish_count (initial_fish : ℕ) (given_fish : ℕ) (total_fish : ℕ)
  (h1 : total_fish = initial_fish + given_fish)
  (h2 : total_fish = 69)
  (h3 : given_fish = 47) :
  initial_fish = 22 :=
by
  sorry

end NUMINAMATH_GPT_initial_fish_count_l1113_111397


namespace NUMINAMATH_GPT_perfect_square_trinomial_k_l1113_111392

theorem perfect_square_trinomial_k (k : ℤ) :
  (∃ a : ℤ, x^2 + k*x + 25 = (x + a)^2 ∧ a^2 = 25) → (k = 10 ∨ k = -10) :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_trinomial_k_l1113_111392


namespace NUMINAMATH_GPT_Taehyung_walked_distance_l1113_111352

variable (step_distance : ℝ) (steps_per_set : ℕ) (num_sets : ℕ)
variable (h1 : step_distance = 0.45)
variable (h2 : steps_per_set = 90)
variable (h3 : num_sets = 13)

theorem Taehyung_walked_distance :
  (steps_per_set * step_distance) * num_sets = 526.5 :=
by 
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_Taehyung_walked_distance_l1113_111352


namespace NUMINAMATH_GPT_inequality_solution_set_l1113_111379

theorem inequality_solution_set (x : ℝ) :
  (4 * x - 2 ≥ 3 * (x - 1)) ∧ ((x - 5) / 2 > x - 4) ↔ (-1 ≤ x ∧ x < 3) := 
by sorry

end NUMINAMATH_GPT_inequality_solution_set_l1113_111379


namespace NUMINAMATH_GPT_curve_intersects_itself_l1113_111360

theorem curve_intersects_itself :
  ∃ t₁ t₂ : ℝ, t₁ ≠ t₂ ∧ (t₁^2 - 3, t₁^3 - 6 * t₁ + 4) = (3, 4) ∧ (t₂^2 - 3, t₂^3 - 6 * t₂ + 4) = (3, 4) :=
sorry

end NUMINAMATH_GPT_curve_intersects_itself_l1113_111360


namespace NUMINAMATH_GPT_sum_of_integers_l1113_111311

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 10) (h3 : x * y = 56) : x + y = 18 := 
sorry

end NUMINAMATH_GPT_sum_of_integers_l1113_111311


namespace NUMINAMATH_GPT_gcd_of_factorials_l1113_111334

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem gcd_of_factorials :
  Nat.gcd (factorial 8) ((factorial 6)^2) = 1440 := by
  sorry

end NUMINAMATH_GPT_gcd_of_factorials_l1113_111334


namespace NUMINAMATH_GPT_relationship_among_a_b_c_l1113_111324

noncomputable def f (x : ℝ) : ℝ := sorry  -- The actual function definition is not necessary for this statement.

-- Lean statements for the given conditions
variables {f : ℝ → ℝ}

-- f is even
def even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x

-- f(x+1) = -f(x)
def periodic_property (f : ℝ → ℝ) := ∀ x, f (x + 1) = - f x

-- f is monotonically increasing on [-1, 0]
def monotonically_increasing_on (f : ℝ → ℝ) := ∀ x y, -1 ≤ x ∧ x ≤ y ∧ y ≤ 0 → f x ≤ f y

-- Define the relationship statement
theorem relationship_among_a_b_c (h1 : even_function f) (h2 : periodic_property f) 
  (h3 : monotonically_increasing_on f) :
  f 3 < f (Real.sqrt 2) ∧ f (Real.sqrt 2) < f 2 :=
sorry

end NUMINAMATH_GPT_relationship_among_a_b_c_l1113_111324


namespace NUMINAMATH_GPT_book_total_pages_l1113_111386

theorem book_total_pages (num_chapters pages_per_chapter : ℕ) (h1 : num_chapters = 31) (h2 : pages_per_chapter = 61) :
  num_chapters * pages_per_chapter = 1891 := sorry

end NUMINAMATH_GPT_book_total_pages_l1113_111386


namespace NUMINAMATH_GPT_num_sides_of_length4_eq_4_l1113_111395

-- Definitions of the variables and conditions
def total_sides : ℕ := 6
def total_perimeter : ℕ := 30
def side_length1 : ℕ := 7
def side_length2 : ℕ := 4

-- The conditions imposed by the problem
def is_hexagon (x y : ℕ) : Prop := x + y = total_sides
def perimeter_condition (x y : ℕ) : Prop := side_length1 * x + side_length2 * y = total_perimeter

-- The proof problem: Prove that the number of sides of length 4 is 4
theorem num_sides_of_length4_eq_4 (x y : ℕ) 
    (h1 : is_hexagon x y) 
    (h2 : perimeter_condition x y) : y = 4 :=
sorry

end NUMINAMATH_GPT_num_sides_of_length4_eq_4_l1113_111395


namespace NUMINAMATH_GPT_find_b_value_l1113_111349

theorem find_b_value (b : ℕ) 
  (h1 : 5 ^ 5 * b = 3 * 15 ^ 5) 
  (h2 : b = 9 ^ 3) : b = 729 :=
by
  sorry

end NUMINAMATH_GPT_find_b_value_l1113_111349


namespace NUMINAMATH_GPT_convex_polyhedron_property_l1113_111340

-- Given conditions as definitions
def num_faces : ℕ := 40
def num_hexagons : ℕ := 8
def num_triangles_eq_twice_pentagons (P : ℕ) (T : ℕ) : Prop := T = 2 * P
def num_pentagons_eq_twice_hexagons (P : ℕ) (H : ℕ) : Prop := P = 2 * H

-- Main statement for the proof problem
theorem convex_polyhedron_property (P T V : ℕ) :
  num_triangles_eq_twice_pentagons P T ∧ num_pentagons_eq_twice_hexagons P num_hexagons ∧ 
  num_faces = T + P + num_hexagons ∧ V = (T * 3 + P * 5 + num_hexagons * 6) / 2 + num_faces - 2 →
  100 * P + 10 * T + V = 535 :=
by
  sorry

end NUMINAMATH_GPT_convex_polyhedron_property_l1113_111340


namespace NUMINAMATH_GPT_rectangle_dimensions_l1113_111387

theorem rectangle_dimensions
  (l w : ℕ)
  (h1 : 2 * l + 2 * w = l * w)
  (h2 : w = l - 3) :
  l = 6 ∧ w = 3 :=
by
  sorry

end NUMINAMATH_GPT_rectangle_dimensions_l1113_111387


namespace NUMINAMATH_GPT_line_equation_l1113_111346

theorem line_equation (a b : ℝ) (h1 : (1, 2) ∈ line) (h2 : ∃ a b : ℝ, b = 2 * a ∧ line = {p : ℝ × ℝ | p.1 / a + p.2 / b = 1}) :
  line = {p : ℝ × ℝ | 2 * p.1 - p.2 = 0} ∨ line = {p : ℝ × ℝ | 2 * p.1 + p.2 - 4 = 0} :=
sorry

end NUMINAMATH_GPT_line_equation_l1113_111346


namespace NUMINAMATH_GPT_hyperbola_min_focal_asymptote_eq_l1113_111301

theorem hyperbola_min_focal_asymptote_eq {x y m : ℝ}
  (h1 : -2 ≤ m)
  (h2 : m < 0)
  (h_eq : x^2 / m^2 - y^2 / (2 * m + 6) = 1)
  (h_min_focal : m = -1) :
  y = 2 * x ∨ y = -2 * x :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_min_focal_asymptote_eq_l1113_111301


namespace NUMINAMATH_GPT_definite_integral_value_l1113_111370

theorem definite_integral_value :
  (∫ x in (0 : ℝ)..Real.arctan (1/3), (8 + Real.tan x) / (18 * Real.sin x^2 + 2 * Real.cos x^2)) 
  = (Real.pi / 3) + (Real.log 2 / 36) :=
by
  -- Proof to be provided
  sorry

end NUMINAMATH_GPT_definite_integral_value_l1113_111370


namespace NUMINAMATH_GPT_point_reflection_l1113_111368

-- Definition of point and reflection over x-axis
def P : ℝ × ℝ := (-2, 3)

def reflect_x_axis (point : ℝ × ℝ) : ℝ × ℝ :=
  (point.1, -point.2)

-- Statement to prove
theorem point_reflection : reflect_x_axis P = (-2, -3) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_point_reflection_l1113_111368


namespace NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1113_111384

theorem arithmetic_sequence_common_difference (a : ℕ → ℤ) (d : ℤ) (h1 : a 3 = 7) (h2 : a 7 = -5)
  (h3 : ∀ n, a (n + 1) = a n + d) : 
  d = -3 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_common_difference_l1113_111384


namespace NUMINAMATH_GPT_matt_minus_sara_l1113_111376

def sales_tax_rate : ℝ := 0.08
def original_price : ℝ := 120.00
def discount_rate : ℝ := 0.25

def matt_total : ℝ := (original_price * (1 + sales_tax_rate)) * (1 - discount_rate)
def sara_total : ℝ := (original_price * (1 - discount_rate)) * (1 + sales_tax_rate)

theorem matt_minus_sara : matt_total - sara_total = 0 :=
by
  sorry

end NUMINAMATH_GPT_matt_minus_sara_l1113_111376


namespace NUMINAMATH_GPT_lawn_care_company_expense_l1113_111355

theorem lawn_care_company_expense (cost_blade : ℕ) (num_blades : ℕ) (cost_string : ℕ) :
  cost_blade = 8 → num_blades = 4 → cost_string = 7 → 
  (num_blades * cost_blade + cost_string = 39) :=
by
  intro h1 h2 h3
  sorry

end NUMINAMATH_GPT_lawn_care_company_expense_l1113_111355


namespace NUMINAMATH_GPT_geom_seq_common_ratio_l1113_111366

variable {a_n : ℕ → ℝ}
variable {q : ℝ}

def is_geom_seq (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

theorem geom_seq_common_ratio (h1 : a_n 0 + a_n 2 = 10)
                              (h2 : a_n 3 + a_n 5 = 5 / 4)
                              (h_geom : is_geom_seq a_n q) :
  q = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_geom_seq_common_ratio_l1113_111366


namespace NUMINAMATH_GPT_tan_div_sin_cos_sin_mul_cos_l1113_111315

theorem tan_div_sin_cos (α : ℝ) (h : Real.tan α = 7) :
  (Real.sin α + Real.cos α) / (2 * Real.sin α - Real.cos α) = 8 / 13 :=
by
  sorry

theorem sin_mul_cos (α : ℝ) (h : Real.tan α = 7) :
  Real.sin α * Real.cos α = 7 / 50 :=
by
  sorry

end NUMINAMATH_GPT_tan_div_sin_cos_sin_mul_cos_l1113_111315


namespace NUMINAMATH_GPT_add_water_to_solution_l1113_111357

noncomputable def current_solution_volume : ℝ := 300
noncomputable def desired_water_percentage : ℝ := 0.70
noncomputable def current_water_volume : ℝ := 0.60 * current_solution_volume
noncomputable def current_acid_volume : ℝ := 0.40 * current_solution_volume

theorem add_water_to_solution (x : ℝ) : 
  (current_water_volume + x) / (current_solution_volume + x) = desired_water_percentage ↔ x = 100 :=
by
  sorry

end NUMINAMATH_GPT_add_water_to_solution_l1113_111357


namespace NUMINAMATH_GPT_tangerines_left_l1113_111333

def total_tangerines : ℕ := 27
def tangerines_eaten : ℕ := 18

theorem tangerines_left : total_tangerines - tangerines_eaten = 9 := by
  sorry

end NUMINAMATH_GPT_tangerines_left_l1113_111333


namespace NUMINAMATH_GPT_sum_of_squares_l1113_111308

theorem sum_of_squares (a b c : ℝ) (h1 : a + b + c = 23) (h2 : a * b + b * c + a * c = 131) :
  a^2 + b^2 + c^2 = 267 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_l1113_111308


namespace NUMINAMATH_GPT_real_solutions_of_polynomial_l1113_111375

theorem real_solutions_of_polynomial :
  ∀ x : ℝ, x^4 - 3 * x^3 + x^2 - 3 * x = 0 ↔ x = 0 ∨ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_real_solutions_of_polynomial_l1113_111375


namespace NUMINAMATH_GPT_sport_formulation_water_l1113_111313

theorem sport_formulation_water
  (f : ℝ) (c : ℝ) (w : ℝ) 
  (f_s : ℝ) (c_s : ℝ) (w_s : ℝ)
  (standard_ratio : f / c = 1 / 12 ∧ f / w = 1 / 30)
  (sport_ratio_corn_syrup : f_s / c_s = 3 * (f / c))
  (sport_ratio_water : f_s / w_s = (1 / 2) * (f / w))
  (corn_syrup_amount : c_s = 3) :
  w_s = 45 :=
by
  sorry

end NUMINAMATH_GPT_sport_formulation_water_l1113_111313


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l1113_111339

theorem number_of_ordered_pairs (p q : ℂ) (h1 : p^4 * q^3 = 1) (h2 : p^8 * q = 1) : (∃ n : ℕ, n = 40) :=
sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_l1113_111339


namespace NUMINAMATH_GPT_sqrt_sqrt_of_81_eq_pm3_and_cube_root_self_l1113_111364

theorem sqrt_sqrt_of_81_eq_pm3_and_cube_root_self (x : ℝ) : 
  (∃ y : ℝ, y^2 = 81 ∧ (x^2 = y → x = 3 ∨ x = -3)) ∧ (∀ z : ℝ, z^3 = z → (z = 1 ∨ z = -1 ∨ z = 0)) := by
  sorry

end NUMINAMATH_GPT_sqrt_sqrt_of_81_eq_pm3_and_cube_root_self_l1113_111364


namespace NUMINAMATH_GPT_no_natural_solution_l1113_111367

theorem no_natural_solution :
  ¬ (∃ (x y : ℕ), 2 * x + 3 * y = 6) :=
by
sorry

end NUMINAMATH_GPT_no_natural_solution_l1113_111367


namespace NUMINAMATH_GPT_quadratic_condition_l1113_111363

theorem quadratic_condition (p q : ℝ) (x1 x2 : ℝ) (hx : x1 + x2 = -p) (hq : x1 * x2 = q) :
  p + q = 0 := sorry

end NUMINAMATH_GPT_quadratic_condition_l1113_111363


namespace NUMINAMATH_GPT_fraction_identity_l1113_111318

def at_op (a b : ℤ) : ℤ := a * b - 3 * b ^ 2
def hash_op (a b : ℤ) : ℤ := a + 2 * b - 2 * a * b ^ 2

theorem fraction_identity : (at_op 8 3) / (hash_op 8 3) = 3 / 130 := by
  sorry

end NUMINAMATH_GPT_fraction_identity_l1113_111318


namespace NUMINAMATH_GPT_angle_A_in_triangle_find_b_c_given_a_and_A_l1113_111322

theorem angle_A_in_triangle (A B C : ℝ) (a b c : ℝ)
  (h1 : 2 * Real.cos (2 * A) + 4 * Real.cos (B + C) + 3 = 0) :
  A = π / 3 :=
by
  sorry

theorem find_b_c_given_a_and_A (b c : ℝ)
  (A : ℝ)
  (a : ℝ := Real.sqrt 3)
  (h1 : 2 * b * Real.cos A + Real.sqrt (0 - c^2 + 6 * c - 9) = a)
  (h2 : b + c = 3)
  (h3 : A = π / 3) :
  (b = 2 ∧ c = 1) ∨ (b = 1 ∧ c = 2) :=
by
  sorry

end NUMINAMATH_GPT_angle_A_in_triangle_find_b_c_given_a_and_A_l1113_111322


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l1113_111305

theorem arithmetic_sequence_general_term (a : ℕ → ℝ)
  (h_arithmetic : ∀ n, a (n + 1) - a n = a 2 - a 1)
  (h_a3 : a 3 = 7)
  (h_a7 : a 7 = 3) :
  ∀ n, a n = -↑n + 10 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l1113_111305


namespace NUMINAMATH_GPT_fewest_printers_l1113_111328

theorem fewest_printers (x y : ℕ) (h1 : 375 * x = 150 * y) : x + y = 7 :=
  sorry

end NUMINAMATH_GPT_fewest_printers_l1113_111328


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l1113_111345

def setA : Set ℤ := {x | abs x < 4}
def setB : Set ℤ := {x | x - 1 ≥ 0}
def setIntersection : Set ℤ := {1, 2, 3}

theorem intersection_of_A_and_B : setA ∩ setB = setIntersection :=
by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l1113_111345


namespace NUMINAMATH_GPT_blue_paint_needed_l1113_111309

theorem blue_paint_needed (F B : ℝ) :
  (6/9 * F = 4/5 * (F * 1/3 + B) → B = 1/2 * F) :=
sorry

end NUMINAMATH_GPT_blue_paint_needed_l1113_111309


namespace NUMINAMATH_GPT_hostel_initial_plan_l1113_111399

variable (x : ℕ) -- representing the initial number of days

-- Define the conditions
def provisions_for_250_men (x : ℕ) : ℕ := 250 * x
def provisions_for_200_men_45_days : ℕ := 200 * 45

-- Prove the statement
theorem hostel_initial_plan (x : ℕ) (h : provisions_for_250_men x = provisions_for_200_men_45_days) :
  x = 36 :=
by
  sorry

end NUMINAMATH_GPT_hostel_initial_plan_l1113_111399


namespace NUMINAMATH_GPT_geometric_sequence_product_l1113_111317

variable {a1 a2 a3 a4 a5 a6 : ℝ}
variable (r : ℝ)
variable (seq : ℕ → ℝ)

-- Conditions defining the terms of a geometric sequence
def is_geometric_sequence (seq : ℕ → ℝ) (a1 r : ℝ) : Prop :=
  ∀ n : ℕ, seq (n + 1) = seq n * r

-- Given condition: a_3 * a_4 = 5
def given_condition (seq : ℕ → ℝ) := (seq 2 * seq 3 = 5)

-- Proving the required question: a_1 * a_2 * a_5 * a_6 = 5
theorem geometric_sequence_product
  (h_geom : is_geometric_sequence seq a1 r)
  (h_given : given_condition seq) :
  seq 0 * seq 1 * seq 4 * seq 5 = 5 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_product_l1113_111317


namespace NUMINAMATH_GPT_complement_union_l1113_111356

-- Definitions
def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 3}
def B : Set ℕ := {2, 3}

-- Theorem Statement
theorem complement_union (hU: U = {0, 1, 2, 3, 4}) (hA: A = {0, 1, 3}) (hB: B = {2, 3}) :
  (U \ (A ∪ B)) = {4} :=
sorry

end NUMINAMATH_GPT_complement_union_l1113_111356


namespace NUMINAMATH_GPT_paintable_wall_area_correct_l1113_111325

noncomputable def paintable_wall_area : Nat :=
  let length := 15
  let width := 11
  let height := 9
  let closet_width := 3
  let closet_length := 4
  let unused_area := 70
  let room_wall_area :=
    2 * (length * height) +
    2 * (width * height)
  let closet_wall_area := 
    2 * (closet_width * height)
  let paintable_area_per_bedroom := 
    room_wall_area - (unused_area + closet_wall_area)
  4 * paintable_area_per_bedroom

theorem paintable_wall_area_correct : paintable_wall_area = 1376 := by
  sorry

end NUMINAMATH_GPT_paintable_wall_area_correct_l1113_111325


namespace NUMINAMATH_GPT_train_length_correct_l1113_111341

-- Define the conditions
def bridge_length : ℝ := 180
def train_speed : ℝ := 15
def time_to_cross_bridge : ℝ := 20
def time_to_cross_man : ℝ := 8

-- Define the length of the train
def length_of_train : ℝ := 120

-- Proof statement
theorem train_length_correct :
  (train_speed * time_to_cross_man = length_of_train) ∧
  (train_speed * time_to_cross_bridge = length_of_train + bridge_length) :=
by
  sorry

end NUMINAMATH_GPT_train_length_correct_l1113_111341


namespace NUMINAMATH_GPT_range_of_m_l1113_111343

theorem range_of_m (m : ℝ) :
  let A := {x : ℝ | m + 1 ≤ x ∧ x ≤ 3 * m - 1}
  let B := {x : ℝ | 1 ≤ x ∧ x ≤ 10}
  (A ⊆ B) ↔ (m ≤ (11:ℝ)/3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_m_l1113_111343


namespace NUMINAMATH_GPT_arithmetic_sequence_26th_term_eq_neg48_l1113_111382

def arithmetic_sequence_term (a₁ d n : ℤ) : ℤ := a₁ + (n - 1) * d

theorem arithmetic_sequence_26th_term_eq_neg48 : 
  arithmetic_sequence_term 2 (-2) 26 = -48 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_26th_term_eq_neg48_l1113_111382


namespace NUMINAMATH_GPT_areasEqualForHexagonAndOctagon_l1113_111344

noncomputable def areaHexagon (s : ℝ) : ℝ :=
  let r := s / Real.sin (Real.pi / 6) -- Circumscribed radius
  let a := s / (2 * Real.tan (Real.pi / 6)) -- Inscribed radius
  Real.pi * (r^2 - a^2)

noncomputable def areaOctagon (s : ℝ) : ℝ :=
  let r := s / Real.sin (Real.pi / 8) -- Circumscribed radius
  let a := s / (2 * Real.tan (3 * Real.pi / 8)) -- Inscribed radius
  Real.pi * (r^2 - a^2)

theorem areasEqualForHexagonAndOctagon :
  let s := 3
  areaHexagon s = areaOctagon s := sorry

end NUMINAMATH_GPT_areasEqualForHexagonAndOctagon_l1113_111344


namespace NUMINAMATH_GPT_speed_of_bus_l1113_111330

def distance : ℝ := 500.04
def time : ℝ := 20.0
def conversion_factor : ℝ := 3.6

theorem speed_of_bus :
  (distance / time) * conversion_factor = 90.0072 := 
sorry

end NUMINAMATH_GPT_speed_of_bus_l1113_111330


namespace NUMINAMATH_GPT_ratio_apples_pie_to_total_is_one_to_two_l1113_111380

variable (x : ℕ) -- number of apples Paul put aside for pie
variable (total_apples : ℕ := 62) 
variable (fridge_apples : ℕ := 25)
variable (muffin_apples : ℕ := 6)

def apples_pie_ratio (x total_apples : ℕ) : ℕ := x / gcd x total_apples

theorem ratio_apples_pie_to_total_is_one_to_two :
  x + fridge_apples + muffin_apples = total_apples -> apples_pie_ratio x total_apples = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_apples_pie_to_total_is_one_to_two_l1113_111380


namespace NUMINAMATH_GPT_part1_part2_l1113_111369

def A (x y : ℝ) : ℝ := 3 * x ^ 2 + 2 * x * y - 2 * x - 1
def B (x y : ℝ) : ℝ := - x ^ 2 + x * y - 1

theorem part1 (x y : ℝ) : A x y + 3 * B x y = 5 * x * y - 2 * x - 4 := by
  sorry

theorem part2 (y : ℝ) : (∀ x : ℝ, 5 * x * y - 2 * x - 4 = -4) → y = 2 / 5 := by
  sorry

end NUMINAMATH_GPT_part1_part2_l1113_111369


namespace NUMINAMATH_GPT_remainder_of_7_pow_12_mod_100_l1113_111329

theorem remainder_of_7_pow_12_mod_100 : (7 ^ 12) % 100 = 1 := 
by sorry

end NUMINAMATH_GPT_remainder_of_7_pow_12_mod_100_l1113_111329


namespace NUMINAMATH_GPT_product_not_end_in_1999_l1113_111335

theorem product_not_end_in_1999 (a b c d e : ℕ) (h : a + b + c + d + e = 200) : 
  ¬(a * b * c * d * e % 10000 = 1999) := 
by
  sorry

end NUMINAMATH_GPT_product_not_end_in_1999_l1113_111335


namespace NUMINAMATH_GPT_part_I_part_II_l1113_111336

def sequence_def (x : ℕ → ℝ) (p : ℝ) : Prop :=
  x 1 = 1 ∧ ∀ n ∈ (Nat.succ <$> {n | n > 0}), x (n + 1) = 1 + x n / (p + x n)

theorem part_I (x : ℕ → ℝ) (p : ℝ) (h_seq : sequence_def x p) :
  p = 2 → ∀ n ∈ (Nat.succ <$> {n | n > 0}), x n < Real.sqrt 2 :=
sorry

theorem part_II (x : ℕ → ℝ) (p : ℝ) (h_seq : sequence_def x p) :
  (∀ n ∈ (Nat.succ <$> {n | n > 0}), x (n + 1) > x n) → ¬ ∃ M ∈ {n | n > 0}, ∀ n > 0, x M ≥ x n :=
sorry

end NUMINAMATH_GPT_part_I_part_II_l1113_111336


namespace NUMINAMATH_GPT_total_end_of_year_students_l1113_111398

theorem total_end_of_year_students :
  let start_fourth := 33
  let start_fifth := 45
  let start_sixth := 28
  let left_fourth := 18
  let joined_fourth := 14
  let left_fifth := 12
  let joined_fifth := 20
  let left_sixth := 10
  let joined_sixth := 16

  let end_fourth := start_fourth - left_fourth + joined_fourth
  let end_fifth := start_fifth - left_fifth + joined_fifth
  let end_sixth := start_sixth - left_sixth + joined_sixth
  
  end_fourth + end_fifth + end_sixth = 116 := by
    sorry

end NUMINAMATH_GPT_total_end_of_year_students_l1113_111398


namespace NUMINAMATH_GPT_remainder_m_squared_plus_4m_plus_6_l1113_111358

theorem remainder_m_squared_plus_4m_plus_6 (m : ℤ) (k : ℤ) (hk : m = 100 * k - 2) :
  (m ^ 2 + 4 * m + 6) % 100 = 2 := 
sorry

end NUMINAMATH_GPT_remainder_m_squared_plus_4m_plus_6_l1113_111358


namespace NUMINAMATH_GPT_triangle_area_l1113_111303

/-- The area of the triangle enclosed by a line with slope -1/2 passing through (2, -3) and the coordinate axes is 4. -/
theorem triangle_area {l : ℝ → ℝ} (h1 : ∀ x, l x = -1/2 * x + b)
  (h2 : l 2 = -3) : 
  ∃ (A : ℝ) (B : ℝ), 
  ((l 0 = B) ∧ (l A = 0) ∧ (A ≠ 0) ∧ (B ≠ 0)) ∧
  (1/2 * |A| * |B| = 4) := 
sorry

end NUMINAMATH_GPT_triangle_area_l1113_111303


namespace NUMINAMATH_GPT_license_plates_count_l1113_111373

-- Definitions from conditions
def num_digits : ℕ := 4
def num_digits_choices : ℕ := 10
def num_letters : ℕ := 3
def num_letters_choices : ℕ := 26

-- Define the blocks and their possible arrangements
def digits_permutations : ℕ := num_digits_choices^num_digits
def letters_permutations : ℕ := num_letters_choices^num_letters
def block_positions : ℕ := 5

-- We need to show that total possible license plates is 878,800,000.
def total_plates : ℕ := digits_permutations * letters_permutations * block_positions

-- The theorem statement
theorem license_plates_count :
  total_plates = 878800000 := by
  sorry

end NUMINAMATH_GPT_license_plates_count_l1113_111373


namespace NUMINAMATH_GPT_abs_inequality_solution_l1113_111319

theorem abs_inequality_solution (x : ℝ) : (|x - 1| < 2) ↔ (x > -1 ∧ x < 3) := 
sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1113_111319


namespace NUMINAMATH_GPT_sector_central_angle_l1113_111394

-- Defining the problem as a theorem in Lean 4
theorem sector_central_angle (r θ : ℝ) (h1 : 2 * r + r * θ = 4) (h2 : (1 / 2) * r^2 * θ = 1) : θ = 2 :=
by
  sorry

end NUMINAMATH_GPT_sector_central_angle_l1113_111394


namespace NUMINAMATH_GPT_num_squares_figure8_perimeter_figure12_perimeter_figureC_eq_38_ratio_perimeter_figure29_figureD_l1113_111348

-- Condition: Figure 1 is formed by 3 identical squares of side length 1 cm.
def squares_in_figure1 : ℕ := 3

-- Condition: Perimeter of Figure 1 is 8 cm.
def perimeter_figure1 : ℝ := 8

-- Condition: Each subsequent figure adds 2 squares.
def squares_in_figure (n : ℕ) : ℕ :=
  squares_in_figure1 + 2 * (n - 1)

-- Condition: Each subsequent figure increases perimeter by 2 cm.
def perimeter_figure (n : ℕ) : ℝ :=
  perimeter_figure1 + 2 * (n - 1)

-- Proof problem (a): Prove that the number of squares in Figure 8 is 17.
theorem num_squares_figure8 :
  squares_in_figure 8 = 17 :=
sorry

-- Proof problem (b): Prove that the perimeter of Figure 12 is 30 cm.
theorem perimeter_figure12 :
  perimeter_figure 12 = 30 :=
sorry

-- Proof problem (c): Prove that the positive integer C for which the perimeter of Figure C is 38 cm is 16.
theorem perimeter_figureC_eq_38 :
  ∃ C : ℕ, perimeter_figure C = 38 :=
sorry

-- Proof problem (d): Prove that the positive integer D for which the ratio of the perimeter of Figure 29 to the perimeter of Figure D is 4/11 is 85.
theorem ratio_perimeter_figure29_figureD :
  ∃ D : ℕ, (perimeter_figure 29 / perimeter_figure D) = (4 / 11) :=
sorry

end NUMINAMATH_GPT_num_squares_figure8_perimeter_figure12_perimeter_figureC_eq_38_ratio_perimeter_figure29_figureD_l1113_111348


namespace NUMINAMATH_GPT_diameter_increase_l1113_111350

theorem diameter_increase (h : 0.628 = π * d) : d = 0.2 := 
sorry

end NUMINAMATH_GPT_diameter_increase_l1113_111350


namespace NUMINAMATH_GPT_compute_fraction_sum_l1113_111381

variable (a b c : ℝ)
open Real

theorem compute_fraction_sum (h1 : (a * c) / (a + b) + (b * a) / (b + c) + (c * b) / (c + a) = -15)
                            (h2 : (b * c) / (a + b) + (c * a) / (b + c) + (a * b) / (c + a) = 6) :
  (b / (a + b) + c / (b + c) + a / (c + a)) = 12 := 
sorry

end NUMINAMATH_GPT_compute_fraction_sum_l1113_111381


namespace NUMINAMATH_GPT_train_speed_in_kmph_l1113_111302

def train_length : ℕ := 125
def time_to_cross_pole : ℕ := 9
def conversion_factor : ℚ := 18 / 5

theorem train_speed_in_kmph
  (d : ℕ := train_length)
  (t : ℕ := time_to_cross_pole)
  (cf : ℚ := conversion_factor) :
  d / t * cf = 50 := 
sorry

end NUMINAMATH_GPT_train_speed_in_kmph_l1113_111302


namespace NUMINAMATH_GPT_largest_m_dividing_factorials_l1113_111390

noncomputable def factorial (n : ℕ) : ℕ :=
if h : n = 0 then 1 else n * factorial (n - 1)

theorem largest_m_dividing_factorials (m : ℕ) :
  (∀ k : ℕ, k ≤ m → factorial k ∣ (factorial 100 + factorial 99 + factorial 98)) ↔ m = 98 :=
by
  sorry

end NUMINAMATH_GPT_largest_m_dividing_factorials_l1113_111390


namespace NUMINAMATH_GPT_problem_divisibility_l1113_111320

theorem problem_divisibility (a : ℤ) (h1 : 0 ≤ a) (h2 : a ≤ 13) (h3 : (51^2012 + a) % 13 = 0) : a = 12 :=
by
  sorry

end NUMINAMATH_GPT_problem_divisibility_l1113_111320


namespace NUMINAMATH_GPT_jack_mopping_time_l1113_111312

-- Definitions for the conditions
def bathroom_area : ℝ := 24
def kitchen_area : ℝ := 80
def mopping_rate : ℝ := 8

-- The proof problem: Prove Jack will spend 13 minutes mopping
theorem jack_mopping_time : (bathroom_area + kitchen_area) / mopping_rate = 13 := by
  sorry

end NUMINAMATH_GPT_jack_mopping_time_l1113_111312


namespace NUMINAMATH_GPT_debby_bottles_l1113_111323

noncomputable def number_of_bottles_initial : ℕ := 301
noncomputable def number_of_bottles_drank : ℕ := 144
noncomputable def number_of_bottles_left : ℕ := 157

theorem debby_bottles:
  (number_of_bottles_initial - number_of_bottles_drank) = number_of_bottles_left :=
sorry

end NUMINAMATH_GPT_debby_bottles_l1113_111323


namespace NUMINAMATH_GPT_combined_area_of_triangles_l1113_111354

noncomputable def area_of_rectangle (length width : ℝ) : ℝ :=
  length * width

noncomputable def first_triangle_area (x : ℝ) : ℝ :=
  5 * x

noncomputable def second_triangle_area (base height : ℝ) : ℝ :=
  (base * height) / 2

theorem combined_area_of_triangles (length width x base height : ℝ)
  (h1 : area_of_rectangle length width / first_triangle_area x = 2 / 5)
  (h2 : base + height = 20)
  (h3 : second_triangle_area base height / first_triangle_area x = 3 / 5)
  (length_value : length = 6)
  (width_value : width = 4)
  (base_value : base = 8) :
  first_triangle_area x + second_triangle_area base height = 108 := 
by
  sorry

end NUMINAMATH_GPT_combined_area_of_triangles_l1113_111354


namespace NUMINAMATH_GPT_sports_club_problem_l1113_111306

theorem sports_club_problem (total_members : ℕ) (members_playing_badminton : ℕ) 
  (members_playing_tennis : ℕ) (members_not_playing_either : ℕ) 
  (h_total_members : total_members = 100) (h_badminton : members_playing_badminton = 60) 
  (h_tennis : members_playing_tennis = 70) (h_neither : members_not_playing_either = 10) : 
  (members_playing_badminton + members_playing_tennis - 
   (total_members - members_not_playing_either) = 40) :=
by {
  sorry
}

end NUMINAMATH_GPT_sports_club_problem_l1113_111306


namespace NUMINAMATH_GPT_negative_solution_condition_l1113_111361

-- Define the system of equations
def system_of_equations (a b c x y : ℝ) :=
  a * x + b * y = c ∧
  b * x + c * y = a ∧
  c * x + a * y = b

-- State the theorem
theorem negative_solution_condition (a b c : ℝ) :
  (∃ x y : ℝ, x < 0 ∧ y < 0 ∧ system_of_equations a b c x y) ↔ a + b + c = 0 :=
by
  sorry

end NUMINAMATH_GPT_negative_solution_condition_l1113_111361


namespace NUMINAMATH_GPT_Anthony_vs_Jim_l1113_111331

variable (Scott_pairs : ℕ)
variable (Anthony_pairs : ℕ)
variable (Jim_pairs : ℕ)

axiom Scott_value : Scott_pairs = 7
axiom Anthony_value : Anthony_pairs = 3 * Scott_pairs
axiom Jim_value : Jim_pairs = Anthony_pairs - 2

theorem Anthony_vs_Jim (Scott_pairs Anthony_pairs Jim_pairs : ℕ) 
  (Scott_value : Scott_pairs = 7) 
  (Anthony_value : Anthony_pairs = 3 * Scott_pairs) 
  (Jim_value : Jim_pairs = Anthony_pairs - 2) :
  Anthony_pairs - Jim_pairs = 2 := 
sorry

end NUMINAMATH_GPT_Anthony_vs_Jim_l1113_111331


namespace NUMINAMATH_GPT_possible_values_of_q_l1113_111359

theorem possible_values_of_q {q : ℕ} (hq : q > 0) :
  (∃ k : ℕ, (5 * q + 35) = k * (3 * q - 7) ∧ k > 0) ↔
  q = 3 ∨ q = 4 ∨ q = 5 ∨ q = 7 ∨ q = 9 ∨ q = 15 ∨ q = 21 ∨ q = 31 :=
by
  sorry

end NUMINAMATH_GPT_possible_values_of_q_l1113_111359


namespace NUMINAMATH_GPT_sqrt_sum_inequality_l1113_111321

-- Define variables a and b as positive real numbers
variable {a b : ℝ}

-- State the theorem to be proved
theorem sqrt_sum_inequality (ha : 0 < a) (hb : 0 < b) : 
  (a.sqrt + b.sqrt)^8 ≥ 64 * a * b * (a + b)^2 :=
sorry

end NUMINAMATH_GPT_sqrt_sum_inequality_l1113_111321


namespace NUMINAMATH_GPT_find_three_numbers_l1113_111347

theorem find_three_numbers :
  ∃ (x1 x2 x3 k1 k2 k3 : ℕ),
  x1 = 2500 * k1 / (3^k1 - 1) ∧
  x2 = 2500 * k2 / (3^k2 - 1) ∧
  x3 = 2500 * k3 / (3^k3 - 1) ∧
  k1 ≠ k2 ∧ k1 ≠ k3 ∧ k2 ≠ k3 ∧
  x1 ≠ x2 ∧ x1 ≠ x3 ∧ x2 ≠ x3 :=
by
  sorry

end NUMINAMATH_GPT_find_three_numbers_l1113_111347


namespace NUMINAMATH_GPT_calc_expression_l1113_111383

theorem calc_expression : (3.242 * 14) / 100 = 0.45388 :=
by
  sorry

end NUMINAMATH_GPT_calc_expression_l1113_111383


namespace NUMINAMATH_GPT_cost_of_5kg_l1113_111391

def cost_of_seeds (x : ℕ) : ℕ :=
  if x ≤ 2 then 5 * x else 4 * x + 2

theorem cost_of_5kg : cost_of_seeds 5 = 22 := by
  sorry

end NUMINAMATH_GPT_cost_of_5kg_l1113_111391


namespace NUMINAMATH_GPT_min_value_of_a_l1113_111374

theorem min_value_of_a 
  (a b x1 x2 : ℕ) 
  (h1 : a = b - 2005) 
  (h2 : (x1 + x2) = a) 
  (h3 : (x1 * x2) = b) 
  (h4 : x1 > 0 ∧ x2 > 0) : 
  a ≥ 95 :=
sorry

end NUMINAMATH_GPT_min_value_of_a_l1113_111374


namespace NUMINAMATH_GPT_marble_ratio_l1113_111393

theorem marble_ratio
  (L_b : ℕ) (J_y : ℕ) (A : ℕ)
  (A_b : ℕ) (A_y : ℕ) (R : ℕ)
  (h1 : L_b = 4)
  (h2 : J_y = 22)
  (h3 : A = 19)
  (h4 : A_y = J_y / 2)
  (h5 : A = A_b + A_y)
  (h6 : A_b = L_b * R) :
  R = 2 := by
  sorry

end NUMINAMATH_GPT_marble_ratio_l1113_111393


namespace NUMINAMATH_GPT_find_number_l1113_111342

theorem find_number (x : ℝ) (h : 0.8 * x = (2/5 : ℝ) * 25 + 22) : x = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1113_111342


namespace NUMINAMATH_GPT_disc_thickness_l1113_111353

theorem disc_thickness (r_sphere : ℝ) (r_disc : ℝ) (h : ℝ)
  (h_radius_sphere : r_sphere = 3)
  (h_radius_disc : r_disc = 10)
  (h_volume_constant : (4/3) * Real.pi * r_sphere^3 = Real.pi * r_disc^2 * h) :
  h = 9 / 25 :=
by
  sorry

end NUMINAMATH_GPT_disc_thickness_l1113_111353


namespace NUMINAMATH_GPT_infinite_not_expressible_as_sum_of_three_squares_l1113_111372

theorem infinite_not_expressible_as_sum_of_three_squares :
  ∃ (n : ℕ), ∃ (infinitely_many_n : ℕ → Prop), (∀ m:ℕ, (infinitely_many_n m ↔ m ≡ 7 [MOD 8])) ∧ ∀ a b c : ℕ, n ≠ a^2 + b^2 + c^2 := 
by
  sorry

end NUMINAMATH_GPT_infinite_not_expressible_as_sum_of_three_squares_l1113_111372


namespace NUMINAMATH_GPT_expected_waiting_time_correct_l1113_111316

noncomputable def combined_average_bites_per_5_minutes := 6
def average_waiting_time_for_first_bite_in_seconds : ℝ := 50

theorem expected_waiting_time_correct :
  (1 / combined_average_bites_per_5_minutes) * 300 = average_waiting_time_for_first_bite_in_seconds :=
by
  sorry

end NUMINAMATH_GPT_expected_waiting_time_correct_l1113_111316


namespace NUMINAMATH_GPT_eight_digit_number_divisible_by_101_l1113_111385

def repeat_twice (x : ℕ) : ℕ := 100 * x + x

theorem eight_digit_number_divisible_by_101 (ef gh ij kl : ℕ) 
  (hef : ef < 100) (hgh : gh < 100) (hij : ij < 100) (hkl : kl < 100) :
  (100010001 * repeat_twice ef + 1000010 * repeat_twice gh + 10010 * repeat_twice ij + 10 * repeat_twice kl) % 101 = 0 := sorry

end NUMINAMATH_GPT_eight_digit_number_divisible_by_101_l1113_111385


namespace NUMINAMATH_GPT_minimum_value_128_l1113_111389

theorem minimum_value_128 (a b c : ℝ) (h_pos: a > 0 ∧ b > 0 ∧ c > 0) (h_prod: a * b * c = 8) : 
  (2 * a + b) * (a + 3 * c) * (b * c + 2) ≥ 128 := 
by
  sorry

end NUMINAMATH_GPT_minimum_value_128_l1113_111389


namespace NUMINAMATH_GPT_power_calc_l1113_111337

theorem power_calc : (3^2)^4 = 6561 := 
by
  sorry

end NUMINAMATH_GPT_power_calc_l1113_111337
