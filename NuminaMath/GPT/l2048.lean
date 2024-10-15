import Mathlib

namespace NUMINAMATH_GPT_slant_height_of_cone_l2048_204800

theorem slant_height_of_cone (r : ℝ) (h : ℝ) (s : ℝ) (unfolds_to_semicircle : s = π) (base_radius : r = 1) : s = 2 :=
by
  sorry

end NUMINAMATH_GPT_slant_height_of_cone_l2048_204800


namespace NUMINAMATH_GPT_average_last_12_results_l2048_204835

theorem average_last_12_results (S25 S12 S_last12 : ℕ) (A : ℕ) 
  (h1 : S25 = 25 * 24) 
  (h2: S12 = 12 * 14) 
  (h3: 12 * A = S_last12)
  (h4: S25 = S12 + 228 + S_last12) : A = 17 := 
by
  sorry

end NUMINAMATH_GPT_average_last_12_results_l2048_204835


namespace NUMINAMATH_GPT_average_age_of_dance_group_l2048_204863

theorem average_age_of_dance_group (S_f S_m : ℕ) (avg_females avg_males : ℕ) 
(hf : avg_females = S_f / 12) (hm : avg_males = S_m / 18) 
(h1 : avg_females = 25) (h2 : avg_males = 40) : 
  (S_f + S_m) / 30 = 34 :=
by
  sorry

end NUMINAMATH_GPT_average_age_of_dance_group_l2048_204863


namespace NUMINAMATH_GPT_find_n_l2048_204894

theorem find_n (n : ℤ) (hn : -180 ≤ n ∧ n ≤ 180) (hsin : Real.sin (n * Real.pi / 180) = Real.sin (750 * Real.pi / 180)) :
  n = 30 ∨ n = 150 ∨ n = -30 ∨ n = -150 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l2048_204894


namespace NUMINAMATH_GPT_perfect_cubes_in_range_l2048_204873

theorem perfect_cubes_in_range :
  ∃ (n : ℕ), (∀ (k : ℕ), (50 < k^3 ∧ k^3 ≤ 1000) → (k = 4 ∨ k = 5 ∨ k = 6 ∨ k = 7 ∨ k = 8 ∨ k = 9 ∨ k = 10)) ∧
    (∃ m, (m = 7)) :=
by
  sorry

end NUMINAMATH_GPT_perfect_cubes_in_range_l2048_204873


namespace NUMINAMATH_GPT_certain_number_division_l2048_204849

theorem certain_number_division (N G : ℤ) : 
  G = 88 ∧ (∃ k : ℤ, N = G * k + 31) ∧ (∃ m : ℤ, 4521 = G * m + 33) → 
  N = 4519 := 
by
  sorry

end NUMINAMATH_GPT_certain_number_division_l2048_204849


namespace NUMINAMATH_GPT_domain_of_f_l2048_204858

theorem domain_of_f :
  (∀ x : ℝ, (0 < 1 - x) ∧ (0 < 3 * x + 1) ↔ ( - (1 / 3 : ℝ) < x ∧ x < 1)) :=
by
  sorry

end NUMINAMATH_GPT_domain_of_f_l2048_204858


namespace NUMINAMATH_GPT_smallest_number_property_l2048_204862

theorem smallest_number_property : 
  ∃ n, ((n - 7) % 12 = 0) ∧ ((n - 7) % 16 = 0) ∧ ((n - 7) % 18 = 0) ∧ ((n - 7) % 21 = 0) ∧ ((n - 7) % 28 = 0) ∧ n = 1015 :=
by
  sorry  -- Proof is omitted

end NUMINAMATH_GPT_smallest_number_property_l2048_204862


namespace NUMINAMATH_GPT_range_of_m_l2048_204803

theorem range_of_m : 3 < 3 * Real.sqrt 2 - 1 ∧ 3 * Real.sqrt 2 - 1 < 4 :=
by
  have h1 : 3 * Real.sqrt 2 < 5 := sorry
  have h2 : 4 < 3 * Real.sqrt 2 := sorry
  exact ⟨by linarith, by linarith⟩

end NUMINAMATH_GPT_range_of_m_l2048_204803


namespace NUMINAMATH_GPT_length_ST_l2048_204839

theorem length_ST (PQ QR RS SP SQ PT RT : ℝ) 
  (h1 : PQ = 6) (h2 : QR = 6)
  (h3 : RS = 6) (h4 : SP = 6)
  (h5 : SQ = 6) (h6 : PT = 14)
  (h7 : RT = 14) : 
  ∃ ST : ℝ, ST = 10 := 
by
  -- sorry is used to complete the theorem without a proof
  sorry

end NUMINAMATH_GPT_length_ST_l2048_204839


namespace NUMINAMATH_GPT_shoveling_driveways_l2048_204811

-- Definitions of the conditions
def cost_of_candy_bars := 2 * 0.75
def cost_of_lollipops := 4 * 0.25
def total_cost := cost_of_candy_bars + cost_of_lollipops
def portion_of_earnings := total_cost * 6
def charge_per_driveway := 1.50
def number_of_driveways := portion_of_earnings / charge_per_driveway

-- The theorem to prove Jimmy shoveled 10 driveways
theorem shoveling_driveways :
  number_of_driveways = 10 := 
by
  sorry

end NUMINAMATH_GPT_shoveling_driveways_l2048_204811


namespace NUMINAMATH_GPT_division_quotient_remainder_l2048_204879

theorem division_quotient_remainder (A : ℕ) (h1 : A / 9 = 2) (h2 : A % 9 = 6) : A = 24 := 
by
  sorry

end NUMINAMATH_GPT_division_quotient_remainder_l2048_204879


namespace NUMINAMATH_GPT_smallest_positive_angle_same_terminal_side_l2048_204852

theorem smallest_positive_angle_same_terminal_side : 
  ∃ k : ℤ, (∃ α : ℝ, α > 0 ∧ α = -660 + k * 360) ∧ (∀ β : ℝ, β > 0 ∧ β = -660 + k * 360 → β ≥ α) :=
sorry

end NUMINAMATH_GPT_smallest_positive_angle_same_terminal_side_l2048_204852


namespace NUMINAMATH_GPT_quadratic_inequality_solution_l2048_204881

theorem quadratic_inequality_solution
  (a b c : ℝ)
  (h1: ∀ x : ℝ, (-1/3 < x ∧ x < 2) → (ax^2 + bx + c) > 0)
  (h2: a < 0):
  ∀ x : ℝ, ((-3 < x ∧ x < 1/2) ↔ (cx^2 + bx + a) < 0) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_l2048_204881


namespace NUMINAMATH_GPT_richmond_more_than_victoria_l2048_204818

-- Defining the population of Beacon
def beacon_people : ℕ := 500

-- Defining the population of Victoria based on Beacon's population
def victoria_people : ℕ := 4 * beacon_people

-- Defining the population of Richmond
def richmond_people : ℕ := 3000

-- The proof problem: calculating the difference
theorem richmond_more_than_victoria : richmond_people - victoria_people = 1000 := by
  -- The statement of the theorem
  sorry

end NUMINAMATH_GPT_richmond_more_than_victoria_l2048_204818


namespace NUMINAMATH_GPT_discriminant_nonnegative_l2048_204832

theorem discriminant_nonnegative (x : ℤ) (h : x^2 * (25 - 24 * x^2) ≥ 0) : x = 0 ∨ x = 1 ∨ x = -1 :=
by sorry

end NUMINAMATH_GPT_discriminant_nonnegative_l2048_204832


namespace NUMINAMATH_GPT_range_of_a_l2048_204801

noncomputable def quadratic_inequality_condition (a : ℝ) (x : ℝ) : Prop :=
  x^2 - 2 * (a - 2) * x + a > 0

theorem range_of_a :
  (∀ x : ℝ, (x < 1 ∨ x > 5) → quadratic_inequality_condition a x) ↔ (1 < a ∧ a ≤ 5) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2048_204801


namespace NUMINAMATH_GPT_simplify_expression_l2048_204802

open Real

theorem simplify_expression (a b : ℝ) 
  (h1 : b ≠ 0) (h2 : b ≠ -3 * a) (h3 : b ≠ a) (h4 : b ≠ -a) : 
  ((2 * b + a - (4 * a ^ 2 - b ^ 2) / a) / (b ^ 3 + 2 * a * b ^ 2 - 3 * a ^ 2 * b)) *
  ((a ^ 3 * b - 2 * a ^ 2 * b ^ 2 + a * b ^ 3) / (a ^ 2 - b ^ 2)) = 
  (a - b) / (a + b) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l2048_204802


namespace NUMINAMATH_GPT_quadratic_roots_square_cube_sum_l2048_204828

theorem quadratic_roots_square_cube_sum
  (a b c : ℝ) (h : a ≠ 0) (x1 x2 : ℝ)
  (hx : ∀ (x : ℝ), a * x^2 + b * x + c = 0 ↔ x = x1 ∨ x = x2) :
  (x1^2 + x2^2 = (b^2 - 2 * a * c) / a^2) ∧
  (x1^3 + x2^3 = (3 * a * b * c - b^3) / a^3) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_roots_square_cube_sum_l2048_204828


namespace NUMINAMATH_GPT_ben_bonus_amount_l2048_204860

variables (B : ℝ)

-- Conditions
def condition1 := B - (1/22) * B - (1/4) * B - (1/8) * B = 867

-- Theorem statement
theorem ben_bonus_amount (h : condition1 B) : B = 1496.50 := 
sorry

end NUMINAMATH_GPT_ben_bonus_amount_l2048_204860


namespace NUMINAMATH_GPT_range_of_a_l2048_204867

noncomputable def f (a x : ℝ) : ℝ := (a * x + 1) / (x + 2)

theorem range_of_a (a : ℝ) : (∀ x y : ℝ, -2 < x → -2 < y → x < y → f a x < f a y) → (a > 1/2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2048_204867


namespace NUMINAMATH_GPT_point_relationship_l2048_204825

variable {m : ℝ}

theorem point_relationship
    (hA : ∃ y1 : ℝ, y1 = (-4 : ℝ)^2 - 2 * (-4 : ℝ) + m)
    (hB : ∃ y2 : ℝ, y2 = (0 : ℝ)^2 - 2 * (0 : ℝ) + m)
    (hC : ∃ y3 : ℝ, y3 = (3 : ℝ)^2 - 2 * (3 : ℝ) + m) :
    (∃ y2 y3 y1 : ℝ, y2 < y3 ∧ y3 < y1) := by
  sorry

end NUMINAMATH_GPT_point_relationship_l2048_204825


namespace NUMINAMATH_GPT_compound_statement_false_l2048_204898

theorem compound_statement_false (p q : Prop) (h : ¬ (p ∧ q)) : ¬ p ∨ ¬ q :=
sorry

end NUMINAMATH_GPT_compound_statement_false_l2048_204898


namespace NUMINAMATH_GPT_total_clothes_l2048_204884

-- Defining the conditions
def shirts := 12
def pants := 5 * shirts
def shorts := (1 / 4) * pants

-- Theorem to prove the total number of pieces of clothes
theorem total_clothes : shirts + pants + shorts = 87 := by
  -- using sorry to skip the proof
  sorry

end NUMINAMATH_GPT_total_clothes_l2048_204884


namespace NUMINAMATH_GPT_total_number_of_toys_l2048_204861

def jaxon_toys := 15
def gabriel_toys := 2 * jaxon_toys
def jerry_toys := gabriel_toys + 8

theorem total_number_of_toys : jaxon_toys + gabriel_toys + jerry_toys = 83 := by
  sorry

end NUMINAMATH_GPT_total_number_of_toys_l2048_204861


namespace NUMINAMATH_GPT_number_of_laborers_in_crew_l2048_204896

theorem number_of_laborers_in_crew (present : ℕ) (percentage : ℝ) (total : ℕ) 
    (h1 : present = 70) (h2 : percentage = 44.9 / 100) (h3 : present = percentage * total) : 
    total = 156 := 
sorry

end NUMINAMATH_GPT_number_of_laborers_in_crew_l2048_204896


namespace NUMINAMATH_GPT_min_value_of_fraction_sum_l2048_204820

theorem min_value_of_fraction_sum (a b : ℤ) (h1 : a = b + 1) : 
  (a > b) -> (∃ x, x > 0 ∧ ((a + b) / (a - b) + (a - b) / (a + b)) = 2) :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_fraction_sum_l2048_204820


namespace NUMINAMATH_GPT_eq_of_frac_eq_and_neq_neg_one_l2048_204899

theorem eq_of_frac_eq_and_neq_neg_one
  (a b c d : ℝ)
  (h : (a + b) / (c + d) = (b + c) / (a + d))
  (h_neq : (a + b) / (c + d) ≠ -1) :
  a = c :=
sorry

end NUMINAMATH_GPT_eq_of_frac_eq_and_neq_neg_one_l2048_204899


namespace NUMINAMATH_GPT_distance_between_foci_l2048_204865

-- Define the given ellipse equation.
def ellipse_eq (x y : ℝ) : Prop := 9 * x^2 + y^2 = 144

-- Provide the values of the semi-major and semi-minor axes.
def a : ℝ := 12
def b : ℝ := 4

-- Define the equation for calculating the distance between the foci.
noncomputable def focal_distance (a b : ℝ) : ℝ :=
  Real.sqrt (a^2 - b^2)

-- The theorem we need to prove.
theorem distance_between_foci : focal_distance a b = 8 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_GPT_distance_between_foci_l2048_204865


namespace NUMINAMATH_GPT_inequality_solution_l2048_204877

theorem inequality_solution 
  (x : ℝ) 
  (h : 2*x^4 + x^2 - 4*x - 3*x^2 * |x - 2| + 4 ≥ 0) : 
  x ∈ Set.Iic (-2) ∪ Set.Icc ((-1 - Real.sqrt 17) / 4) ((-1 + Real.sqrt 17) / 4) ∪ Set.Ici 1 :=
sorry

end NUMINAMATH_GPT_inequality_solution_l2048_204877


namespace NUMINAMATH_GPT_largest_B_at_45_l2048_204841

def binomial_coeff (n k : ℕ) : ℕ := Nat.choose n k

def B (k : ℕ) : ℝ :=
  if k ≤ 500 then (binomial_coeff 500 k) * (0.1)^k else 0

theorem largest_B_at_45 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ 500 → B k ≤ B 45 :=
by
  intros k hk
  sorry

end NUMINAMATH_GPT_largest_B_at_45_l2048_204841


namespace NUMINAMATH_GPT_remainder_sum_modulo_l2048_204886

theorem remainder_sum_modulo :
  (9156 + 9157 + 9158 + 9159 + 9160) % 9 = 7 :=
by
sorry

end NUMINAMATH_GPT_remainder_sum_modulo_l2048_204886


namespace NUMINAMATH_GPT_factor_expression_eq_l2048_204838

theorem factor_expression_eq (x : ℤ) : 75 * x + 50 = 25 * (3 * x + 2) :=
by
  -- The actual proof is omitted
  sorry

end NUMINAMATH_GPT_factor_expression_eq_l2048_204838


namespace NUMINAMATH_GPT_quadratic_intersects_once_l2048_204842

theorem quadratic_intersects_once (c : ℝ) : (∀ x : ℝ, x^2 - 6 * x + c = 0 → x = 3 ) ↔ c = 9 :=
by
  sorry

end NUMINAMATH_GPT_quadratic_intersects_once_l2048_204842


namespace NUMINAMATH_GPT_simplified_expression_num_terms_l2048_204836

noncomputable def num_terms_polynomial (n: ℕ) : ℕ :=
  (n/2) * (1 + (n+1))

theorem simplified_expression_num_terms :
  num_terms_polynomial 2012 = 1012608 :=
by
  sorry

end NUMINAMATH_GPT_simplified_expression_num_terms_l2048_204836


namespace NUMINAMATH_GPT_khali_total_snow_volume_l2048_204813

def length1 : ℝ := 25
def width1 : ℝ := 3
def depth1 : ℝ := 0.75

def length2 : ℝ := 15
def width2 : ℝ := 3
def depth2 : ℝ := 1

def volume1 : ℝ := length1 * width1 * depth1
def volume2 : ℝ := length2 * width2 * depth2
def total_volume : ℝ := volume1 + volume2

theorem khali_total_snow_volume : total_volume = 101.25 := by
  sorry

end NUMINAMATH_GPT_khali_total_snow_volume_l2048_204813


namespace NUMINAMATH_GPT_surface_area_with_holes_l2048_204851

-- Define the cube and holes properties
def edge_length_cube : ℝ := 4
def side_length_hole : ℝ := 2
def number_faces_cube : ℕ := 6

-- Define areas
def area_face_cube := edge_length_cube ^ 2
def area_face_hole := side_length_hole ^ 2
def original_surface_area := number_faces_cube * area_face_cube
def total_hole_area := number_faces_cube * area_face_hole
def new_exposed_area := number_faces_cube * 4 * area_face_hole

-- Calculate the total surface area including holes
def total_surface_area := original_surface_area - total_hole_area + new_exposed_area

-- Lean statement for the proof
theorem surface_area_with_holes :
  total_surface_area = 168 := by
  sorry

end NUMINAMATH_GPT_surface_area_with_holes_l2048_204851


namespace NUMINAMATH_GPT_tangent_line_to_parabola_k_value_l2048_204816

theorem tangent_line_to_parabola_k_value (k : ℝ) :
  (∀ x y : ℝ, 4 * x - 3 * y + k = 0 → y^2 = 16 * x → (4 * x - 3 * y + k = 0 ∧ y^2 = 16 * x) ∧ (144 - 16 * k = 0)) → k = 9 :=
by
  sorry

end NUMINAMATH_GPT_tangent_line_to_parabola_k_value_l2048_204816


namespace NUMINAMATH_GPT_added_water_proof_l2048_204827

variable (total_volume : ℕ) (milk_ratio water_ratio : ℕ) (added_water : ℕ)

theorem added_water_proof 
  (h1 : total_volume = 45) 
  (h2 : milk_ratio = 4) 
  (h3 : water_ratio = 1) 
  (h4 : added_water = 3) 
  (milk_volume : ℕ)
  (water_volume : ℕ)
  (h5 : milk_volume = (milk_ratio * total_volume) / (milk_ratio + water_ratio))
  (h6 : water_volume = (water_ratio * total_volume) / (milk_ratio + water_ratio))
  (new_ratio : ℕ)
  (h7 : new_ratio = milk_volume / (water_volume + added_water)) : added_water = 3 :=
by
  sorry

end NUMINAMATH_GPT_added_water_proof_l2048_204827


namespace NUMINAMATH_GPT_quadrilateral_sum_of_squares_l2048_204834

theorem quadrilateral_sum_of_squares
  (a b c d m n t : ℝ) : 
  a^2 + b^2 + c^2 + d^2 = m^2 + n^2 + 4 * t^2 :=
sorry

end NUMINAMATH_GPT_quadrilateral_sum_of_squares_l2048_204834


namespace NUMINAMATH_GPT_cost_to_make_each_pop_l2048_204826

-- Define the conditions as given in step a)
def selling_price : ℝ := 1.50
def pops_sold : ℝ := 300
def pencil_cost : ℝ := 1.80
def pencils_to_buy : ℝ := 100

-- Define the total revenue from selling the ice-pops
def total_revenue : ℝ := pops_sold * selling_price

-- Define the total cost to buy the pencils
def total_pencil_cost : ℝ := pencils_to_buy * pencil_cost

-- Define the total profit
def total_profit : ℝ := total_revenue - total_pencil_cost

-- Define the cost to make each ice-pop
theorem cost_to_make_each_pop : total_profit / pops_sold = 0.90 :=
by
  sorry

end NUMINAMATH_GPT_cost_to_make_each_pop_l2048_204826


namespace NUMINAMATH_GPT_find_x_for_mean_l2048_204889

theorem find_x_for_mean 
(x : ℝ) 
(h_mean : (3 + 11 + 7 + 9 + 15 + 13 + 8 + 19 + 17 + 21 + 14 + x) / 12 = 12) : 
x = 7 :=
sorry

end NUMINAMATH_GPT_find_x_for_mean_l2048_204889


namespace NUMINAMATH_GPT_arithmetic_progression_sum_squares_l2048_204890

theorem arithmetic_progression_sum_squares (a1 a2 a3 : ℚ)
  (h1 : a2 = (a1 + a3) / 2)
  (h2 : a1 + a2 + a3 = 2)
  (h3 : a1^2 + a2^2 + a3^2 = 14/9) :
  (a1 = 1/3 ∧ a2 = 2/3 ∧ a3 = 1) ∨ (a1 = 1 ∧ a2 = 2/3 ∧ a3 = 1/3) :=
sorry

end NUMINAMATH_GPT_arithmetic_progression_sum_squares_l2048_204890


namespace NUMINAMATH_GPT_cupcakes_left_l2048_204843

def pack_count := 3
def cupcakes_per_pack := 4
def cupcakes_eaten := 5

theorem cupcakes_left : (pack_count * cupcakes_per_pack - cupcakes_eaten) = 7 := 
by 
  sorry

end NUMINAMATH_GPT_cupcakes_left_l2048_204843


namespace NUMINAMATH_GPT_intersection_with_x_axis_l2048_204864

theorem intersection_with_x_axis :
  (∃ x, ∃ y, y = 0 ∧ y = -3 * x + 3 ∧ (x = 1 ∧ y = 0)) :=
by
  -- proof will go here
  sorry

end NUMINAMATH_GPT_intersection_with_x_axis_l2048_204864


namespace NUMINAMATH_GPT_no_square_divisible_by_six_in_range_l2048_204882

theorem no_square_divisible_by_six_in_range : ¬ ∃ y : ℕ, (∃ k : ℕ, y = k * k) ∧ (6 ∣ y) ∧ (50 ≤ y ∧ y ≤ 120) :=
by
  sorry

end NUMINAMATH_GPT_no_square_divisible_by_six_in_range_l2048_204882


namespace NUMINAMATH_GPT_cubic_polynomial_roots_product_l2048_204872

theorem cubic_polynomial_roots_product :
  (∃ a b c : ℝ, (3*a^3 - 9*a^2 + 5*a - 15 = 0) ∧
               (3*b^3 - 9*b^2 + 5*b - 15 = 0) ∧
               (3*c^3 - 9*c^2 + 5*c - 15 = 0) ∧
               a ≠ b ∧ b ≠ c ∧ c ≠ a) →
  ∃ a b c : ℝ, (3*a*b*c = 5) := 
sorry

end NUMINAMATH_GPT_cubic_polynomial_roots_product_l2048_204872


namespace NUMINAMATH_GPT_midpoint_polar_coords_l2048_204871

/-- 
Given two points in polar coordinates: (6, π/6) and (2, -π/6),  
the midpoint of the line segment connecting these points in polar coordinates is (√13, π/6).
-/
theorem midpoint_polar_coords :
  let A := (6, Real.pi / 6)
  let B := (2, -Real.pi / 6)
  let A_cart := (6 * Real.cos (Real.pi / 6), 6 * Real.sin (Real.pi / 6))
  let B_cart := (2 * Real.cos (-Real.pi / 6), 2 * Real.sin (-Real.pi / 6))
  let Mx := ((A_cart.fst + B_cart.fst) / 2)
  let My := ((A_cart.snd + B_cart.snd) / 2)
  let r := Real.sqrt (Mx^2 + My^2)
  let theta := Real.arctan (My / Mx)
  0 <= theta ∧ theta < 2 * Real.pi ∧ r > 0 ∧ (r = Real.sqrt 13 ∧ theta = Real.pi / 6) :=
by 
  sorry

end NUMINAMATH_GPT_midpoint_polar_coords_l2048_204871


namespace NUMINAMATH_GPT_possible_perimeters_l2048_204837

-- Define the condition that the side lengths satisfy the equation
def sides_satisfy_eqn (x : ℝ) : Prop := x^2 - 6 * x + 8 = 0

-- Theorem to prove the possible perimeters
theorem possible_perimeters (x y z : ℝ) (h1 : sides_satisfy_eqn x) (h2 : sides_satisfy_eqn y) (h3 : sides_satisfy_eqn z) :
  (x + y + z = 10) ∨ (x + y + z = 6) ∨ (x + y + z = 12) := by
  sorry

end NUMINAMATH_GPT_possible_perimeters_l2048_204837


namespace NUMINAMATH_GPT_find_a_l2048_204821

theorem find_a (a b : ℤ) 
  (h1: 4181 * a + 2584 * b = 0 ) 
  (h2: 2584 * a + 1597 * b = -1) 
: a = 1597 := 
sorry

end NUMINAMATH_GPT_find_a_l2048_204821


namespace NUMINAMATH_GPT_f_divisible_by_k2_k1_l2048_204878

noncomputable def f (n : ℕ) (x : ℤ) : ℤ :=
  x^(n + 2) + (x + 1)^(2 * n + 1)

theorem f_divisible_by_k2_k1 (n : ℕ) (k : ℤ) (hn : n > 0) : 
  ((k^2 + k + 1) ∣ f n k) :=
sorry

end NUMINAMATH_GPT_f_divisible_by_k2_k1_l2048_204878


namespace NUMINAMATH_GPT_fractional_sides_l2048_204885

variable {F : ℕ} -- Number of fractional sides
variable {D : ℕ} -- Number of diagonals

theorem fractional_sides (h1 : D = 2 * F) (h2 : D = F * (F - 3) / 2) : F = 7 :=
by
  sorry

end NUMINAMATH_GPT_fractional_sides_l2048_204885


namespace NUMINAMATH_GPT_common_ratio_of_series_l2048_204854

-- Define the terms and conditions for the infinite geometric series problem.
def first_term : ℝ := 500
def series_sum : ℝ := 4000

-- State the theorem that needs to be proven: the common ratio of the series is 7/8.
theorem common_ratio_of_series (a S r : ℝ) (h_a : a = 500) (h_S : S = 4000) (h_eq : S = a / (1 - r)) :
  r = 7 / 8 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_of_series_l2048_204854


namespace NUMINAMATH_GPT_percentage_difference_max_min_l2048_204868

-- Definitions for the sector angles of each department
def angle_manufacturing := 162.0
def angle_sales := 108.0
def angle_research_and_development := 54.0
def angle_administration := 36.0

-- Full circle in degrees
def full_circle := 360.0

-- Compute the percentage representations of each department
def percentage_manufacturing := (angle_manufacturing / full_circle) * 100
def percentage_sales := (angle_sales / full_circle) * 100
def percentage_research_and_development := (angle_research_and_development / full_circle) * 100
def percentage_administration := (angle_administration / full_circle) * 100

-- Prove that the percentage difference between the department with the maximum and minimum number of employees is 35%
theorem percentage_difference_max_min : 
  percentage_manufacturing - percentage_administration = 35.0 :=
by
  -- placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_percentage_difference_max_min_l2048_204868


namespace NUMINAMATH_GPT_complement_union_eq_l2048_204806

variable (U : Set ℕ) (A : Set ℕ) (B : Set ℕ)

theorem complement_union_eq :
  U = {1, 2, 3, 4, 5, 6, 7, 8} →
  A = {1, 3, 5, 7} →
  B = {2, 4, 5} →
  U \ (A ∪ B) = {6, 8} :=
by
  intros hU hA hB
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_complement_union_eq_l2048_204806


namespace NUMINAMATH_GPT_question1_question2_l2048_204892

def A (x : ℝ) : Prop := x^2 - 2*x - 3 ≤ 0
def B (m : ℝ) (x : ℝ) : Prop := x^2 - 2*m*x + m^2 - 4 ≤ 0

-- Question 1: If A ∩ B = [1, 3], then m = 3
theorem question1 (m : ℝ) : (∀ x, A x ∧ B m x ↔ (1 ≤ x ∧ x ≤ 3)) → m = 3 :=
sorry

-- Question 2: If A is a subset of the complement of B in ℝ, then m > 5 or m < -3
theorem question2 (m : ℝ) : (∀ x, A x → ¬ B m x) → (m > 5 ∨ m < -3) :=
sorry

end NUMINAMATH_GPT_question1_question2_l2048_204892


namespace NUMINAMATH_GPT_problems_on_each_worksheet_l2048_204870

-- Define the conditions
def worksheets_total : Nat := 9
def worksheets_graded : Nat := 5
def problems_left : Nat := 16

-- Define the number of remaining worksheets and the problems per worksheet
def remaining_worksheets : Nat := worksheets_total - worksheets_graded
def problems_per_worksheet : Nat := problems_left / remaining_worksheets

-- Prove the number of problems on each worksheet
theorem problems_on_each_worksheet : problems_per_worksheet = 4 :=
by
  sorry

end NUMINAMATH_GPT_problems_on_each_worksheet_l2048_204870


namespace NUMINAMATH_GPT_elevator_max_weight_l2048_204824

theorem elevator_max_weight :
  let avg_weight_adult := 150
  let num_adults := 7
  let avg_weight_child := 70
  let num_children := 5
  let orig_max_weight := 1500
  let weight_adults := num_adults * avg_weight_adult
  let weight_children := num_children * avg_weight_child
  let current_weight := weight_adults + weight_children
  let upgrade_percentage := 0.10
  let new_max_weight := orig_max_weight * (1 + upgrade_percentage)
  new_max_weight - current_weight = 250 := 
  by
    sorry

end NUMINAMATH_GPT_elevator_max_weight_l2048_204824


namespace NUMINAMATH_GPT_nonoverlapping_area_difference_l2048_204883

theorem nonoverlapping_area_difference :
  let radius := 3
  let side := 2
  let circle_area := Real.pi * radius^2
  let square_area := side^2
  ∃ (x : ℝ), (circle_area - x) - (square_area - x) = 9 * Real.pi - 4 :=
by
  sorry

end NUMINAMATH_GPT_nonoverlapping_area_difference_l2048_204883


namespace NUMINAMATH_GPT_initial_quantity_of_A_l2048_204850

theorem initial_quantity_of_A (x : ℚ) 
    (h1 : 7 * x = a)
    (h2 : 5 * x = b)
    (h3 : a + b = 12 * x)
    (h4 : a' = a - (7 / 12) * 9)
    (h5 : b' = b - (5 / 12) * 9 + 9)
    (h6 : a' / b' = 7 / 9) : 
    a = 23.625 := 
sorry

end NUMINAMATH_GPT_initial_quantity_of_A_l2048_204850


namespace NUMINAMATH_GPT_odd_cube_difference_divisible_by_power_of_two_l2048_204817

theorem odd_cube_difference_divisible_by_power_of_two {a b n : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 1) :
  (2^n ∣ (a^3 - b^3)) ↔ (2^n ∣ (a - b)) :=
by
  sorry

end NUMINAMATH_GPT_odd_cube_difference_divisible_by_power_of_two_l2048_204817


namespace NUMINAMATH_GPT_missing_digit_divisible_by_9_l2048_204804

theorem missing_digit_divisible_by_9 (x : ℕ) (h : 0 ≤ x ∧ x < 10) : (3 + 5 + 1 + 9 + 2 + x) % 9 = 0 ↔ x = 7 :=
by
  sorry

end NUMINAMATH_GPT_missing_digit_divisible_by_9_l2048_204804


namespace NUMINAMATH_GPT_find_x_l2048_204847

def star (p q : ℤ × ℤ) : ℤ × ℤ :=
  (p.1 + q.1, p.2 - q.2)

theorem find_x : ∃ x : ℤ, ∃ y : ℤ, star (4, 5) (1, 3) = star (x, y) (2, 1) ∧ x = 3 :=
by 
  sorry

end NUMINAMATH_GPT_find_x_l2048_204847


namespace NUMINAMATH_GPT_compute_value_l2048_204869

theorem compute_value {p q : ℝ} (h1 : 3 * p^2 - 5 * p - 8 = 0) (h2 : 3 * q^2 - 5 * q - 8 = 0) :
  (5 * p^3 - 5 * q^3) / (p - q) = 245 / 9 :=
by
  sorry

end NUMINAMATH_GPT_compute_value_l2048_204869


namespace NUMINAMATH_GPT_shaded_area_of_hexagon_with_semicircles_l2048_204891

theorem shaded_area_of_hexagon_with_semicircles :
  let s := 3
  let r := 3 / 2
  let hexagon_area := (3 * Real.sqrt 3 / 2) * s^2
  let semicircle_area := 3 * (1/2 * Real.pi * r^2)
  let shaded_area := hexagon_area - semicircle_area
  shaded_area = 13.5 * Real.sqrt 3 - 27 * Real.pi / 8 :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_of_hexagon_with_semicircles_l2048_204891


namespace NUMINAMATH_GPT_division_simplification_l2048_204848

theorem division_simplification :
  (2 * 4.6 * 9 + 4 * 9.2 * 18) / (1 * 2.3 * 4.5 + 3 * 6.9 * 13.5) = 18 / 7 :=
by
  sorry

end NUMINAMATH_GPT_division_simplification_l2048_204848


namespace NUMINAMATH_GPT_right_triangle_min_area_l2048_204859

theorem right_triangle_min_area (a b : ℕ) (h₁ : a = 6) (h₂ : b = 8) :
  ∃ (c : ℕ), c * c = a * a + b * b ∧ ∃ (A : ℕ), A = (a * b) / 2 ∧ A = 24 :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_min_area_l2048_204859


namespace NUMINAMATH_GPT_largest_sum_of_ABCD_l2048_204887

theorem largest_sum_of_ABCD :
  ∃ (A B C D : ℕ), 10 ≤ A ∧ A < 100 ∧ 10 ≤ B ∧ B < 100 ∧ 10 ≤ C ∧ C < 100 ∧ 10 ≤ D ∧ D < 100 ∧
  B = 3 * C ∧ D = 2 * B - C ∧ A = B + D ∧ A + B + C + D = 204 :=
by
  sorry

end NUMINAMATH_GPT_largest_sum_of_ABCD_l2048_204887


namespace NUMINAMATH_GPT_fill_parentheses_correct_l2048_204829

theorem fill_parentheses_correct (a b : ℝ) :
  (3 * b + a) * (3 * b - a) = 9 * b^2 - a^2 :=
by 
  sorry

end NUMINAMATH_GPT_fill_parentheses_correct_l2048_204829


namespace NUMINAMATH_GPT_count_four_digit_numbers_with_repeated_digits_l2048_204880

def countDistinctFourDigitNumbersWithRepeatedDigits : Nat :=
  let totalNumbers := 4 ^ 4
  let uniqueNumbers := 4 * 3 * 2 * 1
  totalNumbers - uniqueNumbers

theorem count_four_digit_numbers_with_repeated_digits :
  countDistinctFourDigitNumbersWithRepeatedDigits = 232 := by
  sorry

end NUMINAMATH_GPT_count_four_digit_numbers_with_repeated_digits_l2048_204880


namespace NUMINAMATH_GPT_rahul_task_days_l2048_204823

theorem rahul_task_days (R : ℕ) (h1 : ∀ x : ℤ, x > 0 → 1 / R + 1 / 84 = 1 / 35) : R = 70 := 
by
  -- placeholder for the proof
  sorry

end NUMINAMATH_GPT_rahul_task_days_l2048_204823


namespace NUMINAMATH_GPT_div_by_3_9_then_mul_by_5_6_eq_div_by_5_2_l2048_204833

theorem div_by_3_9_then_mul_by_5_6_eq_div_by_5_2 :
  (∀ (x : ℚ), (x / (3/9)) * (5/6) = x / (5/2)) :=
by
  sorry

end NUMINAMATH_GPT_div_by_3_9_then_mul_by_5_6_eq_div_by_5_2_l2048_204833


namespace NUMINAMATH_GPT_corrected_mean_is_36_74_l2048_204874

noncomputable def corrected_mean (incorrect_mean : ℝ) 
(number_of_observations : ℕ) 
(correct_value wrong_value : ℝ) : ℝ :=
(incorrect_mean * number_of_observations - wrong_value + correct_value) / number_of_observations

theorem corrected_mean_is_36_74 :
  corrected_mean 36 50 60 23 = 36.74 :=
by
  sorry

end NUMINAMATH_GPT_corrected_mean_is_36_74_l2048_204874


namespace NUMINAMATH_GPT_roots_rational_l2048_204807

/-- Prove that the roots of the equation x^2 + px + q = 0 are always rational,
given the rational numbers p and q, and a rational n where p = n + q / n. -/
theorem roots_rational
  (n p q : ℚ)
  (hp : p = n + q / n)
  : ∃ x y : ℚ, x^2 + p * x + q = 0 ∧ y^2 + p * y + q = 0 ∧ x ≠ y :=
sorry

end NUMINAMATH_GPT_roots_rational_l2048_204807


namespace NUMINAMATH_GPT_arnold_danny_age_l2048_204814

theorem arnold_danny_age:
  ∃ x : ℝ, (x + 1) * (x + 1) = x * x + 11 ∧ x = 5 :=
by
  sorry

end NUMINAMATH_GPT_arnold_danny_age_l2048_204814


namespace NUMINAMATH_GPT_highway_extension_l2048_204809

def initial_length : ℕ := 200
def final_length : ℕ := 650
def first_day_construction : ℕ := 50
def second_day_construction : ℕ := 3 * first_day_construction
def total_construction : ℕ := first_day_construction + second_day_construction
def total_extension_needed : ℕ := final_length - initial_length
def miles_still_needed : ℕ := total_extension_needed - total_construction

theorem highway_extension : miles_still_needed = 250 := by
  sorry

end NUMINAMATH_GPT_highway_extension_l2048_204809


namespace NUMINAMATH_GPT_radius_of_semi_circle_l2048_204888

-- Given definitions and conditions
def perimeter : ℝ := 33.934511513692634
def pi_approx : ℝ := 3.141592653589793

-- The formula for the perimeter of a semi-circle
def semi_circle_perimeter (r : ℝ) : ℝ := pi_approx * r + 2 * r

-- The theorem we want to prove
theorem radius_of_semi_circle (r : ℝ) (h: semi_circle_perimeter r = perimeter) : r = 6.6 :=
sorry

end NUMINAMATH_GPT_radius_of_semi_circle_l2048_204888


namespace NUMINAMATH_GPT_mark_performance_length_l2048_204822

theorem mark_performance_length :
  ∃ (x : ℕ), (x > 0) ∧ (6 * 5 * x = 90) ∧ (x = 3) :=
by
  sorry

end NUMINAMATH_GPT_mark_performance_length_l2048_204822


namespace NUMINAMATH_GPT_secant_line_slope_positive_l2048_204844

theorem secant_line_slope_positive (f : ℝ → ℝ) (h_deriv : ∀ x : ℝ, 0 < (deriv f x)) :
  ∀ (x1 x2 : ℝ), x1 ≠ x2 → 0 < (f x1 - f x2) / (x1 - x2) :=
by
  intros x1 x2 h_ne
  sorry

end NUMINAMATH_GPT_secant_line_slope_positive_l2048_204844


namespace NUMINAMATH_GPT_range_omega_l2048_204845

noncomputable def f (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := Real.sin (ω * x + φ)
noncomputable def f' (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := ω * Real.cos (ω * x + φ)

theorem range_omega (t ω φ : ℝ) (hω_pos : ω > 0) (hf_t_zero : f t ω φ = 0) (hf'_t_pos : f' t ω φ > 0) (no_min_value : ∀ x, t ≤ x ∧ x < t + 1 → ∃ y, y ≠ x ∧ f y ω φ < f x ω φ) : π < ω ∧ ω ≤ (3 * π / 2) :=
sorry

end NUMINAMATH_GPT_range_omega_l2048_204845


namespace NUMINAMATH_GPT_smallest_natural_number_l2048_204831

theorem smallest_natural_number (a : ℕ) : 
  (∃ a, a % 3 = 0 ∧ (a - 1) % 4 = 0 ∧ (a - 2) % 5 = 0) → a = 57 :=
by
  sorry

end NUMINAMATH_GPT_smallest_natural_number_l2048_204831


namespace NUMINAMATH_GPT_integer_pair_solution_l2048_204819

theorem integer_pair_solution (x y : ℤ) :
  3^4 * 2^3 * (x^2 + y^2) = x^3 * y^3 ↔ (x = 6 ∧ y = 6) ∨ (x = -6 ∧ y = -6) ∨ (x = 0 ∧ y = 0) :=
by
  sorry

end NUMINAMATH_GPT_integer_pair_solution_l2048_204819


namespace NUMINAMATH_GPT_sum_first_five_terms_l2048_204830

variable {a : ℕ → ℝ} -- Define the arithmetic sequence as a function

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a 1 - a 0

noncomputable def S_5 (a : ℕ → ℝ) : ℝ :=
  a 0 + a 1 + a 2 + a 3 + a 4

theorem sum_first_five_terms (a : ℕ → ℝ) (h : is_arithmetic_sequence a) 
  (h2 : a 1 + a 3 = 6) : S_5 a = 15 :=
by
  -- skipping actual proof
  sorry

end NUMINAMATH_GPT_sum_first_five_terms_l2048_204830


namespace NUMINAMATH_GPT_find_x_l2048_204876

theorem find_x (h₁ : 2994 / 14.5 = 175) (h₂ : 29.94 / x = 17.5) : x = 29.94 / 17.5 :=
by
  -- skipping proofs
  sorry

end NUMINAMATH_GPT_find_x_l2048_204876


namespace NUMINAMATH_GPT_largest_4_digit_number_divisible_by_24_l2048_204856

theorem largest_4_digit_number_divisible_by_24 : 
  ∃ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ n % 24 = 0 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m ≤ 9999 ∧ m % 24 = 0 → m ≤ n :=
sorry

end NUMINAMATH_GPT_largest_4_digit_number_divisible_by_24_l2048_204856


namespace NUMINAMATH_GPT_lines_perpendicular_l2048_204897

/-- Given two lines l1: 3x + 4y + 1 = 0 and l2: 4x - 3y + 2 = 0, 
    prove that the lines are perpendicular. -/
theorem lines_perpendicular :
  ∀ (x y : ℝ), (3 * x + 4 * y + 1 = 0) → (4 * x - 3 * y + 2 = 0) → (- (3 / 4) * (4 / 3) = -1) :=
by
  intro x y h₁ h₂
  sorry

end NUMINAMATH_GPT_lines_perpendicular_l2048_204897


namespace NUMINAMATH_GPT_quadratic_distinct_real_roots_l2048_204810

theorem quadratic_distinct_real_roots (k : ℝ) :
  ((k - 1) * x^2 + 6 * x + 3 = 0 → ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ((k - 1) * x1^2 + 6 * x1 + 3 = 0) ∧ ((k - 1) * x2^2 + 6 * x2 + 3 = 0)) ↔ (k < 4 ∧ k ≠ 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_quadratic_distinct_real_roots_l2048_204810


namespace NUMINAMATH_GPT_cube_problem_l2048_204805

-- Define the conditions
def cube_volume (s : ℝ) : ℝ := s^3
def cube_surface_area (s : ℝ) : ℝ := 6 * s^2

theorem cube_problem (x : ℝ) (s : ℝ) :
  cube_volume s = 8 * x ∧ cube_surface_area s = 4 * x → x = 216 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_cube_problem_l2048_204805


namespace NUMINAMATH_GPT_gift_items_l2048_204815

theorem gift_items (x y z : ℕ) : 
  x + y + z = 20 ∧ 60 * x + 50 * y + 10 * z = 720 ↔ 
  ((x = 4 ∧ y = 8 ∧ z = 8) ∨ (x = 8 ∧ y = 3 ∧ z = 9)) :=
by sorry

end NUMINAMATH_GPT_gift_items_l2048_204815


namespace NUMINAMATH_GPT_quadratic_coefficients_sum_l2048_204812

-- Definition of the quadratic function and the conditions
def quadraticFunction (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c

-- Conditions
def vertexCondition (a b c : ℝ) : Prop :=
  quadraticFunction a b c 2 = 3
  
def pointCondition (a b c : ℝ) : Prop :=
  quadraticFunction a b c 3 = 2

-- The theorem to prove
theorem quadratic_coefficients_sum (a b c : ℝ)
  (hv : vertexCondition a b c)
  (hp : pointCondition a b c):
  a + b + 2 * c = 2 :=
sorry

end NUMINAMATH_GPT_quadratic_coefficients_sum_l2048_204812


namespace NUMINAMATH_GPT_factorize_expression_l2048_204853

theorem factorize_expression (x y : ℝ) : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) := by
  sorry

end NUMINAMATH_GPT_factorize_expression_l2048_204853


namespace NUMINAMATH_GPT_boxes_to_eliminate_l2048_204893

noncomputable def total_boxes : ℕ := 26
noncomputable def high_value_boxes : ℕ := 6
noncomputable def threshold_probability : ℚ := 1 / 2

-- Define the condition for having the minimum number of boxes
def min_boxes_needed_for_probability (total high_value : ℕ) (prob : ℚ) : ℕ :=
  total - high_value - ((total - high_value) / 2)

theorem boxes_to_eliminate :
  min_boxes_needed_for_probability total_boxes high_value_boxes threshold_probability = 15 :=
by
  sorry

end NUMINAMATH_GPT_boxes_to_eliminate_l2048_204893


namespace NUMINAMATH_GPT_correct_assignment_statement_l2048_204808

theorem correct_assignment_statement (a b : ℕ) : 
  (2 = a → False) ∧ 
  (a = a + 1 → True) ∧ 
  (a * b = 2 → False) ∧ 
  (a + 1 = a → False) :=
by {
  sorry
}

end NUMINAMATH_GPT_correct_assignment_statement_l2048_204808


namespace NUMINAMATH_GPT_lesser_fraction_l2048_204875

theorem lesser_fraction (x y : ℚ) (h_sum : x + y = 17 / 24) (h_prod : x * y = 1 / 8) : min x y = 1 / 3 := by
  sorry

end NUMINAMATH_GPT_lesser_fraction_l2048_204875


namespace NUMINAMATH_GPT_tangent_line_min_slope_l2048_204855

noncomputable def curve (x : ℝ) : ℝ := x^3 + 3*x - 1

noncomputable def curve_derivative (x : ℝ) : ℝ := 3*x^2 + 3

theorem tangent_line_min_slope :
  ∃ k b : ℝ, (∀ x : ℝ, curve_derivative x ≥ 3) ∧ 
             k = 3 ∧ b = 1 ∧
             (∀ x y : ℝ, y = k * x + b ↔ 3 * x - y + 1 = 0) := 
by {
  sorry
}

end NUMINAMATH_GPT_tangent_line_min_slope_l2048_204855


namespace NUMINAMATH_GPT_profit_percent_is_26_l2048_204857

variables (P C : ℝ)
variables (h1 : (2/3) * P = 0.84 * C)

theorem profit_percent_is_26 :
  ((P - C) / C) * 100 = 26 :=
by
  sorry

end NUMINAMATH_GPT_profit_percent_is_26_l2048_204857


namespace NUMINAMATH_GPT_arithmetic_geometric_means_l2048_204840

theorem arithmetic_geometric_means (a b : ℝ) (h1 : 2 * a = 1 + 2) (h2 : b^2 = (-1) * (-16)) : a * b = 6 ∨ a * b = -6 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_means_l2048_204840


namespace NUMINAMATH_GPT_carter_lucy_ratio_l2048_204866

-- Define the number of pages Oliver can read in 1 hour
def oliver_pages : ℕ := 40

-- Define the number of additional pages Lucy can read compared to Oliver
def additional_pages : ℕ := 20

-- Define the number of pages Carter can read in 1 hour
def carter_pages : ℕ := 30

-- Calculate the number of pages Lucy can read in 1 hour
def lucy_pages : ℕ := oliver_pages + additional_pages

-- Prove the ratio of the number of pages Carter can read to the number of pages Lucy can read is 1/2
theorem carter_lucy_ratio : (carter_pages : ℚ) / (lucy_pages : ℚ) = 1 / 2 := by
  sorry

end NUMINAMATH_GPT_carter_lucy_ratio_l2048_204866


namespace NUMINAMATH_GPT_largest_positive_integer_not_sum_of_multiple_30_and_composite_l2048_204895

def is_composite (n : ℕ) : Prop :=
  2 ≤ n ∧ ∃ m k : ℕ, 2 ≤ m ∧ 2 ≤ k ∧ n = m * k

noncomputable def largest_not_sum_of_multiple_30_and_composite : ℕ :=
  211

theorem largest_positive_integer_not_sum_of_multiple_30_and_composite {m : ℕ} :
  m = largest_not_sum_of_multiple_30_and_composite ↔ 
  (∀ a b : ℕ, a > 0 ∧ b > 0 ∧ (∃ k : ℕ, (b = k * 30) ∨ is_composite b) → m ≠ 30 * a + b) :=
sorry

end NUMINAMATH_GPT_largest_positive_integer_not_sum_of_multiple_30_and_composite_l2048_204895


namespace NUMINAMATH_GPT_total_students_l2048_204846

theorem total_students (f1 f2 f3 total : ℕ)
  (h_ratio : f1 * 2 = f2)
  (h_ratio2 : f1 * 3 = f3)
  (h_f1 : f1 = 6)
  (h_total : total = f1 + f2 + f3) :
  total = 48 :=
by
  sorry

end NUMINAMATH_GPT_total_students_l2048_204846
