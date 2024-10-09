import Mathlib

namespace perspective_square_area_l309_30996

theorem perspective_square_area (a b : ℝ) (ha : a = 4 ∨ b = 4) : 
  a * a = 16 ∨ (2 * b) * (2 * b) = 64 :=
by 
sorry

end perspective_square_area_l309_30996


namespace shaded_area_square_semicircles_l309_30970

theorem shaded_area_square_semicircles :
  let side_length := 2
  let radius_circle := side_length * Real.sqrt 2 / 2
  let area_circle := Real.pi * radius_circle^2
  let area_square := side_length^2
  let area_semicircle := Real.pi * (side_length / 2)^2 / 2
  let total_area_semicircles := 4 * area_semicircle
  let shaded_area := total_area_semicircles - area_circle
  shaded_area = 4 :=
by
  sorry

end shaded_area_square_semicircles_l309_30970


namespace ship_speed_in_still_water_l309_30989

theorem ship_speed_in_still_water
  (x y : ℝ)
  (h1: x + y = 32)
  (h2: x - y = 28)
  (h3: x > y) : 
  x = 30 := 
sorry

end ship_speed_in_still_water_l309_30989


namespace div_identity_l309_30934

theorem div_identity (a b c : ℚ) (h1 : a / b = 3) (h2 : b / c = 2 / 5) : c / a = 5 / 6 := by
  sorry

end div_identity_l309_30934


namespace minimize_dot_product_l309_30985

def vector := ℝ × ℝ

def OA : vector := (2, 2)
def OB : vector := (4, 1)

def dot_product (u v : vector) : ℝ :=
  u.1 * v.1 + u.2 * v.2

def AP (P : vector) : vector :=
  (P.1 - OA.1, P.2 - OA.2)

def BP (P : vector) : vector :=
  (P.1 - OB.1, P.2 - OB.2)

def is_on_x_axis (P : vector) : Prop :=
  P.2 = 0

theorem minimize_dot_product :
  ∃ (P : vector), is_on_x_axis P ∧ dot_product (AP P) (BP P) = ( (P.1 - 3) ^ 2 + 1) ∧ P = (3, 0) :=
by
  sorry

end minimize_dot_product_l309_30985


namespace angle_quadrant_l309_30936

theorem angle_quadrant (α : ℝ) (h1 : Real.sin α > 0) (h2 : Real.cos α < 0) : 
  0 < (π - α) ∧ (π - α) < π  :=
by
  sorry

end angle_quadrant_l309_30936


namespace flour_already_put_in_l309_30909

def total_flour : ℕ := 8
def additional_flour_needed : ℕ := 6

theorem flour_already_put_in : total_flour - additional_flour_needed = 2 := by
  sorry

end flour_already_put_in_l309_30909


namespace most_appropriate_sampling_l309_30942

def total_students := 126 + 280 + 95
def adjusted_total_students := 126 - 1 + 280 + 95
def required_sample_size := 100

def elementary_proportion (total : Nat) (sample : Nat) : Nat := (sample * 126) / total
def middle_proportion (total : Nat) (sample : Nat) : Nat := (sample * 280) / total
def high_proportion (total : Nat) (sample : Nat) : Nat := (sample * 95) / total

theorem most_appropriate_sampling :
  required_sample_size = elementary_proportion adjusted_total_students required_sample_size + 
                         middle_proportion adjusted_total_students required_sample_size + 
                         high_proportion adjusted_total_students required_sample_size :=
by
  sorry

end most_appropriate_sampling_l309_30942


namespace smallest_even_in_sequence_sum_400_l309_30963

theorem smallest_even_in_sequence_sum_400 :
  ∃ (n : ℤ), (n - 6) + (n - 4) + (n - 2) + n + (n + 2) + (n + 4) + (n + 6) = 400 ∧ (n - 6) % 2 = 0 ∧ n - 6 = 52 :=
sorry

end smallest_even_in_sequence_sum_400_l309_30963


namespace carlos_finishes_first_l309_30983

theorem carlos_finishes_first
  (a : ℝ) -- Andy's lawn area
  (r : ℝ) -- Andy's mowing rate
  (hBeth_lawn : ∀ (b : ℝ), b = a / 3) -- Beth's lawn area
  (hCarlos_lawn : ∀ (c : ℝ), c = a / 4) -- Carlos' lawn area
  (hCarlos_Beth_rate : ∀ (rc rb : ℝ), rc = r / 2 ∧ rb = r / 2) -- Carlos' and Beth's mowing rate
  : (∃ (ta tb tc : ℝ), ta = a / r ∧ tb = (2 * a) / (3 * r) ∧ tc = a / (2 * r) ∧ tc < tb ∧ tc < ta) :=
-- Prove that the mowing times are such that Carlos finishes first
sorry

end carlos_finishes_first_l309_30983


namespace condition1_condition2_l309_30913

-- Define the point P
def P (m : ℝ) : ℝ × ℝ := (m + 1, 2 * m - 4)

-- Define the point A
def A : ℝ × ℝ := (-5, 2)

-- Condition 1: P lies on the x-axis
def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

-- Condition 2: AP is parallel to the y-axis
def parallel_y_axis (a p : ℝ × ℝ) : Prop := a.1 = p.1

-- Prove the conditions
theorem condition1 (m : ℝ) (h : on_x_axis (P m)) : P m = (3, 0) :=
by
  sorry

theorem condition2 (m : ℝ) (h : parallel_y_axis A (P m)) : P m = (-5, -16) :=
by
  sorry

end condition1_condition2_l309_30913


namespace prime_k_for_equiangular_polygons_l309_30980

-- Definitions for conditions in Lean 4
def is_equiangular_polygon (n : ℕ) (angle : ℕ) : Prop :=
  angle = 180 - 360 / n

def is_prime (k : ℕ) : Prop :=
  Nat.Prime k

def valid_angle (x : ℕ) (k : ℕ) : Prop :=
  x < 180 / k

-- The main statement
theorem prime_k_for_equiangular_polygons (n1 n2 x k : ℕ) :
  is_equiangular_polygon n1 x →
  is_equiangular_polygon n2 (k * x) →
  1 < k →
  is_prime k →
  k = 3 :=
by sorry -- proof is not required

end prime_k_for_equiangular_polygons_l309_30980


namespace solve_equation_l309_30922

theorem solve_equation (x : ℝ) (hx : x ≠ 0) 
  (h : 1 / 4 + 8 / x = 13 / x + 1 / 8) : 
  x = 40 :=
sorry

end solve_equation_l309_30922


namespace f_neg_two_l309_30991

noncomputable def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^5 + b * x - c / x + 2

theorem f_neg_two (a b c : ℝ) (h : f a b c 2 = 4) : f a b c (-2) = 0 :=
sorry

end f_neg_two_l309_30991


namespace atomic_weight_O_l309_30939

-- We define the atomic weights of sodium and chlorine
def atomic_weight_Na : ℝ := 22.99
def atomic_weight_Cl : ℝ := 35.45

-- We define the molecular weight of the compound
def molecular_weight_compound : ℝ := 74.0

-- We want to prove that the atomic weight of oxygen (O) is 15.56 given the above conditions
theorem atomic_weight_O : 
  (molecular_weight_compound = atomic_weight_Na + atomic_weight_Cl + w -> w = 15.56) :=
by
  sorry

end atomic_weight_O_l309_30939


namespace find_circle_center_l309_30998

def circle_center_condition (x y : ℝ) : Prop :=
  (3 * x - 4 * y = 24 ∨ 3 * x - 4 * y = -12) ∧ 3 * x + 2 * y = 0

theorem find_circle_center :
  ∃ (x y : ℝ), circle_center_condition x y ∧ (x, y) = (2/3, -1) :=
by
  sorry

end find_circle_center_l309_30998


namespace correct_statements_l309_30944

-- Define the propositions p and q
variables (p q : Prop)

-- Define the given statements as logical conditions
def statement1 := (p ∧ q) → (p ∨ q)
def statement2 := ¬(p ∧ q) → (p ∨ q)
def statement3 := (p ∨ q) ↔ ¬¬p
def statement4 := (¬p) → ¬(p ∧ q)

-- Define the proof problem
theorem correct_statements :
  ((statement1 p q) ∧ (¬statement2 p q) ∧ (statement3 p q) ∧ (¬statement4 p q)) :=
by {
  -- Here you would prove that
  -- statement1 is correct,
  -- statement2 is incorrect,
  -- statement3 is correct,
  -- statement4 is incorrect
  sorry
}

end correct_statements_l309_30944


namespace harry_items_left_l309_30984

def sea_stars : ℕ := 34
def seashells : ℕ := 21
def snails : ℕ := 29
def lost_items : ℕ := 25

def total_items : ℕ := sea_stars + seashells + snails
def remaining_items : ℕ := total_items - lost_items

theorem harry_items_left : remaining_items = 59 := by
  -- proof skipped
  sorry

end harry_items_left_l309_30984


namespace find_a_l309_30921

theorem find_a (a : ℝ) (h : ((2 * a + 16) + (3 * a - 8)) / 2 = 89) : a = 34 :=
sorry

end find_a_l309_30921


namespace union_of_sets_l309_30997

-- Define the sets A and B
def A : Set ℝ := { x | x^2 - x - 2 < 0 }
def B : Set ℝ := { x | 1 < x ∧ x < 4 }

-- Define the set representing the union's result
def C : Set ℝ := { x | -1 < x ∧ x < 4 }

-- The theorem statement
theorem union_of_sets : ∀ x : ℝ, (x ∈ (A ∪ B) ↔ x ∈ C) :=
by
  sorry

end union_of_sets_l309_30997


namespace Diane_net_loss_l309_30981

variable (x y a b: ℝ)

axiom h1 : x * a = 65
axiom h2 : y * b = 150

theorem Diane_net_loss : (y * b) - (x * a) = 50 := by
  sorry

end Diane_net_loss_l309_30981


namespace time_to_save_for_downpayment_l309_30916

def annual_salary : ℝ := 120000
def savings_percentage : ℝ := 0.15
def house_cost : ℝ := 550000
def downpayment_percentage : ℝ := 0.25

def annual_savings : ℝ := savings_percentage * annual_salary
def downpayment_needed : ℝ := downpayment_percentage * house_cost

theorem time_to_save_for_downpayment :
  (downpayment_needed / annual_savings) = 7.64 :=
by
  -- Proof to be provided
  sorry

end time_to_save_for_downpayment_l309_30916


namespace find_D_double_prime_l309_30976

def reflectY (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

def translateUp1 (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 + 1)

def reflectYeqX (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

def translateDown1 (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 - 1)

def D'' (D : ℝ × ℝ) : ℝ × ℝ :=
  translateDown1 (reflectYeqX (translateUp1 (reflectY D)))

theorem find_D_double_prime :
  let D := (5, 0)
  D'' D = (-1, 4) :=
by
  sorry

end find_D_double_prime_l309_30976


namespace proof_l309_30974

-- Definition of the logical statements
def all_essays_correct (maria : Type) : Prop := sorry
def passed_course (maria : Type) : Prop := sorry

-- Condition provided in the problem
axiom condition : ∀ (maria : Type), all_essays_correct maria → passed_course maria

-- We need to prove this
theorem proof (maria : Type) : ¬ (passed_course maria) → ¬ (all_essays_correct maria) :=
by sorry

end proof_l309_30974


namespace value_of_f_inv_sum_l309_30951

noncomputable def f (x : ℝ) : ℝ := sorry
noncomputable def f_inv (y : ℝ) : ℝ := sorry

axiom f_inv_is_inverse : ∀ x : ℝ, f (f_inv x) = x ∧ f_inv (f x) = x
axiom f_condition : ∀ x : ℝ, f x + f (-x) = 2

theorem value_of_f_inv_sum (x : ℝ) : f_inv (2008 - x) + f_inv (x - 2006) = 0 :=
sorry

end value_of_f_inv_sum_l309_30951


namespace juanitas_dessert_cost_l309_30933

theorem juanitas_dessert_cost :
  let brownie_cost := 2.50
  let ice_cream_cost := 1.00
  let syrup_cost := 0.50
  let nuts_cost := 1.50
  let num_scoops_ice_cream := 2
  let num_syrups := 2
  let total_cost := brownie_cost + num_scoops_ice_cream * ice_cream_cost + num_syrups * syrup_cost + nuts_cost
  total_cost = 7.00 :=
by
  sorry

end juanitas_dessert_cost_l309_30933


namespace percentage_of_rotten_bananas_l309_30918

-- Define the initial conditions and the question as a Lean theorem statement
theorem percentage_of_rotten_bananas (oranges bananas : ℕ) (perc_rot_oranges perc_good_fruits : ℝ) 
  (total_fruits good_fruits good_oranges good_bananas rotten_bananas perc_rot_bananas : ℝ) :
  oranges = 600 →
  bananas = 400 →
  perc_rot_oranges = 0.15 →
  perc_good_fruits = 0.886 →
  total_fruits = (oranges + bananas) →
  good_fruits = (perc_good_fruits * total_fruits) →
  good_oranges = ((1 - perc_rot_oranges) * oranges) →
  good_bananas = (good_fruits - good_oranges) →
  rotten_bananas = (bananas - good_bananas) →
  perc_rot_bananas = ((rotten_bananas / bananas) * 100) →
  perc_rot_bananas = 6 :=
by
  intros; sorry

end percentage_of_rotten_bananas_l309_30918


namespace maria_initial_cookies_l309_30955

theorem maria_initial_cookies (X : ℕ) 
  (h1: X - 5 = 2 * (5 + 2)) 
  (h2: X ≥ 5)
  : X = 19 := 
by
  sorry

end maria_initial_cookies_l309_30955


namespace largest_integer_b_l309_30960

theorem largest_integer_b (b : ℤ) : (b^2 < 60) → b ≤ 7 :=
by sorry

end largest_integer_b_l309_30960


namespace runs_scored_by_c_l309_30902

-- Definitions
variables (A B C : ℕ)

-- Conditions as hypotheses
theorem runs_scored_by_c (h1 : B = 3 * A) (h2 : C = 5 * B) (h3 : A + B + C = 95) : C = 75 :=
by
  -- Proof will be here
  sorry

end runs_scored_by_c_l309_30902


namespace intersections_of_absolute_value_functions_l309_30919

theorem intersections_of_absolute_value_functions : 
  (∃ x y : ℝ, y = |3 * x + 4| ∧ y = -|4 * x + 3|) → ∃ (x y : ℝ), (x = -1 ∧ y = 1) ∧ ¬(∃ (x' y' : ℝ), y' = |3 * x' + 4| ∧ y' = -|4 * x' + 3| ∧ (x' ≠ -1 ∨ y' ≠ 1)) :=
by
  sorry

end intersections_of_absolute_value_functions_l309_30919


namespace proof_l309_30977

open Set

-- Universal set U
def U : Set ℕ := {x | x ∈ Finset.range 7}

-- Set A
def A : Set ℕ := {1, 3, 5}

-- Set B
def B : Set ℕ := {4, 5, 6}

-- Complement of A in U
def CU (s : Set ℕ) : Set ℕ := U \ s

-- Proof statement
theorem proof : (CU A) ∩ B = {4, 6} :=
by
  sorry

end proof_l309_30977


namespace multiply_and_simplify_l309_30950

variable (a b : ℝ)

theorem multiply_and_simplify :
  (3 * a + 2 * b) * (a - 2 * b) = 3 * a^2 - 4 * a * b - 4 * b^2 :=
by
  sorry

end multiply_and_simplify_l309_30950


namespace min_radius_circle_line_intersection_l309_30935

theorem min_radius_circle_line_intersection (θ : ℝ) (hθ : 0 ≤ θ ∧ θ < 2 * Real.pi) (r : ℝ) (hr : r > 0)
    (intersect : ∃ (x y : ℝ), (x - Real.cos θ)^2 + (y - Real.sin θ)^2 = r^2 ∧ 2 * x - y - 10 = 0) :
    r ≥ 2 * Real.sqrt 5 - 1 :=
  sorry

end min_radius_circle_line_intersection_l309_30935


namespace pascal_triangle_eighth_row_l309_30986

def sum_interior_numbers (n : ℕ) : ℕ :=
  2^(n-1) - 2

def binomial_coefficient (n k : ℕ) : ℕ :=
  Nat.choose (n-1) (k-1) 

theorem pascal_triangle_eighth_row:
  sum_interior_numbers 8 = 126 ∧ binomial_coefficient 8 3 = 21 :=
by
  sorry

end pascal_triangle_eighth_row_l309_30986


namespace smallest_power_of_7_not_palindrome_l309_30964

def is_palindrome (n : ℕ) : Prop :=
  let digits := n.digits 10
  digits = digits.reverse

theorem smallest_power_of_7_not_palindrome : ∃ n : ℕ, n > 0 ∧ 7^n = 2401 ∧ ¬is_palindrome (7^n) ∧ (∀ m : ℕ, m > 0 ∧ ¬is_palindrome (7^m) → 7^n ≤ 7^m) :=
by
  sorry

end smallest_power_of_7_not_palindrome_l309_30964


namespace king_arthur_actual_weight_l309_30969

theorem king_arthur_actual_weight (K H E : ℤ) 
  (h1 : K + E = 19) 
  (h2 : H + E = 101) 
  (h3 : K + H + E = 114) : K = 13 := 
by 
  -- Introduction for proof to be skipped
  sorry

end king_arthur_actual_weight_l309_30969


namespace cyclist_go_south_speed_l309_30994

noncomputable def speed_of_cyclist_go_south (v : ℝ) : Prop :=
  let north_speed := 10 -- speed of cyclist going north in kmph
  let time := 2 -- time in hours
  let distance := 50 -- distance apart in km
  (north_speed + v) * time = distance

theorem cyclist_go_south_speed (v : ℝ) : speed_of_cyclist_go_south v → v = 15 :=
by
  intro h
  -- Proof part is skipped
  sorry

end cyclist_go_south_speed_l309_30994


namespace sum_of_integers_l309_30925

theorem sum_of_integers (m n : ℕ) (h1 : m * n = 2 * (m + n)) (h2 : m * n = 6 * (m - n)) :
  m + n = 9 := by
  sorry

end sum_of_integers_l309_30925


namespace combined_perimeter_of_squares_l309_30905

theorem combined_perimeter_of_squares (p1 p2 : ℝ) (s1 s2 : ℝ) :
  p1 = 40 → p2 = 100 → 4 * s1 = p1 → 4 * s2 = p2 →
  (p1 + p2 - 2 * s1) = 120 :=
by
  intros h1 h2 h3 h4
  sorry

end combined_perimeter_of_squares_l309_30905


namespace value_of_c_l309_30943

theorem value_of_c :
  ∃ (a b c : ℕ), 
  30 = 2 * (10 + a) ∧ 
  b = 2 * (a + 30) ∧ 
  c = 2 * (b + 30) ∧ 
  c = 200 := 
sorry

end value_of_c_l309_30943


namespace largest_A_smallest_A_l309_30924

noncomputable def is_coprime_with_12 (n : Nat) : Prop :=
  Nat.gcd n 12 = 1

noncomputable def rotated_number (n : Nat) : Option Nat :=
  if n < 10^7 then none else
  let b := n % 10
  let k := n / 10
  some (b * 10^7 + k)

noncomputable def satisfies_conditions (B : Nat) : Prop :=
  B > 44444444 ∧ is_coprime_with_12 B

theorem largest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 99999998 :=
sorry

theorem smallest_A :
  ∃ (B : Nat), satisfies_conditions B ∧ rotated_number B = some 14444446 :=
sorry

end largest_A_smallest_A_l309_30924


namespace max_children_l309_30910

theorem max_children (x : ℕ) (h1 : x * (x - 2) + 2 * 5 = 58) : x = 8 :=
by
  sorry

end max_children_l309_30910


namespace vacation_costs_l309_30952

theorem vacation_costs :
  let a := 15
  let b := 22.5
  let c := 22.5
  a + b + c = 45 → b - a = 7.5 := by
sorry

end vacation_costs_l309_30952


namespace ratio_part_to_whole_number_l309_30908

theorem ratio_part_to_whole_number (P N : ℚ) 
  (h1 : (1 / 4) * (1 / 3) * P = 25) 
  (h2 : 0.40 * N = 300) : P / N = 2 / 5 :=
by
  sorry

end ratio_part_to_whole_number_l309_30908


namespace how_many_tuna_l309_30917

-- Definitions for conditions
variables (customers : ℕ) (weightPerTuna : ℕ) (weightPerCustomer : ℕ)
variables (unsatisfiedCustomers : ℕ)

-- Hypotheses based on the problem conditions
def conditions :=
  customers = 100 ∧
  weightPerTuna = 200 ∧
  weightPerCustomer = 25 ∧
  unsatisfiedCustomers = 20

-- Statement to prove how many tuna Mr. Ray needs
theorem how_many_tuna (h : conditions customers weightPerTuna weightPerCustomer unsatisfiedCustomers) : 
  ∃ n, n = 10 :=
by
  sorry

end how_many_tuna_l309_30917


namespace annual_increase_in_living_space_l309_30949

-- Definitions based on conditions
def population_2000 : ℕ := 200000
def living_space_2000_per_person : ℝ := 8
def target_living_space_2004_per_person : ℝ := 10
def annual_growth_rate : ℝ := 0.01
def years : ℕ := 4

-- Goal stated as a theorem
theorem annual_increase_in_living_space :
  let final_population := population_2000 * (1 + annual_growth_rate)^years
  let total_living_space_2004 := target_living_space_2004_per_person * final_population
  let initial_living_space := living_space_2000_per_person * population_2000
  let total_additional_space := total_living_space_2004 - initial_living_space
  let average_annual_increase := total_additional_space / years
  average_annual_increase = 120500.0 :=
sorry

end annual_increase_in_living_space_l309_30949


namespace proof_problem_l309_30946

noncomputable def f (x : ℝ) := Real.tan (x + (Real.pi / 4))

theorem proof_problem :
  (- (3 * Real.pi) / 4 < 1 - Real.pi ∧ 1 - Real.pi < -1 ∧ -1 < 0 ∧ 0 < Real.pi / 4) →
  f 0 > f (-1) ∧ f (-1) > f 1 := by
  sorry

end proof_problem_l309_30946


namespace intersecting_lines_l309_30992

theorem intersecting_lines (p q r s t : ℝ) : (∃ u v : ℝ, p * u^2 + q * v^2 + r * u + s * v + t = 0) →
  ( ∃ p q : ℝ, p * q < 0 ∧ 4 * t = r^2 / p + s^2 / q ) :=
sorry

end intersecting_lines_l309_30992


namespace power_inequality_l309_30979

theorem power_inequality (a b c : ℕ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hcb : c ≥ b) : 
  a^b * (a + b)^c > c^b * a^c := 
sorry

end power_inequality_l309_30979


namespace max_value_of_f_l309_30990

noncomputable def f (x : ℝ) : ℝ := min (2^x) (min (x + 2) (10 - x))

theorem max_value_of_f : ∃ M, (∀ x ≥ 0, f x ≤ M) ∧ (∃ x ≥ 0, f x = M) ∧ M = 6 :=
by
  sorry

end max_value_of_f_l309_30990


namespace points_per_member_l309_30967

def numMembersTotal := 12
def numMembersAbsent := 4
def totalPoints := 64

theorem points_per_member (h : numMembersTotal - numMembersAbsent = 12 - 4) :
  (totalPoints / (numMembersTotal - numMembersAbsent)) = 8 := 
  sorry

end points_per_member_l309_30967


namespace power_mod_7_l309_30930

theorem power_mod_7 {a : ℤ} (h : a = 3) : (a ^ 123) % 7 = 6 := by
  sorry

end power_mod_7_l309_30930


namespace final_range_a_l309_30937

open Real

noncomputable def f (a x : ℝ) : ℝ := log x + x^2 - a * x

lemma increasing_function_range_a (a : ℝ) (h : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0) :
  a ≤ 2 * sqrt 2 :=
sorry

lemma condition_range_a (a : ℝ) (h1 : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0)
  (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x ≤ 1/2 * (3 * x^2 + 1 / x^2 - 6 * x)) :
  2 ≤ a :=
sorry

theorem final_range_a (a : ℝ)
  (h1 : ∀ x : ℝ, x > 0 → deriv (f a) x ≥ 0)
  (h2 : ∀ x : ℝ, 0 < x ∧ x ≤ 1 → f a x ≤ 1/2 * (3 * x^2 + 1 / x^2 - 6 * x)) :
  2 ≤ a ∧ a ≤ 2 * sqrt 2 :=
sorry

end final_range_a_l309_30937


namespace probability_all_same_flips_l309_30987

noncomputable def four_same_flips_probability : ℚ := 
  (∑' n : ℕ, if n > 0 then (1/2)^(4*n) else 0)

theorem probability_all_same_flips : 
  four_same_flips_probability = 1 / 15 := 
sorry

end probability_all_same_flips_l309_30987


namespace chocolate_bars_per_box_l309_30907

theorem chocolate_bars_per_box (total_chocolate_bars num_small_boxes : ℕ) (h1 : total_chocolate_bars = 300) (h2 : num_small_boxes = 15) : 
  total_chocolate_bars / num_small_boxes = 20 :=
by 
  sorry

end chocolate_bars_per_box_l309_30907


namespace arithmetic_series_sum_base6_l309_30927

-- Define the terms in the arithmetic series in base 6
def a₁ := 1
def a₄₅ := 45
def n := a₄₅

-- Sum of arithmetic series in base 6
def sum_arithmetic_series := (n * (a₁ + a₄₅)) / 2

-- Expected result for the arithmetic series sum
def expected_result := 2003

theorem arithmetic_series_sum_base6 :
  sum_arithmetic_series = expected_result := by
  sorry

end arithmetic_series_sum_base6_l309_30927


namespace find_nth_term_of_arithmetic_seq_l309_30962

def is_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

def is_geometric_progression (a1 a2 a5 : ℝ) :=
  a1 * a5 = a2^2

theorem find_nth_term_of_arithmetic_seq (a : ℕ → ℝ) (d : ℝ) (h_arith : is_arithmetic_seq a d)
    (h_a1 : a 1 = 1) (h_nonzero : d ≠ 0) (h_geom : is_geometric_progression (a 1) (a 2) (a 5)) : 
    ∀ n, a n = 2 * n - 1 :=
by
  sorry

end find_nth_term_of_arithmetic_seq_l309_30962


namespace find_oxygen_weight_l309_30968

-- Definitions of given conditions
def molecular_weight : ℝ := 68
def weight_hydrogen : ℝ := 1
def weight_chlorine : ℝ := 35.5

-- Definition of unknown atomic weight of oxygen
def weight_oxygen : ℝ := 15.75

-- Mathematical statement to prove
theorem find_oxygen_weight :
  weight_hydrogen + weight_chlorine + 2 * weight_oxygen = molecular_weight := by
sorry

end find_oxygen_weight_l309_30968


namespace second_smallest_relative_prime_210_l309_30904

theorem second_smallest_relative_prime_210 (x : ℕ) (h1 : x > 1) (h2 : Nat.gcd x 210 = 1) : x = 13 :=
sorry

end second_smallest_relative_prime_210_l309_30904


namespace f_10_l309_30948

noncomputable def f : ℕ → ℕ
| 0       => 1
| (n + 1) => 2 * f n

theorem f_10 : f 10 = 2^10 :=
by
  -- This would be filled in with the necessary proof steps to show f(10) = 2^10
  sorry

end f_10_l309_30948


namespace algebra_ineq_a2_b2_geq_2_l309_30920

theorem algebra_ineq_a2_b2_geq_2
  (a b : ℝ)
  (h1 : a^3 - b^3 = 2)
  (h2 : a^5 - b^5 ≥ 4) :
  a^2 + b^2 ≥ 2 :=
by
  sorry

end algebra_ineq_a2_b2_geq_2_l309_30920


namespace derivative_at_1_l309_30911

noncomputable def f (x : ℝ) : ℝ := Real.sin x + Real.log x

theorem derivative_at_1 : deriv f 1 = 1 + Real.cos 1 :=
by
  sorry

end derivative_at_1_l309_30911


namespace basketball_weight_l309_30906

variable (b c : ℝ)

theorem basketball_weight (h1 : 9 * b = 5 * c) (h2 : 3 * c = 75) : b = 125 / 9 :=
by
  sorry

end basketball_weight_l309_30906


namespace chinese_money_plant_sales_l309_30932

/-- 
Consider a scenario where a plant supplier sells 20 pieces of orchids for $50 each 
and some pieces of potted Chinese money plant for $25 each. He paid his two workers $40 each 
and bought new pots worth $150. The plant supplier had $1145 left from his earnings. 
Prove that the number of pieces of potted Chinese money plants sold by the supplier is 15.
-/
theorem chinese_money_plant_sales (earnings_orchids earnings_per_orchid: ℤ)
  (num_orchids: ℤ)
  (earnings_plants earnings_per_plant: ℤ)
  (worker_wage num_workers: ℤ)
  (new_pots_cost remaining_money: ℤ)
  (earnings: ℤ)
  (P : earnings_orchids = num_orchids * earnings_per_orchid)
  (Q : earnings = earnings_orchids + earnings_plants)
  (R : earnings - (worker_wage * num_workers + new_pots_cost) = remaining_money)
  (conditions: earnings_per_orchid = 50 ∧ num_orchids = 20 ∧ earnings_per_plant = 25 ∧ worker_wage = 40 ∧ num_workers = 2 ∧ new_pots_cost = 150 ∧ remaining_money = 1145):
  earnings_plants / earnings_per_plant = 15 := 
by
  sorry

end chinese_money_plant_sales_l309_30932


namespace probability_of_both_selected_l309_30903

theorem probability_of_both_selected (pX pY : ℚ) (hX : pX = 1/7) (hY : pY = 2/5) : 
  pX * pY = 2 / 35 :=
by {
  sorry
}

end probability_of_both_selected_l309_30903


namespace wendy_packages_chocolates_l309_30938

variable (packages_per_5min : Nat := 2)
variable (dozen_size : Nat := 12)
variable (minutes_in_hour : Nat := 60)
variable (hours : Nat := 4)

theorem wendy_packages_chocolates (h1 : packages_per_5min = 2) 
                                 (h2 : dozen_size = 12) 
                                 (h3 : minutes_in_hour = 60) 
                                 (h4 : hours = 4) : 
    let chocolates_per_5min := packages_per_5min * dozen_size
    let intervals_per_hour := minutes_in_hour / 5
    let chocolates_per_hour := chocolates_per_5min * intervals_per_hour
    let chocolates_in_4hours := chocolates_per_hour * hours
    chocolates_in_4hours = 1152 := 
by
  let chocolates_per_5min := packages_per_5min * dozen_size
  let intervals_per_hour := minutes_in_hour / 5
  let chocolates_per_hour := chocolates_per_5min * intervals_per_hour
  let chocolates_in_4hours := chocolates_per_hour * hours
  sorry

end wendy_packages_chocolates_l309_30938


namespace bike_license_combinations_l309_30975

theorem bike_license_combinations : 
  let letters := 3
  let digits := 10
  let total_combinations := letters * digits^4
  total_combinations = 30000 := by
  let letters := 3
  let digits := 10
  let total_combinations := letters * digits^4
  sorry

end bike_license_combinations_l309_30975


namespace applicants_majored_in_political_science_l309_30912

theorem applicants_majored_in_political_science
  (total_applicants : ℕ)
  (gpa_above_3 : ℕ)
  (non_political_science_and_gpa_leq_3 : ℕ)
  (political_science_and_gpa_above_3 : ℕ) :
  total_applicants = 40 →
  gpa_above_3 = 20 →
  non_political_science_and_gpa_leq_3 = 10 →
  political_science_and_gpa_above_3 = 5 →
  ∃ P : ℕ, P = 15 :=
by
  intros
  sorry

end applicants_majored_in_political_science_l309_30912


namespace minimum_m_value_l309_30900

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^2 + a * Real.log x + 1

theorem minimum_m_value :
  (∀ x1 x2 : ℝ, x1 ∈ Set.Ici (3 : ℝ) → x2 ∈ Set.Ici (3 : ℝ) → x1 ≠ x2 →
     ∃ a : ℝ, a ∈ Set.Icc (1 : ℝ) (2 : ℝ) ∧
     (f x1 a - f x2 a) / (x2 - x1) < m) →
  m ≥ -20 / 3 := sorry

end minimum_m_value_l309_30900


namespace simplify_expression_l309_30973

theorem simplify_expression (x y z : ℝ) : ((x + y) - (z - y)) - ((x + z) - (y + z)) = 3 * y - z := by
  sorry

end simplify_expression_l309_30973


namespace kingfisher_catch_difference_l309_30982

def pelicanFish : Nat := 13
def fishermanFish (K : Nat) : Nat := 3 * (pelicanFish + K)
def fishermanConditionFish : Nat := pelicanFish + 86

theorem kingfisher_catch_difference (K : Nat) (h1 : K > pelicanFish)
  (h2 : fishermanFish K = fishermanConditionFish) :
  K - pelicanFish = 7 := by
  sorry

end kingfisher_catch_difference_l309_30982


namespace inequality_a4_b4_c4_geq_l309_30956

theorem inequality_a4_b4_c4_geq (a b c : ℝ) : 
  a^4 + b^4 + c^4 ≥ a^2 * b^2 + b^2 * c^2 + c^2 * a^2 := 
by
  sorry

end inequality_a4_b4_c4_geq_l309_30956


namespace no_real_roots_l309_30923

theorem no_real_roots 
    (h : ∀ x : ℝ, (3 * x^2 / (x - 2)) - (3 * x + 8) / 2 + (5 - 9 * x) / (x - 2) + 2 = 0) 
    : False := by
  sorry

end no_real_roots_l309_30923


namespace difference_of_sides_l309_30931

-- Definitions based on conditions
def smaller_square_side (s : ℝ) := s
def larger_square_side (S s : ℝ) (h : (S^2 : ℝ) = 4 * s^2) := S

-- Theorem statement based on the proof problem
theorem difference_of_sides (s S : ℝ) (h : (S^2 : ℝ) = 4 * s^2) : S - s = s := 
by
  sorry

end difference_of_sides_l309_30931


namespace sufficient_not_necessary_condition_l309_30993

theorem sufficient_not_necessary_condition (x k : ℝ) (p : x ≥ k) (q : (2 - x) / (x + 1) < 0) :
  (∀ x, x ≥ k → ((2 - x) / (x + 1) < 0)) ∧ (∃ x, (2 - x) / (x + 1) < 0 ∧ x < k) → k > 2 := by
  sorry

end sufficient_not_necessary_condition_l309_30993


namespace perpendicular_lines_a_value_l309_30945

theorem perpendicular_lines_a_value (a : ℝ) :
  (a * (a + 2) = -1) → a = -1 :=
by
  intro h
  sorry

end perpendicular_lines_a_value_l309_30945


namespace rectangle_other_side_l309_30978

theorem rectangle_other_side (A x y : ℝ) (hA : A = 1 / 8) (hx : x = 1 / 2) (hArea : A = x * y) :
    y = 1 / 4 := 
  sorry

end rectangle_other_side_l309_30978


namespace arithmetic_sequence_a4_eq_1_l309_30995

theorem arithmetic_sequence_a4_eq_1 
  (a : ℕ → ℝ)
  (h_arith_seq : ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d)
  (h_pos : ∀ n : ℕ, a n > 0)
  (h_eq : a 2 ^ 2 + 2 * a 2 * a 6 + a 6 ^ 2 - 4 = 0) : 
  a 4 = 1 :=
sorry

end arithmetic_sequence_a4_eq_1_l309_30995


namespace proof_expectation_red_balls_drawn_l309_30988

noncomputable def expectation_red_balls_drawn : Prop :=
  let total_ways := Nat.choose 5 2
  let ways_2_red := Nat.choose 3 2
  let ways_1_red_1_yellow := Nat.choose 3 1 * Nat.choose 2 1
  let p_X_eq_2 := (ways_2_red : ℝ) / total_ways
  let p_X_eq_1 := (ways_1_red_1_yellow : ℝ) / total_ways
  let expectation := 2 * p_X_eq_2 + 1 * p_X_eq_1
  expectation = 1.2

theorem proof_expectation_red_balls_drawn :
  expectation_red_balls_drawn :=
by
  sorry

end proof_expectation_red_balls_drawn_l309_30988


namespace packed_lunch_needs_l309_30966

-- Definitions based on conditions
def students_A : ℕ := 10
def students_B : ℕ := 15
def students_C : ℕ := 20

def total_students : ℕ := students_A + students_B + students_C

def slices_per_sandwich : ℕ := 4
def sandwiches_per_student : ℕ := 2
def bread_slices_per_student : ℕ := sandwiches_per_student * slices_per_sandwich
def total_bread_slices : ℕ := total_students * bread_slices_per_student

def bags_of_chips_per_student : ℕ := 1
def total_bags_of_chips : ℕ := total_students * bags_of_chips_per_student

def apples_per_student : ℕ := 3
def total_apples : ℕ := total_students * apples_per_student

def granola_bars_per_student : ℕ := 1
def total_granola_bars : ℕ := total_students * granola_bars_per_student

-- Proof goals
theorem packed_lunch_needs :
  total_bread_slices = 360 ∧
  total_bags_of_chips = 45 ∧
  total_apples = 135 ∧
  total_granola_bars = 45 :=
by
  sorry

end packed_lunch_needs_l309_30966


namespace time_to_be_100_miles_apart_l309_30926

noncomputable def distance_apart (x : ℝ) : ℝ :=
  Real.sqrt ((12 * x) ^ 2 + (16 * x) ^ 2)

theorem time_to_be_100_miles_apart : ∃ x : ℝ, distance_apart x = 100 ↔ x = 5 :=
by {
  sorry
}

end time_to_be_100_miles_apart_l309_30926


namespace find_third_number_l309_30928
open BigOperators

def LCM (a b : ℕ) : ℕ := a * b / (Nat.gcd a b)

def LCM_of_three (a b c : ℕ) : ℕ := LCM (LCM a b) c

theorem find_third_number (n : ℕ) (h₁ : LCM 15 25 = 75) (h₂ : LCM_of_three 15 25 n = 525) : n = 7 :=
by 
  sorry

end find_third_number_l309_30928


namespace equation1_solution_equation2_solution_l309_30953

theorem equation1_solution (x : ℝ) (h : 2 * (x - 1) = 2 - 5 * (x + 2)) : x = -6 / 7 :=
sorry

theorem equation2_solution (x : ℝ) (h : (5 * x + 1) / 2 - (6 * x + 2) / 4 = 1) : x = 1 :=
sorry

end equation1_solution_equation2_solution_l309_30953


namespace investment_difference_l309_30901

noncomputable def compound_yearly (P : ℕ) (r : ℚ) (t : ℕ) : ℚ :=
  P * (1 + r)^t

noncomputable def compound_monthly (P : ℕ) (r : ℚ) (months : ℕ) : ℚ :=
  P * (1 + r)^(months)

theorem investment_difference :
  let P := 70000
  let r := 0.05
  let t := 3
  let monthly_r := r / 12
  let months := t * 12
  compound_monthly P monthly_r months - compound_yearly P r t = 263.71 :=
by
  sorry

end investment_difference_l309_30901


namespace value_of_a_l309_30915

theorem value_of_a (a b : ℝ) (h1 : b = 2 * a) (h2 : b = 15 - 4 * a) : a = 5 / 2 :=
by
  sorry

end value_of_a_l309_30915


namespace right_triangle_condition_l309_30941

def fib (n : ℕ) : ℕ :=
  match n with
  | 0 => 0
  | 1 => 4
  | 2 => 4
  | n + 3 => fib (n + 2) + fib (n + 1)

theorem right_triangle_condition (n : ℕ) : 
  ∃ a b c, a = fib n * fib (n + 4) ∧ 
           b = fib (n + 1) * fib (n + 3) ∧ 
           c = 2 * fib (n + 2) ∧
           a * a + b * b = c * c :=
by sorry

end right_triangle_condition_l309_30941


namespace paco_cookies_l309_30961

theorem paco_cookies :
  let initial_cookies := 25
  let ate_cookies := 5
  let remaining_cookies_after_eating := initial_cookies - ate_cookies
  let gave_away_cookies := 4
  let remaining_cookies_after_giving := remaining_cookies_after_eating - gave_away_cookies
  let bought_cookies := 3
  let final_cookies := remaining_cookies_after_giving + bought_cookies
  let combined_bought_and_gave_away := gave_away_cookies + bought_cookies
  (ate_cookies - combined_bought_and_gave_away) = -2 :=
by sorry

end paco_cookies_l309_30961


namespace total_rowing_and_hiking_l309_30929

def total_campers : ℕ := 80
def morning_rowing : ℕ := 41
def morning_hiking : ℕ := 4
def morning_swimming : ℕ := 15
def afternoon_rowing : ℕ := 26
def afternoon_hiking : ℕ := 8
def afternoon_swimming : ℕ := total_campers - afternoon_rowing - afternoon_hiking - (total_campers - morning_rowing - morning_hiking - morning_swimming)

theorem total_rowing_and_hiking : 
  (morning_rowing + afternoon_rowing) + (morning_hiking + afternoon_hiking) = 79 :=
by
  sorry

end total_rowing_and_hiking_l309_30929


namespace no_such_base_exists_l309_30914

theorem no_such_base_exists : ¬ ∃ b : ℕ, (b^3 ≤ 630 ∧ 630 < b^4) ∧ (630 % b) % 2 = 1 := by
  sorry

end no_such_base_exists_l309_30914


namespace number_of_common_tangents_between_circleC_and_circleD_l309_30972

noncomputable def circleC := { p : ℝ × ℝ | p.1^2 + p.2^2 = 1 }

noncomputable def circleD := { p : ℝ × ℝ | p.1^2 + p.2^2 - 4 * p.1 + 2 * p.2 - 4 = 0 }

theorem number_of_common_tangents_between_circleC_and_circleD : 
    ∃ (num_tangents : ℕ), num_tangents = 2 :=
by
    -- Proving the number of common tangents is 2
    sorry

end number_of_common_tangents_between_circleC_and_circleD_l309_30972


namespace moms_took_chocolates_l309_30958

theorem moms_took_chocolates (N : ℕ) (A : ℕ) (M : ℕ) : 
  N = 10 → 
  A = 3 * N →
  A - M = N + 15 →
  M = 5 :=
by
  intros h1 h2 h3
  sorry

end moms_took_chocolates_l309_30958


namespace surface_area_of_given_cylinder_l309_30971

noncomputable def surface_area_of_cylinder (length width : ℝ) : ℝ :=
  let r := (length / (2 * Real.pi))
  let h := width
  2 * Real.pi * r^2 + 2 * Real.pi * r * h

theorem surface_area_of_given_cylinder : 
  surface_area_of_cylinder (4 * Real.pi) 2 = 16 * Real.pi :=
by
  -- Proof will be filled here
  sorry

end surface_area_of_given_cylinder_l309_30971


namespace max_area_rectangle_l309_30999

theorem max_area_rectangle (P : ℕ) (hP : P = 40) : ∃ A : ℕ, A = 100 ∧ ∀ x y : ℕ, 2 * (x + y) = P → x * y ≤ A := by
  sorry

end max_area_rectangle_l309_30999


namespace max_area_isosceles_triangle_l309_30947

theorem max_area_isosceles_triangle (b : ℝ) (h : ℝ) (area : ℝ) 
  (h_cond : h^2 = 1 - b^2 / 4)
  (area_def : area = 1 / 2 * b * h) : 
  area ≤ 2 * Real.sqrt 2 / 3 := 
sorry

end max_area_isosceles_triangle_l309_30947


namespace part1_solution_set_part2_min_value_l309_30954

-- Part 1
noncomputable def f (x : ℝ) : ℝ := 2 * |x + 1| + |3 * x|

theorem part1_solution_set :
  {x : ℝ | f x ≥ 3 * |x| + 1} = {x : ℝ | x ≥ -1/2} ∪ {x : ℝ | x ≤ -3/2} :=
by
  sorry

-- Part 2
noncomputable def f_min (x a b : ℝ) : ℝ := 2 * |x + a| + |3 * x - b|

theorem part2_min_value (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : ∃ x, f_min x a b = 2) :
  3 * a + b = 3 :=
by
  sorry

end part1_solution_set_part2_min_value_l309_30954


namespace find_multiple_l309_30965

theorem find_multiple (x k : ℕ) (hx : x > 0) (h_eq : x + 17 = k * (1/x)) (h_x : x = 3) : k = 60 :=
by
  sorry

end find_multiple_l309_30965


namespace shaded_area_is_correct_l309_30940

noncomputable def square_shaded_area (side : ℝ) (beta : ℝ) (cos_beta : ℝ) : ℝ :=
  if (0 < beta) ∧ (beta < 90) ∧ (cos_beta = 3 / 5) ∧ (side = 2) then 3 / 10 
  else 0

theorem shaded_area_is_correct :
  square_shaded_area 2 beta (3 / 5) = 3 / 10 :=
by
  sorry

end shaded_area_is_correct_l309_30940


namespace increasing_interval_of_f_maximum_value_of_f_l309_30959

open Real

def f (x : ℝ) : ℝ := x^2 - 2 * x

-- Consider x in the interval [-2, 4]
def domain_x (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 4

theorem increasing_interval_of_f :
  ∃a b : ℝ, (a, b) = (1, 4) ∧ ∀ x y : ℝ, domain_x x → domain_x y → a ≤ x → x < y → y ≤ b → f x < f y := sorry

theorem maximum_value_of_f :
  ∃ M : ℝ, M = 8 ∧ ∀ x : ℝ, domain_x x → f x ≤ M := sorry

end increasing_interval_of_f_maximum_value_of_f_l309_30959


namespace area_of_rectangular_field_l309_30957

-- Define the conditions
variables (l w : ℝ)

def perimeter_condition : Prop := 2 * l + 2 * w = 100
def length_width_relation : Prop := l = 3 * w

-- Define the area
def area : ℝ := l * w

-- Prove the area given the conditions
theorem area_of_rectangular_field (h1 : perimeter_condition l w) (h2 : length_width_relation l w) : area l w = 468.75 :=
by sorry

end area_of_rectangular_field_l309_30957
