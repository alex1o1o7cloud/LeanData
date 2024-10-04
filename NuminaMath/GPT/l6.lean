import Mathlib

namespace integral_evaluation_l6_6884

theorem integral_evaluation : ∫ x in -2..2, (x^3 + 2) = 8 :=
by
  intervalIntegral calc
    (∫ x in -2..2, (x^3 + 2)) =
      ((1/4 * x^4 + 2 * x) | -2..2) :
    by
      simp [integral]
      sorry
    _ = 8 : 
    by
      -- Evaluate the antiderivative at the limits
      simp only [one_div, pow_four]
      sorry

end integral_evaluation_l6_6884


namespace place_value_accuracy_l6_6314

theorem place_value_accuracy (x : ℝ) (h : x = 3.20 * 10000) :
  ∃ p : ℕ, p = 100 ∧ (∃ k : ℤ, x / p = k) := by
  sorry

end place_value_accuracy_l6_6314


namespace rectangle_area_l6_6818

-- Definitions based on the conditions
def radius := 6
def diameter := 2 * radius
def width := diameter
def length := 3 * width

-- Statement of the theorem
theorem rectangle_area : (width * length = 432) := by
  sorry

end rectangle_area_l6_6818


namespace fill_time_two_pipes_l6_6803

variable (R : ℝ)
variable (c : ℝ)
variable (t1 : ℝ) (t2 : ℝ)

noncomputable def fill_time_with_pipes (num_pipes : ℝ) (time_per_tank : ℝ) : ℝ :=
  time_per_tank / num_pipes

theorem fill_time_two_pipes (h1 : fill_time_with_pipes 3 t1 = 12) 
                            (h2 : c = R)
                            : fill_time_with_pipes 2 (3 * R * t1) = 18 := 
by
  sorry

end fill_time_two_pipes_l6_6803


namespace repeating_decimal_to_fraction_l6_6001

theorem repeating_decimal_to_fraction (h : 0.\overline{02} = 2 / 99) : 
  2.\overline{06} = 68 / 33 := by
  sorry

end repeating_decimal_to_fraction_l6_6001


namespace permutations_with_exactly_four_not_in_original_positions_l6_6510

theorem permutations_with_exactly_four_not_in_original_positions :
  ∃ (S : Finset (Set (Fin 8))) (card_S : S.card = 1) (T : Finset (Set (Fin 8))) (card_T : T.card = 70),
  (∃ (d : Derangements (Fin 8)) (hd : d.card = 9), S.card * T.card * d.card = 630) :=
sorry

end permutations_with_exactly_four_not_in_original_positions_l6_6510


namespace cos_angle_l6_6868

noncomputable def angle := -19 * Real.pi / 6

theorem cos_angle : Real.cos angle = Real.sqrt 3 / 2 :=
by sorry

end cos_angle_l6_6868


namespace repeating_decimal_as_fraction_l6_6036

theorem repeating_decimal_as_fraction : (0.\overline{02} = 2 / 99) → (2.\overline{06} = 68 / 33) :=
by
  sorry

end repeating_decimal_as_fraction_l6_6036


namespace problem_1_problem_2_l6_6138

-- Define set A
def A := {x : ℝ | 6 / (x + 1) ≥ 1}

-- Define set B depending on m (used in Part 2)
def B (m : ℝ) := {x : ℝ | x^2 - 2 * x - m < 0}

-- Problem (1)
theorem problem_1 : (A ∩ {x : ℝ | 3 ≤ x ∨ x ≤ -1}) = {x : ℝ | 3 ≤ x ∧ x ≤ 5} := begin
    sorry
end

-- Problem (2) to find m
theorem problem_2 (m : ℝ) : (A ∩ B m) = {x : ℝ | -1 < x ∧ x < 4} ↔ m = 8 := begin
    sorry
end

end problem_1_problem_2_l6_6138


namespace parabola_y_intercepts_l6_6982

theorem parabola_y_intercepts : 
  (∀ y : ℝ, 3 * y^2 - 6 * y + 1 = 0) → (∃ y1 y2 : ℝ, y1 ≠ y2) :=
by sorry

end parabola_y_intercepts_l6_6982


namespace max_cos_a_l6_6652

theorem max_cos_a (a b c : ℝ) 
  (h1 : Real.sin a = Real.cos b) 
  (h2 : Real.sin b = Real.cos c) 
  (h3 : Real.sin c = Real.cos a) : 
  Real.cos a = Real.sqrt 2 / 2 := by
sorry

end max_cos_a_l6_6652


namespace acute_angle_sine_diff_l6_6551

theorem acute_angle_sine_diff (α β : ℝ) (h₀ : 0 < α ∧ α < π / 2) (h₁ : 0 < β ∧ β < π / 2)
  (h₂ : Real.sin α = (Real.sqrt 5) / 5) (h₃ : Real.sin (α - β) = -(Real.sqrt 10) / 10) : β = π / 4 :=
sorry

end acute_angle_sine_diff_l6_6551


namespace a_n_correct_sum_2023_b_n_l6_6964

-- Given conditions
variables {a : ℕ → ℤ} {b : ℕ → ℝ}

-- Assuming the arithmetic sequence and conditions
axiom a_arith_seq : ∃ d : ℤ, ∀ n : ℕ, a(n + 1) = a n + d
axiom a_2 : a 2 = 3
axiom geom_seq : (a 3), (a 5), (a 8) form a geometric sequence

-- Define the common difference
noncomputable def d : ℤ := sorry -- needs to be proven based on the information

-- Derived formula for a_n
noncomputable def a_n_formula (n : ℕ) : ℤ := n + 1

-- Prove that the derived formula is indeed a_n
theorem a_n_correct (n : ℕ) : a n = a_n_formula n :=
sorry

-- Define b_n and sum the first 2023 terms
noncomputable def b_n (n : ℕ) : ℝ := (a_n_formula n) * (Real.cos (π * (a_n_formula n) / 2))

-- Definitions to compute the sum
noncomputable def sum_b_n : ℕ → ℝ
| 0     := 0
| (n+1) := b_n (n + 1) + sum_b_n n

-- Prove the sum of the first 2023 terms
theorem sum_2023_b_n : sum_b_n 2023 = 1012 :=
sorry

end a_n_correct_sum_2023_b_n_l6_6964


namespace repeating_decimal_as_fraction_l6_6042

theorem repeating_decimal_as_fraction : (0.\overline{02} = 2 / 99) → (2.\overline{06} = 68 / 33) :=
by
  sorry

end repeating_decimal_as_fraction_l6_6042


namespace determine_length_BI_l6_6632

noncomputable def length_BI (AB AC BC : ℝ) (I : Point) (BI : ℝ) : Prop :=
  let ABC : Triangle := {A := A, B := B, C := C}
  ∧ AB = 30
  ∧ AC = 29
  ∧ BC = 27
  ∧ I = internalAngleBisectorsIntersection ABC
  ∧ BI = 13

theorem determine_length_BI : length_BI 30 29 27 I 13 :=
by 
  sorry

end determine_length_BI_l6_6632


namespace affine_parallel_preservation_l6_6298

-- Definition of an affine transformation for the purpose of this proof.
def affine_transformation (T : ℝ × ℝ ⟶ ℝ × ℝ) : Prop :=
  ∀ (x1 x2 x3 : ℝ × ℝ) (p : ℝ), (x2 - x1) = p * (x3 - x1) → (T x2 - T x1) = p * (T x3 - T x1)

-- A line is preserved and transformed under an affine transformation.
def line_preserved (T : ℝ × ℝ ⟶ ℝ × ℝ) (L : set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ × ℝ, ∀ x ∈ L, x = a + b ∨ x = a - b

-- Parallel lines do not intersect.
def parallel_lines (L1 L2 : set (ℝ × ℝ)) : Prop :=
  ∀ x ∈ L1, ∀ y ∈ L2, x ≠ y

theorem affine_parallel_preservation (T : ℝ × ℝ ⟶ ℝ × ℝ) (L1 L2 : set (ℝ × ℝ))
  (T_affine : affine_transformation T)
  (L1_line : line_preserved T L1) (L2_line : line_preserved T L2)
  (L1_L2_parallel : parallel_lines L1 L2) :
  parallel_lines (T '' L1) (T '' L2) :=
sorry

end affine_parallel_preservation_l6_6298


namespace sub_condition_1_sub_condition_2_l6_6085

-- Define the conditions
def total_cities : ℕ := 8
def cities_to_select : ℕ := 6

-- Define the specific cities A and B
def city_A : ℕ := 1
def city_B : ℕ := 2

-- Prove there are 12 methods and 8640 routes under sub-condition (1)
theorem sub_condition_1 :
  let methods_1 : ℕ := 2 * Nat.choose (total_cities - 2) 5 in
  let routes_1 : ℕ := methods_1 * Nat.perm cities_to_select cities_to_select in
  methods_1 = 12 ∧ routes_1 = 8640 :=
by
  sorry

-- Prove there are 27 methods and 19440 routes under sub-condition (2)
theorem sub_condition_2 :
  let methods_1 : ℕ := 2 * Nat.choose (total_cities - 2) 5 in
  let methods_2 := methods_1 + Nat.choose (total_cities - 2) 4 in
  let routes_2 : ℕ := methods_2 * Nat.perm cities_to_select cities_to_select in
  methods_2 = 27 ∧ routes_2 = 19440 :=
by
  sorry

end sub_condition_1_sub_condition_2_l6_6085


namespace recurring_decimal_to_fraction_l6_6053

theorem recurring_decimal_to_fraction
  (h : (2:ℚ) + 3 * (2 / 99) = 2.06) :
  (2:ℚ) + 0.\overline{06} = (68 / 33) :=
by
  -- Given: 0.\overline{02} = 2 / 99
  have h0 : (0.\overline{02} : ℚ) = 2 / 99 := by sorry

  -- 0.\overline{06} = 3 * 0.\overline{02}
  have h1 : (0.\overline{06} : ℚ) = 3 * (0.\overline{02} : ℚ) :=
    by rw [← h0]; sorry

  -- Hence, 0.\overline{06} = 6 / 99 = 2 / 33
  have h2 : (0.\overline{06} : ℚ) = 2 / 33 :=
    by sorry

  -- Therefore, 2.\overline{06} = 2 + 0.\overline{06} = 2 + 2 / 33 = 68 / 33
  show (2:ℚ) + (0.\overline{06} : ℚ) = 68 / 33
    by sorry

end recurring_decimal_to_fraction_l6_6053


namespace plains_routes_count_l6_6176

theorem plains_routes_count (total_cities mountainous_cities plains_cities total_routes routes_mountainous_pairs: ℕ) :
  total_cities = 100 →
  mountainous_cities = 30 →
  plains_cities = 70 →
  total_routes = 150 →
  routes_mountainous_pairs = 21 →
  let endpoints_mountainous := mountainous_cities * 3 in
  let endpoints_mountainous_pairs := routes_mountainous_pairs * 2 in
  let endpoints_mountainous_plains := endpoints_mountainous - endpoints_mountainous_pairs in
  let endpoints_plains := plains_cities * 3 in
  let routes_mountainous_plains := endpoints_mountainous_plains in
  let endpoints_plains_pairs := endpoints_plains - routes_mountainous_plains in
  let routes_plains_pairs := endpoints_plains_pairs / 2 in
  routes_plains_pairs = 81 :=
by
  intros h1 h2 h3 h4 h5
  dsimp
  rw [h1, h2, h3, h4, h5]
  sorry

end plains_routes_count_l6_6176


namespace even_poly_iff_a_zero_l6_6605

theorem even_poly_iff_a_zero (a : ℝ) : 
  (∀ x : ℝ, (x^2 + a*x + 3) = (x^2 - a*x + 3)) → a = 0 :=
by
  sorry

end even_poly_iff_a_zero_l6_6605


namespace original_cost_price_of_car_l6_6833

theorem original_cost_price_of_car
    (S_m S_f C : ℝ)
    (h1 : S_m = 0.86 * C)
    (h2 : S_f = 54000)
    (h3 : S_f = 1.20 * S_m) :
    C = 52325.58 :=
by
    sorry

end original_cost_price_of_car_l6_6833


namespace coefficient_of_x4_in_expansion_l6_6360

theorem coefficient_of_x4_in_expansion :
  (∑ k in Finset.range (8 + 1), (Nat.choose 8 k) * (x : ℝ)^(8 - k) * (3 * Real.sqrt 2)^k).coeff 4 = 22680 :=
by
  sorry

end coefficient_of_x4_in_expansion_l6_6360


namespace affine_parallel_preservation_l6_6297

-- Definition of an affine transformation for the purpose of this proof.
def affine_transformation (T : ℝ × ℝ ⟶ ℝ × ℝ) : Prop :=
  ∀ (x1 x2 x3 : ℝ × ℝ) (p : ℝ), (x2 - x1) = p * (x3 - x1) → (T x2 - T x1) = p * (T x3 - T x1)

-- A line is preserved and transformed under an affine transformation.
def line_preserved (T : ℝ × ℝ ⟶ ℝ × ℝ) (L : set (ℝ × ℝ)) : Prop :=
  ∃ a b : ℝ × ℝ, ∀ x ∈ L, x = a + b ∨ x = a - b

-- Parallel lines do not intersect.
def parallel_lines (L1 L2 : set (ℝ × ℝ)) : Prop :=
  ∀ x ∈ L1, ∀ y ∈ L2, x ≠ y

theorem affine_parallel_preservation (T : ℝ × ℝ ⟶ ℝ × ℝ) (L1 L2 : set (ℝ × ℝ))
  (T_affine : affine_transformation T)
  (L1_line : line_preserved T L1) (L2_line : line_preserved T L2)
  (L1_L2_parallel : parallel_lines L1 L2) :
  parallel_lines (T '' L1) (T '' L2) :=
sorry

end affine_parallel_preservation_l6_6297


namespace axis_of_symmetry_same_value_of_a_l6_6568

theorem axis_of_symmetry_same_value_of_a :
  ∃ a : ℝ, (∀ k n : ℤ, (λ x : ℝ, sin (x + π/6)^2) = (λ x : ℝ, sin (2 * x) + a * cos (2 * x))
    → a = - (sqrt 3) / 3) :=
sorry

end axis_of_symmetry_same_value_of_a_l6_6568


namespace DVDs_sold_168_l6_6171

-- Definitions of the conditions
def CDs_sold := ℤ
def DVDs_sold := ℤ

def ratio_condition (C D : ℤ) : Prop := D = 16 * C / 10
def total_condition (C D : ℤ) : Prop := D + C = 273

-- The main statement to prove
theorem DVDs_sold_168 (C D : ℤ) 
  (h1 : ratio_condition C D) 
  (h2 : total_condition C D) : D = 168 :=
sorry

end DVDs_sold_168_l6_6171


namespace parallel_lines_perpendicular_lines_l6_6571

-- Define the lines l1 and l2 as per the given conditions
def line1 (a : ℝ) : ℝ → ℝ → Prop := λ x y, a * x - y + 2 = 0
def line2 (a : ℝ) : ℝ → ℝ → Prop := λ x y, (a + 2) * x - a * y - 2 = 0

-- Define the slopes of the lines
def slope1 (a : ℝ) := a
def slope2 (a : ℝ) := (a + 2) / a

-- The proof problem for parallel lines
theorem parallel_lines (a : ℝ) (h : slope1 a = slope2 a) : a = 2 := sorry

-- The proof problem for perpendicular lines
theorem perpendicular_lines (a : ℝ) (h : slope1 a * slope2 a = -1) : a = 0 ∨ a = -3 := sorry

end parallel_lines_perpendicular_lines_l6_6571


namespace length_AC_and_area_OAC_l6_6544

open Real EuclideanGeometry

def ellipse (x y : ℝ) : Prop :=
  x^2 + 2 * y^2 = 2

def line_1 (x y : ℝ) : Prop :=
  y = x + 1

def line_2 (B : ℝ × ℝ) (P : ℝ × ℝ) : Prop :=
  B.fst = 3 * P.fst ∧ B.snd = 3 * P.snd

theorem length_AC_and_area_OAC 
  (A C : ℝ × ℝ) 
  (B P : ℝ × ℝ) 
  (O : ℝ × ℝ := (0, 0)) 
  (h1 : ellipse A.fst A.snd) 
  (h2 : ellipse C.fst C.snd) 
  (h3 : line_1 A.fst A.snd) 
  (h4 : line_1 C.fst C.snd) 
  (h5 : line_2 B P) 
  (h6 : (P.fst = (A.fst + C.fst) / 2) ∧ (P.snd = (A.snd + C.snd) / 2)) : 
  |(dist A C)| = 4/3 * sqrt 2 ∧
  (1/2 * abs (A.fst * C.snd - C.fst * A.snd)) = 4/9 := sorry

end length_AC_and_area_OAC_l6_6544


namespace max_real_solutions_l6_6484

noncomputable def max_number_of_real_solutions (n : ℕ) (y : ℝ) : ℕ :=
if (n + 1) % 2 = 1 then 1 else 0

theorem max_real_solutions (n : ℕ) (hn : 0 < n) (y : ℝ) :
  max_number_of_real_solutions n y = 1 :=
by
  sorry

end max_real_solutions_l6_6484


namespace repeating_decimal_as_fraction_l6_6035

theorem repeating_decimal_as_fraction : (0.\overline{02} = 2 / 99) → (2.\overline{06} = 68 / 33) :=
by
  sorry

end repeating_decimal_as_fraction_l6_6035


namespace students_left_l6_6219

theorem students_left (initial students new students end students left : ℕ)
  (h1: initial = 31)
  (h2: new = 11)
  (h3: end = 37)
  (h4: initial + new - left = end)
  : left = 5 :=
by
  sorry

end students_left_l6_6219


namespace largest_circle_area_is_287_l6_6838

noncomputable def largest_circle_area_from_string (length width : ℝ) (h : length = 2 * width) (area_rect : width * length = 200) : ℝ :=
  let perimeter := 2 * (width + length) in
  let radius := perimeter / (2 * Real.pi) in
  Real.pi * radius^2

theorem largest_circle_area_is_287 (length width : ℝ)
  (h1 : length = 2 * width)
  (h2 : width * length = 200)
  (h3 : ∀ (perimeter : ℝ), perimeter = 2 * (width + length)) :
  largest_circle_area_from_string length width h1 h2 ≈ 287 :=
by
  sorry

end largest_circle_area_is_287_l6_6838


namespace find_AB_plus_AC_l6_6476

noncomputable theory
open_locale classical

-- Define the problem conditions
def radius := 5
def center_O_A_dist := 13
def BC_length := 7

-- Define the initial parameters
variables (O A : Point) (ω : Circle O radius) 

-- Hypotheses/assumptions based on problem statement
axiom dist_OA : dist O A = center_O_A_dist
axiom tangent_BC : tangent BC_length ω
axiom points_BC : ∃ B C : Point, tangent B ω ∧ tangent C ω ∧ dist B C = BC_length

-- Result to prove
theorem find_AB_plus_AC : 
  let AB := tangent_length A ω
  let AC := tangent_length A ω in
  AB + AC = 17 :=
sorry

end find_AB_plus_AC_l6_6476


namespace distance_is_30_l6_6279

-- Define given conditions
def total_distance : ℕ := 120
def trips : ℕ := 4

-- Define the distance from Mrs. Hilt's desk to the water fountain
def distance_to_water_fountain : ℕ := total_distance / trips

-- Prove the distance is 30 feet
theorem distance_is_30 : distance_to_water_fountain = 30 :=
by
  -- Utilizing the division defined in distance_to_water_fountain
  sorry

end distance_is_30_l6_6279


namespace wheel_moves_in_one_hour_l6_6852

theorem wheel_moves_in_one_hour
  (rotations_per_minute : ℕ)
  (distance_per_rotation_cm : ℕ)
  (minutes_in_hour : ℕ) :
  rotations_per_minute = 20 →
  distance_per_rotation_cm = 35 →
  minutes_in_hour = 60 →
  let distance_per_rotation_m : ℚ := distance_per_rotation_cm / 100
  let total_rotations_per_hour : ℕ := rotations_per_minute * minutes_in_hour
  let total_distance_in_hour : ℚ := distance_per_rotation_m * total_rotations_per_hour
  total_distance_in_hour = 420 := by
  intros
  sorry

end wheel_moves_in_one_hour_l6_6852


namespace j_mod_2_not_zero_l6_6168

theorem j_mod_2_not_zero (x j : ℤ) (h : 2 * x - j = 11) : j % 2 ≠ 0 :=
sorry

end j_mod_2_not_zero_l6_6168


namespace necessary_but_not_sufficient_l6_6701

-- Definitions based on the conditions
def on_curve (M : ℝ × ℝ) : Prop := M.2 = 4 * M.1 ^ 2
def satisfies_condition (M : ℝ × ℝ) : Prop := sqrt M.2 = 2 * M.1

-- Theorem statement: 
-- Prove that the condition is necessary but not sufficient for the point to be on the curve
theorem necessary_but_not_sufficient (M : ℝ × ℝ) :
  (satisfies_condition M → on_curve M) ∧ ¬(on_curve M → satisfies_condition M) :=
by
  sorry

end necessary_but_not_sufficient_l6_6701


namespace find_certain_fraction_l6_6505

theorem find_certain_fraction (x y : ℚ) : (3/8) / (x/y) = (3/4) / (2/5) → x / y = 1 / 5 :=
begin
  intro h,
  -- Proof would go here
  sorry
end

end find_certain_fraction_l6_6505


namespace find_p_of_hyperbola_and_parabola_l6_6935

theorem find_p_of_hyperbola_and_parabola
  (a b p : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < p)
  (h4 : ∀ {x y : ℝ}, x^2 / a^2 - y^2 / b^2 = 1)
  (h5 : ∀ {x y : ℝ}, y^2 = 2 * p * x)
  (eccentricity : ∀ {c : ℝ}, c = 2 * a)
  (area : ∀ {A B : ℝ × ℝ}, is_triangle_area (A B (0, 0)) (sqrt 3 / 3)) :
  p = 2 * sqrt 3 / 3 :=
by
  sorry

end find_p_of_hyperbola_and_parabola_l6_6935


namespace parabola_properties_l6_6559

noncomputable def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem parabola_properties (a b c : ℝ) (h₀ : a ≠ 0)
    (h₁ : parabola a b c (-1) = -1)
    (h₂ : parabola a b c 0 = 1)
    (h₃ : parabola a b c (-2) > 1) :
    (a * b * c > 0) ∧
    (∃ Δ : ℝ, Δ > 0 ∧ (Δ = b^2 - 4*a*c)) ∧
    (a + b + c > 7) :=
sorry

end parabola_properties_l6_6559


namespace decimal_to_fraction_l6_6018

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l6_6018


namespace positive_slope_asymptote_l6_6879

theorem positive_slope_asymptote (A B : ℝ × ℝ) (x y : ℝ) (PA PB : ℝ) :
  A = (2, 3) → B = (6, 3) → sqrt ((x - 2)^2 + (y - 3)^2) - sqrt ((x - 6)^2 + (y - 3)^2) = 4 →
  PA = sqrt ((x - 2)^2 + (y - 3)^2) → PB = sqrt ((x - 6)^2 + (y - 3)^2) → PA - PB = 4 →
  (∃ b a : ℝ, b = sqrt (5) ∧ a = 2 ∧ (± (b / a)) = ± (sqrt (5) / 2)) :=
by
  intros hA hB hEq hPA hPB hDiff
  use (sqrt 5), 2
  split
  exact sqrt 5
  split
  exact 2
  split
  sorry


end positive_slope_asymptote_l6_6879


namespace probability_event_A_l6_6742

theorem probability_event_A (P : Type) {A B : P → Prop} [probability_space P] :
  (Pr {x | B x} = 0.4) →
  (Pr {x | A x ∧ B x} = 0.25) →
  (Pr {x | A x ∨ B x} = 0.6) →
  Pr {x | A x} = 0.45 :=
by
  intros hPB hPAB hPAB_union
  -- begins the proof which is not required by the prompt
  sorry

end probability_event_A_l6_6742


namespace modulus_Z_l6_6264

theorem modulus_Z (Z : ℂ) (h : Z * (2 - 3 * Complex.I) = 6 + 4 * Complex.I) : Complex.abs Z = 2 := 
sorry

end modulus_Z_l6_6264


namespace max_area_inscribed_circle_total_area_four_quarter_circles_l6_6507

noncomputable def pi : ℝ := Real.pi
def side_length : ℝ := 20
def radius (s : ℝ) : ℝ := s / 2
def area_circle (r : ℝ) : ℝ := pi * r^2
def area_quarter_circle (r : ℝ) : ℝ := (1/4) * area_circle r
def total_area_quarter_circles (r : ℝ) : ℝ := 4 * area_quarter_circle r

theorem max_area_inscribed_circle : area_circle (radius side_length) = 100 * pi :=
by
  have r : ℝ := radius side_length
  have a : ℝ := area_circle r
  have r_eq : r = 10 := by sorry
  have a_eq : a = 100 * pi := by sorry
  exact a_eq

theorem total_area_four_quarter_circles : total_area_quarter_circles (radius side_length) = 100 * pi :=
by
  have r : ℝ := radius side_length
  have total_area : ℝ := total_area_quarter_circles r
  have r_eq : r = 10 := by sorry
  have total_area_eq : total_area = 100 * pi := by sorry
  exact total_area_eq

end max_area_inscribed_circle_total_area_four_quarter_circles_l6_6507


namespace polynomial_factorization_l6_6381

theorem polynomial_factorization :
  (x : ℤ[X]) →
  x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by
  intros x
  sorry

end polynomial_factorization_l6_6381


namespace no_integer_solutions_for_binomial_l6_6295

theorem no_integer_solutions_for_binomial (n k l m : ℕ) (hl : l ≥ 2) (hk : 4 ≤ k ∧ k ≤ n - 4) :
  ¬ ∃ (n k l m : ℕ), (l ≥ 2) ∧ (4 ≤ k) ∧ (k ≤ n - 4) ∧ nat.choose n k = m^l :=
by
  sorry

end no_integer_solutions_for_binomial_l6_6295


namespace sum_of_odd_numbers_less_than_20_l6_6081

theorem sum_of_odd_numbers_less_than_20 : 
  (∑ i in Finset.range 10, (2 * i + 1)) = 100 := 
by 
  sorry

end sum_of_odd_numbers_less_than_20_l6_6081


namespace like_terms_exponents_l6_6985

theorem like_terms_exponents (m n : ℕ) (x y : ℝ) (h : 2 * x^(2*m) * y^6 = -3 * x^8 * y^(2*n)) : m = 4 ∧ n = 3 :=
by 
  sorry

end like_terms_exponents_l6_6985


namespace ages_proof_l6_6243

variable (Claire Jessica Mike Nisha : ℕ)

-- Conditions
def condition1 : Jessica = Claire + 6 := by sorry
def condition2 : Claire = 20 - 2 := by sorry
def condition3 : Mike = 2 * (Jessica - 3) := by sorry
def condition4 : Nisha * Nisha = Claire * Jessica := by sorry

-- Theorem to prove
theorem ages_proof : Mike = 42 ∧ Nisha ≈ 21 := by
  have h1 : Jessica = Claire + 6 := condition1
  have h2 : Claire = 18 := condition2
  have h3 : Mike = 42 := condition3
  have h4 : Nisha ≈ 21 := by 
    -- Using approximation here
    sorry
  exact ⟨h3, h4⟩

end ages_proof_l6_6243


namespace repeating_decimal_to_fraction_l6_6000

theorem repeating_decimal_to_fraction (h : 0.\overline{02} = 2 / 99) : 
  2.\overline{06} = 68 / 33 := by
  sorry

end repeating_decimal_to_fraction_l6_6000


namespace plains_routes_count_l6_6178

theorem plains_routes_count (total_cities mountainous_cities plains_cities total_routes routes_mountainous_pairs: ℕ) :
  total_cities = 100 →
  mountainous_cities = 30 →
  plains_cities = 70 →
  total_routes = 150 →
  routes_mountainous_pairs = 21 →
  let endpoints_mountainous := mountainous_cities * 3 in
  let endpoints_mountainous_pairs := routes_mountainous_pairs * 2 in
  let endpoints_mountainous_plains := endpoints_mountainous - endpoints_mountainous_pairs in
  let endpoints_plains := plains_cities * 3 in
  let routes_mountainous_plains := endpoints_mountainous_plains in
  let endpoints_plains_pairs := endpoints_plains - routes_mountainous_plains in
  let routes_plains_pairs := endpoints_plains_pairs / 2 in
  routes_plains_pairs = 81 :=
by
  intros h1 h2 h3 h4 h5
  dsimp
  rw [h1, h2, h3, h4, h5]
  sorry

end plains_routes_count_l6_6178


namespace breadth_of_rectangle_l6_6802

theorem breadth_of_rectangle (b l : ℝ) (h1 : l * b = 24 * b) (h2 : l - b = 10) : b = 14 :=
by
  sorry

end breadth_of_rectangle_l6_6802


namespace polynomial_factorization_l6_6386

theorem polynomial_factorization :
  ∀ x : ℤ, x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by sorry

end polynomial_factorization_l6_6386


namespace find_r_l6_6907

theorem find_r (r : ℝ) :
  (r ≠ 7) →
  ((r^2 - 6r + 9) / (r^2 - 9r + 14) = (r^2 - 4r - 21) / (r^2 - 2r - 35)) →
  (r = 3 ∨ r = (-1 + real.sqrt 69) / 2 ∨ r = (-1 - real.sqrt 69) / 2) :=
by
  intro h1 h2
  sorry

end find_r_l6_6907


namespace dihedral_angle_AC1K_AC1N_l6_6933

-- Define the geometrical setup of the cube and the inscribed sphere
variables {Point : Type} [metric_space Point] [normed_group Point]
variables (A B C D A1 B1 C1 D1 K N : Point)

-- Assume A is a point on the cube, a plane passing through A and tangent to the sphere
-- intersects the edges A1B1 and A1D1 at points K and N respectively
noncomputable def is_tangent_plane_to_inscribed_sphere (P : Point) (sphere_center : Point) (sphere_radius : ℝ) : Prop :=
∃ plane : set Point, P ∈ plane ∧ (∀ Q ∈ plane, dist Q sphere_center = sphere_radius)

-- Dihedral angle measure between planar faces
noncomputable def dihedral_angle (plane1 plane2 : set Point) : ℝ := 
sorry  -- Dihedral angle computation placeholder

-- Main theorem: Dihedral angle between planes AC1K and AC1N
theorem dihedral_angle_AC1K_AC1N :
  let cube := [A, B, C, D, A1, B1, C1, D1] 
  let sphere_center := /* choose appropriate center of the sphere */
  let sphere_radius := /* choose appropriate radius of the sphere */

  is_tangent_plane_to_inscribed_sphere A sphere_center sphere_radius →
  dihedral_angle {X | X ∈ Planar (A, C1, K)} {X | X ∈ Planar (A, C1, N)} = π / 3 :=
begin
  sorry -- Proof of the theorem will be done here
end

end dihedral_angle_AC1K_AC1N_l6_6933


namespace no_one_scored_15_l6_6411

theorem no_one_scored_15 (n : ℕ) (a : Fin n → ℕ) 
  (h1 : n = 7317)
  (h2 : ∀ i j, i ≠ j → a i ≠ a j) 
  (h3 : ∀ i, a i < ∑ j in Finset.univ.erase i, a j) :
  ¬ ∃ i, a i = 15 :=
by
  sorry

end no_one_scored_15_l6_6411


namespace recurring_decimal_reduced_fraction_l6_6060

noncomputable def recurring_decimal_as_fraction : Prop := 
  ∀ (x y : ℚ), (x = 2.06) ∧ (y = 0.02) → y = 2 / 99 → x = 68 / 33

theorem recurring_decimal_reduced_fraction (x y : ℚ) 
  (h1 : x = 2 + 0.06) (h2 : y = 0.02) (h3 : y = 2 / 99) : 
  x = 68 / 33 := 
begin
  -- Proof here
  sorry
end

end recurring_decimal_reduced_fraction_l6_6060


namespace scale_model_height_l6_6462

theorem scale_model_height :
  let scale_ratio : ℚ := 1 / 25
  let actual_height : ℚ := 151
  let model_height : ℚ := actual_height * scale_ratio
  round model_height = 6 :=
by
  sorry

end scale_model_height_l6_6462


namespace population_exceeds_l6_6618

theorem population_exceeds (n : ℕ) : (∃ n, 4 * 3^n > 200) ∧ ∀ m, m < n → 4 * 3^m ≤ 200 := by
  sorry

end population_exceeds_l6_6618


namespace area_of_triangle_l6_6745

theorem area_of_triangle (a b c : ℝ) (h1 : a = 18) (h2 : b = 80) (h3 : c = 82) 
  (h_right : a^2 + b^2 = c^2) : (1/2 * a * b) = 720 :=
begin
  sorry
end

end area_of_triangle_l6_6745


namespace probability_two_hearts_is_one_seventeenth_l6_6079

-- Define the problem parameters
def totalCards : ℕ := 52
def hearts : ℕ := 13
def drawCount : ℕ := 2

-- Define function to calculate combinations
def combination (n k : ℕ) : ℕ :=
  n.choose k

-- Define the probability calculation
def probability_drawing_two_hearts : ℚ :=
  (combination hearts drawCount) / (combination totalCards drawCount)

-- State the theorem to be proved
theorem probability_two_hearts_is_one_seventeenth :
  probability_drawing_two_hearts = 1 / 17 :=
by
  -- Proof not required, so provide sorry
  sorry

end probability_two_hearts_is_one_seventeenth_l6_6079


namespace recurring_decimal_reduced_fraction_l6_6062

noncomputable def recurring_decimal_as_fraction : Prop := 
  ∀ (x y : ℚ), (x = 2.06) ∧ (y = 0.02) → y = 2 / 99 → x = 68 / 33

theorem recurring_decimal_reduced_fraction (x y : ℚ) 
  (h1 : x = 2 + 0.06) (h2 : y = 0.02) (h3 : y = 2 / 99) : 
  x = 68 / 33 := 
begin
  -- Proof here
  sorry
end

end recurring_decimal_reduced_fraction_l6_6062


namespace construct_polygon_a_construct_polygon_b_construct_polygon_c_l6_6394

theorem construct_polygon_a (n : ℕ) (circle : Set Point) (M : Fin n → Point) (l : Fin n → Set Point) :
  ∃ polygon : Polygon, (∀ i, vertex polygon i ∈ l i) ∧ (∀ i, side polygon i ∋ M i) :=
  sorry

theorem construct_polygon_b (n : ℕ) (circle : Set Point) (M : Fin n → Point) :
  ∃ polygon : Polygon, (∃ v, v ∈ circle) ∧ (∀ i, side polygon i ∋ M i) :=
  sorry

theorem construct_polygon_c (n : ℕ) (circle : Set Point) (M : List (Option Point)) (l : List (Option (Set Point))) (lengths : List (Option ℝ)) :
  ∃ polygon : Polygon, 
    (∀ i, match M[i] with | some p => side polygon i ∋ p | none => True end) ∧
    (∀ i, match l[i] with | some line => side_polygon i ∥ line | none => True end) ∧
    (∀ i, match lengths[i] with | some len => length (side_polygon i) = len | none => True end) :=
  sorry

end construct_polygon_a_construct_polygon_b_construct_polygon_c_l6_6394


namespace zero_in_interval_l6_6239

def f (x : ℝ) : ℝ := x^3 + 3 * x - 1

theorem zero_in_interval :
  (f 0 < 0) → (f 0.5 > 0) → (f 0.25 < 0) → ∃ x, 0.25 < x ∧ x < 0.5 ∧ f x = 0 :=
by
  intro h0 h05 h025
  -- This is just the statement; the proof is not required as per instructions
  sorry

end zero_in_interval_l6_6239


namespace number_of_plains_routes_is_81_l6_6188

-- Define the number of cities in each region
def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70

-- Define the number of routes established over three years
def total_routes : ℕ := 150
def routes_per_year : ℕ := 50

-- Define the number of routes connecting pairs of mountainous cities
def mountainous_routes : ℕ := 21

-- Define a function to calculate the number of routes connecting pairs of plains cities
def plains_routes : ℕ :=
  let total_endpoints := total_routes * 2
  let mountainous_endpoints := mountainous_cities * 3
  let plains_endpoints := plains_cities * 3
  let mountainous_pair_endpoints := mountainous_routes * 2
  let mountain_plain_routes := (mountainous_endpoints - mountainous_pair_endpoints) / 2
  let plain_only_endpoints := plains_endpoints - mountain_plain_routes
  plain_only_endpoints / 2

theorem number_of_plains_routes_is_81 : plains_routes = 81 := 
  sorry

end number_of_plains_routes_is_81_l6_6188


namespace plains_routes_count_l6_6194

-- Defining the total number of cities and the number of cities in each region
def total_cities : Nat := 100
def mountainous_cities : Nat := 30
def plains_cities : Nat := total_cities - mountainous_cities

-- Defining the number of routes established each year and over three years
def routes_per_year : Nat := 50
def total_routes : Nat := routes_per_year * 3

-- Defining the number of routes connecting pairs of mountainous cities
def mountainous_routes : Nat := 21

-- The statement to prove the number of routes connecting pairs of plains cities
theorem plains_routes_count :
  plains_cities = 70 →
  total_routes = 150 →
  mountainous_routes = 21 →
  3 * mountainous_cities - 2 * mountainous_routes = 48 →
  3 * plains_cities - 48 = 162 →
  81 = 81 := sorry

end plains_routes_count_l6_6194


namespace shaded_area_of_square_l6_6450

theorem shaded_area_of_square (s r : ℝ) (hs : s = 15) (hr : r = 5) : 
  let square_area := s * s,
      quarter_circle_area := (1 / 4) * π * (r * r),
      total_circle_area := 4 * quarter_circle_area,
      shaded_area := square_area - total_circle_area
  in shaded_area = 225 - 25 * π := 
by {
  sorry
}

end shaded_area_of_square_l6_6450


namespace abs_eq_imp_b_eq_2_l6_6134

theorem abs_eq_imp_b_eq_2 (b : ℝ) (h : |1 - b| = |3 - b|) : b = 2 := 
sorry

end abs_eq_imp_b_eq_2_l6_6134


namespace inequality_solution_sets_l6_6124

theorem inequality_solution_sets (a b : ℝ) :
  (∀ x : ℝ, ax^2 - 5 * x + b > 0 ↔ x < -1 / 3 ∨ x > 1 / 2) →
  (∀ x : ℝ, bx^2 - 5 * x + a > 0 ↔ -3 < x ∧ x < 2) :=
by sorry

end inequality_solution_sets_l6_6124


namespace ellipse_equation_x_axis_ellipse_equation_y_axis_parabola_equation_x_axis_parabola_equation_y_axis_l6_6504

-- Define the conditions for the ellipse problem
def major_axis_length : ℝ := 10
def focal_length : ℝ := 4

-- Define the conditions for the parabola problem
def point_P : ℝ × ℝ := (-2, -4)

-- The equations to be proven
theorem ellipse_equation_x_axis :
  2 * (5 : ℝ) = major_axis_length ∧ 2 * (2 : ℝ) = focal_length →
  (5 : ℝ)^2 - (2 : ℝ)^2 = 21 →
  (∀ x y : ℝ, x^2 / 25 + y^2 / 21 = 1) := sorry

theorem ellipse_equation_y_axis :
  2 * (5 : ℝ) = major_axis_length ∧ 2 * (2 : ℝ) = focal_length →
  (5 : ℝ)^2 - (2 : ℝ)^2 = 21 →
  (∀ x y : ℝ, y^2 / 25 + x^2 / 21 = 1) := sorry

theorem parabola_equation_x_axis :
  point_P = (-2, -4) →
  (∀ x y : ℝ, y^2 = -8 * x) := sorry

theorem parabola_equation_y_axis :
  point_P = (-2, -4) →
  (∀ x y : ℝ, x^2 = -y) := sorry

end ellipse_equation_x_axis_ellipse_equation_y_axis_parabola_equation_x_axis_parabola_equation_y_axis_l6_6504


namespace find_two_digit_number_with_conditions_l6_6991

theorem find_two_digit_number_with_conditions :
  ∃ (c : ℕ), c >= 10 ∧ c < 100 ∧ 
    (let x := c / 10 in
    let y := c % 10 in
    x + y = 10 ∧ x * y = 25 ∧ c = 55) :=
sorry

end find_two_digit_number_with_conditions_l6_6991


namespace cost_of_shoes_l6_6457

-- Define individual prices of items (excluding the shoes)
def shirt_price : ℕ := 30
def pants_price : ℕ := 46
def coat_price : ℕ := 38
def socks_price : ℕ := 11
def belt_price : ℕ := 18
def necktie_price : ℕ := 22

-- Budget and remaining amount after discount coupon
def budget : ℕ := 200
def remaining_amount : ℕ := 16

-- Prove that the cost of the shoes is $19
theorem cost_of_shoes : 
  let total_cost_of_items := shirt_price + pants_price + coat_price + socks_price + belt_price + necktie_price in
  let total_spent := budget - remaining_amount in
  let cost_of_shoes := total_spent - total_cost_of_items in
  cost_of_shoes = 19 := by
  -- Proof context can be added here
  sorry

end cost_of_shoes_l6_6457


namespace price_reduction_l6_6455

theorem price_reduction (x : ℝ) : 
  188 * (1 - x) ^ 2 = 108 :=
sorry

end price_reduction_l6_6455


namespace find_g_9_l6_6710

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_of_3 : g 3 = 4

theorem find_g_9 : g 9 = 64 := by
  sorry

end find_g_9_l6_6710


namespace beef_stew_duration_l6_6795

noncomputable def original_portions : ℕ := 14
noncomputable def your_portion : ℕ := 1
noncomputable def roommate_portion : ℕ := 3
noncomputable def guest_portion : ℕ := 4
noncomputable def total_daily_consumption : ℕ := your_portion + roommate_portion + guest_portion
noncomputable def days_stew_lasts : ℕ := original_portions / total_daily_consumption

theorem beef_stew_duration : days_stew_lasts = 2 :=
by
  sorry

end beef_stew_duration_l6_6795


namespace angle_AOB_in_tangent_triangle_l6_6766

-- Statement of the problem
theorem angle_AOB_in_tangent_triangle (P A B O: Type) 
  [Is_Triangle PAB] [TangentToCircle P A B O] 
  (h_angle_APB : ∠ APB = 50°) :
  ∠ AOB = 130° :=
sorry

end angle_AOB_in_tangent_triangle_l6_6766


namespace main_problem_l6_6509

noncomputable def no_valid_pairs (x y : ℝ) : Prop := 9^(x^3 + y) + 9^(x + y^3) = 1

theorem main_problem : ∀ x y : ℝ, ¬ no_valid_pairs x y :=
by
  intro x y
  sorry

end main_problem_l6_6509


namespace g_9_l6_6725

variable (g : ℝ → ℝ)

-- Conditions
axiom func_eq : ∀ x y : ℝ, g(x + y) = g(x) * g(y)
axiom g_3 : g 3 = 4

-- Theorem to prove
theorem g_9 : g 9 = 64 :=
by
  sorry

end g_9_l6_6725


namespace right_rectangular_prism_volume_l6_6347

theorem right_rectangular_prism_volume (x y z : ℝ) 
  (h1 : x * y = 72) (h2 : y * z = 75) (h3 : x * z = 80) : 
  x * y * z = 657 :=
sorry

end right_rectangular_prism_volume_l6_6347


namespace recurring_decimal_reduced_fraction_l6_6056

noncomputable def recurring_decimal_as_fraction : Prop := 
  ∀ (x y : ℚ), (x = 2.06) ∧ (y = 0.02) → y = 2 / 99 → x = 68 / 33

theorem recurring_decimal_reduced_fraction (x y : ℚ) 
  (h1 : x = 2 + 0.06) (h2 : y = 0.02) (h3 : y = 2 / 99) : 
  x = 68 / 33 := 
begin
  -- Proof here
  sorry
end

end recurring_decimal_reduced_fraction_l6_6056


namespace rhombus_diagonals_perpendicular_and_bisect_l6_6794

theorem rhombus_diagonals_perpendicular_and_bisect (Q : Type) [quadrilateral Q] :
  (∀ (P : rhombus Q), 
    let d1 := diagonal1 P
    let d2 := diagonal2 P
    perpendicular d1 d2 ∧ bisect d1 ∧ bisect d2) :=
sorry

end rhombus_diagonals_perpendicular_and_bisect_l6_6794


namespace has_inverse_a_has_inverse_b_has_inverse_c_has_inverse_d_has_inverse_e_has_inverse_f_has_inverse_g_has_inverse_h_l6_6657

noncomputable def a (x : ℝ) : ℝ := sqrt (2 - x)
noncomputable def b (x : ℝ) : ℝ := x^2 - x + 1
noncomputable def c (x : ℝ) : ℝ := 2*x - 1/x
noncomputable def d (x : ℝ) : ℝ := x^2 / (3 + x)
noncomputable def e (x : ℝ) : ℝ := (x-2)^2 + (x+3)^2
noncomputable def f (x : ℝ) : ℝ := 2^x + 9^x
noncomputable def g (x : ℝ) : ℝ := x + sqrt x
noncomputable def h (x : ℝ) : ℝ := x / 3

theorem has_inverse_a : ∃ a_inv, ∀ x (hx : x ∈ Set.Iic 2), a (a_inv x) = x := sorry
theorem has_inverse_b : ∃ b_inv, ∀ x (hx : x ∈ Set.Ici 0), b (b_inv x) = x := sorry
theorem has_inverse_c : ∃ c_inv, ∀ x (hx : x ∈ Set.Ioi 0), c (c_inv x) = x := sorry
theorem has_inverse_d : ∃ d_inv, ∀ x (hx : x ∈ Set.Ici 0), d (d_inv x) = x := sorry
theorem has_inverse_e : ∃ e_inv, ∀ x, e (e_inv x) = x := sorry
theorem has_inverse_f : ∃ f_inv, ∀ x, f (f_inv x) = x := sorry
theorem has_inverse_g : ∃ g_inv, ∀ x (hx : x ∈ Set.Ici 0), g (g_inv x) = x := sorry
theorem has_inverse_h : ∃ h_inv, ∀ x (hx : x ∈ Set.Ico (-3) 9), h (h_inv x) = x := sorry

end has_inverse_a_has_inverse_b_has_inverse_c_has_inverse_d_has_inverse_e_has_inverse_f_has_inverse_g_has_inverse_h_l6_6657


namespace geometric_locus_is_convex_polygon_l6_6543

-- Define the nature of acute-angled triangles
open Set

-- Define the triangle and its properties
variables {A B C X : Point}
variable {triangle_ABC : Triangle}
variable {acute_ABC : acute triangle_ABC}

-- Define the property for triangles ABX, BCX, and CAX being acute-angled
def acute_triangle (A B C : Point) : Prop :=
  ∀ X, acute (Triangle X A B) ∧ acute (Triangle X B C) ∧ acute (Triangle X C A)

-- Define the geometric locus
def geometric_locus (A B C : Point) : Set Point :=
  { X | acute_triangle(A, B, C) }

-- State the problem
theorem geometric_locus_is_convex_polygon :
  geometric_locus(A, B, C) = interior(convex_hull(A, B, C)) :=
sorry

end geometric_locus_is_convex_polygon_l6_6543


namespace total_games_played_l6_6400

theorem total_games_played (n : ℕ) (h : n = 14) : combinatorics.choose n 2 = 91 :=
by {
  sorry
}

end total_games_played_l6_6400


namespace smallest_a_exists_l6_6262

def f (x : ℕ) : ℕ :=
  if x % 2 = 0 && x % 5 = 0 then x / 10
  else if x % 5 = 0 then 2 * x
  else if x % 2 = 0 then 5 * x
  else x + 2

noncomputable def f_iter (a : ℕ) (x : ℕ) : ℕ :=
  Nat.iterate (λ y, f y) a x

theorem smallest_a_exists :
  ∃ a, a > 1 ∧ f_iter a 3 = f 3 ∧ (∀ b, b > 1 ∧ f_iter b 3 = f 3 → a ≤ b) := sorry

end smallest_a_exists_l6_6262


namespace charge_R_12_5_percent_more_l6_6700

-- Let R be the charge for a single room at hotel R.
-- Let G be the charge for a single room at hotel G.
-- Let P be the charge for a single room at hotel P.

def charge_R (R : ℝ) : Prop := true
def charge_G (G : ℝ) : Prop := true
def charge_P (P : ℝ) : Prop := true

axiom hotel_P_20_less_R (R P : ℝ) : charge_R R → charge_P P → P = 0.80 * R
axiom hotel_P_10_less_G (G P : ℝ) : charge_G G → charge_P P → P = 0.90 * G

theorem charge_R_12_5_percent_more (R G : ℝ) :
  charge_R R → charge_G G → (∃ P, charge_P P ∧ P = 0.80 * R ∧ P = 0.90 * G) → R = 1.125 * G :=
by sorry

end charge_R_12_5_percent_more_l6_6700


namespace find_t_l6_6163

noncomputable def f (t : ℝ) : ℝ → ℝ := λ x, t * Real.log x
def g : ℝ → ℝ := λ x, x^2 - 1

theorem find_t (t : ℝ) : 
  ∀ (x : ℝ), (f t x = 0 ∧ g x = 0 ∧ (f t).deriv x = (g).deriv x)  → t = 2 :=
begin
  intros x hx,
  have htangent : (f t).deriv 1 = (g).deriv 1, {
    cases hx with hfx hx,
    cases hx with hg dx,
    exact dx,
  },
  dsimp [(f t).deriv, g.deriv] at htangent,
  simp only [deriv_const_mul, deriv_log, deriv_id'] at htangent,
  rw [← mul_one t] at htangent,
  norm_num at htangent,
  exact htangent,
end

#eval find_t

end find_t_l6_6163


namespace remainder_of_sum_of_combinations_divided_by_9_l6_6536

theorem remainder_of_sum_of_combinations_divided_by_9:
  let S := (Finset.range 28).sum (λ k, nat.choose 27 k)
  in S % 9 = 7 := by
sorry

end remainder_of_sum_of_combinations_divided_by_9_l6_6536


namespace instantaneous_velocity_zero_instants_l6_6974

noncomputable def s (t : ℝ) : ℝ := (1 / 4) * t^4 - 4 * t^3 + 16 * t^2

theorem instantaneous_velocity_zero_instants :
  ∃ t : ℝ, s' t = 0 :=
sorry

end instantaneous_velocity_zero_instants_l6_6974


namespace total_black_dots_l6_6341

def num_butterflies : ℕ := 397
def black_dots_per_butterfly : ℕ := 12

theorem total_black_dots : num_butterflies * black_dots_per_butterfly = 4764 := by
  sorry

end total_black_dots_l6_6341


namespace church_tower_height_l6_6346

theorem church_tower_height :
  ∃ h : ℝ, (150)^2 + h^2 = (200)^2 + (200)^2 :=
begin
  use real.sqrt 57500,
  simp [real.sqrt_eq_rpow, real.sqrt_eq_rpow],
  nth_rewrite 0 ←real.sqrt_mul_self_eq (150^2 + (real.sqrt 57500)^2),
  nth_rewrite 1 ←real.sqrt_mul_self_eq (200^2 + 200^2),
  rw [←eq_square_of_mul_eq_mul_left (real.sqrt_nonneg _) (add_nonneg (sq_nonneg 200) (sq_nonneg 200)), sqrt_inj],
  { ring },
end

end church_tower_height_l6_6346


namespace uncle_dave_ice_cream_l6_6353

def ice_cream_sandwiches_per_niece (total_sandwiches: ℝ) (num_nieces: ℝ) : ℝ :=
  total_sandwiches / num_nieces

theorem uncle_dave_ice_cream (total_sandwiches: ℝ) (num_nieces: ℝ) (h1: total_sandwiches = 1573) (h2: num_nieces = 11.0) :
  ice_cream_sandwiches_per_niece total_sandwiches num_nieces = 143 :=
by
  rw [h1, h2]
  norm_num
  sorry

end uncle_dave_ice_cream_l6_6353


namespace dan_spent_more_on_chocolate_l6_6480

def cost_candy_bar : ℤ := 2
def cost_chocolate : ℤ := 3
def cost_difference : ℤ := cost_chocolate - cost_candy_bar

theorem dan_spent_more_on_chocolate : cost_difference = 1 := by
  unfold cost_difference
  simp
  sorry

end dan_spent_more_on_chocolate_l6_6480


namespace odd_function_f_l6_6483

noncomputable def f (x : ℝ) : ℝ :=
if h : 0 ≤ x then x * (1 - x) else x * (1 + x)

theorem odd_function_f :
  ∀ x : ℝ, f (-x) = -f (x) := by
  intro x
  unfold f
  split_ifs with h
  { -- Case 0 ≤ -x
    have hx : x ≤ 0 := by
      linarith
    rw [if_neg hx, if_pos (lt_of_le_of_lt h (neg_pos_of_neg hx))],
    ring
  }
  { -- Case 0 > -x
    have hx : x > 0 := by
      linarith
    rw [if_pos hx, if_neg (not_le_of_gt (neg_lt_of_lt (lt_neg_of_gt hx))),
        ← neg_mul_eq_mul_neg, ← neg_add_eq_add_neg],
    ring
  }

end odd_function_f_l6_6483


namespace diagonals_from_one_vertex_l6_6338

theorem diagonals_from_one_vertex (x : ℕ) (h : (x - 2) * 180 = 1800) : (x - 3) = 9 :=
  by
  sorry

end diagonals_from_one_vertex_l6_6338


namespace olivia_remaining_usd_l6_6658

def initial_usd : ℝ := 78
def initial_eur : ℝ := 50
def exchange_rate : ℝ := 1.20
def spent_usd_supermarket : ℝ := 15
def book_eur : ℝ := 10
def spent_usd_lunch : ℝ := 12

theorem olivia_remaining_usd :
  let total_usd := initial_usd + (initial_eur * exchange_rate)
  let remaining_after_supermarket := total_usd - spent_usd_supermarket
  let remaining_after_book := remaining_after_supermarket - (book_eur * exchange_rate)
  let final_remaining := remaining_after_book - spent_usd_lunch
  final_remaining = 99 :=
by
  sorry

end olivia_remaining_usd_l6_6658


namespace max_value_sqrt_ab_sqrt_1_a_1_b_l6_6644

theorem max_value_sqrt_ab_sqrt_1_a_1_b (a b : ℝ) (h1 : 0 ≤ a) (h2 : a ≤ 1) (h3 : 0 ≤ b) (h4 : b ≤ 1) :
  \(\sqrt{a * b} + \sqrt{(1 - a) * (1 - b)} ≤ 1\) :=
sorry

end max_value_sqrt_ab_sqrt_1_a_1_b_l6_6644


namespace limit_exp_neg2x_div_x_plus_sin_x_sq_eq_3_l6_6478

open Real 

theorem limit_exp_neg2x_div_x_plus_sin_x_sq_eq_3 :
  (∃ L, tendsto (λ x : ℝ, (exp x - exp (-2 * x)) / (x + sin (x ^ 2))) (nhds 0) (nhds L) ∧ L = 3) :=
by
  sorry

end limit_exp_neg2x_div_x_plus_sin_x_sq_eq_3_l6_6478


namespace find_smallest_n_l6_6942

def arithmetic_sequence (a₁ d n : ℕ) : Int := a₁ + (n - 1) * d

def sum_of_first_n_terms (a₁ d n : ℕ) : Int :=
  n * a₁ + d * (n * (n - 1) / 2)

theorem find_smallest_n 
  (a₁ : ℕ) (d : Int) (h₁ : a₁ = 7) (h₂ : d = -2) :
  ∃ n : ℕ, (n > 0) ∧ (sum_of_first_n_terms a₁ d n < 0) ∧ (∀ m : ℕ, m < n → sum_of_first_n_terms a₁ d m ≥ 0) := 
sorry

end find_smallest_n_l6_6942


namespace coeff_x4_in_expansion_l6_6365

theorem coeff_x4_in_expansion (x : ℝ) (sqrt2 : ℝ) (h₁ : sqrt2 = real.sqrt 2) :
  let c := (70 : ℝ) * (3^4 : ℝ) * (sqrt2^4 : ℝ) in
  c = 22680 :=
by
  sorry

end coeff_x4_in_expansion_l6_6365


namespace proof_problem_l6_6977

theorem proof_problem (a b c : ℕ) (h1 : {a, b, c}.to_finset = {0, 1, 2}) 
(h2 : (¬ (a ≠ 2) ∧ b = 2 ∧ ¬ (c ≠ 0)) ∨ (a ≠ 2 ∧ ¬ (b = 2) ∧ ¬ (c ≠ 0)) ∨ (a ≠ 2 ∧ b = 2 ∧ c ≠ 0)) 
(h3 : (a ≠ 2 ∧ ¬ (b = 2) ∧ ¬ (c ≠ 0) ∨ ¬ (a ≠ 2) ∧ b = 2 ∧ ¬ (c ≠ 0) ∨ ¬ (a ≠ 2) ∧ ¬ (b = 2) ∧ c ≠ 0) → False) : 
  10 * a + 2 * b + c = 21 := by
  sorry

end proof_problem_l6_6977


namespace derivative_at_two_l6_6369

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

theorem derivative_at_two (a b : ℝ) (h1 : f 1 a b = -2) (h2 : ∀ x, f x a b ≤ f 1 a b) :
  (a = -2 ∧ b = -2) → deriv (λ x, f x a b) 2 = -1/2 :=
begin
  intros hab,
  obtain ⟨ha, hb⟩ := hab,
  -- omitted steps for clarity
  exact sorry
end

end derivative_at_two_l6_6369


namespace isabella_hair_length_end_of_year_l6_6636

/--
Isabella's initial hair length.
-/
def initial_hair_length : ℕ := 18

/--
Isabella's hair growth over the year.
-/
def hair_growth : ℕ := 6

/--
Prove that Isabella's hair length at the end of the year is 24 inches.
-/
theorem isabella_hair_length_end_of_year : initial_hair_length + hair_growth = 24 := by
  sorry

end isabella_hair_length_end_of_year_l6_6636


namespace one_kilogram_cotton_not_lighter_than_iron_l6_6812

theorem one_kilogram_cotton_not_lighter_than_iron : 
  ∀ (cotton iron : ℝ), (cotton = 1) ∧ (iron = 1) → ¬(cotton < iron) :=
begin
  -- Assume we have 1 kilogram of each
  intros cotton iron h,
  cases h with h1 h2,
  -- kg of cotton equals kg of iron
  rw [h1, h2],
  -- Proof that 1 is not less than 1
  linarith,
end

end one_kilogram_cotton_not_lighter_than_iron_l6_6812


namespace max_integer_solutions_l6_6442

noncomputable def semi_centered (p : ℕ → ℤ) :=
  ∃ k : ℕ, p k = k + 50 - 50 * 50

theorem max_integer_solutions (p : ℕ → ℤ) (h1 : semi_centered p) (h2 : ∀ x : ℕ, ∃ c : ℤ, p x = c * x^2) (h3 : p 50 = 50) :
  ∃ n ≤ 6, ∀ k : ℕ, (p k = k^2) → k ∈ Finset.range (n+1) :=
sorry

end max_integer_solutions_l6_6442


namespace trigonometric_identity_l6_6881

theorem trigonometric_identity :
  cos (43 * (Real.pi / 180)) * cos (77 * (Real.pi / 180)) + 
  sin (43 * (Real.pi / 180)) * cos (167 * (Real.pi / 180)) = -1 / 2 := 
by 
  sorry

end trigonometric_identity_l6_6881


namespace newspaper_target_l6_6682

theorem newspaper_target (total_collected_2_weeks : Nat) (needed_more : Nat) (sections : Nat) (kilos_per_section_2_weeks : Nat)
  (h1 : sections = 6)
  (h2 : kilos_per_section_2_weeks = 280)
  (h3 : total_collected_2_weeks = sections * kilos_per_section_2_weeks)
  (h4 : needed_more = 320)
  : total_collected_2_weeks + needed_more = 2000 :=
by
  sorry

end newspaper_target_l6_6682


namespace simplify_sqrt_expression_l6_6681

theorem simplify_sqrt_expression :
  (sqrt 192 / sqrt 27 - sqrt 500 / sqrt 125) = 2 / 3 :=
by
  sorry

end simplify_sqrt_expression_l6_6681


namespace distance_is_30_l6_6280

-- Define given conditions
def total_distance : ℕ := 120
def trips : ℕ := 4

-- Define the distance from Mrs. Hilt's desk to the water fountain
def distance_to_water_fountain : ℕ := total_distance / trips

-- Prove the distance is 30 feet
theorem distance_is_30 : distance_to_water_fountain = 30 :=
by
  -- Utilizing the division defined in distance_to_water_fountain
  sorry

end distance_is_30_l6_6280


namespace min_length_sum_given_length_sum_l6_6537

variables {α β : Type}
variables (l : affine_plane α) (A B : point α)

-- Problem 77(a)
def sym_point (B : point α) (l : affine_plane α) : point α := sorry

theorem min_length_sum (X : point α) (l : affine_plane α) (A B : point α) : 
  let B' := sym_point B l in 
  X ∈ line l ∧ X ∈ line (A, B') → (AX + XB) = (AX + XB') :=
sorry

-- Problem 77(b)
variables (a : ℝ)
def tangent_point (A A' B : point α) (l : affine_plane α) (a : ℝ) : point α := sorry

theorem given_length_sum (X : point α) (l : affine_plane α) (A B : point α) (a : ℝ) :
  let A' := sym_point A l in 
  let X := tangent_point A A' B l a in 
  X ∈ line l ∧ (AX + XB) = a := 
sorry

end min_length_sum_given_length_sum_l6_6537


namespace Chandler_bicycle_l6_6474

theorem Chandler_bicycle (b p e : ℝ) (x : ℝ) 
  (hb : b = 60 + 45 + 20)
  (hp : p = 600)
  (he : e = 20)
  (h : b + e * x = p) : 
  x = 24 :=
by
  rw [hb, hp, he] at h
  linarith

end Chandler_bicycle_l6_6474


namespace proposition_1_proposition_2_proposition_3_proposition_4_correct_propositions_l6_6563

-- Define the conditions and propositions
variables {m n α : Type*} [LinearOrder m] [LinearOrder n] [LinearOrder α]
variables parallel : m → n → Prop
variables perpendicular : m → α → Prop

-- Proof of correctness of propositions
theorem proposition_1 (h1 : parallel m n) (h2 : perpendicular m α) : perpendicular n α := sorry
theorem proposition_2 (h3 : perpendicular m α) (h4 : perpendicular n α) : parallel m n := sorry
theorem proposition_3 (h5 : perpendicular m α) (h6 : perpendicular m n) : ¬ parallel n α := sorry
theorem proposition_4 (h7 : parallel m α) (h8 : perpendicular m n) : ¬ perpendicular n α := sorry

-- Combining the propositions correctness
theorem correct_propositions : 
  (proposition_1 = true) ∧ 
  (proposition_2 = true) ∧ 
  (proposition_3 = false) ∧
  (proposition_4 = false) := 
begin
  split, sorry, -- proving prop 1
  split, sorry, -- proving prop 2
  split, sorry, -- proving prop 3
  sorry, -- proving prop 4
end

end proposition_1_proposition_2_proposition_3_proposition_4_correct_propositions_l6_6563


namespace part1_part2_l6_6247

noncomputable def f (x : ℝ) : ℝ := 0
noncomputable def g (x : ℝ) : ℝ := (2 / (3 - Real.exp 2)) * Real.exp x + x

theorem part1 (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 1) :
  f(x) = ∫ t in 0..1, Real.exp (x + t) * f(t) := by
  sorry

theorem part2 (x : ℝ) (h₀ : 0 ≤ x ∧ x ≤ 1) :
  g(x) = (∫ t in 0..1, Real.exp (x + t) * g(t)) + x := by
  sorry

end part1_part2_l6_6247


namespace perimeter_of_semicircles_around_pentagon_l6_6843

def pentagon_side_length : ℝ := 5 / Real.pi

def semicircle_perimeter (d : ℝ) : ℝ :=
  d * Real.pi / 2

def total_perimeter (n : ℕ) (d : ℝ) : ℝ :=
  n * semicircle_perimeter(d)

theorem perimeter_of_semicircles_around_pentagon :
  total_perimeter 5 pentagon_side_length = 25 / 2 :=
by
  sorry

end perimeter_of_semicircles_around_pentagon_l6_6843


namespace probability_three_nine_l6_6963

noncomputable def X (σ : ℝ) (hσ : σ > 0) : ProbabilitySpace ℝ :=
normalDist 6 σ

theorem probability_three_nine (σ : ℝ) (hσ : σ > 0) :
  (∫ x in (3 : ℝ)..9, pdf (X σ hσ) x) = 0.6 :=
sorry

end probability_three_nine_l6_6963


namespace find_x_l6_6922

namespace ProofProblem

def δ (x : ℚ) : ℚ := 5 * x + 6
def φ (x : ℚ) : ℚ := 9 * x + 4

theorem find_x (x : ℚ) : (δ (φ x) = 14) ↔ (x = -4 / 15) :=
by
  sorry

end ProofProblem

end find_x_l6_6922


namespace affine_parallel_l6_6300

-- Define an affine transformation T
def affine_transformation (A : ℝ → ℝ → ℝ → ℝ) (b : ℝ × ℝ) (x : ℝ × ℝ) : ℝ × ℝ :=
  (A x.1 x.2 + b.1, A x.2 x.1 + b.2)

-- Assume L1 and L2 are parallel lines in ℝ²
def parallel_lines (L1 L2 : ℝ → ℝ × ℝ) : Prop :=
  ∀ t1 t2, L1 t1 = L2 t2

-- The theorem statement
theorem affine_parallel {A : ℝ → ℝ → ℝ → ℝ} {b : ℝ × ℝ} {L1 L2 : ℝ → ℝ × ℝ} :
  parallel_lines L1 L2 →
  parallel_lines (affine_transformation A b ∘ L1) (affine_transformation A b ∘ L2) :=
by
  intros
  sorry

end affine_parallel_l6_6300


namespace num_satisfying_integers_l6_6913

theorem num_satisfying_integers :
  {n : ℤ // √(2 * n) ≤ √(5 * n - 8) ∧ √(5 * n - 8) < √(3 * n + 7)}.finset.card = 5 :=
by
  sorry

end num_satisfying_integers_l6_6913


namespace find_k_l6_6740

theorem find_k (k : ℤ) : 
  (∀ (x1 y1 x2 y2 x3 y3 : ℤ),
    (x1, y1) = (2, 9) ∧ (x2, y2) = (5, 18) ∧ (x3, y3) = (8, 27) ∧ 
    ∃ m b : ℤ, y1 = m * x1 + b ∧ y2 = m * x2 + b ∧ y3 = m * x3 + b) 
  ∧ ∃ m b : ℤ, k = m * 42 + b
  → k = 129 :=
sorry

end find_k_l6_6740


namespace plains_routes_l6_6209

theorem plains_routes 
  (total_cities : ℕ)
  (mountainous_cities : ℕ)
  (plains_cities : ℕ)
  (total_routes : ℕ)
  (mountainous_routes : ℕ)
  (num_pairs_with_mount_to_mount : ℕ)
  (routes_per_year : ℕ)
  (years : ℕ)
  (mountainous_roots_connections : ℕ)
  : (mountainous_cities = 30) →
    (plains_cities = 70) →
    (total_cities = mountainous_cities + plains_cities) →
    (routes_per_year = 50) →
    (years = 3) →
    (total_routes = routes_per_year * years) →
    (mountainous_routes = num_pairs_with_mount_to_mount * 2) →
    (num_pairs_with_mount_to_mount = 21) →
    let num_endpoints_per_city_route = 2 in
    let mountainous_city_endpoints = mountainous_cities * 3 in
    let mountainous_endpoints = mountainous_routes in
    let mountain_to_plains_endpoints = mountainous_city_endpoints - mountainous_endpoints in
    let total_endpoints = total_routes * num_endpoints_per_city_route in
    let plains_city_endpoints = plains_cities * 3 in
    let routes_between_plain_and_mountain = mountain_to_plains_endpoints in
    let plain_to_plain_endpoints = plains_city_endpoints - routes_between_plain_and_mountain in
    let plain_to_plain_routes = plain_to_plain_endpoints / 2 in
    plain_to_plain_routes = 81 :=
sorry

end plains_routes_l6_6209


namespace smallest_number_is_900_l6_6784

noncomputable def smallest_number_satisfying_conditions : ℕ :=
  let x := 900 in
  if (∀ y, (y % 18 = 0 ∧ y % 30 = 0 ∧ ∃ k : ℕ, y = k * k) → x ≤ y) then
    x
  else
    0  -- This case should never happen under the described problem.

theorem smallest_number_is_900 :
  (∃ x : ℕ, x = smallest_number_satisfying_conditions ∧ x = 900) :=
by
  use 900
  unfold smallest_number_satisfying_conditions
  split_ifs
  · exact ⟨rfl⟩
  · contradiction

end smallest_number_is_900_l6_6784


namespace length_chord_AB_l6_6288

-- Define the basic setup
noncomputable def sphere_center : Type := ℝ × ℝ × ℝ
def point_A : (ℝ × ℝ × ℝ) := (0, 3, 0)
def radius_OA : ℝ := 3

-- Define the chords and their properties
def chord_length (A B : (ℝ × ℝ × ℝ)) : ℝ := 
  let (x1, y1, z1) := A in 
  let (x2, y2, z2) := B in 
  ( (x2 - x1)^2 + (y2 - y1)^2 + (z2 - z1)^2 )^(1/2)

def angle_between_chords : ℝ := 60 -- Angle in degrees

-- The theorem to prove the length of chord AB
theorem length_chord_AB :
  ∃ AB_length : ℝ,
  (chord_length point_A (0, 0, 0) = 3) ∧ 
  (∃ B C D : ℝ × ℝ × ℝ, chord_length (0, 0, 0) B = chord_length point_A B ∧ chord_length point_A B = chord_length point_A C ∧
   chord_length point_A C = chord_length point_A D ∧
   angle_between_chords = 60 ∧
   chord_length point_A B = 2 * real.sqrt 6)
:=
begin
  apply sorry
end

end length_chord_AB_l6_6288


namespace broken_line_length_le_200_l6_6931

-- Define the chessboard and conditions.
constant chessboard : Type
variable {C : chessboard}
variable (n : ℕ)
variable (size : n = 15)

-- Define the closed broken line without self-intersections.
constant broken_line : set (chessboard × chessboard)
variable (closed : ∀ (x y : chessboard), (x, y) ∈ broken_line → ∃ p, (y, x) ∈ broken_line = p : y → x)

-- Define the symmetric property with respect to the main diagonal.
constant symmetric : ∀ (x : chessboard), (x, x) ∈ broken_line

-- Define the length condition.
constant length (P : set (chessboard × chessboard)) : ℕ

-- The theorem and the assertion that needs to be proved.
theorem broken_line_length_le_200 (H1 : size)
  (H2 : ∀ C, (C, C) ∈ symmetric) : length broken_line ≤ 200 :=
sorry

end broken_line_length_le_200_l6_6931


namespace inscribed_triangles_from_ten_points_l6_6340

theorem inscribed_triangles_from_ten_points :
  let n := 10 in
  let k := 3 in
  let binomial (n k : ℕ) : ℕ := n.choose k in
  binomial n k = 120 := by
  sorry

end inscribed_triangles_from_ten_points_l6_6340


namespace proof_f_g_3_l6_6599

def f (x : ℝ) := 4 - real.sqrt x
def g (x : ℝ) := 3 * x + 3 * x^2

theorem proof_f_g_3 :
  f (g (-3)) = 4 - 3 * real.sqrt 2 :=
by
  sorry

end proof_f_g_3_l6_6599


namespace fixed_point_HN_through_l6_6957

-- Define the conditions and entities
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + (y^2 / 4) = 1

def pointA : ℝ × ℝ := (0, -2)
def pointB : ℝ × ℝ := (3/2, -1)

-- Main statement of the problem
theorem fixed_point_HN_through (P : ℝ × ℝ) (P1 : P = (1, -2)) 
    (H : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ) : 
  ellipse (fst pointA) (snd pointA) →
  ellipse (fst pointB) (snd pointB) →
  (∀ M N : ℝ × ℝ, (ellipse (fst M) (snd M) ∧ ellipse (fst N) (snd N)) ↔ 
    ((P = (1, -2)) ∧ 
    (H M T = T)) → 
  ∃ K : ℝ × ℝ, K = (0, -2) ∧ collinear K H N :=
sorry

end fixed_point_HN_through_l6_6957


namespace planA_cost_correct_electricity_usage_60kwh_planA_cheaper_range_l6_6339

def planA_cost (x : ℝ) : ℝ :=
if x ≤ 30 then 2 + 0.5 * x
else 0.6 * x - 1

def planB_cost (x : ℝ) : ℝ := 0.58 * x

theorem planA_cost_correct :
  (∀ x, 0 ≤ x → x ≤ 30 → planA_cost x = 2 + 0.5 * x) ∧
  (∀ x, 30 < x → planA_cost x = 0.6 * x - 1) :=
by
sorry

theorem electricity_usage_60kwh (x : ℝ) :
  planA_cost x = 35 → 30 < x ∧ x = 60 :=
by
sorry

theorem planA_cheaper_range (x : ℝ) :
  (planA_cost x < planB_cost x → 25 < x ∧ x < 50) :=
by
sorry

end planA_cost_correct_electricity_usage_60kwh_planA_cheaper_range_l6_6339


namespace locus_of_feet_of_altitudes_l6_6845

theorem locus_of_feet_of_altitudes
  (A B C : Point)
  (S1 S2 : Circle)
  (h1 : right_triangle A B C)
  (h2 : B ∈ S1)
  (h3 : C ∈ S2)
  (h4 : externally_tangent S1 S2 A)
  (h5 : fixed_circles S1 S2) :
  locus_of_feet_of_altitudes A S1 S2 = arc_of_circle_diameter_tangents A S1 S2 :=
sorry

end locus_of_feet_of_altitudes_l6_6845


namespace perfect_square_polynomial_l6_6595

theorem perfect_square_polynomial (k : ℝ) :
  (∃ p : ℝ[x], p^2 = X^2 + k * X + 36) → (k = 12 ∨ k = -12) :=
by
  sorry

end perfect_square_polynomial_l6_6595


namespace max_rectangular_box_length_l6_6853

theorem max_rectangular_box_length
  (wooden_box_length_m : ℕ)
  (wooden_box_width_m : ℕ)
  (wooden_box_height_m : ℕ)
  (rectangular_box_width_cm : ℕ)
  (rectangular_box_height_cm : ℕ)
  (max_boxes : ℕ)
  (volume_wooden_box_cm : ℕ)
  (volume_one_rect_box_cm : ℕ → ℕ) :
  max_boxes * volume_one_rect_box_cm (4) ≤ volume_wooden_box_cm →
  wooden_box_length_m = 8 →
  wooden_box_width_m = 7 →
  wooden_box_height_m = 6 →
  rectangular_box_width_cm = 7 →
  rectangular_box_height_cm = 6 →
  max_boxes = 2000000 →
  volume_wooden_box_cm = (wooden_box_length_m * 100) * (wooden_box_width_m * 100) * (wooden_box_height_m * 100) →
  volume_one_rect_box_cm x = x * rectangular_box_width_cm * rectangular_box_height_cm →
  x ≤ 4 :=
by
  intros * h_volume_constraint h_len h_width h_height h_rect_width h_rect_height h_max_boxes h_volume_wooden_box_eq h_volume_rect_box_eq
  sorry

end max_rectangular_box_length_l6_6853


namespace modulus_of_complex_l6_6161

variable (z : ℂ)

-- Given: z = (1 + 3*Complex.I) / (1 - 2*Complex.I)
theorem modulus_of_complex :
  z = (1 + 3 * Complex.i) / (1 - 2 * Complex.i) → Complex.abs z = Real.sqrt 2 := by
  intro hz
  sorry

end modulus_of_complex_l6_6161


namespace range_B_is_open_pos_inf_range_A_not_open_pos_inf_range_C_not_open_pos_inf_range_D_not_open_pos_inf_l6_6078

-- Define the functions as given in the problem statement
def f_A (x : ℝ) : ℝ := 5 ^ (1 / (2 - x))
def f_B (x : ℝ) : ℝ := (1 / 3) ^ (1 - x)
def f_C (x : ℝ) : ℝ := sqrt (1 - 2 ^ x)
def f_D (x : ℝ) : ℝ := sqrt ((1 / 2) ^ x - 1)

-- Define the ranges of the functions
def range_A : Set ℝ := {y | ∃ x : ℝ, y = 5 ^ (1 / (2 - x))}
def range_B : Set ℝ := {y | ∃ x : ℝ, y = (1 / 3) ^ (1 - x)}
def range_C : Set ℝ := {y | ∃ x : ℝ, y = sqrt (1 - 2 ^ x)}
def range_D : Set ℝ := {y | ∃ x : ℝ, y = sqrt ((1 / 2) ^ x - 1)}

-- State the theorem to be proved
theorem range_B_is_open_pos_inf : range_B = {y : ℝ | 0 < y} :=
sorry

theorem range_A_not_open_pos_inf : range_A ≠ {y : ℝ | 0 < y} :=
sorry

theorem range_C_not_open_pos_inf : range_C ≠ {y : ℝ | 0 < y} :=
sorry

theorem range_D_not_open_pos_inf : range_D ≠ {y : ℝ | 0 < y} :=
sorry

end range_B_is_open_pos_inf_range_A_not_open_pos_inf_range_C_not_open_pos_inf_range_D_not_open_pos_inf_l6_6078


namespace repeating_decimal_fraction_l6_6064

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l6_6064


namespace player_B_wins_when_k_is_10_player_A_wins_when_k_is_15_l6_6708

variable (k : Nat) (a : Fin (2 * k) → Nat)
-- Given conditions for the digits chosen by players A and B:
-- Assume that a_i are chosen from {1,2,3,4,5}
-- Assume player A and player B choose the digits alternately

def S (n : Nat) : Nat := ∑ i in Finset.range n, a i

theorem player_B_wins_when_k_is_10
    (h_digits : ∀ i, a i ∈ {1, 2, 3, 4, 5})
    (h_player_A : ∀ i, i % 2 = 0 → a i % 2 = 1)  -- player A chooses odd indices
    (h_player_B : ∀ i, i % 2 = 1 → a i % 2 = 0)  -- player B chooses even indices
    (h_k : k = 10) :
  ∃ b : Fin (2 * k) → Nat, (S k b) % 9 = 0 :=
sorry

theorem player_A_wins_when_k_is_15
    (h_digits : ∀ i, a i ∈ {1, 2, 3, 4, 5})
    (h_player_A : ∀ i, i % 2 = 0 → a i % 2 = 1)  -- player A chooses odd indices
    (h_player_B : ∀ i, i % 2 = 1 → a i % 2 = 0)  -- player B chooses even indices
    (h_k : k = 15) :
  ∃ a : Fin (2 * k) → Nat, (S k a) % 9 = 0 :=
sorry

end player_B_wins_when_k_is_10_player_A_wins_when_k_is_15_l6_6708


namespace polynomial_factorization_l6_6387

theorem polynomial_factorization :
  ∀ x : ℤ, x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by sorry

end polynomial_factorization_l6_6387


namespace not_prime_power_consecutives_l6_6293

/-!
# Theorem
For any positive integer r, there exist r consecutive positive integers, none of which is a power of a prime.
-/

theorem not_prime_power_consecutives (r : ℕ) (hr : 0 < r) : 
  ∃ x : ℕ, ∀ i : ℕ, i < r → ∃ p₁ p₂ : ℕ, prime p₁ ∧ prime p₂ ∧ p₁ ∣ (x + i) ∧ p₂ ∣ (x + i) :=
begin
  sorry,
end

end not_prime_power_consecutives_l6_6293


namespace surface_area_circumscribed_sphere_regular_tetrahedron_l6_6235

theorem surface_area_circumscribed_sphere_regular_tetrahedron 
  (SA : ℝ) (SABC_regular : ∀ {a b c : ℝ}, a = b ∧ b = c → SABC_regular)
  (midpoint_M : ∀ {x y : ℝ}, x = y → midpoint_M)
  (midpoint_N : ∀ {x y : ℝ}, x = y → midpoint_N)
  (perp_MN_AN : ∀ {x y : ℝ}, x = y → perp_MN_AN) :
  SA = 2 → SABC_regular → midpoint_M → midpoint_N → perp_MN_AN →
  surface_area_circumscribed_sphere SABC_regular = 36 * Real.pi :=
begin
  sorry
end

end surface_area_circumscribed_sphere_regular_tetrahedron_l6_6235


namespace find_m_l6_6320

noncomputable def mean (lst : List ℝ) : ℝ :=
  lst.sum / lst.length

theorem find_m :
  let x := [0, 1, 2, 3, 4]
  let y := [10, 15, 20, m, 35]
  let regression_line := (λ x : ℝ, 6.5 * x + 9)
  let mean_x := mean x
  let mean_y := mean y
  mean_y = regression_line mean_x -> m = 30 :=
by
  sorry

end find_m_l6_6320


namespace ellipse_eccentricity_area_triangle_ratio_l6_6946

-- Definitions for the conditions
variables (a b c : ℝ) (h_ab : a > b) (h_b0 : b > 0) (h_slope : ∀ (k_PM k_PN : ℝ), k_PM * k_PN = -3 / 4) 
          (x y : ℝ) (ellipse_eq : x ^ 2 / a ^ 2 + y ^ 2 / b ^ 2 = 1)
          (F_ab : F = -c) (A B : ℝ × ℝ) (G_mpt : G = midpoint A B)
          (D E : ℝ × ℝ) (perp_bis : perpendicularBisector A B D E) 
          (S1 S2 : ℝ) (h_Area1 : S1 = areaTriangle G F D) (h_Area2 : S2 = areaTriangle O E D)

-- Another variable specific to the problem where O is the origin
variables (O : ℝ × ℝ) -- Origin

-- Prove the eccentricity
theorem ellipse_eccentricity (h_ellipse : a = 2 * c ∧ b = sqrt 3 * c) : 
  ∃ (e : ℝ), e = 1 / 2 :=
by
  obtain (ha_eq : a = 2 * c) (hb_eq : b = sqrt 3 * c),
    exact ⟨sqrt (1 - (b ^ 2) / (a ^ 2)), by rw [ha_eq, hb_eq]; sorry⟩

-- Prove the range of values
theorem area_triangle_ratio (h_AB_line : ∇AB(k,x+c)) :
  ∃ ratio : ℝ, ratio ∈ Ioo 0 (9 / 41) :=
by
  intro k; sorry

end ellipse_eccentricity_area_triangle_ratio_l6_6946


namespace mrs_hilt_apples_per_hour_l6_6277

-- Defining the conditions
def total_apples : ℕ := 15
def total_hours : ℕ := 3
def apples_per_hour : ℕ := total_apples / total_hours

-- The theorem to prove
theorem mrs_hilt_apples_per_hour : apples_per_hour = 5 :=
by
  -- Directly proving the statement we only derive from the conditions
  rw [total_apples, total_hours, Nat.div_eq_of_eq_mul_right (by decide : 3 ≠ 0) rfl]
  rfl

end mrs_hilt_apples_per_hour_l6_6277


namespace recurring_decimal_to_fraction_l6_6045

theorem recurring_decimal_to_fraction
  (h : (2:ℚ) + 3 * (2 / 99) = 2.06) :
  (2:ℚ) + 0.\overline{06} = (68 / 33) :=
by
  -- Given: 0.\overline{02} = 2 / 99
  have h0 : (0.\overline{02} : ℚ) = 2 / 99 := by sorry

  -- 0.\overline{06} = 3 * 0.\overline{02}
  have h1 : (0.\overline{06} : ℚ) = 3 * (0.\overline{02} : ℚ) :=
    by rw [← h0]; sorry

  -- Hence, 0.\overline{06} = 6 / 99 = 2 / 33
  have h2 : (0.\overline{06} : ℚ) = 2 / 33 :=
    by sorry

  -- Therefore, 2.\overline{06} = 2 + 0.\overline{06} = 2 + 2 / 33 = 68 / 33
  show (2:ℚ) + (0.\overline{06} : ℚ) = 68 / 33
    by sorry

end recurring_decimal_to_fraction_l6_6045


namespace rectangle_y_value_l6_6615

theorem rectangle_y_value (y : ℝ) 
  (H_rect : (-9, y) ≠ (1, y) ∧ (1, -8) ≠ (-9, -8) ∧ 
            ((-9, y), (1, y)), ((1, -8), (-9, -8))) 
  (H_area : abs (1 - (-9)) * abs (y - (-8)) = 90) 
: y = 1 :=
sorry

end rectangle_y_value_l6_6615


namespace recurring_decimal_to_fraction_l6_6033

theorem recurring_decimal_to_fraction
  (h : 0.\overline{02} = (2 : ℝ) / 99) :
  2.\overline{06} = 68 / 33 := by
  sorry

end recurring_decimal_to_fraction_l6_6033


namespace square_playground_properties_l6_6468

-- Define the given conditions
def laps : Int := 2
def total_distance : Int := 400

-- Define what we need to prove: the perimeter and side length of the square
theorem square_playground_properties (l : Int) (d : Int) (h_laps : l = 2) (h_distance : d = 400) :
  ∃ (C a : Int), C = 200 ∧ a = 50 :=
by
  -- Introduce variables for the perimeter (C) and side length (a) of the playground square
  let C := d / l -- Perimeter is total distance divided by laps
  let a := C / 4 -- Side length is perimeter divided by 4
  
  -- We need to show that C = 200 and a = 50
  use [C, a]
  have hC : C = 200 := by
    rw [h_distance, h_laps]
    exact calc (400 : Int) / 2 = 200 : by norm_num
  
  have ha : a = 50 := by
    exact calc (200 : Int) / 4 = 50 : by norm_num
  
  exact ⟨hC, ha⟩
  sorry

end square_playground_properties_l6_6468


namespace find_c_l6_6479

open Real

theorem find_c (c : ℝ) : 
  let v := (matrix.vecCons (-3) (matrix.vecCons c matrix.vecEmpty))
      u := (matrix.vecCons 1 (matrix.vecCons 2 matrix.vecEmpty))
      proj_v_onto_u := (-(7 / 5) : ℝ) • u 
  in (\u v : ℝ) • (u • v / (u • u)) = proj_v_onto_u → c = -2 := 
by
  let v := (matrix.vecCons (-3) (matrix.vecCons c matrix.vecEmpty))
  let u := (matrix.vecCons 1 (matrix.vecCons 2 matrix.vecEmpty))
  let proj_v_onto_u := (-(7 / 5) : ℝ) • u
  have H : (\u v : ℝ) • (u • v / (u • u)) = proj_v_onto_u → c = -2
  sorry

end find_c_l6_6479


namespace pounds_of_coffee_bought_l6_6302

theorem pounds_of_coffee_bought 
  (total_amount_gift_card : ℝ := 70) 
  (cost_per_pound : ℝ := 8.58) 
  (amount_left_on_card : ℝ := 35.68) :
  (total_amount_gift_card - amount_left_on_card) / cost_per_pound = 4 :=
sorry

end pounds_of_coffee_bought_l6_6302


namespace sequence_distinct_l6_6744

theorem sequence_distinct (f : ℕ → ℕ) (h : ∀ n : ℕ, f (f n) = f (n + 1) + f n) :
  ∀ i j : ℕ, i ≠ j → f i ≠ f j :=
by
  sorry

end sequence_distinct_l6_6744


namespace cos_angle_AOB_rectangle_l6_6223

theorem cos_angle_AOB_rectangle (ABCD : rectangle) (AB BC : ℝ)
  (diagonals_intersect : ∃ O, midpoint O A B ∧ midpoint O C D ∧ O ∈ AC ∧ O ∈ BD)
  (hAB : AB = 8) (hBC : BC = 15) :
  ∃ A B O : point, ∃ C D : point, fcos ∠AOB = 8/17 := by
  sorry

end cos_angle_AOB_rectangle_l6_6223


namespace minimum_distance_after_9_minutes_l6_6446

-- Define the initial conditions and movement rules of the robot
structure RobotMovement :=
  (minutes : ℕ)
  (movesStraight : Bool) -- Did the robot move straight in the first minute
  (speed : ℕ)          -- The speed, which is 10 meters/minute
  (turns : Fin (minutes + 1) → ℤ) -- Turns in degrees (-90 for left, 0 for straight, 90 for right)

-- Define the distance function for the robot movement after given minutes
def distanceFromOrigin (rm : RobotMovement) : ℕ :=
  -- This function calculates the minimum distance from the origin where the details are abstracted
  sorry

-- Define the specific conditions of our problem
def robotMovementExample : RobotMovement :=
  { minutes := 9, movesStraight := true, speed := 10,
    turns := λ i => if i = 0 then 0 else -- no turn in the first minute
                      if i % 2 == 0 then 90 else -90 -- Example turning pattern
  }

-- Statement of the proof
theorem minimum_distance_after_9_minutes :
  distanceFromOrigin robotMovementExample = 10 :=
sorry

end minimum_distance_after_9_minutes_l6_6446


namespace sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1_l6_6751

theorem sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1 (t : ℝ) : 
  Real.sqrt (t^4 + t^2) = |t| * Real.sqrt (t^2 + 1) :=
sorry

end sqrt_t4_plus_t2_eq_abs_t_sqrt_t2_plus_1_l6_6751


namespace total_dots_not_visible_l6_6092

-- Define the conditions and variables
def total_dots_one_die : Nat := 1 + 2 + 3 + 4 + 5 + 6
def number_of_dice : Nat := 4
def total_dots_all_dice : Nat := number_of_dice * total_dots_one_die
def visible_numbers : List Nat := [6, 6, 4, 4, 3, 2, 1]

-- The question can be formalized as proving that the total number of dots not visible is 58
theorem total_dots_not_visible :
  total_dots_all_dice - visible_numbers.sum = 58 :=
by
  -- Statement only, proof skipped
  sorry

end total_dots_not_visible_l6_6092


namespace derivative_y₁_derivative_y₂_derivative_y₃_derivative_y₄_l6_6076

-- Definitions of the functions
def y₁ (x : ℝ) : ℝ := log x + 1 / x
def y₂ (x : ℝ) : ℝ := (2 * x ^ 2 - 1) * (3 * x + 1)
def y₃ (x : ℝ) : ℝ := x - (1 / 2) * sin x
def y₄ (x : ℝ) : ℝ := cos x / exp x

-- Proof statements
theorem derivative_y₁ : ∀ x : ℝ, 0 < x → deriv y₁ x = (x - 1) / x^2 := by
sorry

theorem derivative_y₂ : ∀ x : ℝ, deriv y₂ x = 18 * x^2 + 4 * x - 3 := by
sorry

theorem derivative_y₃ : ∀ x : ℝ, deriv y₃ x = 1 - (1 / 2) * cos x := by
sorry

theorem derivative_y₄ : ∀ x : ℝ, deriv y₄ x = (-sin x - cos x) / exp x := by
sorry

end derivative_y₁_derivative_y₂_derivative_y₃_derivative_y₄_l6_6076


namespace train_speed_l6_6393

def train_length : ℝ := 1000  -- train length in meters
def time_to_cross_pole : ℝ := 200  -- time to cross the pole in seconds

theorem train_speed : train_length / time_to_cross_pole = 5 := by
  sorry

end train_speed_l6_6393


namespace find_first_episode_l6_6240

variable (x : ℕ)
variable (w y z : ℕ)
variable (total_minutes: ℕ)
variable (h1 : w = 62)
variable (h2 : y = 65)
variable (h3 : z = 55)
variable (h4 : total_minutes = 240)

theorem find_first_episode :
  x + w + y + z = total_minutes → x = 58 := 
by
  intro h
  rw [h1, h2, h3, h4] at h
  linarith

end find_first_episode_l6_6240


namespace pond_completely_frozen_l6_6445

noncomputable def pond_freezing_day (a : ℝ) (h1 : a > 20) (h2 : (a - 20)^2 = 0.65 * a^2) : ℕ :=
  let n := a / 20
  ⌈n⌉

theorem pond_completely_frozen (a : ℝ) (h1 : a > 20) (h2 : (a - 20)^2 = 0.65 * a^2) :
    pond_freezing_day a h1 h2 = 6 :=
  sorry

end pond_completely_frozen_l6_6445


namespace find_general_formula_count_rational_Tn_l6_6938

-- Define the positive sequence {a_n} and the sum of first n terms as S_n
def a (n : ℕ) : ℕ := n + 1
def S (n : ℕ) : ℕ := (n + 1) * (n + 2) / 2

-- Define the conditions given in the problem
axiom pos_sequence : ∀ n : ℕ, a (n + 1) > 0
axiom sum_condition : ∀ n : ℕ, 2 * S n = a (n + 1)^2 + a (n + 1)

-- Defining b_n and T_n based on the given solution's transformation
def b (n : ℕ) : ℝ := (a n)⁻¹ * (a (n + 1))⁻¹ * 2 / (a n + a (n + 1))
noncomputable def T (n : ℕ) := ∑ i in Finset.range n, b i

-- Proof of the first part of the problem
theorem find_general_formula :
  ∀ n : ℕ, a n = n + 1 :=
  sorry

-- Proof for the number of rational T_n values among the first 100 terms
theorem count_rational_Tn :
  Finset.card (Finset.filter (λ n, ∃ q : ℚ, T n = q) (Finset.range 100)) = 9 :=
  sorry

end find_general_formula_count_rational_Tn_l6_6938


namespace common_roots_product_sum_l6_6707

theorem common_roots_product_sum (C D u v w t p q r : ℝ) (huvw : u^3 + C * u - 20 = 0) (hvw : v^3 + C * v - 20 = 0)
  (hw: w^3 + C * w - 20 = 0) (hut: t^3 + D * t^2 - 40 = 0) (hvw: v^3 + D * v^2 - 40 = 0) 
  (hu: u^3 + D * u^2 - 40 = 0) (h1: u + v + w = 0) (h2: u * v * w = 20) 
  (h3: u * v + u * t + v * t = 0) (h4: u * v * t = 40) :
  p = 4 → q = 3 → r = 5 → p + q + r = 12 :=
by sorry

end common_roots_product_sum_l6_6707


namespace g_at_9_l6_6721

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g_at_3 : g 3 = 4

theorem g_at_9 : g 9 = 64 :=
by
  sorry

end g_at_9_l6_6721


namespace inscribed_vs_circumscribed_sphere_surface_area_ratio_l6_6743

theorem inscribed_vs_circumscribed_sphere_surface_area_ratio :
  ∀ (a : ℝ), a > 0 →
  let r_inscribed := a / 2 in
  let r_circumscribed := a * (real.sqrt 3) / 2 in
  4 * real.pi * r_inscribed^2 / (4 * real.pi * r_circumscribed^2) = 1 / 3 :=
by
  intros a ha
  let r_inscribed := a / 2
  let r_circumscribed := a * (real.sqrt 3) / 2
  sorry

end inscribed_vs_circumscribed_sphere_surface_area_ratio_l6_6743


namespace salt_solution_percentage_l6_6584

theorem salt_solution_percentage :
  ∀ (x y : ℝ), (x = 30) → (y = 30) →
  ((0.60 * x + 0.20 * 30) / (x + 30) * 100 = 40) :=
by
  intros,
  sorry

end salt_solution_percentage_l6_6584


namespace smallest_n_exists_l6_6514

theorem smallest_n_exists (n : ℕ) (h : ∃ (a : ℕ → ℚ), (∀ i, 0 < a i) ∧
  (∃ b : ℕ, ∀ m, m ≥ b → 
    isInt (finset.sum (finset.range (m + 1)) (\i => a i)) ∧ 
    isInt (finset.sum (finset.range (m + 1)) (\i => (a i)⁻¹)))) :
  n = 3 :=
begin
  -- Proof goes here
  sorry
end

end smallest_n_exists_l6_6514


namespace decimal_to_fraction_l6_6019

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l6_6019


namespace farmer_goats_l6_6427

theorem farmer_goats (cows sheep goats : ℕ) (extra_goats : ℕ) 
(hcows : cows = 7) (hsheep : sheep = 8) (hgoats : goats = 6) 
(h : (goats + extra_goats = (cows + sheep + goats + extra_goats) / 2)) : 
extra_goats = 9 := by
  sorry

end farmer_goats_l6_6427


namespace repeating_decimal_as_fraction_l6_6043

theorem repeating_decimal_as_fraction : (0.\overline{02} = 2 / 99) → (2.\overline{06} = 68 / 33) :=
by
  sorry

end repeating_decimal_as_fraction_l6_6043


namespace inequality_necessary_not_sufficient_l6_6877

theorem inequality_necessary_not_sufficient (m : ℝ) : 
  (-3 < m ∧ m < 5) → (5 - m > 0 ∧ m + 3 > 0 ∧ 5 - m ≠ m + 3) :=
by
  intro h
  sorry

end inequality_necessary_not_sufficient_l6_6877


namespace driver_A_more_distance_l6_6804

-- Definitions for the conditions
def distance_between_stops : ℝ := 855
def speed_A : ℝ := 90
def speed_B : ℝ := 80
def time_A_starts_earlier : ℝ := 1

-- Statement to prove
theorem driver_A_more_distance : 
  let total_time : ℝ := (distance_between_stops - speed_A * time_A_starts_earlier) / (speed_A + speed_B),
      distance_A : ℝ := speed_A * total_time + speed_A * time_A_starts_earlier,
      distance_B : ℝ := speed_B * total_time
  in distance_A - distance_B = 135 :=
by
  -- Proof initially omitted
  sorry

end driver_A_more_distance_l6_6804


namespace pagoda_lights_l6_6225

theorem pagoda_lights (a1 : ℕ) (n : ℕ) (r : ℕ) (S_n : ℕ) :
  n = 7 → r = 2 → S_n = 381 → S_n = a1 * (1 - r^n) / (1 - r) → a1 = 3 :=
by
  intros h_n h_r h_S_n h_formula
  -- Assign the values from the assumptions
  rw [h_n, h_r] at h_formula
  -- The sum formula assumption gives: 381 = a1 * (1 - 2^7) / (1 - 2)
  -- Begin by continuing with the calculations
  have h2 : 2^7 = 128 := by norm_num
  rw h2 at h_formula
  -- Simplify the right-hand side expression
  have h3 : 1 - 128 = -127 := by norm_num
  rw h3 at h_formula
  have h4 : 1 - 2 = -1 := by norm_num
  rw h4 at h_formula
  -- This gives: 381 = a1 * (-127) / (-1)
  -- Simplify the fraction
  have h5 : -127 / -1 = 127 := by norm_num
  rw h5 at h_formula
  -- This results in: 381 = 127 * a1
  -- Finally solving for a1
  have h6 : a1 = 381 / 127 := by field_simp
  rw h6
  norm_num
  sorry

end pagoda_lights_l6_6225


namespace hair_color_cost_l6_6469

-- Definitions of the conditions based on a)
def total_items : Nat := 18
def slippers_quantity : Nat := 6
def slippers_price_each : Float := 2.5
def lipsticks_quantity : Nat := 4
def lipsticks_price_each : Float := 1.25
def hair_colors_quantity : Nat := 8
def total_paid : Float := 44

-- The theorem to prove that the cost of each hair color is $3
theorem hair_color_cost :
  let total_slippers_cost := slippers_quantity * slippers_price_each
  let total_lipsticks_cost := lipsticks_quantity * lipsticks_price_each
  let total_known_cost := total_slippers_cost + total_lipsticks_cost
  let hair_color_total_cost := total_paid - total_known_cost
  let hair_color_each_cost := hair_color_total_cost / hair_colors_quantity
  hair_color_each_cost = 3 := sorry

end hair_color_cost_l6_6469


namespace black_pigeons_count_l6_6829

theorem black_pigeons_count
    (total_pigeons : ℕ)
    (half_black : total_pigeons / 2)
    (percent_male : ℕ → Prop)
    (eighty_percent_female : ∀ b, percent_male b → b * 4 = total_pigeons / 2)
    (male_black_pigeons female_black_pigeons: ℕ) :
    total_pigeons = 70 →
    half_black = 35 →
    percent_male 20 →
    male_black_pigeons = 7 →
    female_black_pigeons = half_black - male_black_pigeons →
    female_black_pigeons - male_black_pigeons = 21 :=
by
  intros _ _ _ _ _ _ _ _ _ _ _
  sorry

end black_pigeons_count_l6_6829


namespace neither_long_furred_nor_brown_dogs_is_8_l6_6999

def total_dogs : ℕ := 45
def long_furred_dogs : ℕ := 29
def brown_dogs : ℕ := 17
def long_furred_and_brown_dogs : ℕ := 9

def neither_long_furred_nor_brown_dogs : ℕ :=
  total_dogs - (long_furred_dogs + brown_dogs - long_furred_and_brown_dogs)

theorem neither_long_furred_nor_brown_dogs_is_8 :
  neither_long_furred_nor_brown_dogs = 8 := 
by 
  -- Here we can use substitution and calculation steps used in the solution
  sorry

end neither_long_furred_nor_brown_dogs_is_8_l6_6999


namespace range_of_a_l6_6110

variables {x a : ℝ}

def p : Prop := x^2 + 2 * x - 3 > 0
def q : Prop := x > a
def neg_p : Prop := -3 ≤ x ∧ x ≤ 1
def neg_q : Prop := x ≤ a

theorem range_of_a (h1 : ∀ x, p ↔ (x < -3 ∨ x > 1))
  (h2 : ∀ x, q ↔ (x > a))
  (suff_not_necess_cond : ∀ x, neg_p x → neg_q x):
  ∀ a, a ≥ 1 :=
  begin
  sorry
end

end range_of_a_l6_6110


namespace min_distance_from_start_after_9_minutes_l6_6448

noncomputable def robot_min_distance : ℝ :=
  let movement_per_minute := 10
  sorry

theorem min_distance_from_start_after_9_minutes :
  robot_min_distance = 10 :=
sorry

end min_distance_from_start_after_9_minutes_l6_6448


namespace recurring_decimal_to_fraction_l6_6030

theorem recurring_decimal_to_fraction
  (h : 0.\overline{02} = (2 : ℝ) / 99) :
  2.\overline{06} = 68 / 33 := by
  sorry

end recurring_decimal_to_fraction_l6_6030


namespace sum_of_complex_numbers_l6_6152

-- Define the given complex numbers
def B : ℂ := 3 + 2 * Complex.i
def Q : ℂ := -5
def R : ℂ := -2 * Complex.i
def T : ℂ := 3 + 5 * Complex.i

-- Prove that the sum is 1 + 5i
theorem sum_of_complex_numbers :
  B - Q + R + T = 1 + 5 * Complex.i :=
by
  -- Using placeholder proof for now
  sorry

end sum_of_complex_numbers_l6_6152


namespace isabella_hair_length_l6_6638

def initial_length : ℕ := 18
def growth : ℕ := 6
def final_length : ℕ := initial_length + growth

theorem isabella_hair_length : final_length = 24 :=
by
  simp [final_length, initial_length, growth]
  sorry

end isabella_hair_length_l6_6638


namespace simplify_expression_l6_6308

theorem simplify_expression (x : ℝ) : (3 * x + 6 - 5 * x) / 3 = - (2 / 3) * x + 2 :=
by
  sorry

end simplify_expression_l6_6308


namespace total_number_of_pets_l6_6342

-- Define the numbers given in the conditions
def dogs : ℕ := 43
def fish : ℕ := 72
def cats : ℕ := 34
def chickens : ℕ := 120
def rabbits : ℕ := 57
def parrots : ℕ := 89

-- The goal is to prove the total number of pets is 415
theorem total_number_of_pets : dogs + fish + cats + chickens + rabbits + parrots = 415 :=
by simp [dogs, fish, cats, chickens, rabbits, parrots]; exact sorry

end total_number_of_pets_l6_6342


namespace price_difference_eq_l6_6475

-- Define the problem conditions
variable (P : ℝ) -- Original price
variable (H1 : P - 0.15 * P = 61.2) -- Condition 1: 15% discount results in $61.2
variable (H2 : P * (1 - 0.15) = 61.2) -- Another way to represent Condition 1 (if needed)
variable (H3 : 61.2 * 1.25 = 76.5) -- Condition 4: Price raises by 25% after the 15% discount
variable (H4 : 76.5 * 0.9 = 68.85) -- Condition 5: Additional 10% discount after raise
variable (H5 : P = 72) -- Calculated original price

-- Define the theorem to prove
theorem price_difference_eq :
  (P - 68.85 = 3.15) := 
by
  sorry

end price_difference_eq_l6_6475


namespace rowing_distance_l6_6336

theorem rowing_distance (D : ℝ) : 
  (D / 14 + D / 2 = 120) → D = 210 := by
  sorry

end rowing_distance_l6_6336


namespace problem_statement_l6_6565

noncomputable def f (x a : ℝ) : ℝ := x^2 + (2 * a - 8) * x

theorem problem_statement
  (f : ℝ → ℝ → ℝ)
  (sol_set : Set ℝ)
  (cond1 : ∀ a : ℝ, sol_set = {x : ℝ | -1 ≤ x ∧ x ≤ 5} → ∀ x : ℝ, f x a ≤ 5 ↔ x ∈ sol_set)
  (cond2 : ∀ x : ℝ, ∀ m : ℝ, f x 2 ≥ m^2 - 4 * m - 9) :
  (∃ a : ℝ, a = 2) ∧ (∀ m : ℝ, -1 ≤ m ∧ m ≤ 5) :=
by
  sorry

end problem_statement_l6_6565


namespace AngleMeasure_l6_6405

-- Define necessary terms and conditions using Lean 4.
def ScaleneTriangle (A B C : Point) : Prop :=
  A ≠ B ∧ B ≠ C ∧ C ≠ A ∧ ∠A ≠ ∠B ∧ ∠B ≠ ∠C ∧ ∠C ≠ ∠A

noncomputable def Circumcenter (A B C : Point) : Point := sorry
noncomputable def Orthocenter (A B C : Point) : Point := sorry
noncomputable def AngleBisector (A B C : Point) (α : Angle) : Line := sorry

theorem AngleMeasure (A B C O H : Point) (ON : Line) :
    ScaleneTriangle A B C →
    O = Circumcenter A B C →
    H = Orthocenter A B C →
    Parallel ON (AngleBisector A B C ∠C) →
    ∠C = 120 :=
  by sorry

end AngleMeasure_l6_6405


namespace min_bills_required_l6_6761

-- Conditions
def ten_dollar_bills := 13
def five_dollar_bills := 11
def one_dollar_bills := 17
def total_amount := 128

-- Prove that Tim can pay exactly $128 with the minimum number of bills being 16
theorem min_bills_required : (∃ ten five one : ℕ, 
    ten ≤ ten_dollar_bills ∧
    five ≤ five_dollar_bills ∧
    one ≤ one_dollar_bills ∧
    ten * 10 + five * 5 + one = total_amount ∧
    ten + five + one = 16) :=
by
  -- We will skip the proof for now
  sorry

end min_bills_required_l6_6761


namespace installation_cost_l6_6674

theorem installation_cost (P I : ℝ) (h₁ : 0.80 * P = 12500)
  (h₂ : 18400 = 1.15 * (12500 + 125 + I)) :
  I = 3375 :=
by
  sorry

end installation_cost_l6_6674


namespace polynomial_degree_l6_6797

/-
C_{k+1}^{k} S_{k}(n)+C_{k+1}^{k-1} S_{k-1}(n)+...+C_{k+1}^{1} S_{1}(n)+S_{0}(n)
= (n+1)^{k+1} - 1
-/
lemma sum_identity (k n : ℕ) :
  ∑ i in range k.succ, Nat.binomial (k + 1) i * S i n = (n + 1)^(k + 1) - 1 :=
sorry

/-
For a fixed k, S_k(n) is a polynomial of degree k+1 in n with a leading coefficient of n^(k+1)/(k+1)
-/
theorem polynomial_degree (k : ℕ) :
  ∃ p : polynomial ℕ, (∀ n, S k n = p.eval n) ∧ p.degree = k + 1 ∧ p.leading_coeff = (1 : ℕ) / (k + 1) :=
sorry

end polynomial_degree_l6_6797


namespace terminal_side_quadrant_l6_6996

theorem terminal_side_quadrant (θ : ℝ) (h1 : tan θ < 0) (h2 : cos θ > 0) : 
  (1 / 2 * sin (2 * θ) < 0) ∧ (tan θ < 0) → θ ∈ ({θ : ℝ | θ > 3 * π / 2 ∧ θ < 2 * π}) :=
by { sorry }

end terminal_side_quadrant_l6_6996


namespace symmetry_y_axis_l6_6789

theorem symmetry_y_axis (A B C D : ℝ → ℝ → Prop) 
  (A_eq : ∀ x y : ℝ, A x y ↔ (x^2 - x + y^2 = 1))
  (B_eq : ∀ x y : ℝ, B x y ↔ (x^2 * y + x * y^2 = 1))
  (C_eq : ∀ x y : ℝ, C x y ↔ (x^2 - y^2 = 1))
  (D_eq : ∀ x y : ℝ, D x y ↔ (x - y = 1)) : 
  (∀ x y : ℝ, C x y ↔ C (-x) y) ∧ 
  ¬(∀ x y : ℝ, A x y ↔ A (-x) y) ∧ 
  ¬(∀ x y : ℝ, B x y ↔ B (-x) y) ∧ 
  ¬(∀ x y : ℝ, D x y ↔ D (-x) y) :=
by
  -- Proof goes here
  sorry

end symmetry_y_axis_l6_6789


namespace _l6_6645

variables 
  (A B C P D E F : Type)
  [Point : A] [Point : B] [Point : C] [Point : P] 
  [Point : D] [Point : E] [Point : F]
  [Triangle : ∀ {a b c : Type}, (Point a) → (Point b) → (Point c) → Prop]
  [Inside : ∀ {a b c p : Type}, (Point a) → (Point b) → (Point c) → (Point p) → Prop]
  [Intersect : ∀ {p1 p2 p3 p4 : Type}, (Point p1) → (Point p2) → (Point p3) → (Point p4) → Prop]
  [Area : ∀ {A B C : Type}, (Triangle A B C) → ℝ]

noncomputable def inequality_theorem 
  (ABC : Triangle A B C)
  (inside_point : Inside A B C P)
  (intersect_D : Intersect A P B C D)
  (intersect_E : Intersect B P A C E)
  (intersect_F : Intersect C P A B F) :
  AD * BE * CF ≥ 2 * (Area (Triangle D E F)) * (AB + BC + CA) :=
sorry

end _l6_6645


namespace ratio_of_perimeters_l6_6846

theorem ratio_of_perimeters (L : ℝ) (H : ℝ) (hL1 : L = 8) 
  (hH1 : H = 8) (hH2 : H = 2 * (H / 2)) (hH3 : 4 > 0) (hH4 : 0 < 4 / 3)
  (hW1 : ∀ a, a / 3 > 0 → 8 = L )
  (hPsmall : ∀ P, P = 2 * ((4 / 3) + 8) )
  (hPlarge : ∀ P, P = 2 * ((H - 4 / 3) + 8) )
  :
  (2 * ((4 / 3) + 8)) / (2 * ((8 - (4 / 3)) + 8)) = (7 / 11) := by
  sorry

end ratio_of_perimeters_l6_6846


namespace principal_amount_is_900_l6_6335

-- Define the given conditions
def SimpleInterest : ℝ := 160
def Rate : ℝ := 4.444444444444445
def Time : ℝ := 4

-- Define the main theorem to prove that the principal amount equals 900
theorem principal_amount_is_900 (P : ℝ) :
  P = SimpleInterest * 100 / (Rate * Time) → P = 900 :=
by
  assume h1 : P = SimpleInterest * 100 / (Rate * Time)
  sorry

end principal_amount_is_900_l6_6335


namespace num_integers_between_negative_5pi_and_7pi_l6_6983

theorem num_integers_between_negative_5pi_and_7pi : 
  (∃ (n : ℤ), -5 * Real.pi ≤ n ∧ n ≤ 7 * Real.pi) :=
begin
  sorry
end

end num_integers_between_negative_5pi_and_7pi_l6_6983


namespace weight_of_one_book_l6_6417

theorem weight_of_one_book (total_weight : ℝ) (empty_bag_weight : ℝ) (num_books : ℕ) (book_weight : ℝ) :
  total_weight = 11.14 ∧ empty_bag_weight = 0.5 ∧ num_books = 14 ∧ book_weight = 0.76 →
  (total_weight - empty_bag_weight) / num_books = book_weight :=
by
  intro h
  cases h with h_total_weight h_rest
  cases h_rest with h_empty_bag_weight h_rest'
  cases h_rest' with h_num_books h_book_weight
  rw [h_total_weight, h_empty_bag_weight, h_num_books, h_book_weight]
  norm_num
  rfl

end weight_of_one_book_l6_6417


namespace population_growth_correct_l6_6436

theorem population_growth_correct :
  (∀ (species : Type) (environment : Type) (area : environment),
    (population_change species area = "J" shaped growth then "S" shaped growth)) :=
begin
  sorry
end

end population_growth_correct_l6_6436


namespace solvable_eq_l6_6502

theorem solvable_eq (x : ℝ) :
    Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6 →
    (x = 2 ∨ x = -2) :=
by
  sorry

end solvable_eq_l6_6502


namespace white_squares_95th_figure_l6_6496

theorem white_squares_95th_figure : ∀ (T : ℕ → ℕ),
  T 1 = 8 →
  (∀ n ≥ 1, T (n + 1) = T n + 5) →
  T 95 = 478 :=
by
  intros T hT1 hTrec
  -- Skipping the proof
  sorry

end white_squares_95th_figure_l6_6496


namespace number_of_correct_statements_l6_6252

def floor (x : ℝ) : ℝ := ⌊x⌋₊.toReal
def f (x : ℝ) : ℝ := x - floor x

def is_range_01 (f : ℝ → ℝ) : Prop :=
  ∀ y, 0 ≤ y ∧ y < 1 → ∃ x, f x = y

def has_inf_solutions (f : ℝ → ℝ) (y : ℝ) : Prop :=
  ∀ k : ℤ, ∃ x, f x = y ∧ x = (y + k : ℝ)

def is_periodic (f : ℝ → ℝ) : Prop :=
  ∃ p, p > 0 ∧ ∀ x, f (x + p) = f x

def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂, x₁ ≤ x₂ → f x₁ ≤ f x₂

theorem number_of_correct_statements :
  let statements := [
    is_range_01 f,
    has_inf_solutions f (1/2),
    is_periodic f,
    is_increasing f
  ] in (statements.count id) = 2 :=
by sorry

end number_of_correct_statements_l6_6252


namespace range_of_a_l6_6984

theorem range_of_a (a : ℝ) : ¬ (∃ x : ℝ, a * x^2 - 3 * a * x + 9 ≤ 0) → a ∈ Set.Ico 0 4 := by
  sorry

end range_of_a_l6_6984


namespace complement_of_A_is_CR_A_l6_6093

-- Definitions
def A : Set ℝ := {x | log x > 0}
def CR_A : Set ℝ := {x | x ≤ 1}

-- Theorem statement
theorem complement_of_A_is_CR_A : (C_R A) = {x : ℝ | x ≤ 1} :=
by
  sorry

end complement_of_A_is_CR_A_l6_6093


namespace amy_tickets_initial_l6_6466

theorem amy_tickets_initial (x : ℕ) (h1 : x + 21 = 54) : x = 33 :=
by sorry

end amy_tickets_initial_l6_6466


namespace max_integer_solutions_l6_6444

theorem max_integer_solutions (p : ℤ[X]) (h₀ : p.coeffs ∈ (set.range (coe : ℤ → ℤ))) 
(h₁ : p.eval 50 = 50) : 
  ∃ k₁ k₂ k₃ k₄ k₅ k₆ : ℤ, 
    p.eval k₁ = k₁ ^ 2 ∧ 
    p.eval k₂ = k₂ ^ 2 ∧ 
    p.eval k₃ = k₃ ^ 2 ∧ 
    p.eval k₄ = k₄ ^ 2 ∧ 
    p.eval k₅ = k₅ ^ 2 ∧ 
    p.eval k₆ = k₆ ^ 2 ∧
    ((set.to_finset {k₁, k₂, k₃, k₄, k₅, k₆}).card ≤ 6) := 
sorry

end max_integer_solutions_l6_6444


namespace ice_skating_rinks_and_ski_resorts_2019_l6_6689

theorem ice_skating_rinks_and_ski_resorts_2019 (x y : ℕ) :
  x + y = 1230 →
  2 * x + 212 + y + 288 = 2560 →
  x = 830 ∧ y = 400 :=
by {
  sorry
}

end ice_skating_rinks_and_ski_resorts_2019_l6_6689


namespace recurring_decimal_reduced_fraction_l6_6055

noncomputable def recurring_decimal_as_fraction : Prop := 
  ∀ (x y : ℚ), (x = 2.06) ∧ (y = 0.02) → y = 2 / 99 → x = 68 / 33

theorem recurring_decimal_reduced_fraction (x y : ℚ) 
  (h1 : x = 2 + 0.06) (h2 : y = 0.02) (h3 : y = 2 / 99) : 
  x = 68 / 33 := 
begin
  -- Proof here
  sorry
end

end recurring_decimal_reduced_fraction_l6_6055


namespace profit_initial_percentage_eq_61_54_l6_6213

variable (C S : ℝ)
variable (h1 : 0 < C) (h2 : 0 < S)

theorem profit_initial_percentage_eq_61_54
  (h1 : 0 < C) -- initial cost is positive
  (h2 : 0 < S) -- initial selling price is positive
  (h3 : ((S - 1.12 * C) / S) * 100 ≈ 56.92307692307692) -- condition about new profit percentage after cost increase
  : ((S - C) / C) * 100 ≈ 61.53846153846154 := sorry

end profit_initial_percentage_eq_61_54_l6_6213


namespace part1_part2_l6_6530

open Real

theorem part1 (x : ℝ) (t : ℝ) : (|x-1| - |x-2| ≥ t) ↔ t ≤ 1 := sorry

theorem part2 {m n : ℝ} (h₁ : 1 < m) (h₂ : 1 < n) : 
  (∀ {t : ℝ}, t ∈ Iic 1 → log 3 m * log 3 n ≥ t) → m + n = 6 := sorry

end part1_part2_l6_6530


namespace recurring_decimal_reduced_fraction_l6_6054

noncomputable def recurring_decimal_as_fraction : Prop := 
  ∀ (x y : ℚ), (x = 2.06) ∧ (y = 0.02) → y = 2 / 99 → x = 68 / 33

theorem recurring_decimal_reduced_fraction (x y : ℚ) 
  (h1 : x = 2 + 0.06) (h2 : y = 0.02) (h3 : y = 2 / 99) : 
  x = 68 / 33 := 
begin
  -- Proof here
  sorry
end

end recurring_decimal_reduced_fraction_l6_6054


namespace solutions_to_equation_l6_6499

theorem solutions_to_equation :
  ∀ x : ℝ, 
  sqrt ((3 + sqrt 5) ^ x) + sqrt ((3 - sqrt 5) ^ x) = 6 ↔ (x = 2 ∨ x = -2) :=
by
  intros x
  sorry

end solutions_to_equation_l6_6499


namespace eccentricity_of_ellipse_l6_6733

variables {a b c e : ℝ}
variables (F1 F2 A B : ℝ × ℝ)

-- Given conditions
def ellipse (a b : ℝ) (x y : ℝ) := (x^2) / (a^2) + (y^2) / (b^2) = 1
def points_on_ellipse (a b : ℝ) (A B : ℝ × ℝ) : Prop :=
  ellipse a b A.1 A.2 ∧ ellipse a b B.1 B.2
def collinear (A B F1 : ℝ × ℝ) : Prop :=
  B.1 - A.1 = 3 * (F1.1 - B.1) ∧ B.2 - A.2 = 3 * (F1.2 - B.2)
def right_angle (A B F2 : ℝ × ℝ) : Prop :=
  (B.1 - A.1) * (F2.1 - A.1) + (B.2 - A.2) * (F2.2 - A.2) = 0
def foci (a b : ℝ) : ℝ := sqrt (a^2 - b^2)

theorem eccentricity_of_ellipse 
  (h₀ : a > b > 0)
  (h₁ : points_on_ellipse a b A B)
  (h₂ : collinear A B F1)
  (h₃ : right_angle A B F2) :
  let c := foci a b in
  e = c / a :=
by
  sorry

end eccentricity_of_ellipse_l6_6733


namespace find_ks_l6_6890

theorem find_ks (
  k : ℕ
) : k = 1 ∨ k = 2 ∨ k = 4 ↔ 
  ∀ (p : ℕ) (h_prime_p: p.prime) (a b : ℕ) (h_pos_a: 0 < a) (h_pos_b: 0 < b), 
  (p^2 = a^2 + k * b^2) → 
  ∃ (x y : ℕ), (0 < x) ∧ (0 < y) ∧ (p = x^2 + k * y^2) := 
sorry

end find_ks_l6_6890


namespace sum_x_y_is_9_l6_6607

-- Definitions of the conditions
variables (x y S : ℝ)
axiom h1 : x + y = S
axiom h2 : x - y = 3
axiom h3 : x^2 - y^2 = 27

-- The theorem to prove
theorem sum_x_y_is_9 : S = 9 :=
by
  -- Placeholder for the proof
  sorry

end sum_x_y_is_9_l6_6607


namespace recurring_decimal_to_fraction_l6_6048

theorem recurring_decimal_to_fraction
  (h : (2:ℚ) + 3 * (2 / 99) = 2.06) :
  (2:ℚ) + 0.\overline{06} = (68 / 33) :=
by
  -- Given: 0.\overline{02} = 2 / 99
  have h0 : (0.\overline{02} : ℚ) = 2 / 99 := by sorry

  -- 0.\overline{06} = 3 * 0.\overline{02}
  have h1 : (0.\overline{06} : ℚ) = 3 * (0.\overline{02} : ℚ) :=
    by rw [← h0]; sorry

  -- Hence, 0.\overline{06} = 6 / 99 = 2 / 33
  have h2 : (0.\overline{06} : ℚ) = 2 / 33 :=
    by sorry

  -- Therefore, 2.\overline{06} = 2 + 0.\overline{06} = 2 + 2 / 33 = 68 / 33
  show (2:ℚ) + (0.\overline{06} : ℚ) = 68 / 33
    by sorry

end recurring_decimal_to_fraction_l6_6048


namespace find_repeating_digits_l6_6451

theorem find_repeating_digits (c d : ℕ) (h1 : 0 < c ∧ c < 10) (h2 : 0 <= d ∧ d < 10)
  (h3 : 42 * (1 + (c + d / 10) * 1/9) - 42 * (1 + c / 10 + d / 100) = 0.8) :
  c * 10 + d = 19 :=
sorry

end find_repeating_digits_l6_6451


namespace polynomial_factorization_l6_6382

theorem polynomial_factorization : 
  (x : ℤ) → (x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1)) := 
by
  sorry

end polynomial_factorization_l6_6382


namespace range_of_a_l6_6975

theorem range_of_a (a : ℝ) :
  (a > 0 ∧ (∃ x, x^2 - 4 * a * x + 3 * a^2 < 0)) →
  (∃ x, x^2 - x - 6 ≤ 0 ∧ x^2 + 2 * x - 8 > 0) →
  (2 < a ∧ a ≤ 2) := sorry

end range_of_a_l6_6975


namespace hardest_and_least_work_difference_l6_6694

theorem hardest_and_least_work_difference :
  ∃ (x : ℕ), (2 * x) + (3 * x) + (4 * x) = 90 → (4 * x) - (2 * x) = 20 :=
begin
  use 10,
  intros h,
  have : 9 * 10 = 90 := by norm_num,
  rw this at h,
  norm_num,
end

end hardest_and_least_work_difference_l6_6694


namespace recurring_decimal_to_fraction_l6_6050

theorem recurring_decimal_to_fraction
  (h : (2:ℚ) + 3 * (2 / 99) = 2.06) :
  (2:ℚ) + 0.\overline{06} = (68 / 33) :=
by
  -- Given: 0.\overline{02} = 2 / 99
  have h0 : (0.\overline{02} : ℚ) = 2 / 99 := by sorry

  -- 0.\overline{06} = 3 * 0.\overline{02}
  have h1 : (0.\overline{06} : ℚ) = 3 * (0.\overline{02} : ℚ) :=
    by rw [← h0]; sorry

  -- Hence, 0.\overline{06} = 6 / 99 = 2 / 33
  have h2 : (0.\overline{06} : ℚ) = 2 / 33 :=
    by sorry

  -- Therefore, 2.\overline{06} = 2 + 0.\overline{06} = 2 + 2 / 33 = 68 / 33
  show (2:ℚ) + (0.\overline{06} : ℚ) = 68 / 33
    by sorry

end recurring_decimal_to_fraction_l6_6050


namespace max_integer_solutions_l6_6441

noncomputable def semi_centered (p : ℕ → ℤ) :=
  ∃ k : ℕ, p k = k + 50 - 50 * 50

theorem max_integer_solutions (p : ℕ → ℤ) (h1 : semi_centered p) (h2 : ∀ x : ℕ, ∃ c : ℤ, p x = c * x^2) (h3 : p 50 = 50) :
  ∃ n ≤ 6, ∀ k : ℕ, (p k = k^2) → k ∈ Finset.range (n+1) :=
sorry

end max_integer_solutions_l6_6441


namespace more_black_females_than_males_l6_6831

theorem more_black_females_than_males 
  (total_pigeons : ℕ)
  (half_black : total_pigeons / 2 = 35)
  (percent_male : 20)
  (black_pigeons : total_pigeons / 2 = 35)
  (black_male_pigeons : black_pigeons * percent_male / 100 = 7) :
  (black_pigeons - black_male_pigeons) - black_male_pigeons = 21 :=
by
  sorry

end more_black_females_than_males_l6_6831


namespace total_cost_of_commodities_l6_6702

theorem total_cost_of_commodities (a b : ℕ) (h₁ : a = 477) (h₂ : a - b = 127) : a + b = 827 :=
by
  sorry

end total_cost_of_commodities_l6_6702


namespace vector_inner_sum_l6_6647

variables {E : Type*} [inner_product_space ℝ E]

-- Conditions
variables (u v w : E)
variables (h_norm_u : ∥u∥ = 2) (h_norm_v : ∥v∥ = 3) (h_norm_w : ∥w∥ = 6)
variables (h_sum : u + 2 • v + w = 0)

-- Statement to prove
theorem vector_inner_sum :
  inner u v + inner u w + inner v w = -29 :=
by sorry

end vector_inner_sum_l6_6647


namespace find_a_l6_6083

theorem find_a (a : ℝ) (h : a > 0) 
  (h_expansion : (λ x : ℝ, (1 - x) * (1 + a * x) ^ 6)
       .coeff 2 = 9) : 
  a = 1 :=
sorry

end find_a_l6_6083


namespace length_of_wire_stretched_l6_6698

theorem length_of_wire_stretched : 
  ∀ (d b2 b1 : ℕ), d = 20 → b2 = 3 → b1 = 13 →
  (sqrt (d ^ 2 + b1 ^ 2) = sqrt 569) :=
by
  intros d b2 b1 h_d h_b2 h_b1
  sorry

end length_of_wire_stretched_l6_6698


namespace base_n_product_l6_6520

theorem base_n_product :
  ∏ (n : ℕ) in finset.range 96 + 5, ((n + 2) ^ 2) / (n * (n ^ 3 - 1)) = 20808 / nat.factorial 99 := by
  sorry

end base_n_product_l6_6520


namespace necessary_but_not_sufficient_cond_l6_6112

section
variables {x m : ℝ} (p q : Prop)

def p := abs (1 - (x - 1) / 3) ≤ 2
def q := (x^2 - 2 * x + 1 - m^2) ≤ 0 ∧ m > 0

theorem necessary_but_not_sufficient_cond (h : ¬p → ¬q) : m ≥ 9 :=
by
  sorry
end

end necessary_but_not_sufficient_cond_l6_6112


namespace algebraic_expression_value_l6_6927

theorem algebraic_expression_value (x : ℝ) (h : x = 5) : (3 / (x - 4) - 24 / (x^2 - 16)) = (1 / 3) :=
by
  have hx : x = 5 := h
  sorry

end algebraic_expression_value_l6_6927


namespace sugar_water_sweeter_l6_6125

variable (a b m : ℝ)
variable (a_pos : a > 0) (b_gt_a : b > a) (m_pos : m > 0)

theorem sugar_water_sweeter : (a + m) / (b + m) > a / b :=
by
  sorry

end sugar_water_sweeter_l6_6125


namespace polynomial_factorization_l6_6374

open Polynomial

theorem polynomial_factorization :
  (X ^ 15 + X ^ 10 + X ^ 5 + 1) =
    (X ^ 3 + X ^ 2 + 1) * 
    (X ^ 12 - X ^ 11 + X ^ 9 - X ^ 8 + X ^ 6 - X ^ 5 + X ^ 4 + X ^ 3 + X ^ 2 + X + 1) :=
by
  sorry

end polynomial_factorization_l6_6374


namespace range_of_a_l6_6648

def f (x : ℝ) : ℝ := x^2 - 4*x + 3

theorem range_of_a (a : ℝ) (h_inc : ∀ x : ℝ, x ∈ set.Ici 0 → f(x + a) ≤ f((x + 1) + a)) : 2 ≤ a :=
by
  sorry

end range_of_a_l6_6648


namespace arithmetic_mean_correct_l6_6359

-- Definition of the arithmetic mean
def arithmetic_mean (l : List ℕ) : ℚ :=
  l.sum.toRat / l.length

-- Given data and the condition
def given_numbers : List ℕ := [16, 24, 45, 63, 2 * 16]

-- The statement to be proved
theorem arithmetic_mean_correct : arithmetic_mean given_numbers = 36 := 
  sorry

end arithmetic_mean_correct_l6_6359


namespace side_length_of_square_ABCD_l6_6424

noncomputable def findSquareSideLength
  (circle_tangent_to_extensions_of_two_sides : Bool)
  (circle_cuts_segment_from_vertex_A : ℝ)
  (tangents_drawn_from_point_C : Bool)
  (angle_between_tangents : ℝ)
  (sin_18_deg : ℝ) : ℝ :=
  if circle_tangent_to_extensions_of_two_sides 
     && tangents_drawn_from_point_C 
     && angle_between_tangents = 36 
     && sin_18_deg = (Real.sqrt 5 - 1) / 4 then 
    let segment_length := 6 - 2 * Real.sqrt 5 in
    let side_length := (Real.sqrt 5 - 1) * (2 * Real.sqrt 2 - Real.sqrt 5 + 1) in
    side_length
  else
    0 -- Or some indication of invalid input/condition

theorem side_length_of_square_ABCD 
  (h1 : circle_tangent_to_extensions_of_two_sides = true)
  (h2 : circle_cuts_segment_from_vertex_A = 6 - 2 * Real.sqrt 5)
  (h3 : tangents_drawn_from_point_C = true)
  (h4 : angle_between_tangents = 36)
  (h5 : sin_18_deg = (Real.sqrt 5 - 1) / 4) :
  findSquareSideLength true (6 - 2 * Real.sqrt 5) true 36 ((Real.sqrt 5 - 1) / 4) = 
  (Real.sqrt 5 - 1) * (2 * Real.sqrt 2 - Real.sqrt 5 + 1) :=
sorry

end side_length_of_square_ABCD_l6_6424


namespace interval_of_decrease_l6_6876

def g (x : ℝ) := x^2 - 4 * x

def f (x : ℝ) := log (1 / 2) (g x)

theorem interval_of_decrease :
  ∀ x : ℝ, (4 < x) → f x = log (1 / 2) (g x) → ∀ y : ℝ, (4 < y) → (y < x) → f y > f x :=
sorry

end interval_of_decrease_l6_6876


namespace f_is_even_l6_6258

-- Definition of function h being even
def is_even_function (h : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, h(-x) = h(x)

-- Definition of function f
def f (h : ℝ → ℝ) (x : ℝ) : ℝ :=
  |h (x^5)|

-- The proof statement
theorem f_is_even (h : ℝ → ℝ) (h_even : is_even_function h) : ∀ x : ℝ, f h x = f h (-x) :=
by
  sorry

end f_is_even_l6_6258


namespace find_center_of_circle_l6_6819

-- Define the given conditions
def parabola (x : ℝ) : ℝ := x^2

def tangent_at (p : ℝ × ℝ) : ℝ → ℝ :=
  λ x, 2*(x - 1) + 1

-- The center of the circle to be found
def is_center_of_circle (center : ℝ × ℝ) : Prop :=
  let (a, b) := center in
  (a + b = 1) ∧ ((b - 1) / (a - 1) = -1/2) ∧ 
  ( (a^2 + b^2) = ((a - 1)^2 + (b - 1)^2))

-- The proof problem stating the center is (-1, 2)
theorem find_center_of_circle :
  is_center_of_circle (-1, 2) :=
sorry

end find_center_of_circle_l6_6819


namespace surface_area_of_sphere_eq_l6_6709

-- Definitions for our points and conditions
noncomputable def P : Type := sorry
noncomputable def A : Type := sorry
noncomputable def B : Type := sorry
noncomputable def C : Type := sorry

variables (a : ℝ)

-- Condition: Points are on the same spherical surface
axiom points_on_sphere : ∃ (r : ℝ), ∀ (X : P | A | B | C), dist X P = r

-- Condition: Lines are mutually perpendicular
axiom lines_perpendicular : (dist P A = a) ∧ (dist P B = a) ∧ (dist P C = a) 
                             ∧ (angle P A P B = π / 2) ∧ (angle P A P C = π / 2) ∧ (angle P B P C = π / 2)

theorem surface_area_of_sphere_eq :
  ∃ (r : ℝ), dist P A = r ∧ dist P B = r ∧ dist P C = r ∧ dist P P = r ∧ 
  surface_area_sphere r = 3 * π * a^2 := by
sorry

end surface_area_of_sphere_eq_l6_6709


namespace minimum_distance_after_9_minutes_l6_6447

-- Define the initial conditions and movement rules of the robot
structure RobotMovement :=
  (minutes : ℕ)
  (movesStraight : Bool) -- Did the robot move straight in the first minute
  (speed : ℕ)          -- The speed, which is 10 meters/minute
  (turns : Fin (minutes + 1) → ℤ) -- Turns in degrees (-90 for left, 0 for straight, 90 for right)

-- Define the distance function for the robot movement after given minutes
def distanceFromOrigin (rm : RobotMovement) : ℕ :=
  -- This function calculates the minimum distance from the origin where the details are abstracted
  sorry

-- Define the specific conditions of our problem
def robotMovementExample : RobotMovement :=
  { minutes := 9, movesStraight := true, speed := 10,
    turns := λ i => if i = 0 then 0 else -- no turn in the first minute
                      if i % 2 == 0 then 90 else -90 -- Example turning pattern
  }

-- Statement of the proof
theorem minimum_distance_after_9_minutes :
  distanceFromOrigin robotMovementExample = 10 :=
sorry

end minimum_distance_after_9_minutes_l6_6447


namespace sin_double_angle_l6_6095

theorem sin_double_angle (α : ℝ) (h : Real.sin (α + Real.pi / 4) = 1 / 2) : Real.sin (2 * α) = -1 / 2 :=
sorry

end sin_double_angle_l6_6095


namespace recurring_decimal_to_fraction_l6_6026

theorem recurring_decimal_to_fraction
  (h : 0.\overline{02} = (2 : ℝ) / 99) :
  2.\overline{06} = 68 / 33 := by
  sorry

end recurring_decimal_to_fraction_l6_6026


namespace member_sum_or_double_exists_l6_6463

theorem member_sum_or_double_exists (n : ℕ) (k : ℕ) (P: ℕ → ℕ) (m: ℕ) 
  (h_mem : n = 1978)
  (h_countries : m = 6) : 
  ∃ k, (∃ i j, P i + P j = k ∧ P i = P j)
    ∨ (∃ i, 2 * P i = k) :=
sorry

end member_sum_or_double_exists_l6_6463


namespace g_g_x_has_two_distinct_real_roots_iff_l6_6256

noncomputable def g (d x : ℝ) := x^2 - 4 * x + d

def has_two_distinct_real_roots (f : ℝ → ℝ) : Prop := 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

theorem g_g_x_has_two_distinct_real_roots_iff (d : ℝ) :
  has_two_distinct_real_roots (g d ∘ g d) ↔ d = 8 := sorry

end g_g_x_has_two_distinct_real_roots_iff_l6_6256


namespace hcf_of_two_numbers_l6_6601

theorem hcf_of_two_numbers (A B : ℕ) (h1 : Nat.lcm A B = 750) (h2 : A * B = 18750) : Nat.gcd A B = 25 :=
by
  sorry

end hcf_of_two_numbers_l6_6601


namespace area_of_triangle_MNB_is_sixth_of_ABC_l6_6631

-- Definitions and conditions
variables {A B C D E M N : Type}
variables [has_area A B C] [is_median AD] [is_median CE]

-- Statement of the problem
theorem area_of_triangle_MNB_is_sixth_of_ABC
  (h1 : intersect_at_centroid AD CE M)
  (h2 : is_midpoint N A B) :
  area(M, N, B) = (1 / 6) * area(A, B, C) :=
begin
  sorry,
end

end area_of_triangle_MNB_is_sixth_of_ABC_l6_6631


namespace frustum_lateral_surface_area_l6_6824

-- Definitions of the given constants
def lower_base_radius := 8 -- inches
def upper_base_radius := 2 -- inches
def height := 9 -- inches

-- Compute slant height
def slant_height := Math.sqrt ((lower_base_radius - upper_base_radius) ^ 2 + height ^ 2)

-- Compute lateral surface area
def lateral_surface_area := Real.pi * (upper_base_radius + lower_base_radius) * slant_height

-- Statement of the theorem to prove
theorem frustum_lateral_surface_area
  (r₂ : ℝ) (r₁ : ℝ) (h : ℝ)
  (hr₂ : r₂ = 8) (hr₁ : r₁ = 2) (hh : h = 9) :
  lateral_surface_area = 10 * Real.pi * Math.sqrt 117 :=
by
  -- Skip proof
  sorry

end frustum_lateral_surface_area_l6_6824


namespace coefficient_x3_in_expansion_l6_6230

theorem coefficient_x3_in_expansion (n : ℕ) (x : ℕ) : 
  (∑ i in Finset.range (n + 1), (Nat.choose n i) * 2^(n - i) * (x^i)) = 
  (160 : ℕ) 
:=
by
  let n := 6
  let x := 3
  have binomial_theorem : (∑ i in Finset.range (n + 1), (Nat.choose n i) * 2^(n - i) * (x^i)) = ∑ i in Finset.range (7), (Nat.choose 6 i) * 2^(6 - i) * 3^i :=
    by sorry
  rw binomial_theorem
  -- Using a known result manually just to satisfy the proof structure
  have term_x3 : (∑ i in Finset.range (7), (Nat.choose 6 i) * 2^(6 - i) * 3^i).coeff 3 = 160 :=
    by
    sorry
  exact term_x3

end coefficient_x3_in_expansion_l6_6230


namespace smallest_possible_value_abs_sum_l6_6786

theorem smallest_possible_value_abs_sum :
  ∃ x : ℝ, (∀ y : ℝ, abs (y + 3) + abs (y + 5) + abs (y + 7) ≥ abs (x + 3) + abs (x + 5) + abs (x + 7))
  ∧ (abs (x + 3) + abs (x + 5) + abs (x + 7) = 4) := by
  sorry

end smallest_possible_value_abs_sum_l6_6786


namespace recurring_decimal_to_fraction_l6_6052

theorem recurring_decimal_to_fraction
  (h : (2:ℚ) + 3 * (2 / 99) = 2.06) :
  (2:ℚ) + 0.\overline{06} = (68 / 33) :=
by
  -- Given: 0.\overline{02} = 2 / 99
  have h0 : (0.\overline{02} : ℚ) = 2 / 99 := by sorry

  -- 0.\overline{06} = 3 * 0.\overline{02}
  have h1 : (0.\overline{06} : ℚ) = 3 * (0.\overline{02} : ℚ) :=
    by rw [← h0]; sorry

  -- Hence, 0.\overline{06} = 6 / 99 = 2 / 33
  have h2 : (0.\overline{06} : ℚ) = 2 / 33 :=
    by sorry

  -- Therefore, 2.\overline{06} = 2 + 0.\overline{06} = 2 + 2 / 33 = 68 / 33
  show (2:ℚ) + (0.\overline{06} : ℚ) = 68 / 33
    by sorry

end recurring_decimal_to_fraction_l6_6052


namespace new_alcohol_percentage_l6_6827

noncomputable def initial_capacity : ℝ := 1
noncomputable def original_alcohol_percentage : ℝ := 0.40
noncomputable def replaced_capacity : ℝ := 0.7619047619047619
noncomputable def replaced_alcohol_percentage : ℝ := 0.19

theorem new_alcohol_percentage :
  let original_alcohol := initial_capacity * original_alcohol_percentage in
  let replaced_alcohol := replaced_capacity * replaced_alcohol_percentage in
  let remaining_original_capacity := initial_capacity - replaced_capacity in
  let remaining_original_alcohol := remaining_original_capacity * original_alcohol_percentage in
  let total_alcohol := remaining_original_alcohol + replaced_alcohol in
  let new_alcohol_percentage := total_alcohol / initial_capacity * 100 in
  new_alcohol_percentage ≈ 24.0001 := 
by 
  sorry

end new_alcohol_percentage_l6_6827


namespace product_of_largest_integer_l6_6511

theorem product_of_largest_integer (n : ℕ) (digits : List ℕ) :
  (∀ d ∈ digits, d % 2 = 0) ∧
  (∀ i < digits.length - 1, digits[i] < digits[i+1]) ∧
  (digits.map (λ d => d*d)).sum = 41 →
  (digits.foldr (*) 1) = 12 :=
by
  sorry

end product_of_largest_integer_l6_6511


namespace recurring_decimal_to_fraction_l6_6051

theorem recurring_decimal_to_fraction
  (h : (2:ℚ) + 3 * (2 / 99) = 2.06) :
  (2:ℚ) + 0.\overline{06} = (68 / 33) :=
by
  -- Given: 0.\overline{02} = 2 / 99
  have h0 : (0.\overline{02} : ℚ) = 2 / 99 := by sorry

  -- 0.\overline{06} = 3 * 0.\overline{02}
  have h1 : (0.\overline{06} : ℚ) = 3 * (0.\overline{02} : ℚ) :=
    by rw [← h0]; sorry

  -- Hence, 0.\overline{06} = 6 / 99 = 2 / 33
  have h2 : (0.\overline{06} : ℚ) = 2 / 33 :=
    by sorry

  -- Therefore, 2.\overline{06} = 2 + 0.\overline{06} = 2 + 2 / 33 = 68 / 33
  show (2:ℚ) + (0.\overline{06} : ℚ) = 68 / 33
    by sorry

end recurring_decimal_to_fraction_l6_6051


namespace parallel_lines_coefficient_l6_6558

theorem parallel_lines_coefficient (a : ℝ) : 
  (∀ x y : ℝ, (a * x + 2 * y + 2 = 0) → (3 * x - y - 2 = 0)) → a = -6 :=
  by
    sorry

end parallel_lines_coefficient_l6_6558


namespace find_angle_AOB_l6_6771

noncomputable def angle_AOB (triangle_PAB : Triangle) (angle_APB : ℝ) (tangents_to_circle : Triangle ∧ tangent T : Circle) (circumcircle_O : Circle) : Prop :=
  ∠AOB = 65

theorem find_angle_AOB (P A B : Point) (O : Circle) (tangent_lines : Tangent P A O ∧ Tangent P B O ∧ Tangent A B O) (angle_APB : ∠ APB = 50) : ∠ AOB = 65 :=
begin
  sorry
end

end find_angle_AOB_l6_6771


namespace max_min_f_sum_l6_6552

noncomputable def f (x : ℝ) : ℝ :=
  ((2 ^ x + 1) ^ 2) / (2 ^ x * x) + 1

theorem max_min_f_sum :
  let interval := set.Ico (-2018 : ℝ) 0 ∪ set.Ioc 0 2018 in
  let M := real.Sup (f '' interval) in
  let N := real.Inf (f '' interval) in
  M + N = 2 :=
sorry

end max_min_f_sum_l6_6552


namespace louisa_second_day_miles_l6_6286

theorem louisa_second_day_miles (T1 T2 : ℕ) (speed miles_first_day miles_second_day : ℕ)
  (h1 : speed = 25) 
  (h2 : miles_first_day = 100)
  (h3 : T1 = miles_first_day / speed) 
  (h4 : T2 = T1 + 3) 
  (h5 : miles_second_day = speed * T2) :
  miles_second_day = 175 := 
by
  -- We can add the necessary calculations here, but for now, sorry is used to skip the proof.
  sorry

end louisa_second_day_miles_l6_6286


namespace dice_probability_l6_6348

-- The context that there are three six-sided dice
def total_outcomes : ℕ := 6 * 6 * 6

-- Function to count the number of favorable outcomes where two dice sum to the third
def favorable_outcomes : ℕ :=
  let sum_cases := [1, 2, 3, 4, 5]
  sum_cases.sum
  -- sum_cases is [1, 2, 3, 4, 5] each mapping to the number of ways to form that sum with a third die

theorem dice_probability : 
  (favorable_outcomes * 3) / total_outcomes = 5 / 24 := 
by 
  -- to prove: the probability that the values on two dice sum to the value on the remaining die is 5/24
  sorry

end dice_probability_l6_6348


namespace area_of_section_of_prism_l6_6211

variables (a : ℝ) (α : ℝ)
-- To avoid division by zero, assume α is such that cos α ≠ 0 because cosine of 90 degrees is 0.
-- This is implicit in the wording because the angle is acute.
noncomputable def area_of_section : ℝ :=
  (3 * a^2 * real.sqrt 3) / (2 * real.cos α)

theorem area_of_section_of_prism (h₁ : 0 < a) (h₂ : 0 < α ∧ α < real.pi / 2) :
  area_of_section a α = (3 * a^2 * real.sqrt 3) / (2 * real.cos α) :=
by 
  sorry

end area_of_section_of_prism_l6_6211


namespace bus_travel_time_increased_speed_l6_6420

theorem bus_travel_time_increased_speed :
  let distance := 180
  let increased_speed_1 := 55
  let increased_speed_2 := 75
  let increased_speed_3 := 65
  let segment_distance := 60
  let time_1 := segment_distance / increased_speed_1
  let time_2 := segment_distance / increased_speed_2
  let time_3 := segment_distance / increased_speed_3
  let total_time := time_1 + time_2 + time_3
  abs(total_time - 14.07) < 0.01 :=
by
  sorry

end bus_travel_time_increased_speed_l6_6420


namespace find_a_range_l6_6088

noncomputable def f (x : ℝ) : ℝ := exp (x-1) + x - 2
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := x^2 - a * x - a + 3

def is_adj_zero_functions (f g : ℝ → ℝ) (alpha beta : ℝ) : Prop :=
  f alpha = 0 ∧ g beta = 0 ∧ abs (alpha - beta) ≤ 1

theorem find_a_range (a : ℝ) :
  (∃ beta, g beta a = 0 ∧ 0 ≤ beta ∧ beta ≤ 2) →
  (-a + 3 ≥ 0) →
  ((a / 2)^2 - (a * (a / 2) - a + 3) ≤ 0) →
  2 ≤ a ∧ a ≤ 3 :=
by
  sorry

end find_a_range_l6_6088


namespace Z_evaluation_l6_6988

def Z (x y : ℕ) : ℕ := x^2 - x * y + y^2

theorem Z_evaluation : Z 5 3 = 19 := by
  sorry

end Z_evaluation_l6_6988


namespace total_amount_is_105_l6_6392

theorem total_amount_is_105 (x_amount y_amount z_amount : ℝ) 
  (h1 : ∀ x, y_amount = x * 0.45) 
  (h2 : ∀ x, z_amount = x * 0.30) 
  (h3 : y_amount = 27) : 
  (x_amount + y_amount + z_amount = 105) := 
sorry

end total_amount_is_105_l6_6392


namespace parabola_proposition_l6_6322

theorem parabola_proposition :
  ∀ (p : ℝ), p > 0 → 
  (F := (1 : ℝ, 0 : ℝ)) →
  (parabola_focus := F) →
  (directrix := - (p / 2)) →
  (line_MN_length := 2 * p) →
  (tangent_check := ∀ (x y : ℝ), x = 1 → y = x + 1 → y = 2) →
  ¬ (directrix = -2) :=
by
  intros p hp F parabola_focus directrix line_MN_length tangent_check
  sorry

end parabola_proposition_l6_6322


namespace non_intersecting_chords_with_sum_l6_6357

theorem non_intersecting_chords_with_sum :
  ∃ (pairs : Finset (ℕ × ℕ)),
    (pairs.card = 1010) ∧
    (∀ (a b c d : ℕ), (a, b) ∈ pairs → (c, d) ∈ pairs → 
      (a = c ∨ a = d ∨ b = c ∨ b = d ∨ ¬segment_intersection a b c d)) ∧
    (∑ (p : ℕ × ℕ) in pairs, abs (p.2 - p.1) = 1010^2) :=
sorry

end non_intersecting_chords_with_sum_l6_6357


namespace angle_AOB_in_tangent_triangle_l6_6765

-- Statement of the problem
theorem angle_AOB_in_tangent_triangle (P A B O: Type) 
  [Is_Triangle PAB] [TangentToCircle P A B O] 
  (h_angle_APB : ∠ APB = 50°) :
  ∠ AOB = 130° :=
sorry

end angle_AOB_in_tangent_triangle_l6_6765


namespace angle_BPN_is_30deg_angle_NHC_is_60deg_l6_6260

-- Define the geometric properties and points
variables (A B C D S N H P : Type) 
variables [AffineSpace ℝ A] [AffineSpace ℝ B] [AffineSpace ℝ C] [AffineSpace ℝ D]
variables [AffineSpace ℝ S] [AffineSpace ℝ N] [AffineSpace ℝ H] [AffineSpace ℝ P]
variables [Square A B C D] [EquilateralTriangle B C S]
variables [Midpoint N A S] [Midpoint H C D] [Midpoint P B S]

-- State the problems as hypotheses
axiom Hypothesis1 : isSquare A B C D
axiom Hypothesis2 : isEquilateralTriangle B C S
axiom Hypothesis3 : isMidpoint N A S
axiom Hypothesis4 : isMidpoint H C D
axiom Hypothesis5 : isMidpoint P B S

-- Define the final angles
noncomputable def angleBPN : ℝ := 30
noncomputable def angleNHC : ℝ := 60

-- Statements to be proved
theorem angle_BPN_is_30deg : angle B P N = 30 :=
by
  sorry

theorem angle_NHC_is_60deg : angle N H C = 60 :=
by
  sorry

end angle_BPN_is_30deg_angle_NHC_is_60deg_l6_6260


namespace percentage_of_women_attended_picnic_l6_6172

variable (E : ℝ) -- total number of employees
variable (M : ℝ) -- number of men
variable (W : ℝ) -- number of women

-- 45% of all employees are men
axiom h1 : M = 0.45 * E
-- Rest of employees are women
axiom h2 : W = E - M
-- 20% of men attended the picnic
variable (x : ℝ) -- percentage of women who attended the picnic
axiom h3 : 0.20 * M + (x / 100) * W = 0.31000000000000007 * E

theorem percentage_of_women_attended_picnic : x = 40 :=
by
  sorry

end percentage_of_women_attended_picnic_l6_6172


namespace periodic_decimal_to_fraction_l6_6009

theorem periodic_decimal_to_fraction
  (h : ∀ n : ℕ, 0.<digit>02 n / 99) :
  2.0<digit>06 = 68 / 33 :=
sorry

end periodic_decimal_to_fraction_l6_6009


namespace perimeter_of_triangle_l6_6575

def distance (p q : ℝ × ℝ) : ℝ :=
  real.sqrt ((p.1 - q.1) ^ 2 + (p.2 - q.2) ^ 2)

def A : ℝ × ℝ := (3, 7)
def B : ℝ × ℝ := (-5, 2)
def C : ℝ × ℝ := (3, 2)

theorem perimeter_of_triangle : 
  distance A B + distance B C + distance A C = real.sqrt 89 + 13 :=
by
  sorry

end perimeter_of_triangle_l6_6575


namespace arithmetic_seq_ratio_l6_6140

theorem arithmetic_seq_ratio
  (a b : ℕ → ℝ)
  (S T : ℕ → ℝ)
  (H_seq_a : ∀ n, S n = (n * (a 1 + a n)) / 2)
  (H_seq_b : ∀ n, T n = (n * (b 1 + b n)) / 2)
  (H_ratio : ∀ n, S n / T n = (2 * n - 3) / (4 * n - 3)) :
  (a 3 + a 15) / (2 * (b 3 + b 9)) + a 3 / (b 2 + b 10) = 19 / 41 :=
by
  sorry

end arithmetic_seq_ratio_l6_6140


namespace greatest_value_of_x_for_equation_l6_6366

theorem greatest_value_of_x_for_equation :
  ∃ x : ℝ, (4 * x - 5) ≠ 0 ∧ ((5 * x - 20) / (4 * x - 5)) ^ 2 + ((5 * x - 20) / (4 * x - 5)) = 18 ∧ x = 50 / 29 :=
sorry

end greatest_value_of_x_for_equation_l6_6366


namespace bea_mom_planted_narra_trees_l6_6865

noncomputable def initialNarraTrees (N M total_fallen_trees n_fallen m_fallen T tp np mp) : 
  ℕ :=
  N

theorem bea_mom_planted_narra_trees (N M total_fallen_trees n_fallen m_fallen T tp np mp) :
  N = 30 :=
by
  have h1 : M = 50 := sorry
  have h2 : n_fallen + m_fallen = total_fallen_trees := sorry
  have h3 : m_fallen = n_fallen + 1 := sorry
  have h4 : total_fallen_trees = 5 := sorry
  have h5 : np = 2 * n_fallen := sorry
  have h6 : mp = 3 * m_fallen := sorry
  have h7 : tp + np + mp = T := sorry
  have h8 : T = 88 := sorry
  have h9 : tp = N + M - total_fallen_trees := sorry
  calc
    N + 58 = 88    : sorry
    N = 88 - 58    : sorry
    N = 30         : sorry

end bea_mom_planted_narra_trees_l6_6865


namespace trapezoid_base_count_l6_6695

theorem trapezoid_base_count (A h : ℕ) (multiple : ℕ) (bases_sum pairs_count : ℕ) : 
  A = 1800 ∧ h = 60 ∧ multiple = 10 ∧ pairs_count = 4 ∧ 
  bases_sum = (A / (1/2 * h)) / multiple → pairs_count > 3 := 
by 
  sorry

end trapezoid_base_count_l6_6695


namespace trajectory_of_B_is_ellipse_l6_6103

noncomputable def ellipse_trajectory (A C : ℝ × ℝ) (a b c : ℝ) : Prop :=
  (A = (-1, 0) ∧ C = (1, 0)) ∧ 
  (2 * b = a + c) →
  (∀ (B : ℝ × ℝ), B ≠ A ∧ B ≠ C → (x ≠ 2) ∧ (x ≠ -2) → 
  (∃ (x y : ℝ),
  ((x, y) = B) ∧
  (x^2 / 4 + y^2 / 3 = 1)))

theorem trajectory_of_B_is_ellipse (A C : ℝ × ℝ) (a b c : ℝ) :
  ellipse_trajectory A C a b c :=
begin
  sorry -- proof not required, statement only
end

end trajectory_of_B_is_ellipse_l6_6103


namespace arithmetic_seq_geom_seq_l6_6106

theorem arithmetic_seq_geom_seq {a : ℕ → ℝ} 
  (h1 : ∀ n, 0 < a n)
  (h2 : a 2 + a 3 + a 4 = 15)
  (h3 : (a 1 + 2) * (a 6 + 16) = (a 3 + 4) ^ 2) :
  a 10 = 19 :=
sorry

end arithmetic_seq_geom_seq_l6_6106


namespace decimal_to_fraction_l6_6014

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l6_6014


namespace range_of_a_l6_6323

-- Define the function g(x) = x^3 - 3ax - a
def g (a x : ℝ) : ℝ := x^3 - 3*a*x - a

-- Define the derivative of g(x) which is g'(x) = 3x^2 - 3a
def g' (a x : ℝ) : ℝ := 3*x^2 - 3*a

theorem range_of_a (a : ℝ) : g a 0 * g a 1 < 0 → 0 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l6_6323


namespace max_value_of_function_l6_6331

theorem max_value_of_function : ∀ x : ℝ, 
  (max_value : ℝ),
  y = 3 - cos(1/2 * x)
  ∃ y, y = max_value ∧ max_value = 4 :=
by
  sorry

end max_value_of_function_l6_6331


namespace plains_routes_count_l6_6196

-- Defining the total number of cities and the number of cities in each region
def total_cities : Nat := 100
def mountainous_cities : Nat := 30
def plains_cities : Nat := total_cities - mountainous_cities

-- Defining the number of routes established each year and over three years
def routes_per_year : Nat := 50
def total_routes : Nat := routes_per_year * 3

-- Defining the number of routes connecting pairs of mountainous cities
def mountainous_routes : Nat := 21

-- The statement to prove the number of routes connecting pairs of plains cities
theorem plains_routes_count :
  plains_cities = 70 →
  total_routes = 150 →
  mountainous_routes = 21 →
  3 * mountainous_cities - 2 * mountainous_routes = 48 →
  3 * plains_cities - 48 = 162 →
  81 = 81 := sorry

end plains_routes_count_l6_6196


namespace grab_packet_probability_l6_6610

theorem grab_packet_probability :
  let n := 4 in
  let k := 3 in
  let favorable_outcomes := 12 in
  let total_outcomes := 24 in
  (favorable_outcomes / total_outcomes) = (1/2) :=
by
  -- The proof steps would go here.
  sorry

end grab_packet_probability_l6_6610


namespace original_faculty_members_correct_l6_6662

noncomputable def original_faculty_members : ℝ := 282

theorem original_faculty_members_correct:
  ∃ F : ℝ, (0.6375 * F = 180) ∧ (F = original_faculty_members) :=
by
  sorry

end original_faculty_members_correct_l6_6662


namespace find_b_l6_6237

def a : ℝ := 3 * Real.sqrt 3
def c : ℝ := 2
def angle_B : ℝ := Real.pi * (5 / 6) -- 150 degrees in radians
noncomputable def cos_angle_B : ℝ := Real.cos angle_B

theorem find_b (h_a : a = 3 * Real.sqrt 3) (h_c : c = 2) (h_B : angle_B = Real.pi * (5 / 6)) : ∃ b : ℝ, b = 7 :=
by
  use 7
  sorry

end find_b_l6_6237


namespace coeff_x4_in_expansion_l6_6364

theorem coeff_x4_in_expansion (x : ℝ) (sqrt2 : ℝ) (h₁ : sqrt2 = real.sqrt 2) :
  let c := (70 : ℝ) * (3^4 : ℝ) * (sqrt2^4 : ℝ) in
  c = 22680 :=
by
  sorry

end coeff_x4_in_expansion_l6_6364


namespace find_a_l6_6136

theorem find_a (a : ℝ) : 
  (∃ x y : ℝ, y^2 = 4 * x ∧ ax - y + 1 = 0 ∧ (x, y) = (1, 0)) → a = -1 :=
by
  intro h
  cases h with x hx
  cases hx with y hy
  cases hy with hy1 hy2
  cases hy2 with hy3 hy3_eq
  -- Proof could follow based on the conditions, but we simply state sorry here
  sorry

end find_a_l6_6136


namespace problem_inequality_alpha_beta_l6_6113

variable {R : Type} [RealField R]

theorem problem_inequality_alpha_beta
  (n : ℕ)
  (α β : Fin n → R)
  (hα : ∑ i, (α i) ^ 2 = 1)
  (hβ : ∑ i, (β i) ^ 2 = 1)
  (hzero : ∑ i, (α i) * (β i) = 0) :
  (α 0) ^ 2 + (β 0) ^ 2 ≤ 1 := 
sorry

end problem_inequality_alpha_beta_l6_6113


namespace complex_in_second_quadrant_l6_6227

noncomputable def z : ℂ := Complex.ofReal (Real.cos 3) + Complex.I * Complex.ofReal (Real.sin 3)

theorem complex_in_second_quadrant : (Real.pi / 2 < 3) ∧ (3 < Real.pi) → 
  Real.sin 3 > 0 ∧ Real.cos 3 < 0 → 
  ∃ quadrant, quadrant = 2 :=
by
  intros h1 h2
  use 2
  sorry

end complex_in_second_quadrant_l6_6227


namespace arrange_in_order_l6_6119

noncomputable def a := (Real.sqrt 2 / 2) * (Real.sin (17 * Real.pi / 180) + Real.cos (17 * Real.pi / 180))
noncomputable def b := 2 * (Real.cos (13 * Real.pi / 180))^2 - 1
noncomputable def c := Real.sqrt 3 / 2

theorem arrange_in_order : c < a ∧ a < b := 
by
  sorry

end arrange_in_order_l6_6119


namespace cos_squared_sum_l6_6097

theorem cos_squared_sum {A B C : ℝ} 
  (h1 : sin A + sin B + sin C = 0)
  (h2 : cos A + cos B + cos C = 0) : 
  cos A ^ 2 + cos B ^ 2 + cos C ^ 2 = 3 / 2 := 
by 
  sorry

end cos_squared_sum_l6_6097


namespace plains_routes_count_l6_6174

theorem plains_routes_count (total_cities mountainous_cities plains_cities total_routes routes_mountainous_pairs: ℕ) :
  total_cities = 100 →
  mountainous_cities = 30 →
  plains_cities = 70 →
  total_routes = 150 →
  routes_mountainous_pairs = 21 →
  let endpoints_mountainous := mountainous_cities * 3 in
  let endpoints_mountainous_pairs := routes_mountainous_pairs * 2 in
  let endpoints_mountainous_plains := endpoints_mountainous - endpoints_mountainous_pairs in
  let endpoints_plains := plains_cities * 3 in
  let routes_mountainous_plains := endpoints_mountainous_plains in
  let endpoints_plains_pairs := endpoints_plains - routes_mountainous_plains in
  let routes_plains_pairs := endpoints_plains_pairs / 2 in
  routes_plains_pairs = 81 :=
by
  intros h1 h2 h3 h4 h5
  dsimp
  rw [h1, h2, h3, h4, h5]
  sorry

end plains_routes_count_l6_6174


namespace total_scenarios_count_most_probable_sum_l6_6640

-- Definitions for the given problem
def box := finset {1, 2}
def draw (a b c : box) := (a.to_list.product (b.to_list.product c.to_list)).map (λ ⟨x, ⟨y, z⟩⟩, (x, y, z))

-- Questions reformulated as statements to be proved
theorem total_scenarios_count (A B C : box) : ↑((draw A B C).to_finset.card) = 8 :=
sorry

theorem most_probable_sum (A B C : box) : 
  let sums := ((draw A B C).map (λ ⟨x, y, z⟩, x + y + z)) in 
  sums.to_finset.max' (by sorry) = 4 ∨ sums.to_finset.max' (by sorry) = 5 :=
sorry

end total_scenarios_count_most_probable_sum_l6_6640


namespace complex_expression_result_l6_6867

open Complex

def a : ℂ := 3 + complex.I * 2
def b : ℂ := 2 - complex.I * 3

theorem complex_expression_result : 3 * a + 4 * b + a ^ 2 + b ^ 2 = 35 - 6 * complex.I :=
by
  sorry

end complex_expression_result_l6_6867


namespace find_missing_number_l6_6165

theorem find_missing_number
  (x : ℝ)
  (h1 : (12 + x + y + 78 + 104) / 5 = 62)
  (h2 : (128 + 255 + 511 + 1023 + x) / 5 = 398.2) : 
  y = 42 :=
  sorry

end find_missing_number_l6_6165


namespace max_intersections_perpendiculars_l6_6626
open Classical

theorem max_intersections_perpendiculars {P : ℕ → ℝ × ℝ} {Q : ℕ → ℝ × ℝ}
  (hP4_line : ∀ i j : ℕ, i < 4 → j < 4 → (P i).1 * (P j).2 = (P j).1 * (P i).2)
  (hQ_not_line : ∀ Q_idx : ℕ, 3 ≤ Q_idx ∧ Q_idx ≤ 5 → (Q Q_idx).1 * (P 0).2 ≠ (P 0).1 * (Q Q_idx).2) :
  let num_perp_each_Q := 10,
      total_perpendiculars := 3 * num_perp_each_Q,
      max_intersections := (total_perpendiculars * (total_perpendiculars - 1) / 2) - 17 + 3
  in max_intersections = 421 :=
by
  sorry

end max_intersections_perpendiculars_l6_6626


namespace numerical_expression_as_sum_of_squares_l6_6301

theorem numerical_expression_as_sum_of_squares : 
  2 * (2009:ℕ)^2 + 2 * (2010:ℕ)^2 = (4019:ℕ)^2 + (1:ℕ)^2 := 
by
  sorry

end numerical_expression_as_sum_of_squares_l6_6301


namespace imaginary_part_of_z_l6_6121

variable (z : ℂ)

def abs_4_3i : ℂ := complex.abs (4 + 3 * complex.I)

noncomputable def z_expression := (3 - 4 * complex.I) * z = abs_4_3i

theorem imaginary_part_of_z : z_expression z → z.im = -4 / 5 :=
by
  intros h
  have : abs_4_3i = 5 := by sorry -- Original problem condition states |4+3i| = 5
  rw this at h
  -- Additional Lean-specific steps go here
  sorry

end imaginary_part_of_z_l6_6121


namespace intersection_eq_singleton_l6_6978

def M : set (ℝ × ℝ) := {p | p.1 + p.2 = 3}
def N : set (ℝ × ℝ) := {p | p.1 - p.2 = 5}
def intersection : set (ℝ × ℝ) := {p | p ∈ M ∧ p ∈ N}

theorem intersection_eq_singleton :
  intersection = { (4 : ℝ, -1 : ℝ) } :=
by
  sorry

end intersection_eq_singleton_l6_6978


namespace lights_on_top_layer_l6_6624

theorem lights_on_top_layer
  (x : ℕ)
  (H1 : x + 2 * x + 4 * x + 8 * x + 16 * x + 32 * x + 64 * x = 381) :
  x = 3 :=
  sorry

end lights_on_top_layer_l6_6624


namespace hypotenuse_length_l6_6616

theorem hypotenuse_length (x : ℝ) (c : ℝ) (h1 : 0 < x) (h2 : x < Real.pi / 2)
  (h_AD : AD = Real.tan x) (h_AE : AE = Real.cot x) :
  BC = 4 * Real.sin (2 * x) ^ 2 := sorry

end hypotenuse_length_l6_6616


namespace volume_ratio_midsection_l6_6860

-- Definitions for the proof problem
variables {A B C D E F G : Point}
variables (T : Tetrahedron A B C D)
variables (S : Triangle A B C)
variables (DE_is_midline : isMidline S DE)
variables (EFG : Plane E F G)
variables (BCD : Plane B C D)

-- Lean 4 statement of the proof problem
theorem volume_ratio_midsection (h : isMidline S DE)
  (h₂ : parallel EFG BCD) :
  volume (cutTetrahedron T EFG) / volume T = 1 / 8 :=
by
  sorry

end volume_ratio_midsection_l6_6860


namespace largest_possible_integer_in_list_l6_6432

theorem largest_possible_integer_in_list (L : List ℕ)
  (h_length : L.length = 5)
  (h_pos : ∀ x ∈ L, x > 0)
  (h_trice : ∃! x, L.count x = 3)
  (h_median : L.nth (L.length / 2) = some 12)
  (h_mean : L.sum / 5 = 15) :
  ∃ b, b ∈ L ∧ b = 38 := 
sorry

end largest_possible_integer_in_list_l6_6432


namespace unpainted_area_of_board_l6_6351

theorem unpainted_area_of_board 
  (width1 : ℝ) (width2 : ℝ) (angle : ℝ)
  (h1 : width1 = 5) (h2 : width2 = 8) (h3 : angle = π / 4)
  : (width1 * width2 * sin angle) = 20 * Real.sqrt 2 :=
sorry

end unpainted_area_of_board_l6_6351


namespace decimal_to_fraction_l6_6021

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l6_6021


namespace equation_of_E_HN_fixed_point_l6_6959

-- Condition definitions
def center_origin : Prop := true 
def axes_of_symmetry_origin : Prop := true
def passes_through_A : Prop := (0, -2) ∈ { (x, y) | x^2 / 3 + y^2 / 4 = 1 }
def passes_through_B : Prop := (3 / 2, -1) ∈ { (x, y) | x^2 / 3 + y^2 / 4 = 1 }

-- Main proofs
theorem equation_of_E :
  center_origin →
  axes_of_symmetry_origin →
  passes_through_A →
  passes_through_B →
  (∀ (x y : ℝ), (x, y) ∈ { (x, y) | x^2 / 3 + y^2 / 4 = 1 }) :=
by
  intros _ _ _ _
  sorry

-- Separate geometric conditions for part (2)
def point_P : (ℝ × ℝ) := (1, -2)
def intersection_with_E (x y : ℝ) : Prop := (x, y) ∈ { (x, y) | x^2 / 3 + y^2 / 4 = 1 }
def parallel_to_x_axis (y : ℝ) : ℝ → ℝ := λ x, y
def segment_AB (x : ℝ) : ℝ := (2/3) * x - 2
def point_H (Mx My Tx : ℝ) : (ℝ × ℝ) := (2 * Tx - Mx, My)

-- Fixed point theorem
theorem HN_fixed_point :
  ∀ (M N : ℝ × ℝ), 
  let HN_line (x1 y1 x2 y2 Tx : ℝ) := ∃ k b, y2 = k * x2 + b ∧ (0, -2) ∈ { (x, y) | y = k * x + b } in
  center_origin →
  axes_of_symmetry_origin →
  passes_through_A →
  passes_through_B →
  intersection_with_E M.1 M.2 →
  intersection_with_E N.1 N.2 →
  HN_line M.1 M.2 N.1 N.2 (M.1) :=
by
  intros M N _ _ _ _ _ _ _
  sorry

end equation_of_E_HN_fixed_point_l6_6959


namespace correct_proposition_is_1_l6_6577

-- Define basic objects: lines and planes.
axiom line : Type
axiom plane : Type

-- Define conditions and relationships.
axiom parallel_lines (l1 l2 : line) : Prop
axiom intersect_lines (l1 l2 : line) : Prop
axiom line_in_plane (l : line) (p : plane) : Prop
axiom planes_intersect_at_line (p1 p2 : plane) (l : line) : Prop
axiom parallel_planes (p1 p2 : plane) : Prop

-- Given conditions for four propositions:
axiom proposition_1 (m n : line) (α β : plane) : 
  planes_intersect_at_line α β m ∧ line_in_plane n α → (parallel_lines m n ∨ intersect_lines m n)

axiom proposition_2 (m n : line) (α β : plane) : 
  parallel_planes α β ∧ line_in_plane m α ∧ line_in_plane n β → parallel_lines m n

axiom proposition_3 (m n : line) (α : plane) : 
  parallel_lines m n ∧ parallel_lines m α → parallel_lines n α

axiom proposition_4 (m n : line) (α β : plane) : 
  planes_intersect_at_line α β m ∧ parallel_lines m n → parallel_lines n β ∧ parallel_lines n α

-- The main theorem to prove only Proposition ① is correct.
theorem correct_proposition_is_1 (m n : line) (α β : plane) : 
  proposition_1 m n α β ∧ 
  ¬ proposition_2 m n α β ∧ 
  ¬ proposition_3 m n α β ∧ 
  ¬ proposition_4 m n α β := 
  sorry

end correct_proposition_is_1_l6_6577


namespace g_9_l6_6723

variable (g : ℝ → ℝ)

-- Conditions
axiom func_eq : ∀ x y : ℝ, g(x + y) = g(x) * g(y)
axiom g_3 : g 3 = 4

-- Theorem to prove
theorem g_9 : g 9 = 64 :=
by
  sorry

end g_9_l6_6723


namespace x_coordinate_of_q_l6_6628

theorem x_coordinate_of_q : 
  ∃ (Q : ℝ × ℝ), Q.1 = - (7 * Real.sqrt 2) / 10 ∧ 
  Q.2 < 0 ∧ 
  (Q.1 ^ 2 + Q.2 ^ 2 = 1) ∧ 
  ∃ (P : ℝ × ℝ), P = (3 / 5, 4 / 5) ∧
  Real.angle (0, 0) P Q = 3 * Real.pi / 4 :=
by
  sorry

end x_coordinate_of_q_l6_6628


namespace rectangle_width_l6_6315

theorem rectangle_width (side_square : ℝ) (length_rectangle : ℝ) (area_square_eq_rectangle : side_square^2 = (length_rectangle * (side_square^2 / length_rectangle))) :
  (side_square^2 / length_rectangle) = 6 :=
by
  have square_side : side_square = 12 := sorry
  have rect_length : length_rectangle = 24 := sorry
  have area_equiv : side_square^2 = 144 := sorry
  have width := side_square^2 / length_rectangle
  have width_calc : width = 6 := by
    rw [area_equiv, <-rect_length]
    sorry
  exact width_calc

end rectangle_width_l6_6315


namespace problem_solution_l6_6114

theorem problem_solution (a b c : ℝ) 
  (h1 : 14^a = 2) 
  (h2 : 7^b = 2) 
  (h3 : 4^c = 2) : 
  (1 / a - 1 / b + 1 / c) = 3 := 
by
  sorry

end problem_solution_l6_6114


namespace who_broke_the_glass_l6_6440

-- Definitions of the individuals and their statements
def Andrei_statement : Prop := Viktor_broke_glass
def Viktor_statement : Prop := Sergei_broke_glass
def Sergei_statement : Prop := ¬ Viktor_statement
def Yuri_statement : Prop := ¬ Yuri_broke_glass

-- Prove that Yuri broke the glass given only one is telling the truth.
theorem who_broke_the_glass (Viktor_broke_glass Sergei_broke_glass Yuri_broke_glass : Prop) :
  (Andrei_statement = true ∨ Viktor_statement = true ∨ Sergei_statement = true ∨ Yuri_statement = true) ∧
  ((Andrei_statement ∧ ¬ Viktor_statement ∧ ¬ Sergei_statement ∧ ¬ Yuri_statement) ∨
   (¬ Andrei_statement ∧ Viktor_statement ∧ ¬ Sergei_statement ∧ ¬ Yuri_statement) ∨
   (¬ Andrei_statement ∧ ¬ Viktor_statement ∧ Sergei_statement ∧ ¬ Yuri_statement) ∨
   (¬ Andrei_statement ∧ ¬ Viktor_statement ∧ ¬ Sergei_statement ∧ Yuri_statement))
  -> Yuri_broke_glass := sorry

end who_broke_the_glass_l6_6440


namespace min_students_participating_l6_6464

def ratio_9th_to_10th (n9 n10 : ℕ) : Prop := n9 * 4 = n10 * 3
def ratio_10th_to_11th (n10 n11 : ℕ) : Prop := n10 * 6 = n11 * 5

theorem min_students_participating (n9 n10 n11 : ℕ) 
    (h1 : ratio_9th_to_10th n9 n10) 
    (h2 : ratio_10th_to_11th n10 n11) : 
    n9 + n10 + n11 = 59 :=
sorry

end min_students_participating_l6_6464


namespace equilateral_octagon_problem_l6_6257

theorem equilateral_octagon_problem :
  ∀ (m n : ℕ), 
  (m > 0) → (n > 0) → (Nat.gcd m n = 1) → 
  (∃ (H : 3 * (1 * 1) = 3),
  ∃ (s θ : ℝ), 
  (s * Real.sin θ = 1 / 2) ∧ 
  (s = 1 / (2 * Real.sin θ)) ∧ 
  (Real.sin(2 * θ) = ↑m / ↑n)) → 
  (100 * m + n = 405) := 
begin
  intros m n m_pos n_pos gcd_mn h,
  rcases h with ⟨_, _, ⟨s, θ, eq1, eq2, eq3⟩⟩,
  sorry
end

end equilateral_octagon_problem_l6_6257


namespace largest_real_constant_ineq_l6_6084

theorem largest_real_constant_ineq (n : ℕ) (h_n : n ≥ 2) (a : Fin n → ℝ) (h_a : ∀ i, 0 < a i) :
    (Finset.univ.sum (λ i => (a i) ^ 2) / n)  ≥ (Finset.univ.sum a / n) ^ 2 
    + (1 / (2 * n)) * (a 0 - a (n-1)) ^ 2 :=
sorry

end largest_real_constant_ineq_l6_6084


namespace eccentricity_of_ellipse_l6_6956

theorem eccentricity_of_ellipse (k : ℝ) (h_k : k > 0)
  (focus : ∃ (x : ℝ), (x, 0) = ⟨3, 0⟩) :
  ∃ e : ℝ, e = (Real.sqrt 3 / 2) := 
sorry

end eccentricity_of_ellipse_l6_6956


namespace optimal_kn_l6_6539

theorem optimal_kn (n : ℕ) (hn : 0 < n)
  (a : Fin 3n → ℝ)
  (h_ascending : ∀ i j, i ≤ j → a i ≤ a j)
  (h_nonneg : ∀ i, 0 ≤ a i) :
  (∑ i in Finset.range (3 * n), a i) ^ 3 ≥ 27 * n^2 * ∑ i in Finset.range n, a i * a (n + i) * a (2 * n + i) :=
by
  sorry

end optimal_kn_l6_6539


namespace intersection_on_circumcircle_l6_6104

-- Given a triangle ABC
variables {A B C : Point}

-- Definitions of angle bisector of \(\widehat{B}\)
def angle_bisector_B (A B C : Point) : Line := sorry

-- Definitions of perpendicular bisector of [AC]
def perpendicular_bisector_AC (A C : Point) : Line := sorry

-- Intersection point of angle bisector of \( \widehat{B} \) and perpendicular bisector of \([AC]\)
def intersection_point (A B C : Point) : Point := 
  have h1 : Line := angle_bisector_B A B C
  have h2 : Line := perpendicular_bisector_AC A C
  -- Assuming existence and uniqueness of intersection point
  sorry

-- Circumcircle of the triangle ABC
def circumcircle (A B C : Point) : Circle := sorry

-- Proof statement
theorem intersection_on_circumcircle
  (A B C : Point)
  (hD : intersection_point A B C = D)
  (hOnCirc : D ∈ circumcircle A B C) :
  D ∈ circumcircle A B C :=
sorry

end intersection_on_circumcircle_l6_6104


namespace find_angle_AOB_l6_6772

noncomputable def angle_AOB (triangle_PAB : Triangle) (angle_APB : ℝ) (tangents_to_circle : Triangle ∧ tangent T : Circle) (circumcircle_O : Circle) : Prop :=
  ∠AOB = 65

theorem find_angle_AOB (P A B : Point) (O : Circle) (tangent_lines : Tangent P A O ∧ Tangent P B O ∧ Tangent A B O) (angle_APB : ∠ APB = 50) : ∠ AOB = 65 :=
begin
  sorry
end

end find_angle_AOB_l6_6772


namespace plains_routes_l6_6207

theorem plains_routes 
  (total_cities : ℕ)
  (mountainous_cities : ℕ)
  (plains_cities : ℕ)
  (total_routes : ℕ)
  (mountainous_routes : ℕ)
  (num_pairs_with_mount_to_mount : ℕ)
  (routes_per_year : ℕ)
  (years : ℕ)
  (mountainous_roots_connections : ℕ)
  : (mountainous_cities = 30) →
    (plains_cities = 70) →
    (total_cities = mountainous_cities + plains_cities) →
    (routes_per_year = 50) →
    (years = 3) →
    (total_routes = routes_per_year * years) →
    (mountainous_routes = num_pairs_with_mount_to_mount * 2) →
    (num_pairs_with_mount_to_mount = 21) →
    let num_endpoints_per_city_route = 2 in
    let mountainous_city_endpoints = mountainous_cities * 3 in
    let mountainous_endpoints = mountainous_routes in
    let mountain_to_plains_endpoints = mountainous_city_endpoints - mountainous_endpoints in
    let total_endpoints = total_routes * num_endpoints_per_city_route in
    let plains_city_endpoints = plains_cities * 3 in
    let routes_between_plain_and_mountain = mountain_to_plains_endpoints in
    let plain_to_plain_endpoints = plains_city_endpoints - routes_between_plain_and_mountain in
    let plain_to_plain_routes = plain_to_plain_endpoints / 2 in
    plain_to_plain_routes = 81 :=
sorry

end plains_routes_l6_6207


namespace number_of_plains_routes_is_81_l6_6191

-- Define the number of cities in each region
def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70

-- Define the number of routes established over three years
def total_routes : ℕ := 150
def routes_per_year : ℕ := 50

-- Define the number of routes connecting pairs of mountainous cities
def mountainous_routes : ℕ := 21

-- Define a function to calculate the number of routes connecting pairs of plains cities
def plains_routes : ℕ :=
  let total_endpoints := total_routes * 2
  let mountainous_endpoints := mountainous_cities * 3
  let plains_endpoints := plains_cities * 3
  let mountainous_pair_endpoints := mountainous_routes * 2
  let mountain_plain_routes := (mountainous_endpoints - mountainous_pair_endpoints) / 2
  let plain_only_endpoints := plains_endpoints - mountain_plain_routes
  plain_only_endpoints / 2

theorem number_of_plains_routes_is_81 : plains_routes = 81 := 
  sorry

end number_of_plains_routes_is_81_l6_6191


namespace original_deck_size_l6_6467

/-- 
Aubrey adds 2 additional cards to a deck and then splits the deck evenly among herself and 
two other players, each player having 18 cards. 
We want to prove that the original number of cards in the deck was 52. 
-/
theorem original_deck_size :
  ∃ (n : ℕ), (n + 2) / 3 = 18 ∧ n = 52 :=
by
  sorry

end original_deck_size_l6_6467


namespace tan_periodic_mod_l6_6506

theorem tan_periodic_mod (m : ℤ) (h1 : -180 < m) (h2 : m < 180) : 
  (m : ℤ) = 10 := by
  sorry

end tan_periodic_mod_l6_6506


namespace trapezoid_collinear_l6_6296

structure Trapezoid (P Q R S : Type) :=
(midpoint_base1 : P)
(midpoint_base2 : Q)
(intersection_diagonals : R)
(intersection_extensions : S)

theorem trapezoid_collinear
  {P Q R S : Type}
  [midpoint1 : P] [midpoint2 : Q]
  [intersection_diag : R] [intersection_ext : S]
  [T : Trapezoid P Q R S] :
  ∃ N, collinear {T.intersection_extensions, T.intersection_diagonals, N} :=
sorry

end trapezoid_collinear_l6_6296


namespace problem_1_problem_2a_problem_2b_l6_6780

noncomputable def same_function_q1_r_s (r s : ℝ) : Prop :=
  y1 = 3 * x ^ 2 + r * x + 1 ∧ y2 = s * x + 1 ∧
  (sqrt(3 - s) + abs(r - s) = 0)

theorem problem_1 (r s : ℝ) : same_function_q1_r_s r s → r = 3 ∧ s = 3 :=
by
  sorry

noncomputable def bounded_c_condition (a b c : ℝ) : Prop :=
  ∀ x ∈ Icc(-1, 1), abs(a * x ^ 2 + b * x + c) ≤ 1

theorem problem_2a (a b c : ℝ) (h : bounded_c_condition a b c) : abs(c) ≤ 1 :=
by
  sorry

noncomputable def quadratic_expression_function (a : ℝ) : ℝ → ℝ :=
  if a > 0 then 2 * x ^ 2 - 1 else -2 * x ^ 2 + 1

noncomputable def same_function_q2_expression (a b c m n : ℝ) : Prop :=
  y1 = quadratic_expression_function a ∧ y2 = m * x + n ∧
  (m = a ∧ b = n) ∧
  (∀ x ∈ Icc(-1, 1), abs(a * x ^ 2 + b * x + c) ≤ 1)

theorem problem_2b (a b c : ℝ) (h : same_function_q2_expression a b c 0 0) : y1 = 2 * x ^ 2 - 1 ∨ y1 = -2 * x ^ 2 + 1 :=
by
  sorry

end problem_1_problem_2a_problem_2b_l6_6780


namespace t_f_8_eq_l6_6654

def t (x : ℝ) := Real.sqrt (5 * x + 2)
def f (x : ℝ) := 8 - t x

theorem t_f_8_eq : t (f 8) = Real.sqrt (42 - 5 * Real.sqrt 42) := by
  sorry

end t_f_8_eq_l6_6654


namespace transformed_data_sum_l6_6562

variables {x : ℕ → ℝ}
variables {mean_std_data : ℝ}

-- Given conditions
def data_points_mean := (∑ i in finset.range 8, x i) / 8 = 6
def data_points_std := real.sqrt ((∑ i in finset.range 8, (x i - 6) ^ 2) / 8) = 2

-- Definition of transformed data points
def y (i : ℕ) : ℝ := 3 * x i - 5

-- Required proof
theorem transformed_data_sum (x : ℕ → ℝ)
  (h_mean : (∑ i in finset.range 8, x i) / 8 = 6)
  (h_std : real.sqrt ((∑ i in finset.range 8, (x i - 6) ^ 2) / 8) = 2) :
  let a := (∑ i in finset.range 8, y i) / 8,
      b := (∑ i in finset.range 8, (y i - a) ^ 2) / 8
  in a + b = 49 :=
sorry

end transformed_data_sum_l6_6562


namespace correct_statements_l6_6087

noncomputable def f (x : ℝ) := 3 * Real.sin (2 * x + Real.pi / 6)

theorem correct_statements (x : ℝ) :
  (f(x - Real.pi / 6) = f(x)) ∧ ∀ x, f(x) ≤ 3 :=
by
  sorry

end correct_statements_l6_6087


namespace proof_f_g_3_l6_6600

def f (x : ℝ) := 4 - real.sqrt x
def g (x : ℝ) := 3 * x + 3 * x^2

theorem proof_f_g_3 :
  f (g (-3)) = 4 - 3 * real.sqrt 2 :=
by
  sorry

end proof_f_g_3_l6_6600


namespace pow_prime_quad_congruence_l6_6259

theorem pow_prime_quad_congruence {p a b : ℕ} (hp : Prime p) (h_sum_squares : p = a^2 + b^2) :
  ∃ x ∈ {a, -a, b, -b}, x ≡ (1/2) * (Nat.choose ((p-1)/2) ((p-1)/4)) [MOD p] :=
by
  sorry

end pow_prime_quad_congruence_l6_6259


namespace exists_irreducible_fractions_l6_6862

theorem exists_irreducible_fractions:
  ∃ (f : Fin 2018 → ℚ), 
    (∀ i j : Fin 2018, i ≠ j → (f i).den ≠ (f j).den) ∧ 
    (∀ i j : Fin 2018, i ≠ j → ∀ d : ℚ, d = f i - f j → d ≠ 0 → d.den < (f i).den ∧ d.den < (f j).den) :=
by
  -- proof is omitted
  sorry

end exists_irreducible_fractions_l6_6862


namespace polynomial_factorization_l6_6380

theorem polynomial_factorization :
  (x : ℤ[X]) →
  x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by
  intros x
  sorry

end polynomial_factorization_l6_6380


namespace fraction_goldfish_preference_l6_6861

theorem fraction_goldfish_preference
  (students_per_class : ℕ)
  (students_prefer_golfish_miss_johnson : ℕ)
  (students_prefer_golfish_ms_henderson : ℕ)
  (students_prefer_goldfish_total : ℕ)
  (miss_johnson_fraction : ℚ)
  (ms_henderson_fraction : ℚ)
  (total_students_prefer_goldfish_feldstein : ℕ)
  (feldstein_fraction : ℚ) :
  miss_johnson_fraction = 1/6 ∧
  ms_henderson_fraction = 1/5 ∧
  students_per_class = 30 ∧
  students_prefer_golfish_miss_johnson = miss_johnson_fraction * students_per_class ∧
  students_prefer_golfish_ms_henderson = ms_henderson_fraction * students_per_class ∧
  students_prefer_goldfish_total = 31 ∧
  students_prefer_goldfish_total = students_prefer_golfish_miss_johnson + students_prefer_golfish_ms_henderson + total_students_prefer_goldfish_feldstein ∧
  feldstein_fraction * students_per_class = total_students_prefer_goldfish_feldstein
  →
  feldstein_fraction = 2 / 3 :=
by 
  sorry

end fraction_goldfish_preference_l6_6861


namespace laura_owes_amount_l6_6246

def principal : ℝ := 35
def rate : ℝ := 0.05
def time : ℝ := 1
def interest (P R T : ℝ) := P * R * T
def totalAmountOwed (P I : ℝ) := P + I

theorem laura_owes_amount : totalAmountOwed principal (interest principal rate time) = 36.75 :=
by
  sorry

end laura_owes_amount_l6_6246


namespace max_product_of_sum_1000_l6_6574

theorem max_product_of_sum_1000 :
  ∃ (n : ℕ) (x : Fin n → ℕ), (∑ i, x i = 1000) ∧ (∀ i, 1 ≤ x i) ∧ 
  (∀ x', (∑ i, x' i = 1000) ∧ (∀ i, 1 ≤ x' i) → ∏ i, x i ≥ ∏ i, x' i) ∧ 
  (∏ i, x i = 2^2 * 3^332) :=
sorry

end max_product_of_sum_1000_l6_6574


namespace Evil_Mobile_speed_l6_6452

variable (y : ℝ) -- speed of the Evil-Mobile in miles per hour

-- Condition 1: Superhero's speed calculation
def superhero_speed : ℝ := (10 / 4) * 60

-- Condition 2: Superhero's speed is 50 miles per hour faster than the Evil-Mobile's speed
def condition : Prop := superhero_speed = y + 50

theorem Evil_Mobile_speed : y = 100 :=
  by
    -- Assume the condition as given
    assume h : superhero_speed = y + 50
    -- Solve for y
    sorry

end Evil_Mobile_speed_l6_6452


namespace m_eq_neg6_iff_parallel_l6_6143

def a : (ℝ × ℝ) := (-1, 2)
def b (m : ℝ) : (ℝ × ℝ) := (3, m)

theorem m_eq_neg6_iff_parallel (m : ℝ) : m = -6 ↔ ∃ k : ℝ, a = k • (prod.fst a + prod.fst (b m), prod.snd a + prod.snd (b m)) :=
by
  sorry

end m_eq_neg6_iff_parallel_l6_6143


namespace periodic_decimal_to_fraction_l6_6008

theorem periodic_decimal_to_fraction
  (h : ∀ n : ℕ, 0.<digit>02 n / 99) :
  2.0<digit>06 = 68 / 33 :=
sorry

end periodic_decimal_to_fraction_l6_6008


namespace length_more_than_breadth_l6_6328

theorem length_more_than_breadth
  (b x : ℝ)
  (h1 : b + x = 60)
  (h2 : 4 * b + 2 * x = 200) :
  x = 20 :=
by
  sorry

end length_more_than_breadth_l6_6328


namespace number_of_plains_routes_is_81_l6_6190

-- Define the number of cities in each region
def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70

-- Define the number of routes established over three years
def total_routes : ℕ := 150
def routes_per_year : ℕ := 50

-- Define the number of routes connecting pairs of mountainous cities
def mountainous_routes : ℕ := 21

-- Define a function to calculate the number of routes connecting pairs of plains cities
def plains_routes : ℕ :=
  let total_endpoints := total_routes * 2
  let mountainous_endpoints := mountainous_cities * 3
  let plains_endpoints := plains_cities * 3
  let mountainous_pair_endpoints := mountainous_routes * 2
  let mountain_plain_routes := (mountainous_endpoints - mountainous_pair_endpoints) / 2
  let plain_only_endpoints := plains_endpoints - mountain_plain_routes
  plain_only_endpoints / 2

theorem number_of_plains_routes_is_81 : plains_routes = 81 := 
  sorry

end number_of_plains_routes_is_81_l6_6190


namespace recurring_decimal_to_fraction_l6_6028

theorem recurring_decimal_to_fraction
  (h : 0.\overline{02} = (2 : ℝ) / 99) :
  2.\overline{06} = 68 / 33 := by
  sorry

end recurring_decimal_to_fraction_l6_6028


namespace largest_reciprocal_l6_6791

-- Definitions of the given numbers
def num1 := 1 / 6
def num2 := 2 / 7
def num3 := (2 : ℝ)
def num4 := (8 : ℝ)
def num5 := (1000 : ℝ)

-- The main problem: prove that the reciprocal of 1/6 is the largest
theorem largest_reciprocal :
  (1 / num1 > 1 / num2) ∧ (1 / num1 > 1 / num3) ∧ (1 / num1 > 1 / num4) ∧ (1 / num1 > 1 / num5) :=
by
  sorry

end largest_reciprocal_l6_6791


namespace find_valid_a_l6_6527

noncomputable def f (a x : ℝ) : ℝ :=
  if x >= 0 then abs (x + a) + abs (x - 2) else x^2 - a * x + 1/2 * a + 1

def has_min_value (a : ℝ) : Prop :=
  ∃ x : ℝ, (∀ y : ℝ, f a y ≥ f a x) ∧ f a x = 2 * a

theorem find_valid_a :
  {a : ℝ | has_min_value a} = {-sqrt 13 - 3} :=
sorry

end find_valid_a_l6_6527


namespace value_of_a_l6_6923

theorem value_of_a (a : ℕ) (h : ∀ x, ((a - 2) * x > a - 2) ↔ (x < 1)) : a = 0 ∨ a = 1 := by
  sorry

end value_of_a_l6_6923


namespace sum_of_common_divisors_l6_6906

-- Define the conditions
def is_divisor (a b : ℕ) : Prop := b % a = 0

def common_divisors (lst : List ℕ) : List ℕ :=
  List.filter (λ d, List.all lst (is_divisor d)) (List.range (List.minimum lst + 1))

-- Statement of the problem
theorem sum_of_common_divisors :
  let lst := [48, 96, 16, 144, 192]
  let cds := common_divisors lst
  List.sum cds = 15 := sorry

end sum_of_common_divisors_l6_6906


namespace eval_expression_l6_6871

theorem eval_expression : (2^0 : ℝ) + real.sqrt 9 - |(-4 : ℝ)| = 0 := 
by 
  sorry

end eval_expression_l6_6871


namespace max_plane_division_by_ellipses_l6_6529

theorem max_plane_division_by_ellipses (n : ℕ) (h : 0 < n) : 
  ∃ a_n : ℕ, a_n = 2 * n * (n - 1) + 2 :=
by
  use 2 * n * (n - 1) + 2
  sorry

end max_plane_division_by_ellipses_l6_6529


namespace beads_necklace_l6_6413

theorem beads_necklace :
  ∀ (total amethyst amber turquoise : ℕ),
  total = 40 →
  amethyst = 7 →
  amber = 2 * amethyst →
  turquoise = total - amethyst - amber →
  turquoise = 19 :=
by
  intros total amethyst amber turquoise h_total h_amethyst h_amber h_turquoise
  rw [h_total, h_amethyst, h_amber] at h_turquoise
  exact h_turquoise

end beads_necklace_l6_6413


namespace cone_volume_l6_6732

theorem cone_volume 
  (r l : ℝ)
  (h : ℝ)
  (hlateral_surface_sector_angle : ℝ := sqrt 3 * π) 
  (hlateral_surface_area : ℝ := 2 * sqrt 3 * π)
  (hcondition1 : (2 * π * r) / l = sqrt 3 * π)
  (hcondition2 : 1 / 2 * 2 * π * r * l = 2 * sqrt 3 * π)
  (hheight_condition : h = sqrt (l^2 - r^2)) :
  (1 / 3) * π * r^2 * h = π :=
  sorry

end cone_volume_l6_6732


namespace arithmetic_series_sum_l6_6516

theorem arithmetic_series_sum :
  let a1 := 16
  let an := 32
  let d := (1/3 : ℚ)
  let n := (((an - a1) * 3) + 1) -- corresponding to solving (an = a1 + (n-1)d)
  n = 49 :=
  let S := (n * (a1 + an)) / 2 -- corresponding to the sum formula S = n(a1 + an) / 2
  S = 1176 :=

begin
  -- Using the conditions to show each step
  let a1 := 16,
  let an := 32,
  let d := (1/3 : ℚ),
  let n := (((an - a1) * 3) + 1),
  have h_n : n = 49 := by simp [a1, an, d, n],
  let S := (n * (a1 + an)) / 2,
  have h_S : S = 1176 := by simp [a1, an, n, S],
  exact h_S
end

end arithmetic_series_sum_l6_6516


namespace euston_carriages_l6_6373

-- Definitions of the conditions
def E (N : ℕ) : ℕ := N + 20
def No : ℕ := 100
def FS : ℕ := No + 20
def total_carriages (E N : ℕ) : ℕ := E + N + No + FS

theorem euston_carriages (N : ℕ) (h : total_carriages (E N) N = 460) : E N = 130 :=
by
  -- Proof goes here
  sorry

end euston_carriages_l6_6373


namespace monotonicity_of_f_range_of_y_intercept_l6_6130

noncomputable def f (a x : ℝ) : ℝ := real.exp x * (x^2 - 2 * x + a)

theorem monotonicity_of_f (a : ℝ) :
  (a ≥ 2 → ∀ x y, x < y → f a x < f a y) ∧
  (a < 2 → ∀ x y, 
    ((x < -real.sqrt (2 - a) ∨ x > real.sqrt (2 - a)) ∧ (y < -real.sqrt (2 - a) ∨ y > real.sqrt (2 - a)) ∧ x < y → f a x < f a y) ∧
    (-real.sqrt (2 - a) < x ∧ x < real.sqrt (2 - a) ∧ -real.sqrt (2 - a) < y ∧ y < real.sqrt (2 - a) ∧ x < y → f a x > f a y)) :=
sorry

noncomputable def g (a : ℝ) : ℝ := real.exp a * (-a^3 + a)

theorem range_of_y_intercept :
  1 ≤ a ∧ a ≤ 3 → -24 * real.exp 3 ≤ g a ∧ g a ≤ 0 :=
sorry

end monotonicity_of_f_range_of_y_intercept_l6_6130


namespace probability_two_first_grade_pens_l6_6418

theorem probability_two_first_grade_pens :
  let total_pens := 6
  let first_grade_pens := 3
  let second_grade_pens := 2
  let third_grade_pens := 1
  let total_combinations := Nat.choose total_pens 2
  let first_grade_combinations := Nat.choose first_grade_pens 2
  (↑first_grade_combinations / ↑total_combinations) = (1 : ℚ / 5) := 
by
  unfold total_combinations
  unfold first_grade_combinations
  sorry

end probability_two_first_grade_pens_l6_6418


namespace unique_solution_quadratic_l6_6902

theorem unique_solution_quadratic (n : ℕ) : (∀ x : ℝ, 4 * x^2 + n * x + 4 = 0) → n = 8 :=
by
  intros h
  sorry

end unique_solution_quadratic_l6_6902


namespace average_letters_per_day_l6_6801

theorem average_letters_per_day (letters_tuesday : Nat) (letters_wednesday : Nat) (total_days : Nat) 
  (h_tuesday : letters_tuesday = 7) (h_wednesday : letters_wednesday = 3) (h_days : total_days = 2) : 
  (letters_tuesday + letters_wednesday) / total_days = 5 :=
by 
  sorry

end average_letters_per_day_l6_6801


namespace minimum_swaps_to_sort_l6_6249

theorem minimum_swaps_to_sort (n k : ℕ) (h : 1 ≤ k ∧ k ≤ n) : 
  ∃ m, m = n - Nat.gcd n k := 
begin
  use n - Nat.gcd n k,
  refl,
end

end minimum_swaps_to_sort_l6_6249


namespace minimum_magnitude_l6_6139

noncomputable def a (t : ℝ) : ℝ × ℝ × ℝ := (1 - t, 1 - t, t)
noncomputable def b (t : ℝ) : ℝ × ℝ × ℝ := (2, t, t)

noncomputable def magnitude {α : Type*} [inner_product_space ℝ α] (v : α) : ℝ :=
  ⟪v, v⟫^.real.sqrt

theorem minimum_magnitude : 
  (∃ t : ℝ, magnitude (b t - a t) = ∃ t : ℝ, ((1 + t) ^ 2 + (2 * t - 1) ^ 2 ).sqrt = (3 * (5^.sqrt)) / 5) :=
sorry

end minimum_magnitude_l6_6139


namespace beads_necklace_l6_6412

theorem beads_necklace :
  ∀ (total amethyst amber turquoise : ℕ),
  total = 40 →
  amethyst = 7 →
  amber = 2 * amethyst →
  turquoise = total - amethyst - amber →
  turquoise = 19 :=
by
  intros total amethyst amber turquoise h_total h_amethyst h_amber h_turquoise
  rw [h_total, h_amethyst, h_amber] at h_turquoise
  exact h_turquoise

end beads_necklace_l6_6412


namespace repeating_decimal_fraction_l6_6073

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l6_6073


namespace average_ratio_is_one_l6_6848

-- Defining the initial conditions
def correct_average (xs : List ℝ) : ℝ :=
  List.sum xs / (xs.length : ℝ)

def erroneous_average (xs : List ℝ) (avg : ℝ) : ℝ :=
  (List.sum xs + avg) / ((xs.length + 1) : ℝ)

-- Theorem statement
theorem average_ratio_is_one (xs : List ℝ) (len_pos : xs.length = 35) :
  let avg := correct_average xs in
  let err_avg := erroneous_average xs avg in
  avg / err_avg = 1 :=
by
  sorry

end average_ratio_is_one_l6_6848


namespace distance_between_points_l6_6869

-- Definition of distance between points in 2D space
def distance (p1 p2 : (ℝ × ℝ)) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

-- Points A and B given
def A : (ℝ × ℝ) := (2, 3)
def B : (ℝ × ℝ) := (5, 9)

-- Lean statement for the proof problem
theorem distance_between_points : distance A B = 3 * Real.sqrt 5 :=
by
  -- computation placeholders
  sorry

end distance_between_points_l6_6869


namespace one_even_iff_all_odd_or_at_least_two_even_l6_6778

theorem one_even_iff_all_odd_or_at_least_two_even (a b c : ℕ) :
  (∃ x : ℕ, (x = a ∨ x = b ∨ x = c) ∧ x % 2 = 0 ∧ (x ≠ a ∨ x ≠ b ∨ x ≠ c) ∧ a % 2 ≠ b % 2 ∧ b % 2 ≠ c % 2 ∧ a % 2 ≠ c % 2) ↔ 
  (∀ x : ℕ, (x = a ∧ x % 2 ≠ 0) ∧ (x = b ∧ x % 2 ≠ 0) ∧ (x = c ∧ x % 2 ≠ 0)) ∨ 
  (∃ x x' : ℕ, (x = a ∨ x = b ∨ x = c) ∧ (x' = a ∨ x' = b ∨ x' = c) ∧ x ≠ x' ∧ x % 2 = 0 ∧ x' % 2 = 0) → False :=
begin
  sorry
end

end one_even_iff_all_odd_or_at_least_two_even_l6_6778


namespace find_omega_phi_l6_6350

def f (x : ℝ) : ℝ := (sqrt 3) * sin (1/4 * x) * cos (1/4 * x) + (cos (1/4 * x))^2 - 1/2

def g (x : ℝ) (ω : ℝ) (φ : ℝ) : ℝ := sin ((1/2) * (ω * x) + (1/2) * φ + π/6)

theorem find_omega_phi :
  (0 < φ) ∧ (φ < π) ∧ (0 < ω) ∧ (even_function (g (ω := 4) (φ := 2 * π / 3))) ∧ (periodic (g (ω := 4) (φ := 2 * π / 3)) π) :=
by
  sorry

end find_omega_phi_l6_6350


namespace plains_routes_count_l6_6192

-- Defining the total number of cities and the number of cities in each region
def total_cities : Nat := 100
def mountainous_cities : Nat := 30
def plains_cities : Nat := total_cities - mountainous_cities

-- Defining the number of routes established each year and over three years
def routes_per_year : Nat := 50
def total_routes : Nat := routes_per_year * 3

-- Defining the number of routes connecting pairs of mountainous cities
def mountainous_routes : Nat := 21

-- The statement to prove the number of routes connecting pairs of plains cities
theorem plains_routes_count :
  plains_cities = 70 →
  total_routes = 150 →
  mountainous_routes = 21 →
  3 * mountainous_cities - 2 * mountainous_routes = 48 →
  3 * plains_cities - 48 = 162 →
  81 = 81 := sorry

end plains_routes_count_l6_6192


namespace tangent_plane_at_M_normal_line_at_M_l6_6477

-- Definitions of the mathematical problem
def F (x y z : ℝ) : ℝ := 4 * z - (x * y) + 2 * x + 4 * y - 8

-- Given conditions
def M : ℝ × ℝ × ℝ := (-4, -2, 8)

-- Tangent plane equation
def tangent_plane (x y z : ℝ) : Prop := x + 2 * y + z = 0

-- Normal line equations
def normal_line (x y z : ℝ) : Prop :=
  (x + 4 = (y + 2) / 2) ∧ ((y + 2) / 2 = z - 8)

-- Statement to prove the tangent plane equation
theorem tangent_plane_at_M :
  F M.1 M.2 M.3 = 0 →
  tangent_plane M.1 M.2 M.3 :=
sorry

-- Statement to prove the normal line equations
theorem normal_line_at_M :
  F M.1 M.2 M.3 = 0 →
  normal_line M.1 M.2 M.3 :=
sorry

end tangent_plane_at_M_normal_line_at_M_l6_6477


namespace sq_97_l6_6874

theorem sq_97 : 97^2 = 9409 :=
by
  sorry

end sq_97_l6_6874


namespace remainder_when_divided_by_11_l6_6488

theorem remainder_when_divided_by_11 :
  (7 * 10^20 + 2^20) % 11 = 8 := by
sorry

end remainder_when_divided_by_11_l6_6488


namespace concyclic_points_l6_6109

variables {Point : Type} [AffinePlane Point]
variables (A B C M P X Y : Point)
variables (h_isosceles : dist A B = dist A C) 
variables (h_midpoint : midpoint B C = M)
variables (h_parallel : parallel (line_through P A) (line_through B C))
variables (h_PB : between P B X) 
variables (h_PC : between P C Y)
variables (h_angles : ∠ P X M = ∠ P Y M)

theorem concyclic_points :
  concyclic A P X Y :=
sorry

end concyclic_points_l6_6109


namespace max_integer_solutions_l6_6443

theorem max_integer_solutions (p : ℤ[X]) (h₀ : p.coeffs ∈ (set.range (coe : ℤ → ℤ))) 
(h₁ : p.eval 50 = 50) : 
  ∃ k₁ k₂ k₃ k₄ k₅ k₆ : ℤ, 
    p.eval k₁ = k₁ ^ 2 ∧ 
    p.eval k₂ = k₂ ^ 2 ∧ 
    p.eval k₃ = k₃ ^ 2 ∧ 
    p.eval k₄ = k₄ ^ 2 ∧ 
    p.eval k₅ = k₅ ^ 2 ∧ 
    p.eval k₆ = k₆ ^ 2 ∧
    ((set.to_finset {k₁, k₂, k₃, k₄, k₅, k₆}).card ≤ 6) := 
sorry

end max_integer_solutions_l6_6443


namespace necessary_but_not_sufficient_for_purely_imaginary_l6_6319

-- Definitions
variables {a b : ℝ}
def complex (a b : ℝ) := a + b * complex.i

-- Conditions
def purely_imaginary (z : ℂ) : Prop := z.re = 0

-- Statement to prove
theorem necessary_but_not_sufficient_for_purely_imaginary :
  (∀ (a b : ℝ), ∀ (z : ℂ), z = complex a b → a = 0 → purely_imaginary z) ∧
  ¬ (∀ (a b : ℝ), ∀ (z : ℂ), z = complex a b → purely_imaginary z → a = 0 → (z = complex 0 b)) :=
sorry

end necessary_but_not_sufficient_for_purely_imaginary_l6_6319


namespace triangle_max_third_side_l6_6310

theorem triangle_max_third_side (A B C : ℝ) (a b : ℝ)
  (h1 : cos (3 * A) + cos (3 * B) + cos (3 * C) = 1)
  (h2 : a = 8) 
  (h3 : b = 15) 
  (h4 : C = π * 5 / 6) : 
  ∃ (c : ℝ), c = Real.sqrt (289 + 120 * Real.sqrt 3) :=
by
  sorry

end triangle_max_third_side_l6_6310


namespace turquoise_beads_count_l6_6415

-- Define the conditions
def num_beads_total : ℕ := 40
def num_amethyst : ℕ := 7
def num_amber : ℕ := 2 * num_amethyst

-- Define the main theorem to prove
theorem turquoise_beads_count :
  num_beads_total - (num_amethyst + num_amber) = 19 :=
by
  sorry

end turquoise_beads_count_l6_6415


namespace length_of_rectangle_l6_6739

noncomputable def side_of_square : ℝ :=
  28.27 / (Real.pi / 2 + 1)

def perimeter_square := 4 * side_of_square

def breadth_of_rectangle : ℝ := 16

def perimeter_rectangle (l : ℝ) : ℝ :=
  2 * l + 2 * breadth_of_rectangle

theorem length_of_rectangle :
  ∃ l : ℝ, perimeter_square = perimeter_rectangle l ∧ l = 6 :=
by
  use 6
  have h1 : 4 * side_of_square = perimeter_rectangle 6 := 
    calc
      4 * side_of_square = 4 * (28.27 / (Real.pi / 2 + 1)) : rfl
                      ... = 2 * 6 + 2 * 16 : sorry -- skipped detailed proof steps involving numerical values
  have h2 : 6 = 6 := rfl
  exact ⟨h1, h2⟩

end length_of_rectangle_l6_6739


namespace mean_home_runs_correct_l6_6727

def mean_home_runs (players: List ℕ) (home_runs: List ℕ) : ℚ :=
  let total_runs := (List.zipWith (· * ·) players home_runs).sum
  let total_players := players.sum
  total_runs / total_players

theorem mean_home_runs_correct :
  mean_home_runs [6, 4, 3, 1, 1, 1] [6, 7, 8, 10, 11, 12] = 121 / 16 :=
by
  -- The proof should go here
  sorry

end mean_home_runs_correct_l6_6727


namespace newspaper_target_l6_6683

theorem newspaper_target (total_collected_2_weeks : Nat) (needed_more : Nat) (sections : Nat) (kilos_per_section_2_weeks : Nat)
  (h1 : sections = 6)
  (h2 : kilos_per_section_2_weeks = 280)
  (h3 : total_collected_2_weeks = sections * kilos_per_section_2_weeks)
  (h4 : needed_more = 320)
  : total_collected_2_weeks + needed_more = 2000 :=
by
  sorry

end newspaper_target_l6_6683


namespace quadratic_function_points_relationship_l6_6665

theorem quadratic_function_points_relationship (c y1 y2 y3 : ℝ) 
  (h₁ : y1 = -((-1) ^ 2) + 2 * (-1) + c)
  (h₂ : y2 = -(2 ^ 2) + 2 * 2 + c)
  (h₃ : y3 = -(5 ^ 2) + 2 * 5 + c) :
  y2 > y1 ∧ y1 > y3 :=
by
  sorry

end quadratic_function_points_relationship_l6_6665


namespace smallest_prime_after_seven_nonprimes_l6_6080

/-- 
Prove that the smallest prime number following a sequence 
of seven consecutive nonprime positive integers is 97.
-/
theorem smallest_prime_after_seven_nonprimes : ∃ p : ℕ, prime p ∧ 
  ∀ (a b c d e f g : ℕ), 
    ¬prime a ∧ ¬prime b ∧ ¬prime c ∧ ¬prime d ∧ ¬prime e ∧ ¬prime f ∧ ¬prime g ∧ 
    a + 1 = b ∧ b + 1 = c ∧ c + 1 = d ∧ d + 1 = e ∧ e + 1 = f ∧ f + 1 = g ∧ g + 1 = p ∧ 
    (∀ q : ℕ, prime q → (∃ (u v w x y z : ℕ), 
      ¬prime u ∧ ¬prime v ∧ ¬prime w ∧ ¬prime x ∧ ¬prime y ∧ ¬prime z ∧ 
      u + 1 = v ∧ v + 1 = w ∧ w + 1 = x ∧ x + 1 = y ∧ y + 1 = z ∧ z + 1 = q) →
      p ≤ q) := 
  sorry

end smallest_prime_after_seven_nonprimes_l6_6080


namespace remainder_of_large_number_l6_6513

theorem remainder_of_large_number :
  (102938475610 % 12) = 10 :=
by
  have h1 : (102938475610 % 4) = 2 := sorry
  have h2 : (102938475610 % 3) = 1 := sorry
  sorry

end remainder_of_large_number_l6_6513


namespace a_leq_neg1_iff_inequality_holds_l6_6973

-- Define the function f(x) = x^2 - 4x - 1
def f (x : ℝ) : ℝ := x^2 - 4 * x - 1

-- Prove that if x is in the interval [1, 4], then x^2 - 4x - a - 1 ≥ 0 is true if and only if a ≤ -1
theorem a_leq_neg1_iff_inequality_holds (a : ℝ) :
  (∀ x ∈ set.Icc 1 4, f x - a ≥ 0) ↔ a ≤ -1 :=
sorry

end a_leq_neg1_iff_inequality_holds_l6_6973


namespace quadratic_function_points_relationship_l6_6666

theorem quadratic_function_points_relationship (c y1 y2 y3 : ℝ) 
  (h₁ : y1 = -((-1) ^ 2) + 2 * (-1) + c)
  (h₂ : y2 = -(2 ^ 2) + 2 * 2 + c)
  (h₃ : y3 = -(5 ^ 2) + 2 * 5 + c) :
  y2 > y1 ∧ y1 > y3 :=
by
  sorry

end quadratic_function_points_relationship_l6_6666


namespace complement_example_l6_6979

open Set

variable (U : Set ℝ) (M : Set ℝ)

def universal_set := {x : ℝ | True}
def set_M := {x : ℝ | real.log x / real.log 2 > 0}

theorem complement_example : 
  U = universal_set ∧ M = set_M →
  (complement U M) = {x : ℝ | x ≤ 1} :=
by
  intros hU hM
  sorry

end complement_example_l6_6979


namespace domain_eq_monotonic_intervals_eq_symmetry_center_eq_l6_6895

noncomputable def tan_func (x : ℝ) := real.tan (x / 2 - real.pi / 3)

def domain_tan_func (x : ℝ) : Prop :=
∀ (k : ℤ), x ≠ 2 * k * real.pi + 5 * real.pi / 3

def monotonic_intervals_tan_func (x : ℝ) : Prop :=
∃ (k : ℤ), 2 * k * real.pi - real.pi / 3 < x ∧ x < 2 * k * real.pi + 5 * real.pi / 3

def symmetry_center_tan_func (x y : ℝ) : Prop :=
∃ (k : ℤ), x = k * real.pi + 2 * real.pi / 3 ∧ y = 0

theorem domain_eq :
  (domain_tan_func x) ↔ (∀ k : ℤ, x ≠ 2 * k * real.pi + 5 * real.pi / 3) :=
by sorry

theorem monotonic_intervals_eq :
  (monotonic_intervals_tan_func x) ↔ (∃ k : ℤ, 2 * k * real.pi - real.pi / 3 < x ∧ x < 2 * k * real.pi + 5 * real.pi / 3) :=
by sorry

theorem symmetry_center_eq :
  (symmetry_center_tan_func x y) ↔ (∃ k : ℤ, x = k * real.pi + 2 * real.pi / 3 ∧ y = 0) :=
by sorry

end domain_eq_monotonic_intervals_eq_symmetry_center_eq_l6_6895


namespace domain_union_l6_6705

theorem domain_union (x : ℝ) : 
  (x ≥ 2 ∨ x < 1) ↔ (x ∈ (-∞:ℝ, 1) ∪ [2, +∞)) :=
by
  sorry

end domain_union_l6_6705


namespace smallest_positive_solution_l6_6904

theorem smallest_positive_solution (x : ℝ) (h : tan (4 * x) + tan (5 * x) = cot (4 * x)) : 
  x = π / 18 :=
sorry

end smallest_positive_solution_l6_6904


namespace true_statements_l6_6283

theorem true_statements (b x y : ℝ) (h1 : b * (x + y) = b * x + b * y)
                        (h4 : (log x) / (log b) = log b x)
                        (h5 : b * (x / y) = (b * x) / (b * y)) :
                        (b * (x + y) = b * x + b * y) ∧
                        ((log x) / (log b) = log b x) ∧
                        (b * (x / y) = (b * x) / (b * y)) ∧
                        (¬ (b ^ (x + y) = b ^ x + b ^ y)) ∧
                        (¬ (log b (x + y) = log b x + log b y)) :=
by 
  sorry

end true_statements_l6_6283


namespace problem_solution_l6_6567

theorem problem_solution (ω : ℝ) 
  (hω_pos : ω > 0) 
  (h_monotonic : ∀ x y, (π / 6 < x ∧ x < π / 2) → (π / 6 < y ∧ y < π / 2) → x ≤ y → f x ≤ f y)
  (h_neg : f (π / 3) < 0) :
  (∀ x, f ((x - π) / (6 * ω)) = sin (x / 6)) ∧
  (∀ x g, g x = sin (ω * x + π / 3) → g (x + π / 12) = f x) ∧
  (∀ n : ℤ, ω ≠ n) ∧
  (∃ x1 x2, 0 < x1 ∧ x1 < π ∧ 0 < x2 ∧ x2 < π ∧ f x1 = 0 ∧ f x2 = 0 ∧ x1 ≠ x2) :=
sorry

end problem_solution_l6_6567


namespace Sarah_finished_problems_l6_6678

open Nat

theorem Sarah_finished_problems
  (total_problems : Nat)
  (pages_left : Nat)
  (problems_per_page : Nat)
  (remaining_problems : Nat)
  (finished_problems : Nat)
  (h1 : total_problems = 60)
  (h2 : pages_left = 5)
  (h3 : problems_per_page = 8)
  (h4 : remaining_problems = pages_left * problems_per_page)
  (h5 : finished_problems = total_problems - remaining_problems) :
  finished_problems = 20 := 
by 
  simp [h1, h2, h3, h4, h5]
  sorry

end Sarah_finished_problems_l6_6678


namespace seating_arrangement_count_l6_6754

-- We define the constants representing the 5 students and 2 teachers.
constant students : ℕ := 5
constant teachers : ℕ := 2

-- Defining conditions
def teachers_cannot_be_at_ends (arrangement : List ℕ) : Prop :=
  arrangement.head ≠ 6 ∧ arrangement.head ≠ 7 ∧ arrangement.last ≠ 6 ∧ arrangement.last ≠ 7

def teachers_must_be_together (arrangement : List ℕ) : Prop :=
  ∃ i, arrangement[i] = 6 ∧ arrangement[i + 1] = 7 ∨ arrangement[i] = 7 ∧ arrangement[i + 1] = 6

-- Main statement to prove the number of arrangements given the conditions
theorem seating_arrangement_count : 
  ∃ (arrangements : List (List ℕ)), 
    arrangements.length = 960 ∧ 
    ∀ a ∈ arrangements, teachers_cannot_be_at_ends a ∧ teachers_must_be_together a :=
sorry

end seating_arrangement_count_l6_6754


namespace trigonometric_identity_l6_6120

noncomputable def alpha := -35 / 6 * Real.pi

theorem trigonometric_identity :
  (2 * Real.sin (Real.pi + alpha) * Real.cos (Real.pi - alpha)
    - Real.sin (3 * Real.pi / 2 + alpha)) /
  (1 + Real.sin (alpha) ^ 2 - Real.cos (Real.pi / 2 + alpha)
    - Real.cos (Real.pi + alpha) ^ 2) = -Real.sqrt 3 := by
  sorry

end trigonometric_identity_l6_6120


namespace probability_no_adjacent_same_color_correct_l6_6343

-- Define conditions
def num_red_balls : Nat := 6
def num_white_balls : Nat := 3
def num_yellow_balls : Nat := 3
def total_balls : Nat := num_red_balls + num_white_balls + num_yellow_balls

-- Define function to calculate multinomial coefficient
noncomputable def multinomial (n : Nat) (ks : List Nat) : Nat :=
  ks.foldl Nat.mul 1 • (Nat.factorial n) / List.foldl (•) 1 (ks.map Nat.factorial)

-- Define the probability calculation function
noncomputable def probability_no_adjacent_same_color : ℚ :=
  let total_arrangements := multinomial total_balls [num_red_balls, num_white_balls, num_yellow_balls]
  let valid_arrangements : Nat := 350 -- Calculated from the provided valid ways in solution
  valid_arrangements / total_arrangements

-- State the theorem to be proven
theorem probability_no_adjacent_same_color_correct :
  probability_no_adjacent_same_color = 5 / 924 :=
by
  -- Proof is omitted, provide the statement only as instructed
  sorry

end probability_no_adjacent_same_color_correct_l6_6343


namespace vertical_increase_is_100m_l6_6837

theorem vertical_increase_is_100m 
  (a b x : ℝ)
  (hypotenuse : a = 100 * Real.sqrt 5)
  (slope_ratio : b = 2 * x)
  (pythagorean_thm : x^2 + b^2 = a^2) : 
  x = 100 :=
by
  sorry

end vertical_increase_is_100m_l6_6837


namespace find_foot_of_perpendicular_l6_6538

def is_foot_of_perpendicular (P Q : ℝ × ℝ × ℝ) : Prop :=
  ∃ x y, P = (x, y, P.2.2) ∧ Q = (x, y, 0)

theorem find_foot_of_perpendicular 
  (P : ℝ × ℝ × ℝ) (hP : P = (1, real.sqrt 2, real.sqrt 3)) :
  ∃ Q : ℝ × ℝ × ℝ, is_foot_of_perpendicular P Q ∧ Q = (1, real.sqrt 2, 0) :=
by 
  sorry

end find_foot_of_perpendicular_l6_6538


namespace coeff_x4_in_expansion_l6_6363

theorem coeff_x4_in_expansion (x : ℝ) (sqrt2 : ℝ) (h₁ : sqrt2 = real.sqrt 2) :
  let c := (70 : ℝ) * (3^4 : ℝ) * (sqrt2^4 : ℝ) in
  c = 22680 :=
by
  sorry

end coeff_x4_in_expansion_l6_6363


namespace find_k_is_34_l6_6311

noncomputable def solve_k (a b k : ℝ) : Prop :=
  a = k * b ∧ k > 1 ∧ b > 0 ∧ (a + b) / 2 = 3 * (real.sqrt (a * b)) 

theorem find_k_is_34 (k a b : ℝ) (h : solve_k a b k) : k = 34 :=
by {
  sorry
}

end find_k_is_34_l6_6311


namespace decimal_to_fraction_l6_6022

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l6_6022


namespace polynomial_P_has_no_integer_root_polynomial_P_has_root_mod_n_l6_6635

noncomputable def polynomial_P : Polynomial ℤ :=
    (Polynomial.C 1) * (Polynomial.X ^ 6 + Polynomial.C 1) *
    (Polynomial.X ^ 2 + Polynomial.C 1) *
    (Polynomial.X ^ 2 + Polynomial.C 2) *
    (Polynomial.X ^ 2 - Polynomial.C 2)

theorem polynomial_P_has_no_integer_root
    (x : ℤ)
    (hx : polynomial_P.eval x = 0) : false :=
begin
    sorry
end

theorem polynomial_P_has_root_mod_n (n : ℕ) (hn : n > 0) :
    ∃ x : ℤ, polynomial_P.eval x % n = 0 :=
begin
    sorry
end

end polynomial_P_has_no_integer_root_polynomial_P_has_root_mod_n_l6_6635


namespace probability_three_members_three_days_l6_6430

theorem probability_three_members_three_days :
  let total_outcomes := 7^3
  let favorable_outcomes := 7 * 6 * 5
  let probability := favorable_outcomes.to_rat / total_outcomes.to_rat
  probability = 30 / 49 :=
by
  sorry

end probability_three_members_three_days_l6_6430


namespace solutions_to_equation_l6_6498

theorem solutions_to_equation :
  ∀ x : ℝ, 
  sqrt ((3 + sqrt 5) ^ x) + sqrt ((3 - sqrt 5) ^ x) = 6 ↔ (x = 2 ∨ x = -2) :=
by
  intros x
  sorry

end solutions_to_equation_l6_6498


namespace number_of_math_fun_books_l6_6762

def intelligence_challenge_cost := 18
def math_fun_cost := 8
def total_spent := 92

theorem number_of_math_fun_books (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ 5) (hy : intelligence_challenge_cost * x + math_fun_cost * y = total_spent) : y = 7 := 
by
  sorry

end number_of_math_fun_books_l6_6762


namespace third_defective_on_fourth_test_l6_6344

theorem third_defective_on_fourth_test :
  let num_test_methods := 288
  in ∃ (n_defective n_non_defective n_products : ℕ), n_products = 7 ∧ n_defective = 4 ∧ n_non_defective = 3 ∧
  num_test_methods = 288 :=
begin
  use [4, 3, 7],
  split,
  { refl },
  split,
  { refl },
  split,
  { refl },
  { refl }
end

end third_defective_on_fourth_test_l6_6344


namespace probability_reaching_13_11_l6_6233

noncomputable def probability_Ma_Long_serves_score : ℚ := 2 / 3
noncomputable def probability_Fan_Zhendong_serves_score : ℚ := 1 / 2

theorem probability_reaching_13_11 :
  let P1 := probability_Ma_Long_serves_score * (1 - probability_Fan_Zhendong_serves_score) * probability_Ma_Long_serves_score * (1 - probability_Fan_Zhendong_serves_score) in
  let P2 := (1 - probability_Ma_Long_serves_score) * probability_Fan_Zhendong_serves_score * probability_Ma_Long_serves_score * (1 - probability_Fan_Zhendong_serves_score) in
  let P3 := (1 - probability_Ma_Long_serves_score) * (1 - probability_Fan_Zhendong_serves_score) * (1 - probability_Ma_Long_serves_score) * probability_Fan_Zhendong_serves_score in
  let P4 := probability_Ma_Long_serves_score * (1 - probability_Fan_Zhendong_serves_score) * (1 - probability_Ma_Long_serves_score) * probability_Fan_Zhendong_serves_score in
  P1 + P2 + P3 + P4 = 1 / 4 :=
by
  sorry

end probability_reaching_13_11_l6_6233


namespace find_m_and_period_find_intervals_of_increase_l6_6566

noncomputable def f (x m : ℝ) : ℝ := 
  m * Real.sin (2 * x) - (Real.cos x) ^ 2 - 1/2

variables (α : ℝ)

-- Conditions
axiom tan_alpha : Real.tan α = 2 * Real.sqrt 3
axiom f_alpha : f α m = -3 / 26

-- Solve for m and the smallest period
theorem find_m_and_period : 
  (∃ m : ℝ, f α m = -3 / 26) → 
  (∃ T : ℝ, T = π ∧ (∀ x : ℝ, f (x + T) m = f x m)) := 
sorry

-- Finding intervals of increase
theorem find_intervals_of_increase (m : ℝ) (H_m : m = Real.sqrt 3 / 2) :
  (∀ x y : ℝ, 0 ≤ x ∧ x ≤ y ∧ y ≤ π → f x m ≤ f y m) :=
sorry

end find_m_and_period_find_intervals_of_increase_l6_6566


namespace hyperbola_C2_equation_constant_ratio_kAM_kBN_range_of_w_kAM_kBN_l6_6355

-- Definitions for conditions of the problem
def ellipse_C1 (x y : ℝ) (b : ℝ) : Prop := (x^2) / 4 + (y^2) / (b^2) = 1

def is_sister_conic_section (e1 e2 : ℝ) : Prop :=
  e1 * e2 = Real.sqrt 15 / 4

def hyperbola_C2 (x y : ℝ) : Prop := (x^2) / 4 - y^2 = 1

variable {b : ℝ} (hb : 0 < b ∧ b < 2)
variable {e1 e2 : ℝ} (heccentricities : is_sister_conic_section e1 e2)

theorem hyperbola_C2_equation :
  ∃ (x y : ℝ), ellipse_C1 x y b → hyperbola_C2 x y := sorry

theorem constant_ratio_kAM_kBN (G : ℝ × ℝ) :
  G = (4,0) → 
  ∀ (M N : ℝ × ℝ) (kAM kBN : ℝ), 
  (kAM / kBN = -1/3) := sorry

theorem range_of_w_kAM_kBN (kAM kBN : ℝ) :
  ∃ (w : ℝ),
  w = kAM^2 + (2 / 3) * kBN →
  (w ∈ Set.Icc (-3 / 4) (-11 / 36) ∪ Set.Icc (13 / 36) (5 / 4)) := sorry

end hyperbola_C2_equation_constant_ratio_kAM_kBN_range_of_w_kAM_kBN_l6_6355


namespace distance_between_places_l6_6421

theorem distance_between_places
  (d : ℝ) -- let d be the distance between A and B
  (v : ℝ) -- let v be the original speed
  (h1 : v * 4 = d) -- initially, speed * time = distance
  (h2 : (v + 20) * 3 = d) -- after speed increase, speed * new time = distance
  : d = 240 :=
sorry

end distance_between_places_l6_6421


namespace angle_MEN_right_angle_MN_length_is_25_l6_6809

variables (ABC : Triangle) (H : Point) (M N : Point)
variables (AB HC : ℝ)

-- Given conditions
axiom ABC_acute : IsAcuteTriangle ABC
axiom H_orthocenter : IsOrthocenter H ABC
axiom AB_length : AB = 48
axiom HC_length : HC = 14
axiom M_mid_AB : IsMidpoint M (side AB)
axiom N_mid_HC : IsMidpoint N (segment HC)

-- Part (a) Prove the angle between M and N is a right angle
theorem angle_MEN_right_angle : 
  ∠ (line M E N) = 90 := sorry

-- Part (b) Prove the length of the segment MN is 25
theorem MN_length_is_25 : 
  Length (segment M N) = 25 := sorry

end angle_MEN_right_angle_MN_length_is_25_l6_6809


namespace sum_of_arithmetic_sequence_l6_6368

-- Define the conditions
def is_arithmetic_sequence (first_term last_term : ℕ) (terms : ℕ) : Prop :=
  ∃ (a l : ℕ) (n : ℕ), a = first_term ∧ l = last_term ∧ n = terms ∧ n > 1

-- State the theorem
theorem sum_of_arithmetic_sequence (a l n : ℕ) (h_arith: is_arithmetic_sequence 5 41 10):
  n = 10 ∧ a = 5 ∧ l = 41 → (n * (a + l) / 2) = 230 :=
by
  intros h
  sorry

end sum_of_arithmetic_sequence_l6_6368


namespace repeating_decimal_as_fraction_l6_6037

theorem repeating_decimal_as_fraction : (0.\overline{02} = 2 / 99) → (2.\overline{06} = 68 / 33) :=
by
  sorry

end repeating_decimal_as_fraction_l6_6037


namespace volume_comparison_l6_6144

def volume_of_tetrahedron (a : ℝ) : ℝ :=
  -- Formula for the volume of a regular tetrahedron
  (a ^ 3 / (6 * sqrt 2))

def volume_of_octahedron (a : ℝ) : ℝ :=
  -- Formula for the volume of a regular octahedron
  (sqrt 2 * a ^ 3 / 3)

theorem volume_comparison (a : ℝ) (ha : a > 0) :
  volume_of_octahedron a = 4 * volume_of_tetrahedron a :=
by
  -- Proof goes here
  sorry

end volume_comparison_l6_6144


namespace polynomial_factorization_l6_6389

theorem polynomial_factorization :
  ∀ x : ℤ, x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by sorry

end polynomial_factorization_l6_6389


namespace fibonacci_factorial_sum_last_two_digits_l6_6870

theorem fibonacci_factorial_sum_last_two_digits :
  ((1! % 100) + (1! % 100) + (2! % 100) + (3! % 100) + (5! % 100) + (8! % 100) + (13! % 100) + (21! % 100)) % 100 = 50 := 
by
  sorry

end fibonacci_factorial_sum_last_two_digits_l6_6870


namespace max_value_f_l6_6643

theorem max_value_f (r : Fin n.succ → ℚ) (n : ℕ) (hpos : ∀ i, 0 < r i) (hsum : (Finset.finRange n.succ).sum r = 1) :
  (∃ ε > 0, ∀ n, f n ≤ n - (n.succ : ℕ) + ε ∧ ε < 1) := 
sorry
  
where
  f(n : ℕ) := n - (Finset.sum (Finset.finRange n.succ) (λ i, floor (r i * n)))


end max_value_f_l6_6643


namespace length_of_ST_l6_6169

theorem length_of_ST (LM MN NL: ℝ) (LR : ℝ) (LT TR LS SR: ℝ) 
  (h1: LM = 8) (h2: MN = 10) (h3: NL = 6) (h4: LR = 6) 
  (h5: LT = 8 / 3) (h6: TR = 10 / 3) (h7: LS = 9 / 4) (h8: SR = 15 / 4) :
  LS - LT = -5 / 12 :=
by
  sorry

end length_of_ST_l6_6169


namespace area_of_triangle_ABC_l6_6611

open Real

def point := ℝ × ℝ

def A : point := (0, 0)
def B : point := (5, 0)
def C : point := (0, 3)

def base (A B : point) : ℝ := |B.1 - A.1|
def height (A C : point) : ℝ := |C.2 - A.2|

def area_of_triangle (A B C : point) : ℝ :=
  0.5 * base A B * height A C

theorem area_of_triangle_ABC :
  area_of_triangle A B C = 7.5 :=
by
  sorry

end area_of_triangle_ABC_l6_6611


namespace find_g_9_l6_6712

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_of_3 : g 3 = 4

theorem find_g_9 : g 9 = 64 := by
  sorry

end find_g_9_l6_6712


namespace find_eccentricity_l6_6972

-- Let (a, b : ℝ) be positive real numbers.
-- Define a hyperbola with the equation x²/a² - y²/b² = 1.
-- The right vertex of the hyperbola is defined as (a, 0).
-- A circle is centered at (a, 0) with radius b.
-- The circle intersects one asymptote of the hyperbola at points M and N.
-- Given that the angle ∠MAN = 60°.
-- We are required to prove that eccentricity e of the hyperbola is 2√3 / 3.

variables {a b c : ℝ} (C : Type*) [mul_action ℝ C] [has_scalar ℝ (submodule ℝ C)] 
  (M N : C) (A : C) 

axiom a_pos : a > 0
axiom b_pos : b > 0
axiom ecc_formula : 3 * c^2 = 4 * a^2
axiom right_vertex : A = ⟨a, 0⟩
axiom angle_man : ∠MAN = real.pi / 3 -- 60 degrees in radians

def eccentricity (c a : ℝ) : ℝ := c / a

theorem find_eccentricity : 
  let e := eccentricity c a in
  3 * c^2 = 4 * a^2 → e = 2 * real.sqrt 3 / 3 :=
sorry

end find_eccentricity_l6_6972


namespace rectangle_perimeter_l6_6796

theorem rectangle_perimeter (l d : ℝ) (h_l : l = 8) (h_d : d = 17) :
  ∃ w : ℝ, (d^2 = l^2 + w^2) ∧ (2*l + 2*w = 46) :=
by
  sorry

end rectangle_perimeter_l6_6796


namespace distance_between_home_and_school_l6_6419

theorem distance_between_home_and_school :
  ∃ d t : ℝ, d = 4 * (t + 7 / 60) ∧ d = 8 * (t - 8 / 60) ∧ d = 2 :=
by
  use 2, 23 / 60 -- assuming these values solve the equations
  split
  { nth_rewrite 0 (2 : ℝ),
    ring_nf,
    sorry,
  }
  { nth_rewrite 0 (2 : ℝ),
    ring_nf,
    sorry,
  }

end distance_between_home_and_school_l6_6419


namespace expected_value_P_deriv_at_0_is_half_N_plus_one_l6_6248

variables (N : ℕ) (hN : 0 < N)
noncomputable def S := finset.range (N + 1) \ {0}
noncomputable def F := {f : ℕ → ℕ // ∀ i ∈ S N, f i ≥ i}

noncomputable def P (f : F N) : polynomial ℝ :=
  polynomial.interp (S N).val.to_list (λ i, f.val i)

noncomputable def P_deriv_at_0 (f : F N) : ℝ :=
  polynomial.deriv (P N f).eval 0

noncomputable def expected_value_P_deriv_at_0 : ℝ :=
  (finset.univ : finset (F N)).sum (λ f, P_deriv_at_0 N f) / (finset.univ : finset (F N)).card

theorem expected_value_P_deriv_at_0_is_half_N_plus_one :
  expected_value_P_deriv_at_0 N hN = (N + 1 : ℝ) / 2 :=
sorry

end expected_value_P_deriv_at_0_is_half_N_plus_one_l6_6248


namespace routes_between_plains_cities_correct_l6_6185

noncomputable def number_of_routes_connecting_pairs_of_plains_cities
    (total_cities : ℕ)
    (mountainous_cities : ℕ)
    (plains_cities : ℕ)
    (total_routes : ℕ)
    (routes_between_mountainous_cities : ℕ) : ℕ :=
let mountainous_city_endpoints := mountainous_cities * 3 in
let routes_between_mountainous_cities_endpoints := routes_between_mountainous_cities * 2 in
let mountainous_to_plains_routes_endpoints := mountainous_city_endpoints - routes_between_mountainous_cities_endpoints in
let plains_city_endpoints := plains_cities * 3 in
let plains_city_to_mountainous_city_routes_endpoints := mountainous_to_plains_routes_endpoints in
let endpoints_fully_in_plains_cities := plains_city_endpoints - plains_city_to_mountainous_city_routes_endpoints in
endpoints_fully_in_plains_cities / 2

theorem routes_between_plains_cities_correct :
    number_of_routes_connecting_pairs_of_plains_cities 100 30 70 150 21 = 81 := by
    sorry

end routes_between_plains_cities_correct_l6_6185


namespace robot_reaches_3_1_in_six_steps_l6_6309

theorem robot_reaches_3_1_in_six_steps :
  let q : ℚ := 37 / 512 in
  let (m, n) := (37, 512) in
  nat.coprime m n ∧ m + n = 549 :=
by
  sorry

end robot_reaches_3_1_in_six_steps_l6_6309


namespace mrs_hilt_initial_money_l6_6278

def initial_amount (pencil_cost candy_cost left_money : ℕ) := 
  pencil_cost + candy_cost + left_money

theorem mrs_hilt_initial_money :
  initial_amount 20 5 18 = 43 :=
by
  -- initial_amount 20 5 18 
  -- = 20 + 5 + 18
  -- = 25 + 18 
  -- = 43
  sorry

end mrs_hilt_initial_money_l6_6278


namespace tan_theta_sub_pi_div_four_eq_neg_seven_l6_6966

theorem tan_theta_sub_pi_div_four_eq_neg_seven (θ : ℝ) 
  (hz : (cos θ - 4/5) + (sin θ - 3/5) * complex.I = 0) : 
  tan (θ - real.pi / 4) = -7 := 
sorry

end tan_theta_sub_pi_div_four_eq_neg_seven_l6_6966


namespace function_2_satisfies_props_l6_6089

noncomputable def f1 (x : ℝ) := abs (x + 2)
noncomputable def f2 (x : ℝ) := abs (x - 2)
noncomputable def f3 (x : ℝ) := cos (x - 2)

def is_even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)

def is_decreasing_on (f : ℝ → ℝ) (s : set ℝ) := ∀ ⦃x y : ℝ⦄, x ∈ s → y ∈ s → x < y → f x ≥ f y
def is_increasing_on (f : ℝ → ℝ) (s : set ℝ) := ∀ ⦃x y : ℝ⦄, x ∈ s → y ∈ s → x < y → f x ≤ f y

theorem function_2_satisfies_props :
  is_even_function (f2 ∘ (λ x, x+2)) ∧
  is_decreasing_on f2 {x | x < 2} ∧ is_increasing_on f2 {x | x > 2} :=
by
  sorry

end function_2_satisfies_props_l6_6089


namespace percentage_exceed_but_not_ticket_is_50_l6_6284

/-- Define the assumptions -/
axiom motorists_count : ℕ
axiom percent_speeding : ℝ
axiom percent_tickets : ℝ
axiom percent_exceed_but_not_ticket : ℝ

/-- Assumption definitions -/
def total_motorists := (motorists_count : ℝ)
def total_exceed := percent_speeding * total_motorists
def total_tickets := percent_tickets * total_motorists
def total_exceed_but_not_ticket := total_exceed - total_tickets

/-- Assumptions given in the problem -/
axiom percent_speeding_value : percent_speeding = 0.20
axiom percent_tickets_value : percent_tickets = 0.10

/-- Theorem to prove the percentage of motorists who exceed the speed limit but do not receive tickets is 50% -/
theorem percentage_exceed_but_not_ticket_is_50 : 
  percent_exceed_but_not_ticket = 0.50 :=
by
  sorry

end percentage_exceed_but_not_ticket_is_50_l6_6284


namespace kilometers_to_centimeters_l6_6548

theorem kilometers_to_centimeters (km_to_m: 1 = 1000) (m_to_cm: 1 = 100) : true := by
  have cm_in_km : 1 * km_to_m * m_to_cm = 100000 := by
    calc
      1 * km_to_m * m_to_cm = 1 * 1000 * 100 := by sorry
  sorry

end kilometers_to_centimeters_l6_6548


namespace distance_between_alice_bob_l6_6855

theorem distance_between_alice_bob :
  let time := 120
  let alice_speed := 1 -- miles per 20 minutes
  let bob_speed := 3 -- miles per 40 minutes
  let alice_distance := (time / 20) * alice_speed
  let bob_distance := (time / 40) * bob_speed
  alice_distance + bob_distance = 15 := 
by
  let time : ℕ := 120
  let alice_speed : ℕ := 1 -- miles per 20 minutes
  let bob_speed : ℕ := 3 -- miles per 40 minutes
  let alice_distance := time / 20 * alice_speed
  let bob_distance := time / 40 * bob_speed
  show alice_distance + bob_distance = 15
  from sorry

end distance_between_alice_bob_l6_6855


namespace increasing_on_1_to_infty_min_value_on_1_to_e_l6_6969

noncomputable def f (x : ℝ) (a : ℝ) := x^2 - a * Real.log x
noncomputable def f' (x : ℝ) (a : ℝ) := (2 * x^2 - a) / x

-- Proof that f(x) is increasing on (1, +∞) when a = 2
theorem increasing_on_1_to_infty (x : ℝ) (h : x > 1) : f' x 2 > 0 := 
  sorry

-- Proof for minimum value of f(x) on [1, e]
theorem min_value_on_1_to_e (a : ℝ) :
  if a ≤ 2 then ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = 1
  else if 2 < a ∧ a < 2 * Real.exp 2 then 
    ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = a / 2 - (a / 2) * Real.log (a / 2)
  else if a ≥ 2 * Real.exp 2 then 
    ∃ c ∈ Set.Icc 1 (Real.exp 1), f c a = Real.exp 2 - a
  else False := 
  sorry

end increasing_on_1_to_infty_min_value_on_1_to_e_l6_6969


namespace singer_winner_is_A_l6_6760

-- Definitions for the statements made by each singer
def statement_A (winner : String) : Prop :=
  winner ≠ "A"

def statement_B (winner : String) : Prop :=
  winner = "C"

def statement_C (winner : String) : Prop :=
  winner = "D"

def statement_D (winner : String) : Prop :=
  winner ≠ "D"

-- The main theorem stating that A is the winner
theorem singer_winner_is_A : ∀ (winner : String), 
  (statement_A winner → ¬(statement_B winner) ∧ ¬(statement_C winner) ∧ statement_D winner) ∧
  (¬(statement_A winner) → (statement_B winner ∧ statement_C winner)) ∧
  (statement_A winner ∧ statement_B winner ∧ ¬(statement_C winner) → ¬statement_D winner) →
  winner = "A" :=
begin
  sorry
end

end singer_winner_is_A_l6_6760


namespace interval_of_monotonicity_l6_6528

noncomputable def f (a x : ℝ) : ℝ := log x - a * x
noncomputable def g (x : ℝ) : ℝ := -x^2 + 2 * x + 1

theorem interval_of_monotonicity 
(a : ℝ) (x_1 : ℝ) (h1 : 1 ≤ x_1 ∧ x_1 ≤ Real.exp 1) :
  (∃ x_2 : ℝ, 0 ≤ x_2 ∧ x_2 ≤ 3 ∧ f a x_1 = g x_2) ↔ 
  (- (1 / Real.exp 1) ≤ a ∧ a ≤ 3 / Real.exp 1) := 
sorry

end interval_of_monotonicity_l6_6528


namespace track_and_field_analysis_l6_6453

theorem track_and_field_analysis :
  let male_athletes := 12
  let female_athletes := 8
  let tallest_height := 190
  let shortest_height := 160
  let avg_male_height := 175
  let avg_female_height := 165
  let total_athletes := male_athletes + female_athletes
  let sample_size := 10
  let prob_selected := 1 / 2
  let prop_male := male_athletes / total_athletes * sample_size
  let prop_female := female_athletes / total_athletes * sample_size
  let overall_avg_height := (male_athletes / total_athletes) * avg_male_height + (female_athletes / total_athletes) * avg_female_height
  (tallest_height - shortest_height = 30) ∧
  (sample_size / total_athletes = prob_selected) ∧
  (prop_male = 6 ∧ prop_female = 4) ∧
  (overall_avg_height = 171) →
  (A = true ∧ B = true ∧ C = false ∧ D = true) :=
by
  sorry

end track_and_field_analysis_l6_6453


namespace total_water_in_bucket_l6_6461

theorem total_water_in_bucket (initial_liters : ℕ) (additional_milliliters : ℕ) (conversion_factor : ℕ) : 
  initial_liters = 2 → additional_milliliters = 460 → conversion_factor = 1000 →
  initial_liters * conversion_factor + additional_milliliters = 2460 := by
  intros h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end total_water_in_bucket_l6_6461


namespace matrix_det_eq_specific_value_l6_6490

theorem matrix_det_eq_specific_value (x : ℝ) : 
  (det ![
    ![x^2, 2*x + 1],
    ![3*x, 4*x + 2]
  ]) = 10 ↔ x = 2 :=
by sorry

end matrix_det_eq_specific_value_l6_6490


namespace determine_angle_E_l6_6222

-- Define the conditions
variables (EFGH : Type) [parallelogram EFGH]
variables (F G H : EFGH)
variables (angle_FGH : ℝ) (h_angle_FGH : angle_FGH = 70)

-- State the theorem
theorem determine_angle_E : angle_E = 110 :=
by
  sorry

end determine_angle_E_l6_6222


namespace smallest_k_l6_6356

theorem smallest_k (k : ℕ) 
  (h1 : 201 % 24 = 9 % 24) 
  (h2 : (201 + k) % (24 + k) = (9 + k) % (24 + k)) : 
  k = 8 :=
by 
  sorry

end smallest_k_l6_6356


namespace minimum_value_l6_6098

theorem minimum_value (a b : ℝ) (h_a_pos : a > 0) (h_b_pos : b > 0) (h_sum : a + b = 2) :
  (1 / a) + (1 / b) ≥ 2 :=
by {
  sorry
}

end minimum_value_l6_6098


namespace angle_AOB_is_65_l6_6769

-- Define the conditions as hypotheses
variables {P A B O : Type} -- Points on the plane
variables (h_triangle_tangent : Triangle_tangent_to_circle P A B O)
variables (h_angle_APB : ∠ APB = 50)

-- Define the theorem to prove the target statement
theorem angle_AOB_is_65 :
  ∠ AOB = 65 :=
sorry

end angle_AOB_is_65_l6_6769


namespace periodic_sequence_exists_l6_6932

noncomputable def bounded_sequence (a : ℕ → ℤ) (M : ℤ) :=
  ∀ n, |a n| ≤ M

noncomputable def satisfies_recurrence (a : ℕ → ℤ) :=
  ∀ n, n ≥ 5 → a n = (a (n - 1) + a (n - 2) + a (n - 3) * a (n - 4)) / (a (n - 1) * a (n - 2) + a (n - 3) + a (n - 4))

theorem periodic_sequence_exists (a : ℕ → ℤ) (M : ℤ) 
  (h_bounded : bounded_sequence a M) (h_rec : satisfies_recurrence a) : 
  ∃ l : ℕ, ∀ n : ℕ, a (l + n) = a (l + n + (l + 1) - l) :=
sorry

end periodic_sequence_exists_l6_6932


namespace line_intersects_circle_l6_6333

noncomputable def center_radius_of_circle (a b : ℝ) (c : ℝ) : (ℝ × ℝ) × ℝ :=
  let center := (a / 2, b / 2)
  let radius := real.sqrt(center.1 ^ 2 + center.2 ^ 2 - c)
  (center, radius)

noncomputable def distance_point_to_line (a b c x₁ y₁ : ℝ) : ℝ :=
  (abs (a * x₁ + b * y₁ + c)) / real.sqrt (a ^ 2 + b ^ 2)

theorem line_intersects_circle : 
  let circle_eq := (x^2 + y^2 - 18 * x - 45 = 0)
  let line_eq := (4 * x - 3 * y = 0)
  let center := (9, 0)
  let radius := real.sqrt 126
  let distance_center_to_line := (36 / 5)
  (distance_center_to_line < radius) →
  ∃ x y : ℝ, (((4 * x - 3 * y = 0) ∧ (x^2 + y^2 - 18 * x - 45 = 0)) :=
by
  skip
  sorry

end line_intersects_circle_l6_6333


namespace base_height_ratio_l6_6317

-- Define the conditions
def cultivation_cost : ℝ := 333.18
def rate_per_hectare : ℝ := 24.68
def base_of_field : ℝ := 300
def height_of_field : ℝ := 300

-- Prove the ratio of base to height is 1
theorem base_height_ratio (b h : ℝ) (cost rate : ℝ)
  (h1 : cost = 333.18) (h2 : rate = 24.68) 
  (h3 : b = 300) (h4 : h = 300) : b / h = 1 :=
by
  sorry

end base_height_ratio_l6_6317


namespace stretching_transformation_eq_curve_l6_6629

variable (x y x₁ y₁ : ℝ)

theorem stretching_transformation_eq_curve :
  (x₁ = 3 * x) →
  (y₁ = y) →
  (x₁^2 + 9 * y₁^2 = 9) →
  (x^2 + y^2 = 1) :=
by
  intros h1 h2 h3
  sorry

end stretching_transformation_eq_curve_l6_6629


namespace periodic_decimal_to_fraction_l6_6005

theorem periodic_decimal_to_fraction
  (h : ∀ n : ℕ, 0.<digit>02 n / 99) :
  2.0<digit>06 = 68 / 33 :=
sorry

end periodic_decimal_to_fraction_l6_6005


namespace plains_routes_count_l6_6201

def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70
def total_routes : ℕ := 150
def mountainous_routes : ℕ := 21

theorem plains_routes_count :
  total_cities = mountainous_cities + plains_cities →
  3 * total_routes = total_cities →
  mountainous_routes * 2 ≤ mountainous_cities * 3 →
  (total_routes - mountainous_routes) * 2 = (70 * 3 - (mountainous_routes * 2)) →
  (total_routes - mountainous_routes * 2) / 2 = 81 :=
begin
  sorry
end

end plains_routes_count_l6_6201


namespace maximum_value_l6_6116

open Real

variables (e1 e2 a : ℝ × ℝ)

def is_unit_vector (v : ℝ × ℝ) : Prop := (v.1 ^ 2 + v.2 ^ 2 = 1)

def less_equal_unit_vector_sum (e1 e2 : ℝ × ℝ) : Prop := 
  (e1.1 + 2 * e2.1) ^ 2 + (e1.2 + 2 * e2.2) ^ 2 ≤ 4

def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (dot_product v v)

noncomputable def max_value_expression (a e1 e2 : ℝ × ℝ) : ℝ := 
  (dot_product a (2 * e1 + e2)) / magnitude a

theorem maximum_value (h1 : is_unit_vector e1) 
                     (h2 : is_unit_vector e2) 
                     (h3 : less_equal_unit_vector_sum e1 e2) 
                     (h4 : a ≠ (0, 0))
                     (h5 : dot_product a e1 ≤ dot_product a e2) : 
  max_value_expression a e1 e2 = 3 * sqrt (6) / 4 := 
sorry

end maximum_value_l6_6116


namespace geom_seq_necessity_geom_seq_not_sufficient_l6_6911

theorem geom_seq_necessity (a₁ q : ℝ) (h₁ : 0 < a₁) (h₂ : a₁ < a₁ * q^2) :
    q > 1 ∨ q < -1 :=
  sorry

theorem geom_seq_not_sufficient (a₁ q : ℝ) (h₁ : 0 < a₁) (h₂ : a₁ < a₁ * q^2) :
    ¬ (q > 1 → a₁ < a₁ * q^2) :=
  sorry

end geom_seq_necessity_geom_seq_not_sufficient_l6_6911


namespace validate_statements_l6_6691

variable {R : Type*} [AddCommGroup R] [OrderedAddCommGroup R]

noncomputable def f : R → R :=
sorry

axiom f_pos : ∀ (x : R), 0 < f x
axiom f_add : ∀ (a b : R), f (a + b) = f a + f b

theorem validate_statements :
  (f 0 = 0) ∧
  (∀ a : R, f (-a) = -f a) ∧
  (∀ a : R, f (2 * a) = 2 * f a) ∧
  ¬(∀ a b : R, a ≤ b → f a ≤ f b) :=
by {
  split,
  { -- Proof for f 0 = 0
    sorry },
  split,
  { -- Proof for ∀ a, f (-a) = -f a
    sorry },
  split,
  { -- Proof for ∀ a, f (2 * a) = 2 * f a
    sorry },
  { -- Proof for ¬(∀ a b, a ≤ b → f a ≤ f b)
    sorry }
}

end validate_statements_l6_6691


namespace affine_division_l6_6651

-- Definitions for points and vectors
variables {K : Type*} [Field K]
variables {V : Type*} [AddCommGroup V] [Module K V]
variables {P : Type*} [AddTorsor V P]

-- Given points and transformation
variables (A B C : P)
variables (L : AffineMap K P P)
variables (A' B' C' : P)

-- Given ratio
variables (p q : K)

-- Hypotheses
hypothesis hC_divides : p • (B -ᵥ C : V) = q • (C -ᵥ A : V)
hypothesis hA' : A' = L A
hypothesis hB' : B' = L B
hypothesis hC' : C' = L C

-- Theorem statement
theorem affine_division (hC_divides : p • (B -ᵥ C : V) = q • (C -ᵥ A : V))
  (hA' : A' = L A) (hB' : B' = L B) (hC' : C' = L C) : 
  p • (B' -ᵥ C' : V) = q • (C' -ᵥ A' : V) := 
sorry

end affine_division_l6_6651


namespace inequality_for_positive_sums_l6_6533

theorem inequality_for_positive_sums (n : ℕ) (hn : 2 ≤ n) (x : Fin n → ℝ) (hx : ∀ i, x i > 0) :
  (∑ i, x i) * (∑ i, (x i)⁻¹) ≥ n^2 :=
sorry

end inequality_for_positive_sums_l6_6533


namespace prime_minister_stays_in_power_l6_6585

theorem prime_minister_stays_in_power (A B: string)
  (h1: A = "Go") (h2: B = "Go"):
  "Stay" = "Stay" :=
by
  sorry

end prime_minister_stays_in_power_l6_6585


namespace find_number_l6_6517

theorem find_number (x : ℝ) (h : x * 9999 = 824777405) : x = 82482.5 :=
by
  sorry

end find_number_l6_6517


namespace johns_overall_profit_l6_6800

def grinder_cost : ℝ := 15000
def mobile_cost : ℝ := 8000
def grinder_loss_percentage : ℝ := 0.04
def mobile_profit_percentage : ℝ := 0.10

def calculate_grinder_loss (cost : ℝ) (loss_percentage : ℝ) : ℝ :=
  loss_percentage * cost

def calculate_mobile_profit (cost : ℝ) (profit_percentage : ℝ) : ℝ :=
  profit_percentage * cost

def calculate_selling_price (cost : ℝ) (change : ℝ) (is_loss : Bool) : ℝ :=
  if is_loss then cost - change else cost + change

theorem johns_overall_profit
  (grinder_cost : ℝ)
  (mobile_cost : ℝ)
  (grinder_loss_percentage : ℝ)
  (mobile_profit_percentage : ℝ) :
  let grinder_loss := calculate_grinder_loss grinder_cost grinder_loss_percentage
  let mobile_profit := calculate_mobile_profit mobile_cost mobile_profit_percentage
  let grinder_selling_price := calculate_selling_price grinder_cost grinder_loss true
  let mobile_selling_price := calculate_selling_price mobile_cost mobile_profit false
  let total_cost_price := grinder_cost + mobile_cost
  let total_selling_price := grinder_selling_price + mobile_selling_price
  let overall_profit := total_selling_price - total_cost_price
  overall_profit = 200 :=
by {
  let grinder_loss :=  calculate_grinder_loss grinder_cost grinder_loss_percentage,
  let mobile_profit := calculate_mobile_profit mobile_cost mobile_profit_percentage,
  let grinder_selling_price := calculate_selling_price grinder_cost grinder_loss true,
  let mobile_selling_price := calculate_selling_price mobile_cost mobile_profit false,
  let total_cost_price := grinder_cost + mobile_cost,
  let total_selling_price := grinder_selling_price + mobile_selling_price,
  let overall_profit := total_selling_price - total_cost_price,
  sorry
}

end johns_overall_profit_l6_6800


namespace ellipse_standard_equations_l6_6880

theorem ellipse_standard_equations :
  ∀ (a b : ℝ), a = 3 → b = 2 →
  ∃ (e : ℝ), e = (sqrt 5) / 5 →
  (∃ (a' b' : ℝ), a' = 5 ∧ b' = 2 * sqrt 5 ∧
    (∀ x y : ℝ, 
      (x^2 / 25) + (y^2 / 20) = 1 ∨
      (y^2 / 25) + (x^2 / 20) = 1)) :=
by
  sorry

end ellipse_standard_equations_l6_6880


namespace identical_prob_of_painted_cubes_l6_6774

/-
  Given:
  - Each face of a cube can be painted in one of 3 colors.
  - Each cube has 6 faces.
  - The total possible ways to paint both cubes is 531441.
  - The total ways to paint them such that they are identical after rotation is 66.

  Prove:
  - The probability of two painted cubes being identical after rotation is 2/16101.
-/
theorem identical_prob_of_painted_cubes :
  let total_ways := 531441
  let identical_ways := 66
  (identical_ways : ℚ) / total_ways = 2 / 16101 := by
  sorry

end identical_prob_of_painted_cubes_l6_6774


namespace ellipse_equation_and_ratio_l6_6108

-- Define the conditions as assumptions in Lean
variables {a b k x y : ℝ}
variables (F : ℝ × ℝ) (B1 B2 : ℝ × ℝ)
variables (ellipse : ℝ → ℝ → Prop)

-- Given Conditions
def ellipseC : Prop := (x^2 / a^2) + (y^2 / b^2) = 1
def focus_cond : Prop := a > b ∧ b > 0 ∧ F = (1, 0)
def minor_axis_cond : Prop := 
  let FB1 := (B1.1 - F.1, B1.2 - F.2) in
  let FB2 := (B2.1 - F.1, B2.2 - F.2) in
  ∃ a, FB1.1 * FB2.1 + FB1.2 * FB2.2 = -a

-- Conditions for the second question
def line_l : Prop := ∀ k : ℝ, k ≠ 0 → ∃ x y : ℝ, y = k * (x - F.1)
def midpoint_P (x1 x2 k : ℝ) : Prop := (4 * k^2 / (3 + 4 * k^2), -3 * k / (3 + 4 * k^2)) = (x1 + x2)/2
def length_MN (k : ℝ) : Prop := √((1 + k^2) * (64 * k^2 / (3 + 4 * k^2)^2 - 4 * (4 * k^2 - 12) / (3 + 4 * k^2))) = 12 * (k^2 + 1) / (4 * k^2 + 3)
def ratio_DP_MN : Prop := ∀ k : ℝ, k ≠ 0 → 0 < 3 * √(k^2 * (k^2 + 1)) / (4 * k^2 + 3) / 12 * (k^2 + 1) / (4 * k^2 + 3) ∧ 1/4 * √(1 - 1 / (k^2 + 1)) < 1/4

-- The proof problem in Lean 4 statement
theorem ellipse_equation_and_ratio (a b : ℝ) (F B1 B2 : ℝ × ℝ) (k : ℝ) (x y : ℝ) : 
  ellipseC x y → focus_cond F → minor_axis_cond F B1 B2 a →
  (a = 2 ∧ b = √3) ∧
  ellipse x y = (x^2 / 4) + (y^2 / 3) = 1 ∧
  ratio_DP_MN k :=
begin
  sorry
end

end ellipse_equation_and_ratio_l6_6108


namespace plains_routes_count_l6_6199

def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70
def total_routes : ℕ := 150
def mountainous_routes : ℕ := 21

theorem plains_routes_count :
  total_cities = mountainous_cities + plains_cities →
  3 * total_routes = total_cities →
  mountainous_routes * 2 ≤ mountainous_cities * 3 →
  (total_routes - mountainous_routes) * 2 = (70 * 3 - (mountainous_routes * 2)) →
  (total_routes - mountainous_routes * 2) / 2 = 81 :=
begin
  sorry
end

end plains_routes_count_l6_6199


namespace coeff_x3_l6_6229

-- Definitions based on problem conditions
def binom (n k : ℕ) := Nat.choose n k
def term1 (k : ℕ) := binom 5 k * 2^(5-k) * x^k
def term2 (k : ℕ) := binom 5 k * 2^(5-k) * x^(k + 1)

noncomputable def expansion_term (k : ℕ) := (if k = 0 then 1 else 0) * term1 k + term2 (k - 1)

-- Theorem to prove the coefficient of x^3
theorem coeff_x3 : 
  let coeff := expansion_term 3;
  coeff = 120 := 
by
  sorry

end coeff_x3_l6_6229


namespace magical_card_in_stack_l6_6699

noncomputable def is_magical_stack (n: ℕ) (card_pos: ℕ) : Prop :=
  let total_cards := 2 * n
  let pile_a := (1 to n).to_list
  let pile_b := (n+1 to total_cards).to_list
  let restacked := List.zipWith (List.append) pile_b pile_a in
  let magical := ∀ i < n, restacked[2*i] == pile_b[i] && restacked[2*i+1] == pile_a[i] in
  (card_pos <= (2*n) ∧ restacked[card_pos - 1] == card_pos)

theorem magical_card_in_stack 
  (n : ℕ) 
  (h1 : odd 201) 
  (h2 : 201 ∈ [(n+1)..(2*n)]) 
  : 2 * 201 = 402 :=
by sorry

end magical_card_in_stack_l6_6699


namespace complex_number_in_second_quadrant_l6_6100

theorem complex_number_in_second_quadrant (z : ℂ) (h : (2 - 3 * complex.I) * z = 1 + complex.I) :
  z.re < 0 ∧ z.im > 0 :=
by
  sorry

end complex_number_in_second_quadrant_l6_6100


namespace anne_bottle_caps_l6_6456

-- Define initial conditions
def anne_initial_bottle_caps : ℕ := 10
def fraction_given_away : ℝ := 2/5
def additional_bottle_caps_found : ℕ := 5

-- Prove the number of bottle caps Anne ends with is 11 given the conditions
theorem anne_bottle_caps : 
  let given_away := (fraction_given_away * (anne_initial_bottle_caps : ℝ)).to_nat in
  let remaining := anne_initial_bottle_caps - given_away in
  let final := remaining + additional_bottle_caps_found in
  final = 11 :=
by
  -- Introduce the formulas to use
  let given_away := (fraction_given_away * (anne_initial_bottle_caps : ℝ)).to_nat
  have h_given_away : given_away = 4 := by norm_num
  let remaining := anne_initial_bottle_caps - given_away
  have h_remaining : remaining = 6 := by norm_num
  let final := remaining + additional_bottle_caps_found
  have h_final : final = 11 := by norm_num
  exact h_final

end anne_bottle_caps_l6_6456


namespace ratio_of_oranges_l6_6244

theorem ratio_of_oranges :
  ∃ (x : ℕ),
    let initial_oranges := 60,
    let eaten_oranges := 10,
    let left_oranges := initial_oranges - eaten_oranges,
    let final_oranges := 30,
    let returned_oranges := 5,
    let stolen_oranges := left_oranges - (final_oranges - returned_oranges)
    in stolen_oranges = x ∧ x / left_oranges = 1 / 2 := 
by {
  let initial_oranges := 60,
  let eaten_oranges := 10,
  let left_oranges := initial_oranges - eaten_oranges,
  let final_oranges := 30,
  let returned_oranges := 5,
  let stolen_oranges := left_oranges - (final_oranges - returned_oranges),
  use stolen_oranges,
  have h1 : left_oranges = 50 := by simp [left_oranges],
  have h2 : final_oranges - returned_oranges = 25 := by simp [final_oranges, returned_oranges],
  have h3 : stolen_oranges = 25 := by simp [left_oranges, final_oranges, returned_oranges],
  have h4 : stolen_oranges / 50 = 1 / 2 := by norm_num [h1, h3],
  exact ⟨h3, h4⟩,
}.

end ratio_of_oranges_l6_6244


namespace length_CF_l6_6863

-- Definitions of the problem conditions
variables (A B C D F E : Type*) -- points
variable [metric_space A] [metric_space B] [metric_space C] [metric_space D] [metric_space E] [metric_space F]
variables (AD BC AB DC : ℝ) (is_isosceles_trapezoid : ∀ (A B C D : Type*), Prop)
variables (midpoint : ∀ (B df : Type*), Prop)

-- Assumptions based on the problem statement
axiom h_isosceles_trapezoid : (is_isosceles_trapezoid A B C D)
axiom h_lengths : (AD = 5) (BC = 5) (AB = 4) (DC = 10)
axiom h_C_on_DF : (C ∈ DF)
axiom h_B_midpoint : midpoint B DF

-- Statement to prove
theorem length_CF (CF : ℝ) : CF = 4 := by sorry

end length_CF_l6_6863


namespace cosine_angle_eq_neg_sqrt21_div_7_l6_6115

variables {E : Type*} [inner_product_space ℝ E]
variables (e1 e2 a b : E)
variables (u : unit_vector e1) (v : unit_vector e2)

-- Conditions
hypothesis h_angle : real.angle u v = real.pi * (2/3)
hypothesis ha : a = 2 • e1 + e2
hypothesis hb : b = e2 - 2 • e1

noncomputable def cosine_angle_between_a_and_b : ℝ :=
  (inner_product_space.to_bilin' a b) / (∥a∥ * ∥b∥)

-- Goal
theorem cosine_angle_eq_neg_sqrt21_div_7 :
  cosine_angle_between_a_and_b a b = - real.sqrt 21 / 7 :=
sorry

end cosine_angle_eq_neg_sqrt21_div_7_l6_6115


namespace f_sum_eq_seven_l6_6131

noncomputable def f : ℝ → ℝ :=
λ x, if x < 1 then x + Real.log (2 - x) / Real.log 3 else 3^x

theorem f_sum_eq_seven (h1 : f (-7) = -5) (h2 : f (Real.log 12 / Real.log 3) = 12) : 
  f (-7) + f (Real.log 12 / Real.log 3) = 7 :=
by
  simp [h1, h2]
  norm_num

-- Since we provided h1 and h2 explicitly, it demonstrates the desired mathematical equivalence.
-- Proof is skipped with sorry, as requested.

end f_sum_eq_seven_l6_6131


namespace g_9_l6_6722

variable (g : ℝ → ℝ)

-- Conditions
axiom func_eq : ∀ x y : ℝ, g(x + y) = g(x) * g(y)
axiom g_3 : g 3 = 4

-- Theorem to prove
theorem g_9 : g 9 = 64 :=
by
  sorry

end g_9_l6_6722


namespace recycling_target_l6_6685

/-- Six Grade 4 sections launched a recycling drive where they collect old newspapers to recycle.
Each section collected 280 kilos in two weeks. After the third week, they found that they need 320 kilos more to reach their target.
  How many kilos of the newspaper is their target? -/
theorem recycling_target (sections : ℕ) (kilos_collected_2_weeks : ℕ) (additional_kilos : ℕ) : 
  sections = 6 ∧ kilos_collected_2_weeks = 280 ∧ additional_kilos = 320 → 
  (sections * (kilos_collected_2_weeks / 2) * 3 + additional_kilos) = 2840 :=
by
  sorry

end recycling_target_l6_6685


namespace max_sum_x_y_l6_6508

theorem max_sum_x_y (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) 
  (h3 : x^3 + y^3 + (x + y)^3 + 36 * x * y = 3456) : x + y ≤ 12 :=
sorry

end max_sum_x_y_l6_6508


namespace polynomial_factorization_l6_6384

theorem polynomial_factorization : 
  (x : ℤ) → (x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1)) := 
by
  sorry

end polynomial_factorization_l6_6384


namespace angle_A_size_max_area_triangle_l6_6608

open Real

variable {A B C a b c : ℝ}

-- Part 1: Prove the size of angle A given the conditions
theorem angle_A_size (h1 : (2 * c - b) / a = cos B / cos A) :
  A = π / 3 :=
sorry

-- Part 2: Prove the maximum area of triangle ABC
theorem max_area_triangle (h2 : a = 2 * sqrt 5) :
  ∃ (S : ℝ), S = 5 * sqrt 3 ∧ ∀ (b c : ℝ), S ≤ 1/2 * b * c * sin (π / 3) :=
sorry

end angle_A_size_max_area_triangle_l6_6608


namespace equilateral_iff_rotation_l6_6261

/-- Representation of the complex rotation factor j -/
def j : ℂ := exp (2 * real.pi * I / 3)

lemma j_cube_root_of_unity : j^3 = 1 := sorry
lemma j_sum_to_zero : (1:ℂ) + j + j^2 = 0 := sorry

/-- A triangle ABC is equilateral if and only if a + jb + j^2c = 0 -/
theorem equilateral_iff_rotation (a b c : ℂ) : 
  (a + j * b + j^2 * c = 0) ↔ 
  -- Conditions to verify if the triangle A, B, C is equilateral.
  -- Declare that ABC is equilateral
  ∀ (θ : ℂ), ∀ (rotation : ℂ -> ℂ), rotation = λ z, j * z ∧
  ∃ (A B C : ℂ), A = a ∧ B = b ∧ C = c ∧
  (C - A = rotation (B - A)) :=
sorry

end equilateral_iff_rotation_l6_6261


namespace sequence_infinite_integers_l6_6936

theorem sequence_infinite_integers (x : ℕ → ℝ) (x1 x2 : ℝ) 
  (h1 : x 1 = x1) 
  (h2 : x 2 = x2) 
  (h3 : ∀ n ≥ 3, x n = x (n - 2) * x (n - 1) / (2 * x (n - 2) - x (n - 1))) : 
  (∃ k : ℤ, x1 = k ∧ x2 = k) ↔ (∀ n, ∃ m : ℤ, x n = m) :=
sorry

end sequence_infinite_integers_l6_6936


namespace eq_pow_four_l6_6955

theorem eq_pow_four (a b : ℝ) (h : a = b + 1) : a^4 = b^4 → a = 1/2 ∧ b = -1/2 :=
by
  sorry

end eq_pow_four_l6_6955


namespace range_of_f_le_2_l6_6266

def f (x : ℝ) : ℝ :=
  if x ≤ 1 then 2^(1 - x) else 1 - Real.log x / Real.log 2

theorem range_of_f_le_2 (x : ℝ) : f(x) ≤ 2 → x ≥ 0 :=
by
  unfold f
  split_ifs with h
  case pos =>
    intro h1
    linarith
  case neg =>
    intro h2
    have hx : Real.log x / Real.log 2 ≥ -1 := by linarith
    have h3 : x > 0 := by linarith [Real.lt_irrefl x]
    sorry

end range_of_f_le_2_l6_6266


namespace cost_split_l6_6397

-- Let total_cost represent the total cost of airfare and hotel.
def total_cost : ℝ := 13500.0

-- Let number_of_people represent the total number of people sharing the cost.
def number_of_people : ℝ := 15.0

-- Define the cost per person as the total cost divided by the number of people.
def cost_per_person : ℝ := total_cost / number_of_people

-- State the theorem to be proved.
theorem cost_split :
  cost_per_person = 900.0 :=
by
  sorry

end cost_split_l6_6397


namespace coefficient_of_x4_in_expansion_l6_6362

theorem coefficient_of_x4_in_expansion :
  (∑ k in Finset.range (8 + 1), (Nat.choose 8 k) * (x : ℝ)^(8 - k) * (3 * Real.sqrt 2)^k).coeff 4 = 22680 :=
by
  sorry

end coefficient_of_x4_in_expansion_l6_6362


namespace polar_to_rectangular_correct_l6_6332

noncomputable def polar_to_rectangular (rho theta x y : ℝ) : Prop :=
  rho = 4 * Real.sin theta + 2 * Real.cos theta ∧
  rho * Real.sin theta = y ∧
  rho * Real.cos theta = x ∧
  (x - 1) ^ 2 + (y - 2) ^ 2 = 5

theorem polar_to_rectangular_correct {rho theta x y : ℝ} :
  (rho = 4 * Real.sin theta + 2 * Real.cos theta) →
  (rho * Real.sin theta = y) →
  (rho * Real.cos theta = x) →
  (x - 1) ^ 2 + (y - 2) ^ 2 = 5 :=
by
  sorry

end polar_to_rectangular_correct_l6_6332


namespace expected_value_is_correct_l6_6676

noncomputable def expected_value_max_two_rolls : ℝ :=
  let p_max_1 := (1/6) * (1/6)
  let p_max_2 := (2/6) * (2/6) - (1/6) * (1/6)
  let p_max_3 := (3/6) * (3/6) - (2/6) * (2/6)
  let p_max_4 := (4/6) * (4/6) - (3/6) * (3/6)
  let p_max_5 := (5/6) * (5/6) - (4/6) * (4/6)
  let p_max_6 := 1 - (5/6) * (5/6)
  1 * p_max_1 + 2 * p_max_2 + 3 * p_max_3 + 4 * p_max_4 + 5 * p_max_5 + 6 * p_max_6

theorem expected_value_is_correct :
  expected_value_max_two_rolls = 4.5 :=
sorry

end expected_value_is_correct_l6_6676


namespace plains_routes_count_l6_6203

def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70
def total_routes : ℕ := 150
def mountainous_routes : ℕ := 21

theorem plains_routes_count :
  total_cities = mountainous_cities + plains_cities →
  3 * total_routes = total_cities →
  mountainous_routes * 2 ≤ mountainous_cities * 3 →
  (total_routes - mountainous_routes) * 2 = (70 * 3 - (mountainous_routes * 2)) →
  (total_routes - mountainous_routes * 2) / 2 = 81 :=
begin
  sorry
end

end plains_routes_count_l6_6203


namespace sector_angle_l6_6962

theorem sector_angle (r : ℝ) (S_sector : ℝ) (h_r : r = 2) (h_S : S_sector = (2 / 5) * π) : 
  (∃ α : ℝ, S_sector = (1 / 2) * α * r^2 ∧ α = (π / 5)) :=
by
  use π / 5
  sorry

end sector_angle_l6_6962


namespace compare_variance_for_uniform_heights_l6_6729

def heights_team_A : List ℝ := [178, 177, 179, 179, 178, 178, 177, 178, 177, 179]
def heights_team_B : List ℝ := [178, 177, 177, 176, 178, 175, 177, 181, 180, 181]

theorem compare_variance_for_uniform_heights :
  (∀ X : List ℝ, ∀ Y : List ℝ, (measure_theory.variance X ≤ measure_theory.variance Y ↔ heights_team_A = X ∧ heights_team_B = Y)) :=
by
  sorry

end compare_variance_for_uniform_heights_l6_6729


namespace club_elections_l6_6289

-- Definitions based on conditions
def club_members : ℕ := 24
def boys : ℕ := 12
def girls : ℕ := 12
def groups : ℕ := 2
def members_per_group_per_gender : ℕ := boys / groups -- 6

-- Proof Statement
theorem club_elections : 
  (∀ p : ℕ, p <= club_members) → 
  (∀ v : ℕ, v <= club_members) → 
  (p_gender : bool) → 
  (v_gender : bool) → 
  p_gender ≠ v_gender → 
  (p_group : ℕ) → 
  (v_group : ℕ) → 
  p_group ≠ v_group → 
  (ways_to_choose_pres : boys + girls = club_members) → 
  (possible_vice_pres : members_per_group_per_gender = 6) → 
  12 * members_per_group_per_gender + 12 * members_per_group_per_gender = 144 :=
  sorry

end club_elections_l6_6289


namespace value_of_p_l6_6587

theorem value_of_p (p : ℤ) : 3^3 * 3^(-1) = 3^p → p = 2 :=
by
  intro h
  have : 3^3 * 3^(-1) = 3^(3-1) := by rw [←pow_add, add_comm, add_neg_self, pow_zero, mul_one]
  rw this at h
  exact pow_injective one_ne_zero h

end value_of_p_l6_6587


namespace ln_tangent_area_at_1_0_l6_6075

noncomputable def ln_tangent_area (f : ℝ → ℝ) (p : ℝ × ℝ) : ℝ :=
  let ⟨x₀, y₀⟩ := p in
  let derivative := 1 / x₀ in
  let tangent_line y := (y - y₀) / (x₀ * derivative) = x₀ in
  let intercept_x := y₀ / (-derivative) + x₀ in
  let intercept_y := y₀ + derivative * (-x₀) in
  0.5 * intercept_x * intercept_y

theorem ln_tangent_area_at_1_0 :
  ln_tangent_area (λ x, Real.log x) (1, 0) = 1 / 2 :=
sorry

end ln_tangent_area_at_1_0_l6_6075


namespace polynomial_factorization_l6_6388

theorem polynomial_factorization :
  ∀ x : ℤ, x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by sorry

end polynomial_factorization_l6_6388


namespace winning_candidate_vote_percentage_l6_6858

theorem winning_candidate_vote_percentage:
  let total_members := 1600
  let votes_cast := 525
  let percentage_winning_votes := 0.60
  let votes_received := percentage_winning_votes * votes_cast
  let percentage_total_membership := (votes_received / total_members) * 100
  percentage_total_membership = 19.69 :=
by
  let total_members := 1600
  let votes_cast := 525
  let percentage_winning_votes := 0.60
  let votes_received := percentage_winning_votes * votes_cast
  let percentage_total_membership := (votes_received / total_members) * 100
  sorry

end winning_candidate_vote_percentage_l6_6858


namespace polynomial_factorization_l6_6385

theorem polynomial_factorization : 
  (x : ℤ) → (x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1)) := 
by
  sorry

end polynomial_factorization_l6_6385


namespace rent_change_percent_l6_6245

open Real

noncomputable def elaine_earnings_last_year (E : ℝ) : ℝ :=
E

noncomputable def elaine_rent_last_year (E : ℝ) : ℝ :=
0.2 * E

noncomputable def elaine_earnings_this_year (E : ℝ) : ℝ :=
1.15 * E

noncomputable def elaine_rent_this_year (E : ℝ) : ℝ :=
0.25 * (1.15 * E)

noncomputable def rent_percentage_change (E : ℝ) : ℝ :=
(elaine_rent_this_year E) / (elaine_rent_last_year E) * 100

theorem rent_change_percent (E : ℝ) :
  rent_percentage_change E = 143.75 :=
by
  sorry

end rent_change_percent_l6_6245


namespace recurring_decimal_to_fraction_l6_6032

theorem recurring_decimal_to_fraction
  (h : 0.\overline{02} = (2 : ℝ) / 99) :
  2.\overline{06} = 68 / 33 := by
  sorry

end recurring_decimal_to_fraction_l6_6032


namespace macey_saving_weeks_l6_6274

-- Definitions for conditions
def shirt_cost : ℝ := 3
def amount_saved : ℝ := 1.5
def weekly_saving : ℝ := 0.5

-- Statement of the proof problem
theorem macey_saving_weeks : (shirt_cost - amount_saved) / weekly_saving = 3 := by
  sorry

end macey_saving_weeks_l6_6274


namespace prob_all_meet_standard_prob_at_least_one_meets_standard_l6_6854

def P_meeting_standard_A := 0.8
def P_meeting_standard_B := 0.6
def P_meeting_standard_C := 0.5

theorem prob_all_meet_standard :
  (P_meeting_standard_A * P_meeting_standard_B * P_meeting_standard_C) = 0.24 :=
by
  sorry

theorem prob_at_least_one_meets_standard :
  (1 - ((1 - P_meeting_standard_A) * (1 - P_meeting_standard_B) * (1 - P_meeting_standard_C))) = 0.96 :=
by
  sorry

end prob_all_meet_standard_prob_at_least_one_meets_standard_l6_6854


namespace equilateral_triangle_inequality_l6_6105

-- Define a triangle with sides a, b, c and area S
noncomputable def triangle (a b c S : ℝ) : Prop :=
  ∃ (α β γ : ℝ) (hα : 0 < α) (hβ : 0 < β) (hγ : 0 < γ) (h_sum : α + β + γ = π),
    S = 0.5 * a * b * sin α ∧
    S = 0.5 * b * c * sin β ∧
    S = 0.5 * c * a * sin γ ∧
    a = (b^2 + c^2 - 2*b*c*cos γ)^(1/2) ∧
    b = (a^2 + c^2 - 2*a*c*cos β)^(1/2) ∧
    c = (a^2 + b^2 - 2*a*b*cos α)^(1/2)

-- Prove the inequality and equality condition if and only if the triangle is equilateral
theorem equilateral_triangle_inequality (a b c S : ℝ) (h_tri : triangle a b c S) :
  a^2 + b^2 + c^2 ≥ 4 * sqrt 3 * S ∧ (a = b ∧ b = c → a^2 + b^2 + c^2 = 4 * sqrt 3 * S) :=
by
  sorry

end equilateral_triangle_inequality_l6_6105


namespace pentagon_AE_AF_EQ_BE_l6_6234

-- Definitions and assumptions
variables (A B C D E F : Type) [RegularPentagon A B C D E]
variable {x : ℝ}  -- side length of the pentagon
variable {AF BE AE F_ext : ℝ}

-- Given conditions
axiom perpendicular_C_CD_AB : Perpendicular (LineThrough C cd) (LineThrough F ab)
axiom meets_F_AB : (LineThrough C cd) ∩ (LineThrough F ab) = F

-- Regular Pentagon properties (could be added according to mathematical structure and necessary imports)
axiom regular_pentagon :
  ∀ A B C D E, RegularPentagon A B C D E → ∑ (sides lengths) = 5 * x 

-- Main theorem statement
theorem pentagon_AE_AF_EQ_BE :
  AE + AF = BE :=
sorry

end pentagon_AE_AF_EQ_BE_l6_6234


namespace g_9_eq_64_l6_6716

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g(x + y) = g(x) * g(y)

axiom g_3_eq_4 : g(3) = 4

theorem g_9_eq_64 : g(9) = 64 := by
  sorry

end g_9_eq_64_l6_6716


namespace area_of_triangle_with_perpendicular_medians_l6_6217

theorem area_of_triangle_with_perpendicular_medians :
  ∀ (T : Type) [triangle T] (m₁ m₂ : median T),
    m₁.length = 18 →
    m₂.length = 24 →
    m₁ ⊥ m₂ →
    area T = 288 :=
by 
  sorry

end area_of_triangle_with_perpendicular_medians_l6_6217


namespace initial_mean_correct_l6_6736

theorem initial_mean_correct (n : ℕ) (correct_val incorrect_val : ℝ) (corrected_mean initial_mean : ℝ) 
  (h1 : n = 50) 
  (h2 : correct_val = 48) 
  (h3 : incorrect_val = 21) 
  (h4 : corrected_mean = 36.54) 
  (h5 : initial_mean * n + (correct_val - incorrect_val) = corrected_mean * n) : 
  initial_mean = 36 := 
by
  rw [← h1, ← h2, ← h3, ← h4] at h5
  rw [← h1]
  rw [mul_comm initial_mean 50] at h5
  linarith


end initial_mean_correct_l6_6736


namespace Δ_unbounded_l6_6909

noncomputable def f : ℕ → ℕ := sorry

def Δ (m n : ℕ) : ℤ := 
  (Nat.iterate f (f n) m : ℕ) - (Nat.iterate f (f m) n : ℕ)

theorem Δ_unbounded (hΔ_nonzero : ∀ m n : ℕ, m ≠ n → Δ m n ≠ 0) : 
  ∀ C : ℕ, ∃ m n : ℕ, |Δ m n| > C :=
sorry

end Δ_unbounded_l6_6909


namespace black_pigeons_count_l6_6830

theorem black_pigeons_count
    (total_pigeons : ℕ)
    (half_black : total_pigeons / 2)
    (percent_male : ℕ → Prop)
    (eighty_percent_female : ∀ b, percent_male b → b * 4 = total_pigeons / 2)
    (male_black_pigeons female_black_pigeons: ℕ) :
    total_pigeons = 70 →
    half_black = 35 →
    percent_male 20 →
    male_black_pigeons = 7 →
    female_black_pigeons = half_black - male_black_pigeons →
    female_black_pigeons - male_black_pigeons = 21 :=
by
  intros _ _ _ _ _ _ _ _ _ _ _
  sorry

end black_pigeons_count_l6_6830


namespace decimal_expansion_2023rd_digit_l6_6893

theorem decimal_expansion_2023rd_digit :
  let seq := "894736842105263157".to_list
  (seq.get (2023 % 18) = '3') :=
by
  sorry

end decimal_expansion_2023rd_digit_l6_6893


namespace cone_volume_l6_6556

theorem cone_volume (r h l : ℝ) (π : ℝ) (r_eq : r = 1) (lateral_surface_unfolds : l = π) (h_eq : h = sqrt 3) :
  (1/3) * π * r * r * h = (sqrt 3 / 3) * π :=
by
  rw [r_eq, h_eq]
  sorry

end cone_volume_l6_6556


namespace number_of_m_values_l6_6142

open Real

noncomputable def point := ℝ × ℝ

def O : point := (0, 0)
def A : point := (4, -1)

def distance_to_line (p : point) (m : ℝ) : ℝ :=
  let (x0, y0) := p in
  let num := abs (m * x0 + m^2 * y0 + 6) in
  let denom := sqrt (m^2 + (m^2)^2) in
  num / denom

theorem number_of_m_values : (set_of (λ m : ℝ, distance_to_line O m = distance_to_line A m)).finite.to_finset.card = 1 := by
  sorry

end number_of_m_values_l6_6142


namespace pencils_multiple_of_40_l6_6423

theorem pencils_multiple_of_40 :
  ∃ n : ℕ, 640 % n = 0 ∧ n ≤ 40 → ∃ m : ℕ, 40 * m = 40 * n :=
by
  sorry

end pencils_multiple_of_40_l6_6423


namespace angle_BAC_75_l6_6236

theorem angle_BAC_75 (A B C X Y D : Type*)
  [triangle : Triangle ABC] (h1 : AX = XY) (h2 : XY = YB)
  (h3 : YB = BD) (h4 : BD = DC)
  (h_angle_ABD : ∠ABD = 150) :
  ∠BAC = 7.5 := 
sorry

end angle_BAC_75_l6_6236


namespace integral_absolute_value_squared_sub_one_l6_6921

open Real

theorem integral_absolute_value_squared_sub_one :
  (∫ x in 0..3, |x^2 - 1|) = 22 / 3 :=
by sorry

end integral_absolute_value_squared_sub_one_l6_6921


namespace clock_angle_at_7_oclock_l6_6782

theorem clock_angle_at_7_oclock : 
  ∀ (hour_angle minute_angle : ℝ), 
    (12 : ℝ) * (30 : ℝ) = 360 →
    (7 : ℝ) * (30 : ℝ) = 210 →
    (210 : ℝ) > 180 →
    (360 : ℝ) - (210 : ℝ) = 150 →
    hour_angle = 7 * 30 →
    minute_angle = 0 →
    min (abs (hour_angle - minute_angle)) (abs ((360 - hour_angle) - minute_angle)) = 150 := by
  sorry

end clock_angle_at_7_oclock_l6_6782


namespace relationship_y1_y2_y3_l6_6668

theorem relationship_y1_y2_y3 (c y1 y2 y3 : ℝ) :
  (y1 = (-(1^2) + 2 * 1 + c))
  ∧ (y2 = (-(2^2) + 2 * 2 + c))
  ∧ (y3 = (-(5^2) + 2 * 5 + c))
  → (y2 > y1 ∧ y1 > y3) :=
by
  intro h
  sorry

end relationship_y1_y2_y3_l6_6668


namespace solve_ellipse_problems_l6_6620

noncomputable def ellipse_eccentricity_problem : Prop :=
  ∃ (a b c : ℝ),
    (a > 0) ∧
    (b > 0) ∧
    (a > b) ∧
    (c = a * (Real.sqrt 2 / 2)) ∧
    (c + a^2 / c = 3) ∧
    (∀ x y : ℝ, (x^2 / a^2) + (y^2 / b^2) = 1 → (x^2 / 2) + y^2 = 1)

noncomputable def line_passing_F_problem : Prop :=
  ∃ (k : ℝ),
    k ≠ 0 ∧
    (∀ x y : ℝ, (x - 1) = k * (y) ∧ (y + k / (1 + 2 * k^2)) = -(1 / k) * (x - 2 * k^2 / (1 + 2 * k^2))
    ∧ ∃ AB : ℝ, AB = 2 * (Real.sqrt 2 / (1 + 2 * k^2) * Real.sqrt (1 + k^2))
    ∧ ∀ x y : ℝ, (y = x - 1 ∨ y = -x + 1))

theorem solve_ellipse_problems:
  ellipse_eccentricity_problem ∧ line_passing_F_problem :=
begin
  split;
  -- sorry proofs
  sorry,
  sorry
end

end solve_ellipse_problems_l6_6620


namespace probabilities_X_l6_6814

noncomputable def probability_X_zero (A B C D E F : Type) : ℝ :=
  let p_red := (1/2 : ℝ) in
  let p_not_red := 1 - p_red in
  let p_AF_FE_red := p_red * p_red in
  let p_AB_black := p_not_red in
  let p_other := 1 - (p_AF_FE_red + (p_AB_black * (1 - p_AF_FE_red))) in
  p_AB_black + p_other

noncomputable def probability_X_two (A B C D E F : Type) : ℝ :=
  let p_red := (1/2 : ℝ) in
  let p_not_red := 1 - p_red in
  let p_AF_FE_red := p_red * p_red in
  let p_AB_BE_red_not_AF_FE :=
    (p_red * p_red) * (1 - p_AF_FE_red - (p_not_red * (1 - p_AF_FE_red))) in
  p_AF_FE_red + p_AB_BE_red_not_AF_FE

noncomputable def probability_X_four (A B C D E F : Type) : ℝ :=
  let p_red := (1/2 : ℝ) in
  let p_not_red := 1 - p_red in
  let p_AB_BC_CD_DE_red_not_previous :=
    (p_red ^ 4) * (1 - (p_red * p_red) - ((1 - p_red) * (1 - (p_red * p_red))) - 
    (p_red * p_red * (1 - (p_red * p_red) - ((1 - p_red) * (1 - (p_red * p_red))))))
  in p_AB_BC_CD_DE_red_not_previous

theorem probabilities_X (A B C D E F : Type) :
  probability_X_zero A B C D E F = 69 / 128 ∧
  probability_X_two A B C D E F = 7 / 16 ∧
  probability_X_four A B C D E F = 3 / 128 :=
by
  sorry

end probabilities_X_l6_6814


namespace quadrilateral_angle_measure_l6_6672

noncomputable def quadrilateral_properties
  (EF FG GH : ℝ) 
  (angle_EFG angle_FGH : ℝ) 
  (h1 : EF = FG) 
  (h2 : FG = GH) 
  (h3 : angle_EFG = 60) 
  (h4 : angle_FGH = 160) : 
  Prop :=
  angle_EHG = 50

theorem quadrilateral_angle_measure 
  (EF FG GH : ℝ) 
  (angle_EFG angle_FGH : ℝ) 
  (h1 : EF = FG) 
  (h2 : FG = GH) 
  (h3 : angle_EFG = 60) 
  (h4 : angle_FGH = 160) :
  quadrilateral_properties EF FG GH angle_EFG angle_FGH := 
sorry

end quadrilateral_angle_measure_l6_6672


namespace solve_inequality_l6_6688

theorem solve_inequality (x : ℝ) :
  (x^2 - 4 * x - 12) / (x - 3) < 0 ↔ (-2 < x ∧ x < 3) ∨ (3 < x ∧ x < 6) := by
  sorry

end solve_inequality_l6_6688


namespace area_of_given_triangle_l6_6398

noncomputable def area_of_triangle (a b c : ℝ) : ℝ :=
  let s := (a + b + c) / 2
  in Real.sqrt (s * (s - a) * (s - b) * (s - c))

theorem area_of_given_triangle : area_of_triangle 30 28 10 = 140 := by
  sorry

end area_of_given_triangle_l6_6398


namespace rect_length_eq_3_l6_6603

variable (sq_side : ℝ)
variable (rect_width rect_length : ℝ)

axiom sq_side_eq_3 : sq_side = 3
axiom rect_width_eq_3 : rect_width = 3
axiom areas_eq : sq_side * sq_side = rect_width * rect_length

theorem rect_length_eq_3 : rect_length = 3 :=
by
  have h1 : sq_side * sq_side = 9 := by rw [sq_side_eq_3]; ring
  have h2 : 9 = rect_width * rect_length := by rw [←h1, areas_eq]
  have h3 : rect_width * rect_length = rect_width * 3 := by rw [rect_width_eq_3]
  linarith

end rect_length_eq_3_l6_6603


namespace find_x_l6_6889

noncomputable def log (b x : ℝ) : ℝ := Real.log x / Real.log b

theorem find_x (x : ℝ) (h₁ : log x 16 = log 4 256) : x = 2 := by
  sorry

end find_x_l6_6889


namespace recurring_decimal_to_fraction_l6_6025

theorem recurring_decimal_to_fraction
  (h : 0.\overline{02} = (2 : ℝ) / 99) :
  2.\overline{06} = 68 / 33 := by
  sorry

end recurring_decimal_to_fraction_l6_6025


namespace decimal_to_fraction_l6_6016

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l6_6016


namespace number_of_teams_l6_6746

-- Define the conditions
def math_club_girls : ℕ := 4
def math_club_boys : ℕ := 7
def team_girls : ℕ := 3
def team_boys : ℕ := 3

-- Compute the number of ways to choose 3 girls from 4 girls
def choose_comb_girls : ℕ := Nat.choose math_club_girls team_girls

-- Compute the number of ways to choose 3 boys from 7 boys
def choose_comb_boys : ℕ := Nat.choose math_club_boys team_boys

-- Formulate the goal statement
theorem number_of_teams : choose_comb_girls * choose_comb_boys = 140 := by
  sorry

end number_of_teams_l6_6746


namespace square_side_length_l6_6409

/-- Define OPEN as a square and T a point on side NO
    such that the areas of triangles TOP and TEN are 
    respectively 62 and 10. Prove that the side length 
    of the square is 12. -/
theorem square_side_length (s x y : ℝ) (T : x + y = s)
    (h1 : 0 < s) (h2 : 0 < x) (h3 : 0 < y)
    (a1 : 1 / 2 * x * s = 62)
    (a2 : 1 / 2 * y * s = 10) :
    s = 12 :=
by
    sorry

end square_side_length_l6_6409


namespace decimal_to_fraction_l6_6020

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l6_6020


namespace hyperbola_equation_l6_6730

theorem hyperbola_equation (a c : ℝ) (e : ℝ) (h1 : 2 * a = 8) (h2 : c / a = 5 / 4) :
  (\exists b : ℝ, (b^2 = c^2 - a^2) ∧ (b = 3) ∧ 
  ( ∀ x y : ℝ, ((x^2 / (a^2) - y^2 / (b^2) = 1) → (x^2 / 16 - y^2 / 9 = 1)))) :=
by
  use 3
  sorry

end hyperbola_equation_l6_6730


namespace pentagon_triangle_area_ratio_l6_6836

theorem pentagon_triangle_area_ratio (b : ℝ) (hb : b > 0) :
  let area_triangle := 1 / 2 * b^2,
      area_rectangle := b * (2 * b),
      total_area := area_rectangle + area_triangle in
  area_triangle / total_area = 1 / 5 :=
by
  sorry

end pentagon_triangle_area_ratio_l6_6836


namespace intersection_points_product_one_l6_6111

def curveC (x y : ℝ) : Prop := (x - 2) ^ 2 + (y - 1) ^ 2 = 4

def lineL (x y t : ℝ) : Prop :=
  x = 3 + (sqrt 3 / 2) * t ∧ y = sqrt 3 + (1 / 2) * t

def polarCurveC (ρ θ : ℝ) : Prop :=
  ρ ^ 2 - 4 * ρ * cos θ - 2 * ρ * sin θ + 1 = 0

def polarLineL (θ : ℝ) : Prop := θ = π / 6

theorem intersection_points_product_one :
  ∀ (ρ₁ ρ₂ : ℝ), polarLineL (π / 6) →
  polarCurveC ρ₁ (π / 6) ∧ polarCurveC ρ₂ (π / 6) →
  |ρ₁ * ρ₂| = 1 :=
by
  sorry

end intersection_points_product_one_l6_6111


namespace discount_difference_l6_6847

theorem discount_difference :
  ∀ (original_price : ℝ),
  let initial_discount := 0.40
  let subsequent_discount := 0.25
  let claimed_discount := 0.60
  let actual_discount := 1 - (1 - initial_discount) * (1 - subsequent_discount)
  let difference := claimed_discount - actual_discount
  actual_discount = 0.55 ∧ difference = 0.05
:= by
  sorry

end discount_difference_l6_6847


namespace repeating_decimal_fraction_l6_6065

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l6_6065


namespace sphere_points_l6_6422

theorem sphere_points :
  ∃ T : ℕ, ∀ S : ℕ, S ≤ 0.72 * T → (S * (S - 1) / 2 = 14) → T = 10 :=
by
  sorry

end sphere_points_l6_6422


namespace sin_neg_five_sixths_pi_l6_6882

theorem sin_neg_five_sixths_pi : Real.sin (- 5 / 6 * Real.pi) = -1 / 2 :=
sorry

end sin_neg_five_sixths_pi_l6_6882


namespace repeating_decimal_fraction_l6_6069

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l6_6069


namespace dish_volume_l6_6704

def hexagon_side_length : ℕ := 1

def volume_of_dish (s : ℕ) : ℚ := 
  let h := (Real.sqrt 3) / 2
  let A1 := (s:ℚ) ^ 2
  let A2 := (2:ℚ) ^ 2
  (1 / 3) * h * (A1 + A2 + Real.sqrt (A1 * A2))

theorem dish_volume (m n : ℕ) (h1 : volume_of_dish hexagon_side_length = Real.sqrt (m / n)) : 
  m = 49 ∧ n = 12 → m + n = 61 :=
by
  intros H
  cases H with H1 H2
  rw [H1, H2]
  norm_num
  sorry

end dish_volume_l6_6704


namespace closest_to_fraction_l6_6883

theorem closest_to_fraction : 
  (closest : ℕ) ∈ {3000, 3100, 3200, 3300, 3400} ∧ 
  (∀ other ∈ {3000, 3100, 3200, 3300, 3400}, 
    abs (3111.7647 - 3100) ≤ abs (3111.7647 - other)) := 
begin
  sorry
end

end closest_to_fraction_l6_6883


namespace relay_count_l6_6759

theorem relay_count (A B C D: Prop) 
    (can_communicate: ∀ X Y: Prop, X ≠ Y → X ↔ Y) 
    (random_send: ∀ X: Prop, X = A ∨ X = B ∨ X = C ∨ X = D ) 
    (not_simultaneous: ∀ X Y Z: Prop, X ≠ Y → X ≠ Z → Y ≠ Z → ¬ (X ∧ Y ∧ Z)) 
    (all_receive: ∃ count, count = 3 ∧ ∃ msg, A ∧ B ∧ C ∧ D) : 
    ∃ ways, ways = 16 :=
by
  sorry

end relay_count_l6_6759


namespace f_at_7_l6_6825

noncomputable def f : ℝ → ℝ :=
  λ v, ( ((v + 3) / 2)^2 + 2 * ((v + 3) / 2) + 2 )

theorem f_at_7 : f 7 = 37 := by
  sorry

end f_at_7_l6_6825


namespace maximum_percent_increase_from_Mar_to_Sep_l6_6334

noncomputable def max_percent_increase : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ 
  | rise1, drop1, gain1, loss1, climb1, decline1 => sorry

theorem maximum_percent_increase_from_Mar_to_Sep (rise1 rise2 : ℝ) (drop1 drop2 : ℝ) (gain1 gain2 : ℝ)
  (loss1 loss2 : ℝ) (climb1 climb2 : ℝ) (decline1 decline2 : ℝ) (max_increase : ℝ) :
  (0.4 ≤ rise1 ∧ rise1 ≤ 0.6) →
  (0.15 ≤ drop1 ∧ drop1 ≤ 0.25) →
  (0.45 ≤ gain1 ∧ gain1 ≤ 0.55) →
  (0.05 ≤ loss1 ∧ loss1 ≤ 0.15) →
  (0.25 ≤ climb1 ∧ climb1 ≤ 0.35) →
  (0.03 ≤ decline1 ∧ decline1 ≤ 0.07) →
  (max_increase = 1) → 
  (max_percent_increase rise1 drop1 gain1 loss1 climb1 decline1 : ℕ) = 94 :=
sorry

end maximum_percent_increase_from_Mar_to_Sep_l6_6334


namespace entry_exit_equivalence_l6_6465

theorem entry_exit_equivalence (entries exits : List ℕ) 
  (h_entry_exit : ∀ n, n ∈ entries ↔ n ∈ exits) :
  entries ~ exits
:=
  sorry

end entry_exit_equivalence_l6_6465


namespace sammy_remaining_problems_l6_6677

variable (total_problems : Nat)
variable (fraction_problems : Nat) (decimal_problems : Nat) (multiplication_problems : Nat) (division_problems : Nat)
variable (completed_fraction_problems : Nat) (completed_decimal_problems : Nat)
variable (completed_multiplication_problems : Nat) (completed_division_problems : Nat)
variable (remaining_problems : Nat)

theorem sammy_remaining_problems
  (h₁ : total_problems = 115)
  (h₂ : fraction_problems = 35)
  (h₃ : decimal_problems = 40)
  (h₄ : multiplication_problems = 20)
  (h₅ : division_problems = 20)
  (h₆ : completed_fraction_problems = 11)
  (h₇ : completed_decimal_problems = 17)
  (h₈ : completed_multiplication_problems = 9)
  (h₉ : completed_division_problems = 5)
  (h₁₀ : remaining_problems =
    fraction_problems - completed_fraction_problems +
    decimal_problems - completed_decimal_problems +
    multiplication_problems - completed_multiplication_problems +
    division_problems - completed_division_problems) :
  remaining_problems = 73 :=
  by
    -- proof to be written
    sorry

end sammy_remaining_problems_l6_6677


namespace no_real_solution_for_exponential_equation_l6_6515

theorem no_real_solution_for_exponential_equation :
  ¬ ∃ x : ℝ, 2^(x^2 - 5*x + 2) = 8^(x - 5) :=
by
  sorry

end no_real_solution_for_exponential_equation_l6_6515


namespace sum_of_f1_possible_values_l6_6254

noncomputable def f : ℝ → ℝ := sorry

theorem sum_of_f1_possible_values :
  (∀ (x y : ℝ), f (f (x - y)) = f x * f y - f x + f y - 2 * x * y) →
  (f 1 = -1) := sorry

end sum_of_f1_possible_values_l6_6254


namespace sum_of_digits_of_d_l6_6285

theorem sum_of_digits_of_d 
  (d : ℕ) 
  (h1 : ∀ (x : ℕ), x * (12 / 8) = (3 / 2) * x) 
  (h2 : (3 / 2) * d - 72 = d) 
  : (d = 144) ∧ (sum_of_digits 144 = 9) := by
  sorry

def sum_of_digits (n : ℕ) : ℕ := sorry

end sum_of_digits_of_d_l6_6285


namespace triangle_properties_l6_6542

noncomputable def point := (ℝ × ℝ)

def A : point := (7, 0)
def B : point := (3, 4)
def C : point := (2, -3)

def ab_line (p : point) : Prop := p.1 + p.2 - 7 = 0
def cd_line (p : point) : Prop := p.1 - p.2 - 5 = 0
def be_line (p : point) : Prop := 3 * p.1 + p.2 - 13 = 0

def length_AB : ℝ := 4 * Real.sqrt 2
def length_CD : ℝ := 4 * Real.sqrt 2
def length_BE : ℝ := 16 * Real.sqrt 10 / 9

def angle_A : ℝ := Real.arctan (4 / 3)

def is_obtuse : Prop := 
    let AB2 := (7 - 3) ^ 2 + (4 - 0) ^ 2
    let AC2 := (7 - 2) ^ 2 + (0 + 3) ^ 2
    let BC2 := (3 - 2) ^ 2 + (4 + 3) ^ 2
    AB2 + AC2 < BC2

def triangle_inequalities (p : point) : Prop := 
    p.1 + p.2 - 7 <= 0 ∧ 
    3 * p.1 - 5 * p.2 - 21 <= 0 ∧ 
    7 * p.1 - p.2 - 17 >= 0

theorem triangle_properties :
  (∀ p : point, ab_line p ↔ p.1 + p.2 - 7 = 0) ∧
  (∀ p : point, cd_line p ↔ p.1 - p.2 - 5 = 0) ∧
  (∀ p : point, be_line p ↔ 3 * p.1 + p.2 - 13 = 0) ∧
  length_AB = 4 * Real.sqrt 2 ∧
  length_CD = 4 * Real.sqrt 2 ∧
  length_BE = 16 * Real.sqrt 10 / 9 ∧
  angle_A = Real.arctan (4 / 3) ∧
  is_obtuse ∧
  ∀ p : point, triangle_inequalities p
:= sorry

end triangle_properties_l6_6542


namespace plains_routes_count_l6_6193

-- Defining the total number of cities and the number of cities in each region
def total_cities : Nat := 100
def mountainous_cities : Nat := 30
def plains_cities : Nat := total_cities - mountainous_cities

-- Defining the number of routes established each year and over three years
def routes_per_year : Nat := 50
def total_routes : Nat := routes_per_year * 3

-- Defining the number of routes connecting pairs of mountainous cities
def mountainous_routes : Nat := 21

-- The statement to prove the number of routes connecting pairs of plains cities
theorem plains_routes_count :
  plains_cities = 70 →
  total_routes = 150 →
  mountainous_routes = 21 →
  3 * mountainous_cities - 2 * mountainous_routes = 48 →
  3 * plains_cities - 48 = 162 →
  81 = 81 := sorry

end plains_routes_count_l6_6193


namespace sector_area_l6_6993

noncomputable def l : ℝ := 4
noncomputable def θ : ℝ := 2
noncomputable def r : ℝ := l / θ

theorem sector_area :
  (1 / 2) * l * r = 4 :=
by
  -- Proof goes here
  sorry

end sector_area_l6_6993


namespace count_four_digit_palindrome_n_and_2n_palindromes_l6_6878

theorem count_four_digit_palindrome_n_and_2n_palindromes :
  ∃ (n_set : Finset ℕ), (∀ n ∈ n_set, 
    (∃ a b : ℕ, 1 ≤ a ∧ a ≤ 4 ∧ 0 ≤ b ∧ b ≤ 4 ∧ n = 1001 * a + 110 * b ∧ 
      let m := 2002 * a + 220 * b in 
      1000 ≤ m ∧ m < 10000 ∧ (reverse_digits m = m)) ∧ 
    1000 ≤ n ∧ n < 10000 ∧ (reverse_digits n = n)) ∧ n_set.card = 20 :=
sorry

/--
Helper function to reverse a number's digits (given a natural number, returns its digits reversed).
-/
noncomputable def reverse_digits (n : ℕ) : ℕ :=
let digits := n.digits 10 in 
digits.foldl (λ acc d, acc * 10 + d) 0

end count_four_digit_palindrome_n_and_2n_palindromes_l6_6878


namespace find_angle_A_l6_6238

theorem find_angle_A (a b c A : ℝ) (h1 : b = c) (h2 : a^2 = 2 * b^2 * (1 - Real.sin A)) : 
  A = Real.pi / 4 :=
by
  sorry

end find_angle_A_l6_6238


namespace prime_divisibility_l6_6153

theorem prime_divisibility (a b : ℕ) (ha_prime : Nat.Prime a) (hb_prime : Nat.Prime b) (ha_gt7 : a > 7) (hb_gt7 : b > 7) :
  290304 ∣ (a^2 - 1) * (b^2 - 1) * (a^6 - b^6) := 
by
  sorry

end prime_divisibility_l6_6153


namespace math_problem_l6_6990

variable {x y : ℝ}
variable (hx : x ≠ 0) (hy : y ≠ 0) (h := y^2 - 1 / x^2 ≠ 0) (h₁ := x^2 * y^2 ≠ 1)

theorem math_problem :
  (x^2 - 1 / y^2) / (y^2 - 1 / x^2) = x^2 / y^2 :=
sorry

end math_problem_l6_6990


namespace find_numbers_l6_6750

-- Given conditions
variables (x y : ℝ)
hypothesis h1 : x + y = 18
hypothesis h2 : x - y = 6

-- Goal to prove
theorem find_numbers (h1 : x + y = 18) (h2 : x - y = 6) : x = 12 ∧ y = 6 :=
by
  sorry

end find_numbers_l6_6750


namespace distance_center_to_line_l6_6621

-- Define the parametric equations of the circle
def circle_param_eq_x (θ : ℝ) : ℝ := 2 * Real.cos θ
def circle_param_eq_y (θ : ℝ) : ℝ := 2 * Real.sin θ + 2

-- Define the equation of the line
def line_eq (x y : ℝ) : Prop := x + y - 6 = 0

-- Define the center of the circle based on its standard equation
def circle_center : (ℝ × ℝ) := (0, 2)

-- Define the formula for distance from a point to a line
noncomputable def distance_point_to_line (x1 y1 A B C : ℝ) : ℝ :=
  (|A * x1 + B * y1 + C|) / Real.sqrt (A^2 + B^2)

-- Prove the required distance
theorem distance_center_to_line : 
  distance_point_to_line 0 2 1 1 (-6) = 2 * Real.sqrt 2 :=
by
  sorry

end distance_center_to_line_l6_6621


namespace tangent_line_equation_l6_6158

theorem tangent_line_equation (n : ℝ) (f : ℝ → ℝ)
  (h1 : f(2) = 8)
  (h2 : ∀ x, f(x) = x^n)
  (h3 : ∀ x, deriv f x = 3 * x^2)
  (tangent_point : (2, 8)) :
  ∃ l : ℝ → ℝ, ∀ x, l(x) = 12 * x - 16 ∧ (l(x) = f(x)) := 
sorry

end tangent_line_equation_l6_6158


namespace ellipse_equation_perimeter_of_triangle_constant_l6_6107

-- Solution for Question 1
theorem ellipse_equation :
  ∃ m n : ℝ, m > 0 ∧ n > 0 ∧ mx^2 + ny^2 = 1 ∧
    (mx * (-2)^2 + ny * 0^2 = 1 ∧ mx * 2^2 + ny * 0^2 = 1 ∧ mx * 1^2 + ny * (3 / 2)^2 = 1) :=
sorry

-- Solution for Question 2
theorem perimeter_of_triangle_constant (a : ℝ) (F1 F2 M N : ℝ × ℝ) :
  ∃ a : ℝ, a = 2 ∧
    let F1M := dist F1 M,
        F2M := dist F2 M,
        F1N := dist F1 N,
        F2N := dist F2 N
    in F1M + F2M = 2 * a ∧ F1N + F2N = 2 * a ∧
       (F1M + F2M + F1N + F2N = 4 * a) :=
sorry

end ellipse_equation_perimeter_of_triangle_constant_l6_6107


namespace min_distance_from_start_after_9_minutes_l6_6449

noncomputable def robot_min_distance : ℝ :=
  let movement_per_minute := 10
  sorry

theorem min_distance_from_start_after_9_minutes :
  robot_min_distance = 10 :=
sorry

end min_distance_from_start_after_9_minutes_l6_6449


namespace lines_and_planes_example_l6_6270

-- Define the entities: lines and planes
def line := ℝ → ℝ³
def plane := ℝ² → ℝ³

-- Define the conditions: perpendicular and parallel
def perp_to_plane (l : line) (α : plane) : Prop := ∀ p : ℝ, ∀ q : ℝ², l p ∈ α q → false
def parallel_to_plane (l : line) (α : plane) : Prop := ∃ q : ℝ², ∀ p : ℝ, l p ∈ α q

-- Define perpendicular between two lines in space
def perp (l1 l2 : line) : Prop := ∃ p1 p2 : ℝ, l1 p1 ⟂ l2 p2

-- Given conditions
variable (l1 l2 : line)
variable (α : plane)
variable (h1 : perp_to_plane l1 α)
variable (h2 : parallel_to_plane l2 α)

-- Goal to prove
theorem lines_and_planes_example : perp l1 l2 :=
sorry

end lines_and_planes_example_l6_6270


namespace equation_solution_inequality_solution_l6_6687

noncomputable def factorial (n : ℕ) : ℕ :=
  if h : n = 0 then 1 else n * factorial (n - 1)

def permutations (n m : ℕ) : ℕ :=
  factorial n / factorial (n - m)

def solveEquation (x : ℕ) : Prop :=
  3 * permutations 8 x = 4 * permutations 9 (x - 1)

theorem equation_solution :
  solveEquation 6 :=
sorry

def solveInequality (x : ℕ) : Prop :=
  (permutations (x - 2) 2 + x ≥ 2) ∧ (x ≥ 4) ∧ (x > 0)

theorem inequality_solution :
  ∀ x : ℕ, solveInequality x ↔ (x ≥ 4) :=
sorry

end equation_solution_inequality_solution_l6_6687


namespace range_of_y_l6_6487

-- Define the function y = sin x + cos x - |sin x - cos x|
def y (x : ℝ) : ℝ := sin x + cos x - abs (sin x - cos x)

-- Define the interval [0, 2π]
def interval : set ℝ := {x | 0 ≤ x ∧ x ≤ 2 * real.pi}

-- State the theorem that the range of y(x) for x in the interval [0, 2π] is [-2, √2]
theorem range_of_y :
  set.range (λ x, y x) = {ρ | -2 ≤ ρ ∧ ρ ≤ real.sqrt 2} :=
sorry

end range_of_y_l6_6487


namespace jennifer_shares_sweets_with_friends_l6_6242

theorem jennifer_shares_sweets_with_friends :
  ∀ (green_sweets : ℕ) (blue_sweets : ℕ) (yellow_sweets : ℕ) (sweets_per_person : ℕ),
    green_sweets = 212 → blue_sweets = 310 → yellow_sweets = 502 → sweets_per_person = 256 →
    let total_sweets := green_sweets + blue_sweets + yellow_sweets in
    let total_people := total_sweets / sweets_per_person in
    let friends := total_people - 1 in
    friends = 3 :=
by
  intros green_sweets blue_sweets yellow_sweets sweets_per_person
  simp
  sorry

end jennifer_shares_sweets_with_friends_l6_6242


namespace integral_evaluation_integral_part1_integral_part2_combined_result_l6_6885

noncomputable def evaluate_integral : ℝ :=
  ∫ x in 1..3, (1 / x + sqrt(1 - (x - 2)^2))

theorem integral_evaluation :
  evaluate_integral = (∫ x in 1..3, (1 / x)) + (∫ x in 1..3, sqrt(1 - (x - 2)^2)) :=
by
  sorry

theorem integral_part1 :
  (∫ x in 1..3, (1 / x)) = Real.log 3 :=
by
  sorry

theorem integral_part2 :
  (∫ x in 1..3, sqrt(1 - (x - 2)^2)) = (π / 2) :=
by
  sorry

theorem combined_result :
  evaluate_integral = Real.log 3 + (π / 2) :=
by
  rw [integral_evaluation, integral_part1, integral_part2]
  sorry

end integral_evaluation_integral_part1_integral_part2_combined_result_l6_6885


namespace sum_F_eq_l6_6912

noncomputable def F (n : ℕ) : ℕ :=
2 * n

theorem sum_F_eq :
  (∑ n in Finset.range 2006 \ Finset.singleton 0 \ Finset.singleton 1, F (n + 2)) = 4032234 :=
by sorry

end sum_F_eq_l6_6912


namespace abc_sum_l6_6593

theorem abc_sum (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a + b + c = 13 * Real.sqrt 6 := 
sorry

end abc_sum_l6_6593


namespace least_positive_integer_divisibility_l6_6783

theorem least_positive_integer_divisibility :
  ∃ M : ℕ, 
  M % 6 = 5 ∧ 
  M % 8 = 7 ∧ 
  M % 9 = 8 ∧ 
  M % 11 = 10 ∧ 
  M % 12 = 11 ∧ 
  M % 13 = 12 ∧ 
  M = 10163 :=
begin
  sorry
end

end least_positive_integer_divisibility_l6_6783


namespace solve_system_l6_6892

theorem solve_system :
  {p : ℝ × ℝ | p.1^3 + p.2^3 = 19 ∧ p.1^2 + p.2^2 + 5 * p.1 + 5 * p.2 + p.1 * p.2 = 12} = {(3, -2), (-2, 3)} :=
sorry

end solve_system_l6_6892


namespace isosceles_trapezoid_area_l6_6358

theorem isosceles_trapezoid_area
  (side : ℝ) (base1 : ℝ) (base2 : ℝ) (h : ℝ)
  (hbases : base1 = 6 ∧ base2 = 12)
  (hside : side = 5)
  (hheight : h = 4)
  (height_correct : h^2 + (3:ℝ)^2 = side^2) :
  let area := (1/2) * (base1 + base2) * h in area = 36 := 
by
  sorry

end isosceles_trapezoid_area_l6_6358


namespace centroid_traced_area_l6_6269

theorem centroid_traced_area (A B C : Point) (diameter_AB : A.distance B = 36)
  (on_circle : ∀ (C : Point), C ≠ A → C ≠ B → ∃ (O : Point) (r : ℝ), (O.distance A = r) ∧ (O.distance B = r) ∧ (O.distance C = r)) :
  ∃ (area : ℝ), area = 36 * Real.pi := 
sorry

end centroid_traced_area_l6_6269


namespace value_of_f_at_3_l6_6987

def f (x : ℚ) : ℚ := (2 * x + 3) / (4 * x - 5)

theorem value_of_f_at_3 : f 3 = 9 / 7 := by
  sorry

end value_of_f_at_3_l6_6987


namespace turquoise_beads_count_l6_6414

-- Define the conditions
def num_beads_total : ℕ := 40
def num_amethyst : ℕ := 7
def num_amber : ℕ := 2 * num_amethyst

-- Define the main theorem to prove
theorem turquoise_beads_count :
  num_beads_total - (num_amethyst + num_amber) = 19 :=
by
  sorry

end turquoise_beads_count_l6_6414


namespace rectangle_tangent_to_circle_l6_6965

theorem rectangle_tangent_to_circle 
    (a b : ℝ) 
    (ha : a > 0) 
    (hb : b > 0) 
    (h : a > b) :
    (∀ P : ℝ × ℝ, (P.1 ^ 2 / a ^ 2) + (P.2 ^ 2 / b ^ 2) = 1 →
        ∃ Q : ℝ × ℝ × ℝ × ℝ, 
            ((Q.1 ^ 2 + Q.2 ^ 2 = 1) ∧ (Q.3 ^ 2 + Q.4 ^ 2 = 1)) ∧
            ((Q.1, Q.2) = P ∨ (Q.3, Q.4) = P) ∧ 
            (∃ R : ℝ × ℝ, R.1 ^ 2 + R.2 ^ 2 = 1)) ↔ 
    (1 / a ^ 2 + 1 / b ^ 2 = 1) :=
sorry

end rectangle_tangent_to_circle_l6_6965


namespace g_9_l6_6724

variable (g : ℝ → ℝ)

-- Conditions
axiom func_eq : ∀ x y : ℝ, g(x + y) = g(x) * g(y)
axiom g_3 : g 3 = 4

-- Theorem to prove
theorem g_9 : g 9 = 64 :=
by
  sorry

end g_9_l6_6724


namespace complement_intersection_nonempty_l6_6546

open Set

variable (R : Type) [LinearOrder R] [TopologicalSpace R] [OrderTopology R]

def A : Set ℝ := {x | x > -2}
def B : Set ℝ := {x | x^2 + 2 * x - 15 ≥ 0}

theorem complement_intersection_nonempty :
  (compl A ∩ compl B).nonempty :=
sorry

end complement_intersection_nonempty_l6_6546


namespace hannah_bought_two_sets_of_measuring_spoons_l6_6582

-- Definitions of conditions
def number_of_cookies_sold : ℕ := 40
def price_per_cookie : ℝ := 0.8
def number_of_cupcakes_sold : ℕ := 30
def price_per_cupcake : ℝ := 2.0
def cost_per_measuring_spoon_set : ℝ := 6.5
def remaining_money : ℝ := 79

-- Definition of total money made from selling cookies and cupcakes
def total_money_made : ℝ := (number_of_cookies_sold * price_per_cookie) + (number_of_cupcakes_sold * price_per_cupcake)

-- Definition of money spent on measuring spoons
def money_spent_on_measuring_spoons : ℝ := total_money_made - remaining_money

-- Theorem statement
theorem hannah_bought_two_sets_of_measuring_spoons :
  (money_spent_on_measuring_spoons / cost_per_measuring_spoon_set) = 2 := by
  sorry

end hannah_bought_two_sets_of_measuring_spoons_l6_6582


namespace hcf_of_two_numbers_l6_6602

theorem hcf_of_two_numbers (A B : ℕ) (h1 : Nat.lcm A B = 750) (h2 : A * B = 18750) : Nat.gcd A B = 25 :=
by
  sorry

end hcf_of_two_numbers_l6_6602


namespace solve_for_x_l6_6148

theorem solve_for_x (x : ℝ) (h : -3 * x - 12 = 8 * x + 5) : x = -17 / 11 :=
by
  sorry

end solve_for_x_l6_6148


namespace find_x_l6_6094

def δ (x : ℝ) : ℝ := 4 * x + 9

def φ (x : ℝ) : ℝ := 9 * x + 6

theorem find_x :
  (δ (φ x) = 10) → x = - (23 / 36) :=
by
  intro h,
  sorry

end find_x_l6_6094


namespace g_50_equals_zero_l6_6325

noncomputable def g : ℝ → ℝ := sorry

theorem g_50_equals_zero (h : ∀ (x y : ℝ), 0 < x → 0 < y → x * g y - y * g x = g ((x + y) / y)) : g 50 = 0 :=
sorry

end g_50_equals_zero_l6_6325


namespace factors_of_n_multiples_of_180_l6_6154

-- Define the given number n
def n : ℕ := 2^12 * 3^15 * 5^9

-- Define what it means to be a multiple of 180
def is_multiple_of_180 (m : ℕ) : Prop := ∃ (a b c : ℕ), m = 2^(2 + a) * 3^(2 + b) * 5^(1 + c)

-- Define the condition that factors of n must satisfy
def is_factor_of_n (m : ℕ) : Prop := m ∣ n

-- State the theorem we want to prove
theorem factors_of_n_multiples_of_180 : 
  (finset.filter (λ m, is_multiple_of_180 m) (finset.filter (λ m, is_factor_of_n m) (finset.range (n + 1)))).card = 1386 := 
sorry

end factors_of_n_multiples_of_180_l6_6154


namespace math_problem_inequality_l6_6096

variable (a b : ℝ)
variable (h1 : a > 0)
variable (h2 : b > 0)
variable (h3 : a + b = 1)

noncomputable def proof : Prop :=
  (a + 1/a)^2 + (b + 1/b)^2 ≥ 25 / 2

theorem math_problem_inequality (h1 : a > 0) (h2 : b > 0) (h3 : a + b = 1) : proof a b h1 h2 h3 :=
sorry

end math_problem_inequality_l6_6096


namespace abc_sum_eq_11sqrt6_l6_6591

theorem abc_sum_eq_11sqrt6 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 :=
sorry

end abc_sum_eq_11sqrt6_l6_6591


namespace sum_of_first_8_terms_l6_6117

def is_arithmetic_sequence (a : ℕ → ℕ) (d : ℕ) : Prop :=
∀ n, a (n + 1) = a n + d

def is_geometric_sequence (a : ℕ → ℕ) : Prop :=
a 2 ^ 2 = a 1 * a 4  -- note in Lean indexes are off by 1, so a2 is a 1, a5 is a 4

def sum_of_first_n_terms (a : ℕ → ℕ) (n : ℕ) : ℕ :=
n * a 1 + (n * (n - 1) * d) / 2

theorem sum_of_first_8_terms 
  (a : ℕ → ℕ) 
  (d : ℕ) 
  (h1 : is_arithmetic_sequence a d) 
  (h2 : a 1 = 1) 
  (h3 : d ≠ 0) 
  (h4 : is_geometric_sequence a) 
  : sum_of_first_n_terms a 8 = 64 :=
by
  sorry

end sum_of_first_8_terms_l6_6117


namespace total_absent_students_l6_6276

section StudentAbsence

variables {T : ℕ}
variable {A1 A2 A3 : ℕ}
variable {P1 : ℕ}

-- Given conditions
def total_students : ℕ := 280
def absent_third_day (T : ℕ) : ℕ := T / 7
def absent_second_day (A3 : ℕ) : ℕ := 2 * A3
def present_first_day (T A2 : ℕ) : ℕ := T - A2
def absent_first_day (T P1 : ℕ) : ℕ := T - P1

-- Total students absent over three days
def total_absent (A1 A2 A3 : ℕ) : ℕ := A1 + A2 + A3

-- Theorem statement
theorem total_absent_students :
  let T := total_students,
      A3 := absent_third_day T,
      A2 := absent_second_day A3,
      P1 := present_first_day T A2,
      A1 := absent_first_day T P1 in
  total_absent A1 A2 A3 = 200 :=
by
  -- Place the proof steps here
  sorry

end StudentAbsence

end total_absent_students_l6_6276


namespace plains_routes_count_l6_6202

def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70
def total_routes : ℕ := 150
def mountainous_routes : ℕ := 21

theorem plains_routes_count :
  total_cities = mountainous_cities + plains_cities →
  3 * total_routes = total_cities →
  mountainous_routes * 2 ≤ mountainous_cities * 3 →
  (total_routes - mountainous_routes) * 2 = (70 * 3 - (mountainous_routes * 2)) →
  (total_routes - mountainous_routes * 2) / 2 = 81 :=
begin
  sorry
end

end plains_routes_count_l6_6202


namespace weeks_to_save_remaining_l6_6271

-- Assuming the conditions
def cost_of_shirt : ℝ := 3
def amount_saved : ℝ := 1.5
def saving_per_week : ℝ := 0.5

-- The proof goal
theorem weeks_to_save_remaining (cost_of_shirt amount_saved saving_per_week : ℝ) :
  cost_of_shirt = 3 ∧ amount_saved = 1.5 ∧ saving_per_week = 0.5 →
  ((cost_of_shirt - amount_saved) / saving_per_week) = 3 := by
  sorry

end weeks_to_save_remaining_l6_6271


namespace problem_conditions_l6_6970

open Real

theorem problem_conditions (t : ℝ) (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (habc : a + b + c = 1) (hx : ∀ x, f x := |x - 4| - t ≤ 2) 
  (hs : ∀ x, -1 ≤ x → x ≤ 5) : t = 1 ∧ (frac a b + frac b c + frac c a ≥ 1) :=
by
  sorry

end problem_conditions_l6_6970


namespace angle_equality_and_length_ratio_l6_6941

open Classical

variables {α : Type}

noncomputable def triangle :=
  {A B C : α // 
    is_acute_angle_triangle A B C}

def tangent_intersect (A B C T U : α) :=
  ∃ Q S P R : α, midpoint Q A P ∧ midpoint S B R ∧ 
  intersect A T B C P ∧ intersect B U A C R 

theorem angle_equality_and_length_ratio 
  {A B C T U : α}
  [triangle A B C]
  (h1 : tangent_intersect A B C T U) :
  ∃ α : ℝ,  ∠A B Q = ∠A B S ∧ side_length_ratio A B C = α :=
by
  sorry

end angle_equality_and_length_ratio_l6_6941


namespace single_elimination_tournament_games_l6_6849

theorem single_elimination_tournament_games (n : ℕ) (h : n = 21) : 
  let total_games := n - 1 in total_games = 20 := by
  have h1 : total_games = 21 - 1 := rfl
  rw h at h1
  exact h1.symm ▸ rfl

end single_elimination_tournament_games_l6_6849


namespace michael_passes_donovan_l6_6799

noncomputable def track_length : ℕ := 600
noncomputable def donovan_lap_time : ℕ := 45
noncomputable def michael_lap_time : ℕ := 40

theorem michael_passes_donovan :
  ∃ n : ℕ, michael_lap_time * n > donovan_lap_time * (n - 1) ∧ n = 9 :=
by
  sorry

end michael_passes_donovan_l6_6799


namespace find_x_for_g_eq_inverse_l6_6650

noncomputable def g (x : ℝ) := 2 * x - 5
noncomputable def g_inv (x : ℝ) := (x + 5) / 2

theorem find_x_for_g_eq_inverse : ∀ (x : ℝ), g x = g_inv x ↔ x = 5 :=
by
  intro x
  split
  { -- proving g(x) = g⁻¹(x) implies x = 5
    intro h
    calc 
      x = 5 : by sorry
  }
  { -- proving x = 5 implies g(x) = g⁻¹(x)
    intro h
    rw h
    simp [g, g_inv]
    sorry
  }

end find_x_for_g_eq_inverse_l6_6650


namespace sum_of_central_angles_pentagon_inscribed_in_circle_l6_6438

theorem sum_of_central_angles_pentagon_inscribed_in_circle
    (pentagon : Type) [inscribed_in_circle pentagon]
    (center : Point)
    (vertices : Fin 5 → Point)
    (h : ∀ i, dist (vertices i) center = circle_radius)
  : ∑ i in Finset.range 5, central_angle (center, vertices i, vertices ((i + 1) % 5)) = 360 :=
sorry

end sum_of_central_angles_pentagon_inscribed_in_circle_l6_6438


namespace find_ellipse_equation_and_x_range_l6_6947

noncomputable def equation_of_ellipse (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

theorem find_ellipse_equation_and_x_range
  (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (e : ℝ) (he : e = sqrt 6 / 3)
  (h1 : b^2 = a^2 * (1 - e^2))
  (h2 : (3 : ℝ)^2 / a^2 + (sqrt 2)^2 / b^2 = 1) :
  equation_of_ellipse 15 5 x y ∧
  (∀ (x0 : ℝ),
    (-sqrt 15 < x0 ∧ x0 < -sqrt 30 / 2) ∨
    (sqrt 30 / 2 < x0 ∧ x0 < sqrt 15)) :=
  sorry

end find_ellipse_equation_and_x_range_l6_6947


namespace repeating_decimal_as_fraction_l6_6038

theorem repeating_decimal_as_fraction : (0.\overline{02} = 2 / 99) → (2.\overline{06} = 68 / 33) :=
by
  sorry

end repeating_decimal_as_fraction_l6_6038


namespace abc_sum_eq_11sqrt6_l6_6590

theorem abc_sum_eq_11sqrt6 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 :=
sorry

end abc_sum_eq_11sqrt6_l6_6590


namespace mean_of_second_set_l6_6995

def mean (s : List ℕ) : ℝ := (s.sum : ℝ) / s.length

theorem mean_of_second_set {M : ℝ} {x : ℕ} :
  mean [28, x, 50, 78, 104] = M →
  mean [48, 62, 98, 124, x] = M + 14.4 :=
by
  intro hMeanFirstSet
  sorry

end mean_of_second_set_l6_6995


namespace repeating_decimal_as_fraction_l6_6034

theorem repeating_decimal_as_fraction : (0.\overline{02} = 2 / 99) → (2.\overline{06} = 68 / 33) :=
by
  sorry

end repeating_decimal_as_fraction_l6_6034


namespace max_value_sum_l6_6535

noncomputable def F (n : ℕ) (x : Fin n → ℝ) : ℝ := 
  ∑ i in Finset.range n, ∑ j in Finset.Icc i n, x i * x j * (x i + x j)

theorem max_value_sum (n : ℕ) (x : Fin n → ℝ) 
  (h_nonneg : ∀ i, 0 ≤ x i) 
  (h_sum : ∑ i in Finset.range n, x i = 1) 
  (h_n : 2 ≤ n) : 
  ∃ (v : Fin n → ℝ), F n v = 1 / 4 := 
sorry

end max_value_sum_l6_6535


namespace vessel_acceleration_for_floating_l6_6619

-- Definitions used in Lean 4 statement
variable (ρ1 ρ0 g a : ℝ)
axiom density_relation : ρ1 = 3 * ρ0

-- Lean statement for the mathematical proof problem
theorem vessel_acceleration_for_floating (ρ1 ρ0 g a : ℝ)
  (h1 : ρ1 = 3 * ρ0)
  (h2 : g - a >= 3 * g) :
  a <= -2 * g :=
by
  calc
    g - a >= 3 * g : h2
    ... : g - a ≥ 3g 
    ... : g - a - g ≥ 3g - g
    ... : -a ≥ 2g
    ... : a ≤ -2g 
  sorry

end vessel_acceleration_for_floating_l6_6619


namespace tennis_tournament_possible_l6_6215

theorem tennis_tournament_possible (p : ℕ) : 
  (∀ i j : ℕ, i ≠ j → ∃ a b c d : ℕ, a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ 
  i = a ∨ i = b ∨ i = c ∨ i = d ∧ j = a ∨ j = b ∨ j = c ∨ j = d) → 
  ∃ k : ℕ, p = 8 * k + 1 := by
  sorry

end tennis_tournament_possible_l6_6215


namespace arithmetic_series_sum_l6_6082

-- Define the parameters of the arithmetic series
def a1 : ℕ := 30
def an : ℕ := 60
def d : ℝ := 1/3
def n : ℕ := 91

-- Prove that the sum of the arithmetic series equals 4095
theorem arithmetic_series_sum : 
  ∑ i in finset.range n, (a1 + i * d) = 4095 := 
by
  sorry

end arithmetic_series_sum_l6_6082


namespace correct_proportion_l6_6170

variables (P Q R S : Type)
variables (p q r u v : ℝ)

-- Conditions
axiom triangle_sides : p > 0 ∧ q > 0 ∧ r > 0
axiom sides_opposite_angles : triangle_sides P Q R p q r
axiom angle_bisector : bisects_angle P S (angle P)

-- Definitions
def QS := u
def RS := v
def QR := p
def angle_bisector_theorem := (u / r = v / q)

-- Theorem to prove
theorem correct_proportion :
  angle_bisector_theorem →
  u + v = p →
  (v / q = p / (r + q)) := by
  sorry

end correct_proportion_l6_6170


namespace max_stores_visited_l6_6756

theorem max_stores_visited (total_stores : ℕ) (total_people : ℕ) 
    (visited_two_stores : ℕ) (total_visits : ℕ) (exactly_two_stores : ℕ) 
    (at_least_one_store : ∀ p, p < total_people → 1 ≤ nvis(p))
    (total_visits_count : ∑ p in finset.range total_people, nvis p = 23) : 
    total_stores = 8 ∧ total_people = 12 ∧ exactly_two_stores = 8 → 
    ∃ max_stores : ℕ, max_stores = 4 :=
begin
  sorry
end

end max_stores_visited_l6_6756


namespace vector_parallel_l6_6980

open Vector

def a (x : ℝ) : ℝ × ℝ := (x, 1)
def b (x : ℝ) : ℝ × ℝ := (-x, x^2)
def c : ℝ × ℝ := (0, 1)
def d : ℝ × ℝ := (1, -1)

theorem vector_parallel (x : ℝ) : ∃ λ : ℝ, a(x) + b(x) = (λ • c) :=
by
  sorry

end vector_parallel_l6_6980


namespace recurring_decimal_reduced_fraction_l6_6058

noncomputable def recurring_decimal_as_fraction : Prop := 
  ∀ (x y : ℚ), (x = 2.06) ∧ (y = 0.02) → y = 2 / 99 → x = 68 / 33

theorem recurring_decimal_reduced_fraction (x y : ℚ) 
  (h1 : x = 2 + 0.06) (h2 : y = 0.02) (h3 : y = 2 / 99) : 
  x = 68 / 33 := 
begin
  -- Proof here
  sorry
end

end recurring_decimal_reduced_fraction_l6_6058


namespace equiangular_parallelogram_is_rectangle_l6_6428

structure Parallelogram (P : Type*) :=
(angle_eq : ∀ {α β : P}, α ≠ β → α = β)
(is_parallelogram : ∀ {α β γ δ : P}, (α ≠ β ∧ β ≠ γ ∧ γ ≠ δ ∧ δ ≠ α) → (α = γ) ∧ (β = δ))

def is_equiangular_parallelogram (P : Type*) [Parallelogram P] : Prop :=
  ∀ {α β : P}, α ≠ β → α = 90 ∧ β = 90

theorem equiangular_parallelogram_is_rectangle (P : Type*) [Parallelogram P] :
  is_equiangular_parallelogram P → Parallelogram P → is_rectangle P :=
by
  sorry

end equiangular_parallelogram_is_rectangle_l6_6428


namespace decreasing_interval_l6_6133

noncomputable def y (a : ℝ) (x : ℝ) : ℝ := a * x^3 - 15 * x^2 + 36 * x - 24

def has_extremum_at (a : ℝ) (x_ext : ℝ) : Prop :=
  deriv (y a) x_ext = 0

theorem decreasing_interval (a : ℝ) (h_extremum_at : has_extremum_at a 3) :
  a = 2 → ∀ x, (2 < x ∧ x < 3) → deriv (y a) x < 0 :=
sorry

end decreasing_interval_l6_6133


namespace find_angle_AOB_l6_6770

noncomputable def angle_AOB (triangle_PAB : Triangle) (angle_APB : ℝ) (tangents_to_circle : Triangle ∧ tangent T : Circle) (circumcircle_O : Circle) : Prop :=
  ∠AOB = 65

theorem find_angle_AOB (P A B : Point) (O : Circle) (tangent_lines : Tangent P A O ∧ Tangent P B O ∧ Tangent A B O) (angle_APB : ∠ APB = 50) : ∠ AOB = 65 :=
begin
  sorry
end

end find_angle_AOB_l6_6770


namespace average_garbage_is_200_correlation_is_strong_l6_6919

noncomputable def average_garbage (y : Fin 20 → ℝ) : ℝ :=
  ∑ i, y i / 20

def correlation_coefficient (x y : Fin 20 → ℝ) :=
  let n := (20 : ℝ)
  let sum_x := ∑ i, x i
  let sum_y := ∑ i, y i
  let mean_x := sum_x / n
  let mean_y := sum_y / n
  let Sxx := ∑ i, (x i - mean_x)^2
  let Syy := ∑ i, (y i - mean_y)^2
  let Sxy := ∑ i, (x i - mean_x) * (y i - mean_y)
  Sxy / Real.sqrt (Sxx * Syy)

theorem average_garbage_is_200 (y : Fin 20 → ℝ) (h1 : ∑ i, y i = 4000) :
  average_garbage y = 200 :=
by {
  unfold average_garbage,
  rw [h1],
  norm_num,
  sorry
}

theorem correlation_is_strong (x y : Fin 20 → ℝ) 
  (h2 : ∑ i, x i = 80)
  (h3 : ∑ i, y i = 4000)
  (h4 : ∑ i, (x i - ∑ i, x i / 20)^2 = 80)
  (h5 : ∑ i, (y i - ∑ i, y i / 20)^2 = 8000)
  (h6 : ∑ i, (x i - ∑ i, x i / 20) * (y i - ∑ i, y i / 20) = 700) :
  correlation_coefficient x y = 0.875 :=
by {
  unfold correlation_coefficient,
  rw [h2, h3, h4, h5, h6],
  norm_num,
  sorry
}

end average_garbage_is_200_correlation_is_strong_l6_6919


namespace find_x_l6_6549

def fractional_part (x : ℝ) : ℝ := x - floor x
def integer_part (x : ℝ) : ℝ := floor x

theorem find_x (x : ℝ) (hx : 4 * x^2 - 5 * integer_part x + 8 * fractional_part x = 19) : x = 5 / 2 :=
by
  sorry

end find_x_l6_6549


namespace isabella_hair_length_l6_6639

def initial_length : ℕ := 18
def growth : ℕ := 6
def final_length : ℕ := initial_length + growth

theorem isabella_hair_length : final_length = 24 :=
by
  simp [final_length, initial_length, growth]
  sorry

end isabella_hair_length_l6_6639


namespace compare_negatives_l6_6873

theorem compare_negatives : (-1.5 : ℝ) < (-1 + -1/5 : ℝ) :=
by 
  sorry

end compare_negatives_l6_6873


namespace recurring_decimal_reduced_fraction_l6_6061

noncomputable def recurring_decimal_as_fraction : Prop := 
  ∀ (x y : ℚ), (x = 2.06) ∧ (y = 0.02) → y = 2 / 99 → x = 68 / 33

theorem recurring_decimal_reduced_fraction (x y : ℚ) 
  (h1 : x = 2 + 0.06) (h2 : y = 0.02) (h3 : y = 2 / 99) : 
  x = 68 / 33 := 
begin
  -- Proof here
  sorry
end

end recurring_decimal_reduced_fraction_l6_6061


namespace coconut_grove_l6_6173

theorem coconut_grove (x N : ℕ) (h1 : (x + 4) * 60 + x * N + (x - 4) * 180 = 3 * x * 100) (hx : x = 8) : N = 120 := 
by
  subst hx
  sorry

end coconut_grove_l6_6173


namespace num_parallelogram_even_l6_6841

-- Define the conditions of the problem in Lean
def isosceles_right_triangle (base_length : ℕ) := 
  base_length = 2

def square (side_length : ℕ) := 
  side_length = 1

def parallelogram (sides_length : ℕ) (diagonals_length : ℕ) := 
  sides_length = 1 ∧ diagonals_length = 1

-- Main statement to prove
theorem num_parallelogram_even (num_triangles num_squares num_parallelograms : ℕ)
  (Htriangle : ∀ t, t < num_triangles → isosceles_right_triangle 2)
  (Hsquare : ∀ s, s < num_squares → square 1)
  (Hparallelogram : ∀ p, p < num_parallelograms → parallelogram 1 1) :
  num_parallelograms % 2 = 0 := 
sorry

end num_parallelogram_even_l6_6841


namespace complex_expression_l6_6561

noncomputable def z : ℂ := 1 + real.sqrt 2 * complex.I 

theorem complex_expression : (z^2 - 2 * z) = -3 :=
by sorry

end complex_expression_l6_6561


namespace repeating_decimal_fraction_l6_6067

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l6_6067


namespace four_friends_meeting_time_l6_6916

theorem four_friends_meeting_time :
  let lcm_four_numbers (a b c d : ℕ) := Nat.lcm (Nat.lcm a b) (Nat.lcm c d) in
  lcm_four_numbers 5 8 9 10 = 360 :=
by
  sorry

end four_friends_meeting_time_l6_6916


namespace probability_of_rolling_two_2s_in_five_rolls_l6_6151

-- Define essential probability values, combinatorial functions, and the probability calculation.

def probability_roll_2_given_fair_die : ℚ := 1 / 6
def probability_not_2_given_fair_die : ℚ := 5 / 6

def binomial_coefficient (n k : ℕ) : ℕ :=
  nat.choose n k

def probability_of_sequence (success_prob fail_prob : ℚ) (success_count fail_count : ℕ) : ℚ :=
  success_prob^success_count * fail_prob^fail_count

def total_probability_of_exactly_two_2s_in_five_rolls : ℚ :=
  (binomial_coefficient 5 2) * (probability_of_sequence probability_roll_2_given_fair_die probability_not_2_given_fair_die 2 3)

theorem probability_of_rolling_two_2s_in_five_rolls :
  total_probability_of_exactly_two_2s_in_five_rolls = 625 / 3888 :=
by sorry

end probability_of_rolling_two_2s_in_five_rolls_l6_6151


namespace power_function_through_point_l6_6573

theorem power_function_through_point (m n : ℝ) (h₁ : ∀ x, f x = m * x^n)
(h₂ : f 2 = 16) : m + n = 5 :=
by sorry

end power_function_through_point_l6_6573


namespace attendance_difference_l6_6659

noncomputable def saturday := 80
noncomputable def monday := saturday - 0.25 * saturday
noncomputable def wednesday := monday + 0.5 * monday
noncomputable def friday := saturday + monday
noncomputable def thursday := 45
noncomputable def sunday := saturday - 0.15 * saturday
noncomputable def expected_total := 350

noncomputable def total_attendance := saturday + monday + wednesday + friday + thursday + sunday
noncomputable def difference := total_attendance - expected_total

theorem attendance_difference : difference = 133 := by
  sorry

end attendance_difference_l6_6659


namespace recurring_decimal_reduced_fraction_l6_6057

noncomputable def recurring_decimal_as_fraction : Prop := 
  ∀ (x y : ℚ), (x = 2.06) ∧ (y = 0.02) → y = 2 / 99 → x = 68 / 33

theorem recurring_decimal_reduced_fraction (x y : ℚ) 
  (h1 : x = 2 + 0.06) (h2 : y = 0.02) (h3 : y = 2 / 99) : 
  x = 68 / 33 := 
begin
  -- Proof here
  sorry
end

end recurring_decimal_reduced_fraction_l6_6057


namespace number_of_plains_routes_is_81_l6_6189

-- Define the number of cities in each region
def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70

-- Define the number of routes established over three years
def total_routes : ℕ := 150
def routes_per_year : ℕ := 50

-- Define the number of routes connecting pairs of mountainous cities
def mountainous_routes : ℕ := 21

-- Define a function to calculate the number of routes connecting pairs of plains cities
def plains_routes : ℕ :=
  let total_endpoints := total_routes * 2
  let mountainous_endpoints := mountainous_cities * 3
  let plains_endpoints := plains_cities * 3
  let mountainous_pair_endpoints := mountainous_routes * 2
  let mountain_plain_routes := (mountainous_endpoints - mountainous_pair_endpoints) / 2
  let plain_only_endpoints := plains_endpoints - mountain_plain_routes
  plain_only_endpoints / 2

theorem number_of_plains_routes_is_81 : plains_routes = 81 := 
  sorry

end number_of_plains_routes_is_81_l6_6189


namespace internal_triangle_area_l6_6214

-- Defining the conditions of the problem
def is_midpoint (p1 p2 m : Point) : Prop := distance p1 m = distance m p2

structure Square where
  side_length : ℝ
  A B C D : Point
  len_AB : distance A B = side_length
  len_BC : distance B C = side_length
  len_CD : distance C D = side_length
  len_DA : distance D A = side_length
  perp_AB_BC : ∠ A B C = π / 2
  perp_BC_CD : ∠ B C D = π / 2
  perp_CD_DA : ∠ C D A = π / 2
  perp_DA_AB : ∠ D A B = π / 2

def midpoints (s : Square) (K L : Point) : Prop := 
  is_midpoint s.A s.D K ∧ is_midpoint s.C s.D L

-- Defining the area calculation for the internal triangle
noncomputable def area_internal_triangle (s : Square) (K L : Point) (h_mid : midpoints s K L) : ℝ :=
let {side_length := a} := s in
3 * a^2 / 8

-- Lean statement for the proof problem
theorem internal_triangle_area (s : Square) (K L : Point) (h_mid : midpoints s K L) : 
  area_internal_triangle s K L h_mid = 3 * s.side_length^2 / 8 :=
by
  sorry

end internal_triangle_area_l6_6214


namespace plains_routes_count_l6_6177

theorem plains_routes_count (total_cities mountainous_cities plains_cities total_routes routes_mountainous_pairs: ℕ) :
  total_cities = 100 →
  mountainous_cities = 30 →
  plains_cities = 70 →
  total_routes = 150 →
  routes_mountainous_pairs = 21 →
  let endpoints_mountainous := mountainous_cities * 3 in
  let endpoints_mountainous_pairs := routes_mountainous_pairs * 2 in
  let endpoints_mountainous_plains := endpoints_mountainous - endpoints_mountainous_pairs in
  let endpoints_plains := plains_cities * 3 in
  let routes_mountainous_plains := endpoints_mountainous_plains in
  let endpoints_plains_pairs := endpoints_plains - routes_mountainous_plains in
  let routes_plains_pairs := endpoints_plains_pairs / 2 in
  routes_plains_pairs = 81 :=
by
  intros h1 h2 h3 h4 h5
  dsimp
  rw [h1, h2, h3, h4, h5]
  sorry

end plains_routes_count_l6_6177


namespace floor_abs_sum_l6_6888

theorem floor_abs_sum (x : ℝ) (h : x = -7.3) : 
    (⌊|x|⌋ + |⌊x⌋|) = 15 := 
by
  rw h
  rw abs_of_neg
  rw floor_neg
  simp
  sorry   -- Placeholder for additional proof steps

end floor_abs_sum_l6_6888


namespace rectangle_side_length_l6_6675

theorem rectangle_side_length (a c : ℝ) (h_ratio : a / c = 3 / 4) (hc : c = 4) : a = 3 :=
by
  sorry

end rectangle_side_length_l6_6675


namespace teacher_proctor_arrangements_l6_6304

theorem teacher_proctor_arrangements {f m : ℕ} (hf : f = 2) (hm : m = 5) :
  (∃ moving_teachers : ℕ, moving_teachers = 1 ∧ (f - moving_teachers) + m = 7 
   ∧ (f - moving_teachers).choose 2 = 21)
  ∧ 2 * 21 = 42 :=
by
    sorry

end teacher_proctor_arrangements_l6_6304


namespace chip_placement_count_l6_6146

def grid := Fin 4 × Fin 3

def grid_positions (n : Nat) := {s : Finset grid // s.card = n}

def no_direct_adjacency (positions : Finset grid) : Prop :=
  ∀ (x y : grid), x ∈ positions → y ∈ positions →
  (x.fst ≠ y.fst ∨ x.snd ≠ y.snd)

noncomputable def count_valid_placements : Nat :=
  -- Function to count valid placements
  sorry

theorem chip_placement_count :
  count_valid_placements = 4 :=
  sorry

end chip_placement_count_l6_6146


namespace point_in_intersection_l6_6268

variables {m n : ℝ}
def U : set (ℝ × ℝ) := {p | true}
def A : set (ℝ × ℝ) := {p | p.1 + p.2 <= m}
def B : set (ℝ × ℝ) := {p | p.1^2 + p.2^2 <= n}
def C_U_A : set (ℝ × ℝ) := {p | p.1 + p.2 <= m}

theorem point_in_intersection (h1 : (1, 2) ∈ C_U_A) (h2 : (1, 2) ∈ B) : m >= 3 ∧ n >= 5 := by {
  sorry
}

end point_in_intersection_l6_6268


namespace all_suits_different_in_groups_of_four_l6_6612

-- Define the alternation pattern of the suits in the deck of 36 cards
def suits : List String := ["spades", "clubs", "hearts", "diamonds"]

-- Formalize the condition that each 4-card group in the deck contains all different suits
def suits_includes_all (cards : List String) : Prop :=
  ∀ i j, i < 4 → j < 4 → i ≠ j → cards.get? i ≠ cards.get? j

-- The main theorem statement
theorem all_suits_different_in_groups_of_four (L : List String)
  (hL : L.length = 36)
  (hA : ∀ n, n < 9 → L.get? (4*n) = some "spades" ∧ L.get? (4*n + 1) = some "clubs" ∧ L.get? (4*n + 2) = some "hearts" ∧ L.get? (4*n + 3) = some "diamonds"):
  ∀ cut reversed_deck, (@List.append String (List.reverse (List.take cut L)) (List.drop cut L) = reversed_deck)
  → ∀ n, n < 9 → suits_includes_all (List.drop (4*n) (List.take 4 reversed_deck)) := sorry

end all_suits_different_in_groups_of_four_l6_6612


namespace four_digit_sum_of_swapped_l6_6390

def isFourDigitNumberUsingDigits {n: ℕ} (digits: Finset ℕ) : Prop :=
  let ns := (toString n).data.map (λ c, c.toNat - '0'.toNat)
  n ≥ 1000 ∧ n < 10000 ∧ digits = ns.toFinset

def swappedMiddleDigits (n: ℕ) : ℕ :=
  let digits := (toString n).data.map (λ c, c.toNat - '0'.toNat)
  let swapped := digits.headI ++ digits.tailI.headI ++ digits.getLastI ++ digits.dropLast.tailI
  swapped.foldl (λ acc d, acc * 10 + d) 0

theorem four_digit_sum_of_swapped {n₁ n₂ : ℕ} 
  (h₁ : isFourDigitNumberUsingDigits n₁ ({4, 5, 8, 9} : Finset ℕ))
  (h₂ : n₁ = 4859) 
  (h₃ : n₂ = swappedMiddleDigits 4859) 
  (h₄ : n₂ = 4958) :
  n₁ + n₂ = 9817 := 
by sorry

end four_digit_sum_of_swapped_l6_6390


namespace repeating_decimal_as_fraction_l6_6039

theorem repeating_decimal_as_fraction : (0.\overline{02} = 2 / 99) → (2.\overline{06} = 68 / 33) :=
by
  sorry

end repeating_decimal_as_fraction_l6_6039


namespace plains_routes_count_l6_6200

def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70
def total_routes : ℕ := 150
def mountainous_routes : ℕ := 21

theorem plains_routes_count :
  total_cities = mountainous_cities + plains_cities →
  3 * total_routes = total_cities →
  mountainous_routes * 2 ≤ mountainous_cities * 3 →
  (total_routes - mountainous_routes) * 2 = (70 * 3 - (mountainous_routes * 2)) →
  (total_routes - mountainous_routes * 2) / 2 = 81 :=
begin
  sorry
end

end plains_routes_count_l6_6200


namespace May4th_Sunday_l6_6433

theorem May4th_Sunday (x : ℕ) (h_sum : x + (x + 7) + (x + 14) + (x + 21) + (x + 28) = 80) : 
  (4 % 7) = 0 :=
by
  sorry

end May4th_Sunday_l6_6433


namespace ant_completes_path_in_finite_time_l6_6805

noncomputable def ant_travel_time (t : ℝ) (k : ℝ) (h : k < 1) : ℝ := 
  t / (1 - k)

theorem ant_completes_path_in_finite_time (t k : ℝ) (h : k < 1) : 
  ∃ T : ℝ, T = ant_travel_time t k h :=
begin
  use ant_travel_time t k h,
  exact rfl,
end

end ant_completes_path_in_finite_time_l6_6805


namespace find_g_expression_l6_6961

theorem find_g_expression (g f : ℝ → ℝ) (h_sym : ∀ x y, g x = y ↔ g (2 - x) = 4 - y)
  (h_f : ∀ x, f x = 3 * x - 1) :
  ∀ x, g x = 3 * x - 1 :=
by
  sorry

end find_g_expression_l6_6961


namespace polynomials_divide_x60_minus_1_l6_6792

-- Define the polynomials
def poly1 := Polynomial.Coeff ℝ 2 + Polynomial.Coeff ℝ 1 + Polynomial.Coeff ℝ 1
def poly2 := Polynomial.Coeff ℝ 4 - Polynomial.Coeff ℝ 1
def poly3 := Polynomial.Coeff ℝ 5 - Polynomial.Coeff ℝ 1
def poly4 := Polynomial.Coeff ℝ 15 - Polynomial.Coeff ℝ 1
def poly60 := Polynomial.Coeff ℝ 60 - Polynomial.Coeff ℝ 1

-- Prove that each polynomial divides x^60 - 1
theorem polynomials_divide_x60_minus_1 :
  poly1 ∣ poly60 ∧ poly2 ∣ poly60 ∧ poly3 ∣ poly60 ∧ poly4 ∣ poly60 := 
sorry

end polynomials_divide_x60_minus_1_l6_6792


namespace solutions_to_equation_l6_6500

theorem solutions_to_equation :
  ∀ x : ℝ, 
  sqrt ((3 + sqrt 5) ^ x) + sqrt ((3 - sqrt 5) ^ x) = 6 ↔ (x = 2 ∨ x = -2) :=
by
  intros x
  sorry

end solutions_to_equation_l6_6500


namespace abs_div_one_add_i_by_i_l6_6550

noncomputable def imaginary_unit : ℂ := Complex.I

/-- The absolute value of the complex number (1 + i)/i is √2. -/
theorem abs_div_one_add_i_by_i : Complex.abs ((1 + imaginary_unit) / imaginary_unit) = Real.sqrt 2 := by
  sorry

end abs_div_one_add_i_by_i_l6_6550


namespace five_digit_numbers_with_alternating_parity_l6_6145

theorem five_digit_numbers_with_alternating_parity : 
  ∃ n : ℕ, n = 5625 ∧ ∀ (x : ℕ), (10000 ≤ x ∧ x < 100000) → 
    (∀ i, i < 4 → (((x / 10^i) % 10) % 2 ≠ ((x / 10^(i+1)) % 10) % 2)) ↔ 
    (x = 5625) := 
sorry

end five_digit_numbers_with_alternating_parity_l6_6145


namespace trapezoid_area_l6_6216

theorem trapezoid_area 
  (PQRS : Type) 
  [Trapezoid PQRS] 
  (P Q R S T : PQRS) 
  (h1 : parallel PQ RS) 
  (h2 : T = PR ∩ QS) 
  (area_PQT : ℝ)
  (area_PRT : ℝ) 
  (h3 : area_PQT = 60) 
  (h4 : area_PRT = 25) : 
  area PQRS = 135 := 
sorry 

end trapezoid_area_l6_6216


namespace hyperbola_eccentricity_proof_l6_6162

noncomputable def hyperbola_eccentricity {a b : ℝ} (h : a > 0) (h' : b > 0) : Prop :=
  let e := 2 in
  e = Real.sqrt (1 + (b^2) / (a^2)) → (b / a = Real.sqrt 3)

theorem hyperbola_eccentricity_proof {a b : ℝ} (h : a > 0) (h' : b > 0) :
  hyperbola_eccentricity h h' :=
by
  sorry

end hyperbola_eccentricity_proof_l6_6162


namespace find_w_l6_6519

variables {x y : ℚ}

def w : ℚ × ℚ := (-48433 / 975, 2058 / 325)

def vec1 : ℚ × ℚ := (3, 2)
def vec2 : ℚ × ℚ := (3, 4)

def proj (u v : ℚ × ℚ) : ℚ × ℚ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let dot_vv := v.1 * v.1 + v.2 * v.2
  (dot_uv / dot_vv * v.1, dot_uv / dot_vv * v.2)

def p1 : ℚ × ℚ := (47 / 13, 31 / 13)
def p2 : ℚ × ℚ := (85 / 25, 113 / 25)

theorem find_w (hw : w = (x, y)) :
  proj ⟨x, y⟩ vec1 = p1 ∧
  proj ⟨x, y⟩ vec2 = p2 :=
sorry

end find_w_l6_6519


namespace range_of_expression_l6_6263

theorem range_of_expression (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 ≤ β ∧ β ≤ π / 2) :
  -π / 6 < 2 * α - β / 3 ∧ 2 * α - β / 3 < π := by
  sorry

end range_of_expression_l6_6263


namespace sum_of_zeros_eq_zero_l6_6749

def f (x : ℝ) : ℝ :=
if x < 0 then (1/2)^x - 2 else x - 1

theorem sum_of_zeros_eq_zero : (zeros : List ℝ) (h : ∀ x ∈ zeros, f x = 0) (hint : zeros.nodup) :=
∑ i in zeros.to_finset, i = 0 :=
sorry

end sum_of_zeros_eq_zero_l6_6749


namespace sin_zero_necessary_not_sufficient_l6_6408

theorem sin_zero_necessary_not_sufficient:
  (∀ α : ℝ, (∃ k : ℤ, α = 2 * k * Real.pi) → (Real.sin α = 0)) ∧
  ¬ (∀ α : ℝ, (Real.sin α = 0) → (∃ k : ℤ, α = 2 * k * Real.pi)) :=
by
  sorry

end sin_zero_necessary_not_sufficient_l6_6408


namespace periodic_decimal_to_fraction_l6_6010

theorem periodic_decimal_to_fraction
  (h : ∀ n : ℕ, 0.<digit>02 n / 99) :
  2.0<digit>06 = 68 / 33 :=
sorry

end periodic_decimal_to_fraction_l6_6010


namespace pyramid_volume_correct_l6_6212

open Real

constant truncated_pyramid_volume : ℝ → ℝ → ℝ → ℝ
constant b1 : ℝ := 1
constant b2 : ℝ := 7
constant h : ℝ 

-- Define the volume of the truncated pyramid
axiom volume_formula :
  truncated_pyramid_volume b1 b2 h = 
  (1 / 3) * h * (b1 ^ 2 + b1 * b2 + b2 ^ 2)

-- Given that the height h 
axiom height_val : 
  h = 2 / (Real.sqrt 5)

-- Let's input these conditions and validate the final volume
theorem pyramid_volume_correct :
  truncated_pyramid_volume b1 b2 h = 38 / (Real.sqrt 5) :=
by
  rw [truncated_pyramid_volume, volume_formula, height_val]
  sorry

end pyramid_volume_correct_l6_6212


namespace angle_AOB_is_65_l6_6767

-- Define the conditions as hypotheses
variables {P A B O : Type} -- Points on the plane
variables (h_triangle_tangent : Triangle_tangent_to_circle P A B O)
variables (h_angle_APB : ∠ APB = 50)

-- Define the theorem to prove the target statement
theorem angle_AOB_is_65 :
  ∠ AOB = 65 :=
sorry

end angle_AOB_is_65_l6_6767


namespace repeating_decimal_fraction_l6_6068

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l6_6068


namespace number_of_books_l6_6755

-- Define the given conditions as variables
def movies_in_series : Nat := 62
def books_read : Nat := 4
def books_yet_to_read : Nat := 15

-- State the proposition we need to prove
theorem number_of_books : (books_read + books_yet_to_read) = 19 :=
by
  sorry

end number_of_books_l6_6755


namespace probability_correct_l6_6950

noncomputable def lengths : List ℕ := [1, 3, 5, 7, 9]

def is_triangle (a b c : ℕ) : Bool :=
  a + b > c ∧ a + c > b ∧ b + c > a

def non_triangle_count : ℕ :=
  lengths.combinations 3 |>.filter (not ∘ Function.uncurry3 is_triangle) |>.length

def total_combinations : ℕ :=
  lengths.combinations 3 |>.length

def probability_non_triangle : ℚ :=
  non_triangle_count /. total_combinations

theorem probability_correct : probability_non_triangle = 7/10 := by
  sorry

end probability_correct_l6_6950


namespace calculate_B_share_l6_6813

-- Definitions for conditions
variables {A B C: ℝ}
#check @∀ (A B C : ℝ), (2 * A = 3 * B) → (B = 4 * C) → 
          -- Total profit is given
          16500 :=
              16500

noncomputable def calculate_share (total_profit : ℝ) : ℝ :=
  let total_ratio := (6 + 4 + 1) * C in
  (4 / total_ratio) * total_profit

-- Theorem to be proved
theorem calculate_B_share (h1 : 2 * A = 3 * B) (h2 : B = 4 * C) :
  calculate_share 16500 = 6000 :=
by 
  sorry

end calculate_B_share_l6_6813


namespace angle_AMO_eq_angle_MAD_l6_6321

-- Define the properties and conditions of the parallelogram and the points
open EuclideanGeometry

variables {A B C D O M : Point}
variables [h_parallelogram : Parallelogram A B C D]
variables [h_diagonals_intersect : DiagonalsIntersect A B C D O]
variables [h_point_M : OnExtensionBeyond A B M]
variables [h_MC_eq_MD : MC = MD]

-- Prove that angle AMO equals angle MAD
theorem angle_AMO_eq_angle_MAD (h_parallelogram : Parallelogram A B C D)
                                 (h_diagonals_intersect : DiagonalsIntersect A B C D O)
                                 (h_point_M : OnExtensionBeyond A B M)
                                 (h_MC_eq_MD : MC = MD) :
  ∠A M O = ∠M A D :=
sorry

end angle_AMO_eq_angle_MAD_l6_6321


namespace probability_C_wins_tournament_l6_6407

/-- Three players A, B, and C participate in a backgammon tournament. The winner of each match proceeds
    to play against C, and the last player to win two successive games wins the tournament. Each player 
    has an equal probability of winning any individual game.
    The probability that player C wins the tournament is 2/7. -/
theorem probability_C_wins_tournament : ∃ (p : ℚ), p = 2 / 7 ∧
  let prob_game := (1 : ℚ) / 2 in
  let scenario_A :=
    (prob_game * prob_game * (1 / (1 - (prob_game^3)) * (1 - prob_game))) in
  let scenario_B :=
    (prob_game * prob_game * (1 / (1 - (prob_game^3)) * (1 - prob_game))) in
  let prob_C_wins :=
    1 - scenario_A - scenario_B in
  prob_C_wins = p :=
sorry

end probability_C_wins_tournament_l6_6407


namespace angle_AOB_is_65_l6_6768

-- Define the conditions as hypotheses
variables {P A B O : Type} -- Points on the plane
variables (h_triangle_tangent : Triangle_tangent_to_circle P A B O)
variables (h_angle_APB : ∠ APB = 50)

-- Define the theorem to prove the target statement
theorem angle_AOB_is_65 :
  ∠ AOB = 65 :=
sorry

end angle_AOB_is_65_l6_6768


namespace coefficient_x3_in_expansion_l6_6231

theorem coefficient_x3_in_expansion (n : ℕ) (x : ℕ) : 
  (∑ i in Finset.range (n + 1), (Nat.choose n i) * 2^(n - i) * (x^i)) = 
  (160 : ℕ) 
:=
by
  let n := 6
  let x := 3
  have binomial_theorem : (∑ i in Finset.range (n + 1), (Nat.choose n i) * 2^(n - i) * (x^i)) = ∑ i in Finset.range (7), (Nat.choose 6 i) * 2^(6 - i) * 3^i :=
    by sorry
  rw binomial_theorem
  -- Using a known result manually just to satisfy the proof structure
  have term_x3 : (∑ i in Finset.range (7), (Nat.choose 6 i) * 2^(6 - i) * 3^i).coeff 3 = 160 :=
    by
    sorry
  exact term_x3

end coefficient_x3_in_expansion_l6_6231


namespace problem_l6_6141

noncomputable def trajectory_C (x y : ℝ) : Prop :=
  y^2 = -8 * x

theorem problem (P : ℝ × ℝ) (k : ℝ) (h : -1 < k ∧ k < 0) 
  (H1 : P.1 = -2 ∨ P.1 = 2)
  (H2 : trajectory_C P.1 P.2) :
  ∃ Q : ℝ × ℝ, Q.1 < -6 :=
  sorry

end problem_l6_6141


namespace BX_equals_BY_l6_6937

variables (A B C D X Y : Point)
          (line_l : Line)

-- Assume ABCD is a parallelogram
axiom parallelogram_ABCD : Parallelogram A B C D

-- Two circles with centers A and C pass through D
axiom circle_A : Circle A = {A, D, X}
axiom circle_C : Circle C = {C, D, Y}

-- Line l passes through D and intersects circles at X and Y
axiom line_through_D_X_Y : line_l ∋ D ∧ line_l ∋ X ∧ line_l ∋ Y

theorem BX_equals_BY : dist B X = dist B Y := by
  sorry

end BX_equals_BY_l6_6937


namespace dot_product_of_vectors_l6_6981

variables (a b : ℝ)
variables (θ : ℝ)
variables (norm_a norm_b : ℝ)

theorem dot_product_of_vectors 
  (h₀ : θ = real.pi / 3) -- 60 degrees in radians
  (h₁ : norm_a = 1)
  (h₂ : norm_b = 2) 
  (ha : ‖a‖ = norm_a)
  (hb : ‖b‖ = norm_b) :
  (a * b * real.cos θ = 1) :=
begin
  sorry
end

end dot_product_of_vectors_l6_6981


namespace chloe_weight_l6_6318

def alice_chloe_weights (a c : ℝ) : Prop :=
  a + c = 200 ∧ a - c = c / 3

theorem chloe_weight : ∃ c : ℝ, (λ c, c = 600 / 7) (c) :=
begin
  use 600 / 7,
  sorry
end

end chloe_weight_l6_6318


namespace math_problem_l6_6952

theorem math_problem (m n : ℝ) 
  (h1 : m = (sqrt (n^2 - 4) + sqrt (4 - n^2) + 4) / (n - 2))
  (h2 : n^2 ≥ 4)
  (h3 : 4 ≥ n^2)
  (h4 : n ≠ 2) :
  |m - 2 * n| + sqrt (8 * m * n) = 7 :=
sorry

end math_problem_l6_6952


namespace negation_of_square_positivity_l6_6738

theorem negation_of_square_positivity :
  (¬ ∀ n : ℕ, n * n > 0) ↔ (∃ n : ℕ, n * n ≤ 0) :=
  sorry

end negation_of_square_positivity_l6_6738


namespace not_perfect_cube_l6_6670

theorem not_perfect_cube (n : ℕ) : ¬ ∃ k : ℕ, k ^ 3 = 2 ^ (2 ^ n) + 1 :=
sorry

end not_perfect_cube_l6_6670


namespace ratio_of_two_numbers_l6_6312

theorem ratio_of_two_numbers (A B : ℕ) (x y : ℕ) (h1 : lcm A B = 60) (h2 : A + B = 50) (h3 : A / B = x / y) (hx : x = 3) (hy : y = 2) : x = 3 ∧ y = 2 := 
by
  -- Conditions provided in the problem
  sorry

end ratio_of_two_numbers_l6_6312


namespace varphi_symmetry_l6_6164

theorem varphi_symmetry (ω φ : ℝ) (h₁ : ω = 2) (h₂ : |φ| < π / 2) :
  (∃ c : ℝ, ∀ x : ℝ, 3 * sin (ω * x - π / 3) = 2 * cos (2 * x + φ)) →
  φ = π / 6 :=
by
  sorry

end varphi_symmetry_l6_6164


namespace total_handshakes_l6_6864

-- Definitions based on conditions
def num_wizards : ℕ := 25
def num_elves : ℕ := 18

-- Each wizard shakes hands with every other wizard
def wizard_handshakes : ℕ := num_wizards * (num_wizards - 1) / 2

-- Each elf shakes hands with every wizard
def elf_wizard_handshakes : ℕ := num_elves * num_wizards

-- Total handshakes is the sum of the above two
theorem total_handshakes : wizard_handshakes + elf_wizard_handshakes = 750 := by
  sorry

end total_handshakes_l6_6864


namespace perimeter_of_inner_pentagon_l6_6403

-- Let sum_of_internal_angles be 180 degrees and star_length be 1
def star_length := 1
def sum_of_internal_angles := 180

theorem perimeter_of_inner_pentagon :
  (let sin_18 := (Math.sin (Real.toRadians 18))
   let perimeter_eq := sin_18 / (1 + sin_18)
   perimeter_eq = sqrt 5 - 2)
  → perimeter_of_inner_pentagon = sqrt 5 - 2
:= sorry

end perimeter_of_inner_pentagon_l6_6403


namespace evaluate_expression_l6_6787

theorem evaluate_expression : 
  ∀ (x y : ℕ), x = 12 → y = 7 → (x - y) * (2 * x + y) = 155 :=
by
  intros x y h1 h2
  rw [h1, h2]
  sorry

end evaluate_expression_l6_6787


namespace find_k_l6_6578

/-
Given vectors a = (1, 2) and b = (-3, 2), 
prove that if (k * a + b) is parallel to (a - 3 * b),
then k = -1 / 3.
-/
theorem find_k (k : ℝ) (a b : ℝ × ℝ) 
    (ha : a = (1, 2)) 
    (hb : b = (-3, 2)) 
    (h : ∃ (λ : ℝ), (k • a + b) = λ • (a - 3 • b)) : 
    k = -1 / 3 :=
sorry

end find_k_l6_6578


namespace find_values_of_x_l6_6497

-- Definitions of integer part and fractional part
def int_part (x : ℝ) : ℤ := ⌊x⌋

def frac_part (x : ℝ) : ℝ := x - int_part x

-- Given conditions in the problem statement
variable {x : ℝ} (h : int_part x + frac_part (2 * x) = 2.5)

-- The theorem we need to prove
theorem find_values_of_x (h : int_part x + frac_part (2 * x) = 2.5) : x = 2.25 ∨ x = 2.75 :=
sorry

end find_values_of_x_l6_6497


namespace decimal_to_fraction_l6_6015

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l6_6015


namespace smallest_value_l6_6649

open Matrix

noncomputable def is_solution (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  (⟦3, 0; 0, 2⟧ ⬝ ⟦a, b; c, d⟧ = ⟦a, b; c, d⟧ ⬝ ⟦18, 12; -20, -13⟧)

theorem smallest_value :
  ∃ a b c d : ℕ, is_solution a b c d ∧ (a + b + c + d = 16) :=
sorry

end smallest_value_l6_6649


namespace part_a_part_b_part_c_part_d_l6_6395

-- Prove that 2^-3 = 1 / 8
theorem part_a : 2^(-3) = 1 / 8 :=
by sorry

-- Prove that (1/3)^-2 = 9
theorem part_b : (1 / 3)^(-2) = 9 :=
by sorry

-- Prove that (2/3)^-4 = 81 / 16
theorem part_c : (2 / 3)^(-4) = 81 / 16 :=
by sorry

-- Prove that (-0.2)^-3 = -125
theorem part_d : (-0.2)^(-3) = -125 :=
by sorry

end part_a_part_b_part_c_part_d_l6_6395


namespace complex_number_fourth_power_l6_6518

theorem complex_number_fourth_power :
    \left(1 + \frac{1}{complex.i}\right)^{4} = -4 := sorry

end complex_number_fourth_power_l6_6518


namespace oldest_child_age_l6_6316

theorem oldest_child_age (a b c : ℕ) (avg_age : ℕ) (h₁ : a = 5) (h₂ : b = 7) (h₃ : c = 9) (h_avg : avg_age = 8) :
  ∃ x : ℕ, (a + b + c + x) / 4 = avg_age ∧ x = 11 :=
by
  use 11
  split
  sorry

end oldest_child_age_l6_6316


namespace geom_seq_formula_sum_b_seq_formula_l6_6102

-- Define the geometric sequence and its required properties
def geometric_seq (q : ℝ) (n : ℕ) : ℝ := 2 * q^(n - 1)

theorem geom_seq_formula :
  (∃ q, geometric_seq q 2 * geometric_seq q 4 = geometric_seq q 6 ∧ geometric_seq q 1 = 2 ∧ ∀ n, geometric_seq q n = 2^n) :=
begin
  use 2,
  split,
  { simp [geometric_seq],
    ring },
  split,
  { simp [geometric_seq] },
  intro n,
  simp [geometric_seq],
  ring_exp
end

-- Define the second sequence and its properties
def b_seq (n : ℕ) : ℝ := 
  let an (n : ℕ) := 2^n in
  1 / (Real.log (an (2*n - 1)) / Real.log 2 * Real.log (an (2*n + 1)) / Real.log 2)
  
theorem sum_b_seq_formula (n : ℕ) :
  (∑ i in Finset.range n, b_seq (i + 1)) = n / (2 * n + 1) :=
begin
  sorry
end

end geom_seq_formula_sum_b_seq_formula_l6_6102


namespace polynomial_factorization_l6_6375

open Polynomial

theorem polynomial_factorization :
  (X ^ 15 + X ^ 10 + X ^ 5 + 1) =
    (X ^ 3 + X ^ 2 + 1) * 
    (X ^ 12 - X ^ 11 + X ^ 9 - X ^ 8 + X ^ 6 - X ^ 5 + X ^ 4 + X ^ 3 + X ^ 2 + X + 1) :=
by
  sorry

end polynomial_factorization_l6_6375


namespace order_of_a_b_c_l6_6924

noncomputable def a := 2 + Real.sqrt 3
noncomputable def b := 1 + Real.sqrt 6
noncomputable def c := Real.sqrt 2 + Real.sqrt 5

theorem order_of_a_b_c : a > c ∧ c > b := 
by {
  sorry
}

end order_of_a_b_c_l6_6924


namespace loan_repayment_l6_6820

variable (M m : ℝ)

-- Assume M and m are positive; otherwise, we need more specific constraints.
theorem loan_repayment (hM : 0 < M) (hm : 0 < m) :
  ∃ a, a = Mm * (1 + m) ^ 10 / ((1 + m) ^ 10 - 1) :=
by
  use Mm * (1 + m) ^ 10 / ((1 + m) ^ 10 - 1)
  sorry

end loan_repayment_l6_6820


namespace solve_for_x_l6_6686

theorem solve_for_x (x : ℝ) :
  (x^2 + x - 2)^3 + (2x^2 - x - 1)^3 = 27 * (x^2 - 1)^3 ↔
  x = 1 ∨ x = -1 ∨ x = -2 ∨ x = -1 / 2 :=
by sorry

end solve_for_x_l6_6686


namespace recurring_decimal_to_fraction_l6_6031

theorem recurring_decimal_to_fraction
  (h : 0.\overline{02} = (2 : ℝ) / 99) :
  2.\overline{06} = 68 / 33 := by
  sorry

end recurring_decimal_to_fraction_l6_6031


namespace monkey_swing_time_l6_6416

-- Definitions based on the given conditions
def speed_running := 15 -- feet per second when running
def time_running := 5 -- seconds
def speed_swinging := 10 -- feet per second when swinging
def total_distance := 175 -- total distance travelled in feet

-- The theorem we want to prove
theorem monkey_swing_time : 
  let distance_running := speed_running * time_running in
  let distance_swinging := total_distance - distance_running in
  let time_swinging := distance_swinging / speed_swinging in
  time_swinging = 10 := 
by
  sorry

end monkey_swing_time_l6_6416


namespace range_of_a_l6_6930

variable (a : ℝ) (x : ℝ)

def p : Prop := -2 ≤ x ∧ x ≤ 10
def q : Prop := 1 - a ≤ x ∧ x ≤ 1 + a

theorem range_of_a :
  (∀ x, p x → q x) ∧ (∃ x, ¬(p x) ∧ q x) ∧ a > 0 ↔ 9 ≤ a := by
  sorry

end range_of_a_l6_6930


namespace repeating_decimal_fraction_l6_6070

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l6_6070


namespace recurring_decimal_to_fraction_l6_6044

theorem recurring_decimal_to_fraction
  (h : (2:ℚ) + 3 * (2 / 99) = 2.06) :
  (2:ℚ) + 0.\overline{06} = (68 / 33) :=
by
  -- Given: 0.\overline{02} = 2 / 99
  have h0 : (0.\overline{02} : ℚ) = 2 / 99 := by sorry

  -- 0.\overline{06} = 3 * 0.\overline{02}
  have h1 : (0.\overline{06} : ℚ) = 3 * (0.\overline{02} : ℚ) :=
    by rw [← h0]; sorry

  -- Hence, 0.\overline{06} = 6 / 99 = 2 / 33
  have h2 : (0.\overline{06} : ℚ) = 2 / 33 :=
    by sorry

  -- Therefore, 2.\overline{06} = 2 + 0.\overline{06} = 2 + 2 / 33 = 68 / 33
  show (2:ℚ) + (0.\overline{06} : ℚ) = 68 / 33
    by sorry

end recurring_decimal_to_fraction_l6_6044


namespace sum_first_30_odd_eq_900_l6_6401

theorem sum_first_30_odd_eq_900 : 
  let n := 30 in
  n^2 = 900 :=
by
  sorry

end sum_first_30_odd_eq_900_l6_6401


namespace valid_n_values_l6_6806

variables (n x y : ℕ)

theorem valid_n_values :
  (n * (x - 3) = y + 3) ∧ (x + n = 3 * (y - n)) →
  (n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 7) :=
by
  sorry

end valid_n_values_l6_6806


namespace gobblean_total_words_l6_6287

-- Define the Gobblean alphabet and its properties.
def gobblean_letters := 6
def max_word_length := 4

-- Function to calculate number of permutations without repetition for a given length.
def num_words (length : ℕ) : ℕ :=
  if length = 1 then 6
  else if length = 2 then 6 * 5
  else if length = 3 then 6 * 5 * 4
  else if length = 4 then 6 * 5 * 4 * 3
  else 0

-- Main theorem stating the total number of possible words.
theorem gobblean_total_words : 
  (num_words 1) + (num_words 2) + (num_words 3) + (num_words 4) = 516 :=
by
  -- Proof is not required
  sorry

end gobblean_total_words_l6_6287


namespace periodic_decimal_to_fraction_l6_6004

theorem periodic_decimal_to_fraction
  (h : ∀ n : ℕ, 0.<digit>02 n / 99) :
  2.0<digit>06 = 68 / 33 :=
sorry

end periodic_decimal_to_fraction_l6_6004


namespace circumcircle_passes_through_focus_l6_6406

/-- Let the tangents to the parabola at points α, β, γ form a triangle ABC. 
    Prove that the circumcircle of triangle ABC passes through the focus of the parabola. -/
theorem circumcircle_passes_through_focus 
  (parabola : Type) 
  (tangents : parabola → parabola → parabola → Type) 
  (focus : parabola)
  (α β γ : parabola) 
  (triangle : tangents α β γ) : 
  circumcircle triangle focus :=
sorry

end circumcircle_passes_through_focus_l6_6406


namespace at_least_one_does_not_land_l6_6811

/-- Proposition stating "A lands within the designated area". -/
def p : Prop := sorry

/-- Proposition stating "B lands within the designated area". -/
def q : Prop := sorry

/-- Negation of proposition p, stating "A does not land within the designated area". -/
def not_p : Prop := ¬p

/-- Negation of proposition q, stating "B does not land within the designated area". -/
def not_q : Prop := ¬q

/-- The proposition "At least one trainee does not land within the designated area" can be expressed as (¬p) ∨ (¬q). -/
theorem at_least_one_does_not_land : (¬p ∨ ¬q) := sorry

end at_least_one_does_not_land_l6_6811


namespace macey_saving_weeks_l6_6273

-- Definitions for conditions
def shirt_cost : ℝ := 3
def amount_saved : ℝ := 1.5
def weekly_saving : ℝ := 0.5

-- Statement of the proof problem
theorem macey_saving_weeks : (shirt_cost - amount_saved) / weekly_saving = 3 := by
  sorry

end macey_saving_weeks_l6_6273


namespace triangle_properties_distance_sum_l6_6634

variable {A B C a b c P : ℝ}
variable {AB AC BC x y d : ℝ}
variable {angleA angleB angleC : ℝ}
variable {sin cos : ℝ → ℝ}

-- Given conditions
def in_triangle (ABC : Type) (AB AC BC : ℝ) (P : Type) (d : ℝ) :=
  AB = 2 ∧ AC = 1 ∧ (cos (2 * angleA) + 2 * (sin ((angleB + angleC) / 2))^2 = 1)

-- Given problem and expected answers 
theorem triangle_properties (h : in_triangle ABC AB AC BC P d) :
  angleA = π / 3 ∧ BC = √3 :=
sorry

theorem distance_sum (hP : in_triangle ABC AB AC BC P d) :
  (d = x + y + (√3 - √3 * x - y)/2) ∧ ∀ x y, (0 ≤ x) ∧ (0 ≤ y) ∧ ∀ d, (d ∈ [√3/2, √3]) :=
sorry

end triangle_properties_distance_sum_l6_6634


namespace length_CE_l6_6763

-- Definitions based on the conditions from step a)
variables (A B C D E F P Q R S : Type)
variables (AB AC BC DE CE : ℝ)
variables (DEF PQRS : Type)

-- Conditions from the problem
def condition1 : AB = 10 := sorry
def condition2 (parallel : ℝ → ℝ → Prop) : parallel DEF AB := sorry
def condition3 (on_segment : Type → Type → Prop) : on_segment D AC := sorry
def condition4 (on_segment : Type → Type → Prop) : on_segment E BC := sorry
def condition5 (bisects_angle : Type → Type → Prop) : bisects_angle (extended AE) (angle FEC) := sorry
def condition6 : DE = 4 := sorry
def condition7 (parallel : Type → Type → Prop) (equal_length : Type → Type → Type → Prop) : 
  parallel PQ AB ∧ equal_length PQ AB := sorry
def condition8 (on_segment : Type → Type → Prop) : on_segment Q BC := sorry
def condition9 (on_segment : Type → Type → Prop) : on_segment P AC := sorry

-- Proof Problem
theorem length_CE : CE = 20 / 3 :=
  sorry

end length_CE_l6_6763


namespace customers_left_l6_6851

theorem customers_left (L : ℕ) (original : ℕ) (new : ℕ) (remaining : ℕ) (h1 : original = 13) (h2 : new = 4) (h3 : remaining = 9) : L = 8 :=
by
  have eq1 : 13 - L + 4 = 9 := by rw [←h1, ←h2, ←h3]
  have eq2 : 13 - L = 5 := by linarith
  have eq3 : -L = -8 := by linarith
  have eq4 : L = 8 := by linarith
  exact eq4

end customers_left_l6_6851


namespace population_capacity_exceeded_in_90_years_l6_6221

def usableLand : ℝ := 32500 
def acresPerPerson : ℝ := 2 
def initialPopulation : ℝ := 500 
def growthFactor : ℝ := 4 
def growthPeriod : ℝ := 30 
def initialYear : ℝ := 2022

def maximumCapacity : ℝ := usableLand / acresPerPerson

theorem population_capacity_exceeded_in_90_years : 
  ∃ yearsAfterInitial, yearsAfterInitial = 90 ∧ 
  initialPopulation * (growthFactor ^ (yearsAfterInitial / growthPeriod)) ≥ maximumCapacity := 
by 
  sorry

end population_capacity_exceeded_in_90_years_l6_6221


namespace sqrt_sum_x_y_eq_two_l6_6531

variable (θ : Real)
noncomputable def x : Real := (3 - Real.cos (4 * θ) + 4 * Real.sin (2 * θ)) / 2
noncomputable def y : Real := (3 - Real.cos (4 * θ) - 4 * Real.sin (2 * θ)) / 2

theorem sqrt_sum_x_y_eq_two (h1 : x + y = 3 - Real.cos (4 * θ)) (h2 : x - y = 4 * Real.sin (2 * θ)) : 
  Real.sqrt x + Real.sqrt y = 2 := sorry

end sqrt_sum_x_y_eq_two_l6_6531


namespace first_person_time_l6_6307

def anthony_time : ℝ := 5
def combined_time : ℝ := 20 / 7

theorem first_person_time : ∃ x : ℝ, (20 / (7 * x) + 20 / 35 = 2) ∧ x = 2 :=
by
  use 2
  have h : 20 / (7 * 2) + 20 / 35 = 2 := by
    calc
      20 / (7 * 2) + 20 / 35
          = 20 / 14 + 20 / 35        : by rfl
      ... = (10 / 7) + (4 / 7)       : by norm_num
      ... = 14 / 7                   : by ring
      ... = 2                        : by norm_num
  exact ⟨h, rfl⟩

end first_person_time_l6_6307


namespace books_brought_back_l6_6753

def initial_books : ℕ := 235
def taken_out_tuesday : ℕ := 227
def taken_out_friday : ℕ := 35
def books_remaining : ℕ := 29

theorem books_brought_back (B : ℕ) :
  B = 56 ↔ (initial_books - taken_out_tuesday + B - taken_out_friday = books_remaining) :=
by
  -- proof steps would go here
  sorry

end books_brought_back_l6_6753


namespace removing_one_has_no_advantage_l6_6781

/-- Defining the list from which one integer will be removed -/
def nums : List ℕ := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

/-- Function to count pairs summing to 13 in a given list -/
def count_pairs_summing_to (s : ℕ) (l : List ℕ) : ℕ :=
l.pairs.filter (λ (p : ℕ × ℕ), p.1 + p.2 = s).length

theorem removing_one_has_no_advantage :
  ∀ n, n ∈ nums →
    count_pairs_summing_to 13 (nums.erase n) = count_pairs_summing_to 13 nums - 1 :=
by 
  sorry

end removing_one_has_no_advantage_l6_6781


namespace number_of_plains_routes_is_81_l6_6186

-- Define the number of cities in each region
def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70

-- Define the number of routes established over three years
def total_routes : ℕ := 150
def routes_per_year : ℕ := 50

-- Define the number of routes connecting pairs of mountainous cities
def mountainous_routes : ℕ := 21

-- Define a function to calculate the number of routes connecting pairs of plains cities
def plains_routes : ℕ :=
  let total_endpoints := total_routes * 2
  let mountainous_endpoints := mountainous_cities * 3
  let plains_endpoints := plains_cities * 3
  let mountainous_pair_endpoints := mountainous_routes * 2
  let mountain_plain_routes := (mountainous_endpoints - mountainous_pair_endpoints) / 2
  let plain_only_endpoints := plains_endpoints - mountain_plain_routes
  plain_only_endpoints / 2

theorem number_of_plains_routes_is_81 : plains_routes = 81 := 
  sorry

end number_of_plains_routes_is_81_l6_6186


namespace find_radius_of_circumsphere_l6_6541

noncomputable def tetrahedron_circumsphere_radius 
  (A B C D : Point ℝ)
  (r small_radius : ℝ)
  (AD : ℝ)
  (angle_BAD angle_CAD angle_BAC : ℝ) 
  (circumcenter_D : Sphere ℝ) 
  (face_sphere : Sphere ℝ) : Prop := 
  small_radius = 1 ∧ 
  AD = 2 * Real.sqrt 3 ∧ 
  angle_BAD = pi / 4 ∧ 
  angle_CAD = pi / 4 ∧ 
  angle_BAC = pi / 3 ∧
  tangential_circumsphere_with_face_sphere face_sphere circumcenter_D A B C D small_radius ∧
  r = 3

axiom tangential_circumsphere_with_face_sphere : ∀ (face_sphere circumsphere : Sphere ℝ) (A B C D : Point ℝ)
  (small_radius : ℝ), 
  face_sphere.radius = 1 -> 
  circumsphere.radius = (r : ℝ) -> 
  face_sphere.is_tangent_at D circumsphere -> 
  face_sphere.is_tangent_to ABC ->
  True ⟫

theorem find_radius_of_circumsphere (A B C D : Point ℝ) 
  (AD : ℝ) 
  (angle_BAD angle_CAD angle_BAC : ℝ) 
  (circumcenter_D face_sphere : Sphere ℝ) 
  (small_radius r : ℝ) :
  tetrahedron_circumsphere_radius A B C D r small_radius AD angle_BAD angle_CAD angle_BAC circumcenter_D face_sphere :=
begin
  sorry
end

end find_radius_of_circumsphere_l6_6541


namespace min_value_fraction_l6_6953

open Real

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (∃ x : ℝ, x = (a / (a + 2 * b) + b / (a + b)) ∧ x ≥ 1 - 1 / (2 * sqrt 2) ∧ x = 1 - 1 / (2 * sqrt 2)) :=
by
  sorry

end min_value_fraction_l6_6953


namespace smallest_n_congruent_5n_eq_n5_mod_7_l6_6489

theorem smallest_n_congruent_5n_eq_n5_mod_7 : ∃ (n : ℕ), n > 0 ∧ (∀ m > 0, 5^m % 7 ≠ m^5 % 7 → m ≥ n) :=
by
  use 6
  -- Proof steps here which are skipped
  sorry

end smallest_n_congruent_5n_eq_n5_mod_7_l6_6489


namespace symmetric_functions_l6_6553

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

theorem symmetric_functions :
  (∀ x : ℝ, f(x) = f(2 - x)) ∧ 
  (∀ x : ℝ, g(x) + g(2 - x) = -4) ∧ 
  (∀ x : ℝ, f(x) + g(x) = 9^x + x^3 + 1) → 
  f(2) * g(2) = 2016 := sorry

end symmetric_functions_l6_6553


namespace Grayson_unanswered_questions_l6_6580

theorem Grayson_unanswered_questions : 
  ∀ (total_questions time_answered min_per_answer : ℕ),
  total_questions = 100 →
  time_answered = 120 →
  min_per_answer = 2 →
  total_questions - (time_answered / min_per_answer) = 40 :=
by
  intros total_questions time_answered min_per_answer h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry -- Proof skipped as instructed

end Grayson_unanswered_questions_l6_6580


namespace corn_bag_price_l6_6372

theorem corn_bag_price
  (cost_seeds: ℕ)
  (cost_fertilizers_pesticides: ℕ)
  (cost_labor: ℕ)
  (total_bags: ℕ)
  (desired_profit_percentage: ℕ)
  (total_cost: ℕ := cost_seeds + cost_fertilizers_pesticides + cost_labor)
  (total_revenue: ℕ := total_cost + (total_cost * desired_profit_percentage / 100))
  (price_per_bag: ℕ := total_revenue / total_bags) :
  cost_seeds = 50 →
  cost_fertilizers_pesticides = 35 →
  cost_labor = 15 →
  total_bags = 10 →
  desired_profit_percentage = 10 →
  price_per_bag = 11 :=
by sorry

end corn_bag_price_l6_6372


namespace altitude_ratio_bound_l6_6655

variable (A B C : Type*) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]

-- Assuming a condition for acute triangle
variable (is_acute_triangle : ∀ (hₐ hᵦ h꜀ : ℝ), hₐ > 0 ∧ hᵦ > 0 ∧ h꜀ > 0)

-- Side lengths of the triangles
variable (a b c : ℝ)

-- Altitudes from vertices A, B, and C
variable (hₐ hᵦ h꜀ : ℝ)

noncomputable def altitude_inequality (hₐ hᵦ h꜀ a b c : ℝ) : Prop :=
  1 / 2 < (hₐ + hᵦ + h꜀) / (a + b + c) ∧ (hₐ + hᵦ + h꜀) / (a + b + c) < 1

theorem altitude_ratio_bound (A B C : Type*) [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C]
  (is_acute_triangle : ∀ (hₐ hᵦ h꜀ : ℝ), hₐ > 0 ∧ hᵦ > 0 ∧ h꜀ > 0)
  (a b c hₐ hᵦ h꜀ : ℝ) : altitude_inequality hₐ hᵦ h꜀ a b c :=
by
  sorry

end altitude_ratio_bound_l6_6655


namespace range_of_a_l6_6976

open Set

def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, p a x → q x) →
  ({ x : ℝ | p a x } ⊆ { x : ℝ | q x }) →
  a ≤ -4 ∨ a ≥ 2 ∨ a = 0 :=
by
  sorry

end range_of_a_l6_6976


namespace distance_to_fountain_l6_6282

def total_distance : ℝ := 120
def number_of_trips : ℝ := 4
def distance_per_trip : ℝ := total_distance / number_of_trips

theorem distance_to_fountain : distance_per_trip = 30 := by
  sorry

end distance_to_fountain_l6_6282


namespace relationship_y1_y2_y3_l6_6667

theorem relationship_y1_y2_y3 (c y1 y2 y3 : ℝ) :
  (y1 = (-(1^2) + 2 * 1 + c))
  ∧ (y2 = (-(2^2) + 2 * 2 + c))
  ∧ (y3 = (-(5^2) + 2 * 5 + c))
  → (y2 > y1 ∧ y1 > y3) :=
by
  intro h
  sorry

end relationship_y1_y2_y3_l6_6667


namespace seventy_fifth_percentile_correct_l6_6680

noncomputable def dataset : List ℝ := [110, 120, 120, 120, 123, 123, 140, 146, 150, 162, 165, 174, 190, 210, 235, 249, 280, 318, 428, 432]

noncomputable def percentile_75 (data : List ℝ) : ℝ :=
  let sorted_data := data.qsort (≤)
  let n := sorted_data.length
  let k := (0.75 * n).floor.toNat
  (sorted_data.get! k + sorted_data.get! (k + 1)) / 2

theorem seventy_fifth_percentile_correct : percentile_75 dataset = 242 := by
  sorry

end seventy_fifth_percentile_correct_l6_6680


namespace ryan_learning_time_l6_6886

theorem ryan_learning_time :
  ∀ (total_time eng_time span_time ch_time : ℝ),
    total_time = 5 →
    eng_time = 2 →
    span_time = 1.5 →
    ch_time = total_time - eng_time - span_time →
    ch_time = 1.5 :=
by
  intros total_time eng_time span_time ch_time
  intros h_total h_eng h_span h_ch
  rw [h_total, h_eng, h_span] at h_ch
  exact h_ch

end ryan_learning_time_l6_6886


namespace arithmetic_geo_sequence_sum_l6_6622

theorem arithmetic_geo_sequence_sum (a : ℕ → ℝ) 
  (h1 : a 2 = 3)
  (h2 : ∃ r : ℝ, r ≠ 0 ∧ a 1 ≠ 0 ∧ a 3 = r * a 1 ∧ a 7 = r^2 * a 1)
  (h3 : ∀ n, a n = 2 + (n - 1) * 1)
  : ∀ (T : ℕ → ℝ) n, 
  (∀ m, T m = ∑ i in finset.range m, (9 / (2 * ∑ j in finset.range (3 * (i + 1)), a (j + 1)))) → 
  (T n = n / (n + 1)) :=
begin
  sorry
end

end arithmetic_geo_sequence_sum_l6_6622


namespace identify_incorrect_quadratic_value_l6_6090

theorem identify_incorrect_quadratic_value
  (a b c : ℝ)
  (f : ℝ → ℝ := λ x, a*x^2 + b*x + c)
  (values : List ℝ := [2500, 2601, 2704, 2900, 3009, 3124, 3249, 3389]) :
  ∃ (incorrect_value : ℝ), incorrect_value = 2900 ∧ incorrect_value ∈ values := 
sorry

end identify_incorrect_quadratic_value_l6_6090


namespace distinct_values_of_T_l6_6986

noncomputable def T (n : ℤ) : ℂ := (1 + complex.I)^n + (1 + complex.I)^(-n)

theorem distinct_values_of_T : (finset.image T (finset.range 8)).card = 2 := 
sorry

end distinct_values_of_T_l6_6986


namespace g_at_9_l6_6718

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g_at_3 : g 3 = 4

theorem g_at_9 : g 9 = 64 :=
by
  sorry

end g_at_9_l6_6718


namespace repeating_decimal_as_fraction_l6_6040

theorem repeating_decimal_as_fraction : (0.\overline{02} = 2 / 99) → (2.\overline{06} = 68 / 33) :=
by
  sorry

end repeating_decimal_as_fraction_l6_6040


namespace smallest_positive_period_range_of_f_l6_6579

noncomputable def ω : ℝ := 5 / 6

def a (x : ℝ) : ℝ × ℝ := (Real.cos (ω * x) - Real.sin (ω * x), Real.sin (ω * x))

def b (x : ℝ) : ℝ × ℝ := (-Real.cos (ω * x) - Real.sin (ω * x), 2 * Real.sqrt 3 * Real.cos (ω * x))

def f (x : ℝ) (λ : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 + λ

theorem smallest_positive_period {λ : ℝ}
  (sym : ∀ x, f (x + 2 * π) λ = f (2 * π - x) λ)
  (pass_through : f (π / 4) λ = 0) :
  ∃ T, T = 6 * π / 5 :=
sorry

theorem range_of_f {λ : ℝ}
  (sym : ∀ x, f (x + 2 * π) λ = f (2 * π - x) λ)
  (pass_through : f (π / 4) λ = 0)
  (hλ : λ = -Real.sqrt 2) :
  ∀ x ∈ Set.Icc 0 (3 * π / 5), f x λ ∈ Set.Icc (-1 - Real.sqrt 2) (2 - Real.sqrt 2) :=
sorry

end smallest_positive_period_range_of_f_l6_6579


namespace isabella_hair_length_end_of_year_l6_6637

/--
Isabella's initial hair length.
-/
def initial_hair_length : ℕ := 18

/--
Isabella's hair growth over the year.
-/
def hair_growth : ℕ := 6

/--
Prove that Isabella's hair length at the end of the year is 24 inches.
-/
theorem isabella_hair_length_end_of_year : initial_hair_length + hair_growth = 24 := by
  sorry

end isabella_hair_length_end_of_year_l6_6637


namespace contestant_needs_median_to_enter_top_5_l6_6224
 
theorem contestant_needs_median_to_enter_top_5
  (scores : List ℝ)
  (h_length : scores.length = 9)
  (h_distinct : scores.nodup) :
  ∃ median, 
  (List.sort ≤ scores) !! 4 = some median :=
by
  sorry

end contestant_needs_median_to_enter_top_5_l6_6224


namespace value_of_a_l6_6326

theorem value_of_a (a : ℝ) (h : a > 0 ∧ a ≠ 1 ∧ (∃ (y : ℝ), y = 2 ∧ 9 = a ^ y)) : a = 3 := 
  by sorry

end value_of_a_l6_6326


namespace triangle_is_obtuse_l6_6734

-- Definitions based on given conditions
def is_obtuse_triangle (a b c : ℝ) : Prop :=
  if a ≥ b ∧ a ≥ c then a^2 > b^2 + c^2
  else if b ≥ a ∧ b ≥ c then b^2 > a^2 + c^2
  else c^2 > a^2 + b^2

def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a

-- Statement to prove
theorem triangle_is_obtuse : is_triangle 4 6 8 ∧ is_obtuse_triangle 4 6 8 :=
by
  sorry

end triangle_is_obtuse_l6_6734


namespace petes_travel_time_to_madison_l6_6866

noncomputable def pete_travel_time (
  distance_in_inches : ℝ := 5,
  map_scale_in_inches_per_mile : ℝ := 0.01282051282051282,
  average_speed_in_mph : ℝ := 60
) : ℝ :=
  let distance_in_miles := distance_in_inches / map_scale_in_inches_per_mile
  distance_in_miles / average_speed_in_mph

theorem petes_travel_time_to_madison : 
  pete_travel_time 5 0.01282051282051282 60 = 6.5 :=
by
  -- these steps transform to the hypothesis calculation
  let distance_in_miles := 5 / 0.01282051282051282
  have actual_distance_miles_approx : distance_in_miles = 390 := by norm_num
  rw [actual_distance_miles_approx]
  norm_num  -- This ensures simplification of the fraction 390 / 60
  exact eq.refl 6.5


end petes_travel_time_to_madison_l6_6866


namespace value_of_a_l6_6482

def star (a b : ℝ) : ℝ := 3 * a - 2 * b ^ 2

theorem value_of_a (a : ℝ) (h : star a 3 = 15) : a = 11 := 
by
  sorry

end value_of_a_l6_6482


namespace find_angle_A_find_sides_b_c_l6_6118

-- Define the types and given conditions
variables {α : Type*} [field α] [has_sqrt α] [has_cos α] [has_sin α]
variables (a b c : α) (A B C : α)

-- Assumptions:
-- α is a type with field operations, sqrt, cosine and sine functions
-- a, b, c are the sides opposite to angles A, B, C respectively in triangle ABC
-- Given conditions for angles and sides
axiom h1 : a * cos C + (sqrt 3) * a * sin C = b + c

-- Problem 1: Find angle A
theorem find_angle_A : A = π / 3 :=
sorry

-- Problem 2: Given a = sqrt 7 and area of triangle ABC
axiom h2 : a = sqrt 7
axiom h3 : 1 / 2 * b * c * sin A = (3 * sqrt 3) / 2
axiom h4 : A = π / 3

-- Calculate the required values
theorem find_sides_b_c : b = 2 ∧ c = 3 ∨ b = 3 ∧ c = 2 :=
sorry

end find_angle_A_find_sides_b_c_l6_6118


namespace recurring_decimal_to_fraction_l6_6049

theorem recurring_decimal_to_fraction
  (h : (2:ℚ) + 3 * (2 / 99) = 2.06) :
  (2:ℚ) + 0.\overline{06} = (68 / 33) :=
by
  -- Given: 0.\overline{02} = 2 / 99
  have h0 : (0.\overline{02} : ℚ) = 2 / 99 := by sorry

  -- 0.\overline{06} = 3 * 0.\overline{02}
  have h1 : (0.\overline{06} : ℚ) = 3 * (0.\overline{02} : ℚ) :=
    by rw [← h0]; sorry

  -- Hence, 0.\overline{06} = 6 / 99 = 2 / 33
  have h2 : (0.\overline{06} : ℚ) = 2 / 33 :=
    by sorry

  -- Therefore, 2.\overline{06} = 2 + 0.\overline{06} = 2 + 2 / 33 = 68 / 33
  show (2:ℚ) + (0.\overline{06} : ℚ) = 68 / 33
    by sorry

end recurring_decimal_to_fraction_l6_6049


namespace animal_arrangement_l6_6313

noncomputable def factorial : ℕ → ℕ
| 0     := 1
| 1     := 1
| (n+1) := (n+1) * factorial n

theorem animal_arrangement :
  let rabbits := 5
  let dogs := 3
  let goats := 4
  let parrots := 2
  factorial 4 * factorial rabbits * factorial dogs * factorial goats * factorial parrots = 414720 :=
by
  sorry

end animal_arrangement_l6_6313


namespace transformed_function_range_l6_6127

theorem transformed_function_range (x : ℝ) (h : 1 ≤ x ∧ x ≤ 3) : 
  let f : ℝ → ℝ := λ x, 2 * x + 1 in
  (f (x - 1) = 2 * x - 1) ∧ (2 ≤ x ∧ x ≤ 4) :=
by
  sorry

end transformed_function_range_l6_6127


namespace simplify_expression_l6_6925

variable (a b c : ℝ) 

theorem simplify_expression (h1 : a ≠ 4) (h2 : b ≠ 5) (h3 : c ≠ 6) :
  (a - 4) / (6 - c) * (b - 5) / (4 - a) * (c - 6) / (5 - b) = -1 :=
by
  sorry

end simplify_expression_l6_6925


namespace part1_part2_l6_6554

noncomputable def angle_A (A : ℝ) : Prop :=
  (sin (π / 3 - A) * cos (π / 6 + A) = 1 / 4) ∧
  A < π / 2

theorem part1 (A : ℝ) : angle_A A → A = π / 6 :=
by
  sorry

noncomputable def area_triangle (a b c : ℝ) (A C B : ℝ) : Prop :=
  a * sin A + c * sin C = 4 * sqrt 3 * sin B ∧
  b = sqrt 3 ∧
  A = π / 6

theorem part2 (a b c : ℝ) (A C B : ℝ) (S : ℝ) :
  area_triangle a b c A C B →
  (S = (1 / 2) * b * c * sin A) → S = (3 * sqrt 3) / 4 :=
by
  sorry

end part1_part2_l6_6554


namespace max_value_of_m_l6_6540

noncomputable theory
open Classical

variable (S : ℕ → ℝ)
variable (a : ℕ → ℝ)
variable (n m : ℕ)

-- Conditions
axiom a1 : a 1 = 1 / 3
axiom a2 : ∀ n ≥ 2, (S n - 1) ^ 2 = a n * S n
axiom a3 : ∀ n ≥ 2, a n = S n - S (n-1)

-- Question:
theorem max_value_of_m (h : S m < 19 / 21) : m ≤ 9 :=
sorry

end max_value_of_m_l6_6540


namespace find_x_l6_6137

def sequence {x : ℝ} (n : ℕ) : ℝ :=
  n * x / finset.prod (finset.range n + 1) (λ k, (k * x + 1))

theorem find_x (x : ℝ) :
  (finset.sum (finset.range 2016) sequence) < 1 →
  x = -11 / 60 :=
begin
  sorry
end

end find_x_l6_6137


namespace three_consecutive_edges_same_color_l6_6821

-- Definitions based on problem conditions
def convex_polyhedron (V : Type) := ∀ v : V, (vertex_degree v = 3 ∨ vertex_degree v = 5 ∧ v = A)

def good_coloring (E : Type) (color : E → ℕ) := 
  ∀ v : vertex_degree v = 3, ∃ c₁ c₂ c₃, (c₁ ≠ c₂ ∧ c₂ ≠ c₃ ∧ c₁ ≠ c₃ ∧ all edges from v have colors c₁, c₂, and c₃ respectively)

-- The given polyhedron cannot have good colorings divisible by 5
axiom not_divisible_by_five (V : Type) [convex_polyhedron V] : ¬ (number_of_good_colorings V) % 5 = 0

-- The theorem to be proved
theorem three_consecutive_edges_same_color (V : Type) [convex_polyhedron V] (E : Type) (color : E → ℕ) :
  ∃ good_c : good_coloring E color, ∃ A_edges : list (E), (A_edges.length = 5 ∧ consecutive_same_color A_edges ∧ vertices A in E ) := sorry

end three_consecutive_edges_same_color_l6_6821


namespace length_more_than_breadth_l6_6329

theorem length_more_than_breadth (b x : ℕ) 
  (h1 : 60 = b + x) 
  (h2 : 4 * b + 2 * x = 200) : x = 20 :=
by {
  sorry
}

end length_more_than_breadth_l6_6329


namespace number_of_period_π_functions_l6_6486

def tan_period := π
def abs_sin_period := π
def cos_shifted_period := π

theorem number_of_period_π_functions :
  (∀ f ∈ {tan_period, abs_sin_period, cos_shifted_period}, f = π) →
  ({tan_period, abs_sin_period, cos_shifted_period}.card = 3) := by
  intros h
  -- insert proof steps here
  sorry

end number_of_period_π_functions_l6_6486


namespace number_of_plains_routes_is_81_l6_6187

-- Define the number of cities in each region
def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70

-- Define the number of routes established over three years
def total_routes : ℕ := 150
def routes_per_year : ℕ := 50

-- Define the number of routes connecting pairs of mountainous cities
def mountainous_routes : ℕ := 21

-- Define a function to calculate the number of routes connecting pairs of plains cities
def plains_routes : ℕ :=
  let total_endpoints := total_routes * 2
  let mountainous_endpoints := mountainous_cities * 3
  let plains_endpoints := plains_cities * 3
  let mountainous_pair_endpoints := mountainous_routes * 2
  let mountain_plain_routes := (mountainous_endpoints - mountainous_pair_endpoints) / 2
  let plain_only_endpoints := plains_endpoints - mountain_plain_routes
  plain_only_endpoints / 2

theorem number_of_plains_routes_is_81 : plains_routes = 81 := 
  sorry

end number_of_plains_routes_is_81_l6_6187


namespace shaded_area_is_64_l6_6228

-- Define the problem conditions
variables (t : ℝ) (ht : t = 12)

-- Define the area of the entire square
def total_area : ℝ := t ^ 2

-- Define the fraction of the shaded area
def shaded_fraction : ℝ := 4 / 9

-- The shaded area based on the given conditions
def shaded_area : ℝ := shaded_fraction * total_area

-- Prove that the shaded area is 64 when t = 12
theorem shaded_area_is_64 (ht : t = 12) : shaded_area t = 64 :=
by sorry

end shaded_area_is_64_l6_6228


namespace runners_meet_again_l6_6349

theorem runners_meet_again :
    ∀ t : ℝ,
      t ≠ 0 →
      (∃ k : ℤ, 3.8 * t - 4 * t = 400 * k) ∧
      (∃ m : ℤ, 4.2 * t - 4 * t = 400 * m) ↔
      t = 2000 := 
by
  sorry

end runners_meet_again_l6_6349


namespace distance_greater_than_two_l6_6660

theorem distance_greater_than_two (x : ℝ) (h : |x| > 2) : x > 2 ∨ x < -2 :=
sorry

end distance_greater_than_two_l6_6660


namespace surface_area_of_solid_l6_6495

theorem surface_area_of_solid (num_unit_cubes : ℕ) (top_layer_cubes : ℕ) 
(bottom_layer_cubes : ℕ) (side_layer_cubes : ℕ) 
(front_and_back_cubes : ℕ) (left_and_right_cubes : ℕ) :
  num_unit_cubes = 15 →
  top_layer_cubes = 5 →
  bottom_layer_cubes = 5 →
  side_layer_cubes = 3 →
  front_and_back_cubes = 5 →
  left_and_right_cubes = 3 →
  let top_and_bottom_surface := top_layer_cubes + bottom_layer_cubes
  let front_and_back_surface := 2 * front_and_back_cubes
  let left_and_right_surface := 2 * left_and_right_cubes
  let total_surface := top_and_bottom_surface + front_and_back_surface + left_and_right_surface
  total_surface = 26 :=
by
  intros h_n h_t h_b h_s h_f h_lr
  let top_and_bottom_surface := top_layer_cubes + bottom_layer_cubes
  let front_and_back_surface := 2 * front_and_back_cubes
  let left_and_right_surface := 2 * left_and_right_cubes
  let total_surface := top_and_bottom_surface + front_and_back_surface + left_and_right_surface
  sorry

end surface_area_of_solid_l6_6495


namespace calculate_midpoint_polar_l6_6614

noncomputable def midpoint_polar := 
  let A := (10, Real.pi / 6)
  let B := (10, 5 * Real.pi / 6)
  let M := (5, Real.pi / 2)
  M

theorem calculate_midpoint_polar:
  let A := (10, Real.pi / 6)
  let B := (10, 5 * Real.pi / 6) 
  let M := (5, Real.pi / 2)
  (r > 0) ∧ (0 ≤ θ < 2 * Real.pi) ∧ 
  ((A, B) = ((10, Real.pi / 6), (10, 5 * Real.pi / 6)) → 
  (midpoint_polar A B) = M) :=
by
  sorry

end calculate_midpoint_polar_l6_6614


namespace value_of_x_l6_6402

noncomputable def k := 9

theorem value_of_x (y : ℝ) (h1 : y = 3) (h2 : ∀ (x : ℝ), x = 2.25 → x = k / (2 : ℝ)^2) : 
  ∃ (x : ℝ), x = 1 := by
  sorry

end value_of_x_l6_6402


namespace routes_between_plains_cities_correct_l6_6183

noncomputable def number_of_routes_connecting_pairs_of_plains_cities
    (total_cities : ℕ)
    (mountainous_cities : ℕ)
    (plains_cities : ℕ)
    (total_routes : ℕ)
    (routes_between_mountainous_cities : ℕ) : ℕ :=
let mountainous_city_endpoints := mountainous_cities * 3 in
let routes_between_mountainous_cities_endpoints := routes_between_mountainous_cities * 2 in
let mountainous_to_plains_routes_endpoints := mountainous_city_endpoints - routes_between_mountainous_cities_endpoints in
let plains_city_endpoints := plains_cities * 3 in
let plains_city_to_mountainous_city_routes_endpoints := mountainous_to_plains_routes_endpoints in
let endpoints_fully_in_plains_cities := plains_city_endpoints - plains_city_to_mountainous_city_routes_endpoints in
endpoints_fully_in_plains_cities / 2

theorem routes_between_plains_cities_correct :
    number_of_routes_connecting_pairs_of_plains_cities 100 30 70 150 21 = 81 := by
    sorry

end routes_between_plains_cities_correct_l6_6183


namespace reachable_town_exists_l6_6757

-- Define the type for nodes (towns)
inductive Town
| A : ℕ → Town

-- Define the type for edges (roads) that can be traveled by motorcycle or by car
inductive Road
| motorcycle : Town → Town → Road
| car : Town → Town → Road

-- Define the graph structure representing the sub-country
structure SubCountry :=
  (towns : fin n → Town)
  (roads : list Road)

-- The main theorem to prove
theorem reachable_town_exists (n : ℕ) (sc : SubCountry) :
  ∃ town : Town, ∀ other_town : Town, 
    (∃ r : Road, r ∈ sc.roads ∧ r = Road.motorcycle town other_town) ∨
    (∃ r : Road, r ∈ sc.roads ∧ r = Road.car town other_town) :=
sorry

end reachable_town_exists_l6_6757


namespace change_received_l6_6693

def smoky_salmon : ℝ := 40
def black_burger : ℝ := 15
def chicken_katsu : ℝ := 25
def seafood_pasta : ℝ := 30
def truffled_mac_and_cheese : ℝ := 20
def bottle_of_wine : ℝ := 50
def food_discount_rate : ℝ := 0.1
def service_charge_rate : ℝ := 0.12
def additional_tip_rate : ℝ := 0.05
def payment_amount : ℝ := 300

/-- The total change Mr. Arevalo receives after paying a $300 bill at the restaurant. -/
theorem change_received :
  let total_food_cost := smoky_salmon + black_burger + chicken_katsu + seafood_pasta + truffled_mac_and_cheese,
      total_cost_including_wine := total_food_cost + bottle_of_wine,
      service_charge := service_charge_rate * total_cost_including_wine,
      total_bill_before_discount := total_cost_including_wine + service_charge,
      discount_on_food := food_discount_rate * total_food_cost,
      total_bill_after_discount := total_bill_before_discount - discount_on_food,
      additional_tip := additional_tip_rate * total_bill_after_discount,
      final_amount := total_bill_after_discount + additional_tip
  in 
  (payment_amount - final_amount) = 101.97 := 
sorry

end change_received_l6_6693


namespace positive_square_root_of_256_l6_6150

theorem positive_square_root_of_256 (y : ℝ) (hy_pos : y > 0) (hy_squared : y^2 = 256) : y = 16 :=
by
  sorry

end positive_square_root_of_256_l6_6150


namespace polynomial_factorization_l6_6376

open Polynomial

theorem polynomial_factorization :
  (X ^ 15 + X ^ 10 + X ^ 5 + 1) =
    (X ^ 3 + X ^ 2 + 1) * 
    (X ^ 12 - X ^ 11 + X ^ 9 - X ^ 8 + X ^ 6 - X ^ 5 + X ^ 4 + X ^ 3 + X ^ 2 + X + 1) :=
by
  sorry

end polynomial_factorization_l6_6376


namespace coeff_x2_in_expansion_l6_6875

theorem coeff_x2_in_expansion : 
  let T (r : ℕ) := (Nat.choose 4 r) * (-2)^r * x^(4 - r)
  ∃ (r : ℕ), 4 - r = 2 ∧ coeff (T r) x^2 = 24 :=
by
  sorry

end coeff_x2_in_expansion_l6_6875


namespace domain_not_neg1_increasing_on_neg1_infty_min_max_on_3_5_l6_6968

noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / (x + 1)

namespace f_props

theorem domain_not_neg1 : ∀ x : ℝ, x ≠ -1 ↔ x ∈ {y | y ≠ -1} :=
by simp [f]

theorem increasing_on_neg1_infty : ∀ x1 x2 : ℝ, -1 < x1 → x1 < x2 → -1 < x2 → f x1 < f x2 :=
sorry

theorem min_max_on_3_5 : (∀ y : ℝ, y = f 3 → y = 5 / 4) ∧ (∀ y : ℝ, y = f 5 → y = 3 / 2) :=
sorry

end f_props

end domain_not_neg1_increasing_on_neg1_infty_min_max_on_3_5_l6_6968


namespace plains_routes_l6_6204

theorem plains_routes 
  (total_cities : ℕ)
  (mountainous_cities : ℕ)
  (plains_cities : ℕ)
  (total_routes : ℕ)
  (mountainous_routes : ℕ)
  (num_pairs_with_mount_to_mount : ℕ)
  (routes_per_year : ℕ)
  (years : ℕ)
  (mountainous_roots_connections : ℕ)
  : (mountainous_cities = 30) →
    (plains_cities = 70) →
    (total_cities = mountainous_cities + plains_cities) →
    (routes_per_year = 50) →
    (years = 3) →
    (total_routes = routes_per_year * years) →
    (mountainous_routes = num_pairs_with_mount_to_mount * 2) →
    (num_pairs_with_mount_to_mount = 21) →
    let num_endpoints_per_city_route = 2 in
    let mountainous_city_endpoints = mountainous_cities * 3 in
    let mountainous_endpoints = mountainous_routes in
    let mountain_to_plains_endpoints = mountainous_city_endpoints - mountainous_endpoints in
    let total_endpoints = total_routes * num_endpoints_per_city_route in
    let plains_city_endpoints = plains_cities * 3 in
    let routes_between_plain_and_mountain = mountain_to_plains_endpoints in
    let plain_to_plain_endpoints = plains_city_endpoints - routes_between_plain_and_mountain in
    let plain_to_plain_routes = plain_to_plain_endpoints / 2 in
    plain_to_plain_routes = 81 :=
sorry

end plains_routes_l6_6204


namespace polynomial_factorization_l6_6378

theorem polynomial_factorization :
  (x : ℤ[X]) →
  x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by
  intros x
  sorry

end polynomial_factorization_l6_6378


namespace train_length_calculation_l6_6454

noncomputable def length_of_train (speed : ℝ) (time_in_sec : ℝ) : ℝ :=
  let time_in_hr := time_in_sec / 3600
  let distance_in_km := speed * time_in_hr
  distance_in_km * 1000

theorem train_length_calculation : 
  length_of_train 60 30 = 500 :=
by
  -- The proof would go here, but we provide a stub with sorry.
  sorry

end train_length_calculation_l6_6454


namespace f_of_g_of_neg3_l6_6597

def f (x : ℝ) : ℝ := 4 - Real.sqrt x
def g (x : ℝ) : ℝ := 3 * x + 3 * x^2

theorem f_of_g_of_neg3 : f (g (-3)) = 4 - 3 * Real.sqrt 2 :=
by
  sorry

end f_of_g_of_neg3_l6_6597


namespace students_contribution_l6_6431

theorem students_contribution (n x : ℕ) 
  (h₁ : ∃ (k : ℕ), k * 9 = 22725)
  (h₂ : n * x = k / 9)
  : (n = 5 ∧ x = 505) ∨ (n = 25 ∧ x = 101) :=
sorry

end students_contribution_l6_6431


namespace plains_routes_count_l6_6198

def total_cities : ℕ := 100
def mountainous_cities : ℕ := 30
def plains_cities : ℕ := 70
def total_routes : ℕ := 150
def mountainous_routes : ℕ := 21

theorem plains_routes_count :
  total_cities = mountainous_cities + plains_cities →
  3 * total_routes = total_cities →
  mountainous_routes * 2 ≤ mountainous_cities * 3 →
  (total_routes - mountainous_routes) * 2 = (70 * 3 - (mountainous_routes * 2)) →
  (total_routes - mountainous_routes * 2) / 2 = 81 :=
begin
  sorry
end

end plains_routes_count_l6_6198


namespace max_value_ineq_l6_6545

theorem max_value_ineq (x y : ℝ) (h : x^2 + y^2 = 20) : xy + 8*x + y ≤ 42 := by
  sorry

end max_value_ineq_l6_6545


namespace equation_of_E_HN_fixed_point_l6_6960

-- Condition definitions
def center_origin : Prop := true 
def axes_of_symmetry_origin : Prop := true
def passes_through_A : Prop := (0, -2) ∈ { (x, y) | x^2 / 3 + y^2 / 4 = 1 }
def passes_through_B : Prop := (3 / 2, -1) ∈ { (x, y) | x^2 / 3 + y^2 / 4 = 1 }

-- Main proofs
theorem equation_of_E :
  center_origin →
  axes_of_symmetry_origin →
  passes_through_A →
  passes_through_B →
  (∀ (x y : ℝ), (x, y) ∈ { (x, y) | x^2 / 3 + y^2 / 4 = 1 }) :=
by
  intros _ _ _ _
  sorry

-- Separate geometric conditions for part (2)
def point_P : (ℝ × ℝ) := (1, -2)
def intersection_with_E (x y : ℝ) : Prop := (x, y) ∈ { (x, y) | x^2 / 3 + y^2 / 4 = 1 }
def parallel_to_x_axis (y : ℝ) : ℝ → ℝ := λ x, y
def segment_AB (x : ℝ) : ℝ := (2/3) * x - 2
def point_H (Mx My Tx : ℝ) : (ℝ × ℝ) := (2 * Tx - Mx, My)

-- Fixed point theorem
theorem HN_fixed_point :
  ∀ (M N : ℝ × ℝ), 
  let HN_line (x1 y1 x2 y2 Tx : ℝ) := ∃ k b, y2 = k * x2 + b ∧ (0, -2) ∈ { (x, y) | y = k * x + b } in
  center_origin →
  axes_of_symmetry_origin →
  passes_through_A →
  passes_through_B →
  intersection_with_E M.1 M.2 →
  intersection_with_E N.1 N.2 →
  HN_line M.1 M.2 N.1 N.2 (M.1) :=
by
  intros M N _ _ _ _ _ _ _
  sorry

end equation_of_E_HN_fixed_point_l6_6960


namespace correct_division_powers_statements_l6_6822

theorem correct_division_powers_statements (a : ℚ) (h : a ≠ 0) :
  (a / a = 1) ∧ ((a / a) / a = 1 / a) ∧ (∀ n : ℕ, even n → ∀ b : ℚ, b < 0 → ((-b)^(2 * n)) > 0) ∧ 
  (∀ n : ℕ, odd n → ∀ b : ℚ, b < 0 → ((-b)^(2 * n + 1)) < 0) :=
by
  sorry

end correct_division_powers_statements_l6_6822


namespace smallest_positive_n_l6_6785

theorem smallest_positive_n (n : ℤ) (h : n > 0) (hmod : 3 * n ≡ 1410 [MOD 24]) : n = 6 := 
sorry

end smallest_positive_n_l6_6785


namespace find_positive_n_unique_solution_l6_6900

theorem find_positive_n_unique_solution (n : ℝ) :
  (∃ x : ℝ, 4 * x^2 + n * x + 4 = 0) → n = 8 :=
by
  intro h
  sorry

end find_positive_n_unique_solution_l6_6900


namespace recycling_target_l6_6684

/-- Six Grade 4 sections launched a recycling drive where they collect old newspapers to recycle.
Each section collected 280 kilos in two weeks. After the third week, they found that they need 320 kilos more to reach their target.
  How many kilos of the newspaper is their target? -/
theorem recycling_target (sections : ℕ) (kilos_collected_2_weeks : ℕ) (additional_kilos : ℕ) : 
  sections = 6 ∧ kilos_collected_2_weeks = 280 ∧ additional_kilos = 320 → 
  (sections * (kilos_collected_2_weeks / 2) * 3 + additional_kilos) = 2840 :=
by
  sorry

end recycling_target_l6_6684


namespace distance_from_M_to_other_focus_is_4_l6_6557

noncomputable def distance_to_other_focus (M : ℝ × ℝ) (focus1 focus2 : ℝ × ℝ) (d : ℝ) : ℝ :=
  let ellipse (x y : ℝ) := x^2 / 16 + y^2 / 8 = 1
  in
  if ellipse M.1 M.2 ∧ dist M focus1 = d ∧ dist M focus1 = 4 then
    dist M focus2
  else
    0

theorem distance_from_M_to_other_focus_is_4 (M : ℝ × ℝ) :
  (M.1^2 / 16 + M.2^2 / 8 = 1) →  
  ∃ focus1 focus2 : ℝ × ℝ, dist M focus1 = 4 ∧ dist M focus2 = 4 :=
by
  assume ellipse_eq : M.1^2 / 16 + M.2^2 / 8 = 1
  have focus_distance_eq : ∀ focus1 focus2 : ℝ × ℝ, dist M focus1 = 4 → dist M focus2 = 4 :=
  sorry
  exact ⟨_, _, focus_distance_eq⟩

end distance_from_M_to_other_focus_is_4_l6_6557


namespace arithmetic_sequence_property_l6_6945

def arith_seq (a : ℕ → ℤ) (a1 a3 : ℤ) (d : ℤ) : Prop :=
  a 1 = a1 ∧ a 3 = a3 ∧ (a 3 - a 1) = 2 * d

theorem arithmetic_sequence_property :
  ∀ (a : ℕ → ℤ), ∃ d : ℤ, arith_seq a 1 (-3) d →
  (1 - (a 2) - a 3 - (a 4) - (a 5) = 17) :=
by
  intros a
  use -2
  simp [arith_seq, *]
  sorry

end arithmetic_sequence_property_l6_6945


namespace circumcircle_with_double_radius_l6_6232

/-- 
  Given a triangle ABC with an incenter I (denoted as \odot I),
  where \odot I is tangent to BC, CA, and AB at points D, E, and F respectively.
  Let AI and BI intersect \odot I at points A1, A0, B1, and B0 with |AA1| < |AA0| and |BB1| < |BB0|.
  Lines parallel to AB through A0 and B0 intersect \odot I at points CA and CB.
  Perpendiculars from F to CAF and CBF intersect CAA1 and CBB1 at points C3 and C4.
  Lines AC3 and BC4 intersect at point C'.
  Similarly, define points A' and B'.
  Prove that the circumcircle of ΔA'B'C' has a radius that is twice the radius of \odot I.
-/
theorem circumcircle_with_double_radius
  (ABC : Triangle)
  (I : Point)
  (r : ℝ) 
  (D E F A1 A0 B1 B0 CA CB C3 C4 C' A' B' : Point) 
  (h_incenter_tangent : I.circle_tangent BC at D ∧ I.circle_tangent CA at E ∧ I.circle_tangent AB at F)
  (h_AI_intersects : [AI meets I at A1 and A0 with |AA1| < |AA0|])
  (h_BI_intersects : [BI meets I at B1 and B0 with |BB1| < |BB0|])
  (h_parallels_intersect_incircle : [lines parallel to AB through A0 and B0 intersect I's circle at CA and CB])
  (h_perp_from_F_to_CAF_and_CBF : [perpendicular from F to CAF intersects CAA1 at C3] ∧ [perpendicular from F to CBF intersects CBB1 at C4])
  (h_lines_intersect : [lines AC3 and BC4 intersect at C'])
  (h_points_defined : [A' and B' defined similarly])
  : circumcircle_radius A' B' C' = 2 * r := sorry

end circumcircle_with_double_radius_l6_6232


namespace correct_options_l6_6790

theorem correct_options :
  (1 + Real.tan 1) * (1 + Real.tan 44) = 2 ∧
  ¬((1 / Real.sin 10) - (Real.sqrt 3 / Real.cos 10) = 2) ∧
  (3 - Real.sin 70) / (2 - (Real.cos 10) ^ 2) = 2 ∧
  ¬(Real.tan 70 * Real.cos 10 * (Real.sqrt 3 * Real.tan 20 - 1) = 2) :=
sorry

end correct_options_l6_6790


namespace smallest_positive_period_monotonic_increase_interval_zero_at_2pi_over_3_and_range_l6_6132

noncomputable def f (x : ℝ) (a : ℝ) := 4 * cos x * cos (x - π / 3) + a

theorem smallest_positive_period 
  (a : ℝ) : 
  (∀ x, f x a = f (x + π) a) := 
sorry

theorem monotonic_increase_interval 
  (a : ℝ) : 
  (∀ x, x ∈ Icc 0 (π / 6) → (2 * sin (2 * x + π / 6) + a + 1) = f x a ∧ 
     (∀ y, (y ∈ Icc 0 x) → f y a ≤ f x a)) :=
sorry

theorem zero_at_2pi_over_3_and_range
  (a : ℝ) 
  (h0 : f (2 * π / 3) a = 0) : 
  (a = 1 ∧ ∀ x, x ∈ Icc 0 (π / 2) → f x a ∈ Icc 1 4) := 
sorry

end smallest_positive_period_monotonic_increase_interval_zero_at_2pi_over_3_and_range_l6_6132


namespace gcd_8154_8640_l6_6897

theorem gcd_8154_8640 : Nat.gcd 8154 8640 = 6 := by
  sorry

end gcd_8154_8640_l6_6897


namespace norm_a_add_b_norm_a_sub_b_angle_between_a_add_b_and_a_sub_b_l6_6534

variables {a b : EuclideanSpace ℝ (Fin 3)}

-- Given conditions
def norm_a : ∥a∥ = 6 := sorry
def norm_b : ∥b∥ = 6 := sorry
def angle_between_a_b : real.angle_between a b = real.pi / 3 := sorry

-- Proof problem 1: Proving |a + b| = 6√3
theorem norm_a_add_b : ∥a + b∥ = 6 * real.sqrt 3 := by
  have h1 : |a + b|^2 = |a|^2 + |b|^2 + 2 * (a • b) := sorry
  have h2 : a • b = |a| * |b| * real.cos (real.pi / 3) := sorry
  have h3 : |a|^2 = 36 := sorry
  have h4 : |b|^2 = 36 := sorry
  have h5 : a • b = 18 := sorry
  show |a + b| = 6 * real.sqrt 3 from sorry

-- Proof problem 2: Proving |a - b| = 6
theorem norm_a_sub_b : ∥a - b∥ = 6 := by
  have h1 : |a - b|^2 = |a|^2 + |b|^2 - 2 * (a • b) := sorry
  have h2 : a • b = |a| * |b| * real.cos (real.pi / 3) := sorry
  have h3 : |a|^2 = 36 := sorry
  have h4 : |b|^2 = 36 := sorry
  have h5 : a • b = 18 := sorry
  show ∥a - b∥ = 6 from sorry

-- Proof problem 3: Proving the angle between (a + b) and (a - b) is π/2
theorem angle_between_a_add_b_and_a_sub_b : 
  real.angle_between (a + b) (a - b) = real.pi / 2 := by
  have h1 : (a + b) • (a - b) = 0 := sorry
  show real.angle (a + b) (a - b) = real.pi / 2 from sorry

end norm_a_add_b_norm_a_sub_b_angle_between_a_add_b_and_a_sub_b_l6_6534


namespace isosceles_parallelogram_perimeter_constant_l6_6918

open Set

-- Define the necessary points for the isosceles triangle and the parallelogram
variables {A B C D E F : Point} {AB AC BC : Line} 

-- Definitions based on the given conditions
def isosceles_triangle (A B C : Point) : Prop := 
AB = AC

def point_on_base (D : Point) (BC : Line) : Prop := 
D ∈ BC

def parallel_to_side_and_intersection (D : Point) (AC : Line) (AB : Line) (E F : Point) : Prop :=
Line.parallel (Line.mk D E) AC ∧ E ∈ AB ∧
Line.parallel (Line.mk D F) AB ∧ F ∈ AC

-- Define the perimeter calculation for parallelogram AEDF
def perimeter (A E D F : Point) : Real :=
dist A E + dist E D + dist D F + dist F A

-- The main theorem statement
theorem isosceles_parallelogram_perimeter_constant 
    (A B C D E F : Point) 
    (AB AC BC : Line)
    (h_iso : isosceles_triangle A B C)
    (h_point : point_on_base D BC)
    (h_par : parallel_to_side_and_intersection D AC AB E F) : 
    perimeter A E D F = 2 * dist A B :=
sorry

end isosceles_parallelogram_perimeter_constant_l6_6918


namespace largest_difference_l6_6367

theorem largest_difference (s : Set ℤ) (h : s = {-20, -8, 0, 3, 7, 15}) : 
  ∃ a b ∈ s, a - b = 35 :=
by
  sorry

end largest_difference_l6_6367


namespace equation_of_parallel_line_l6_6155

theorem equation_of_parallel_line (l : ℝ → ℝ → Prop) (P : ℝ × ℝ)
  (x y : ℝ) (m : ℝ) (H_1 : P = (1, 2)) (H_2 : ∀ x y m, l x y ↔ (2 * x + y + m = 0) )
  (H_3 : l x y) : 
  l 2 (y - 4) := 
  sorry

end equation_of_parallel_line_l6_6155


namespace jason_picked_pears_l6_6241

def jason_picked (total_picked keith_picked mike_picked jason_picked : ℕ) : Prop :=
  jason_picked + keith_picked + mike_picked = total_picked

theorem jason_picked_pears:
  jason_picked 105 47 12 46 :=
by 
  unfold jason_picked
  sorry

end jason_picked_pears_l6_6241


namespace tn_bound_l6_6944

-- Assume S_n is defined for the arithmetic sequence with the given conditions
def S (n : ℕ) : ℝ := n^2 + 2*n

-- Define T_n as the sum of the reciprocals of S_i
def T (n : ℕ) : ℝ := ∑ i in Finset.range n, 1 / (S (i + 1))

-- We need to prove that for all n, T_n < 3 / 4
theorem tn_bound (n : ℕ) : (T n) < 3 / 4 :=
by sorry

end tn_bound_l6_6944


namespace plains_routes_l6_6205

theorem plains_routes 
  (total_cities : ℕ)
  (mountainous_cities : ℕ)
  (plains_cities : ℕ)
  (total_routes : ℕ)
  (mountainous_routes : ℕ)
  (num_pairs_with_mount_to_mount : ℕ)
  (routes_per_year : ℕ)
  (years : ℕ)
  (mountainous_roots_connections : ℕ)
  : (mountainous_cities = 30) →
    (plains_cities = 70) →
    (total_cities = mountainous_cities + plains_cities) →
    (routes_per_year = 50) →
    (years = 3) →
    (total_routes = routes_per_year * years) →
    (mountainous_routes = num_pairs_with_mount_to_mount * 2) →
    (num_pairs_with_mount_to_mount = 21) →
    let num_endpoints_per_city_route = 2 in
    let mountainous_city_endpoints = mountainous_cities * 3 in
    let mountainous_endpoints = mountainous_routes in
    let mountain_to_plains_endpoints = mountainous_city_endpoints - mountainous_endpoints in
    let total_endpoints = total_routes * num_endpoints_per_city_route in
    let plains_city_endpoints = plains_cities * 3 in
    let routes_between_plain_and_mountain = mountain_to_plains_endpoints in
    let plain_to_plain_endpoints = plains_city_endpoints - routes_between_plain_and_mountain in
    let plain_to_plain_routes = plain_to_plain_endpoints / 2 in
    plain_to_plain_routes = 81 :=
sorry

end plains_routes_l6_6205


namespace routes_between_plains_cities_correct_l6_6184

noncomputable def number_of_routes_connecting_pairs_of_plains_cities
    (total_cities : ℕ)
    (mountainous_cities : ℕ)
    (plains_cities : ℕ)
    (total_routes : ℕ)
    (routes_between_mountainous_cities : ℕ) : ℕ :=
let mountainous_city_endpoints := mountainous_cities * 3 in
let routes_between_mountainous_cities_endpoints := routes_between_mountainous_cities * 2 in
let mountainous_to_plains_routes_endpoints := mountainous_city_endpoints - routes_between_mountainous_cities_endpoints in
let plains_city_endpoints := plains_cities * 3 in
let plains_city_to_mountainous_city_routes_endpoints := mountainous_to_plains_routes_endpoints in
let endpoints_fully_in_plains_cities := plains_city_endpoints - plains_city_to_mountainous_city_routes_endpoints in
endpoints_fully_in_plains_cities / 2

theorem routes_between_plains_cities_correct :
    number_of_routes_connecting_pairs_of_plains_cities 100 30 70 150 21 = 81 := by
    sorry

end routes_between_plains_cities_correct_l6_6184


namespace odd_function_passes_through_point_l6_6604

variable {α : Type*} [LinearOrder α] [AddGroup α] [TopologicalSpace α]

def is_odd_function (f : α → α) :=
  ∀ x : α, f (-x) = -f x

theorem odd_function_passes_through_point
  (f : α → α) (a : α) (h : is_odd_function f) :
  ∃ y : α, (y = -f(a) ∧ (−a, y) = (-a, -f(a))) :=
by
  use -f(a)
  split
  · rfl
  · rfl

end odd_function_passes_through_point_l6_6604


namespace recurring_decimal_to_fraction_l6_6047

theorem recurring_decimal_to_fraction
  (h : (2:ℚ) + 3 * (2 / 99) = 2.06) :
  (2:ℚ) + 0.\overline{06} = (68 / 33) :=
by
  -- Given: 0.\overline{02} = 2 / 99
  have h0 : (0.\overline{02} : ℚ) = 2 / 99 := by sorry

  -- 0.\overline{06} = 3 * 0.\overline{02}
  have h1 : (0.\overline{06} : ℚ) = 3 * (0.\overline{02} : ℚ) :=
    by rw [← h0]; sorry

  -- Hence, 0.\overline{06} = 6 / 99 = 2 / 33
  have h2 : (0.\overline{06} : ℚ) = 2 / 33 :=
    by sorry

  -- Therefore, 2.\overline{06} = 2 + 0.\overline{06} = 2 + 2 / 33 = 68 / 33
  show (2:ℚ) + (0.\overline{06} : ℚ) = 68 / 33
    by sorry

end recurring_decimal_to_fraction_l6_6047


namespace selection_probability_l6_6391

/-- Given conditions:
1. Probability of husband's selection: 1/7
2. Probability of wife's selection: 1/5

Prove the probability that only one of them is selected is 2/7.
-/
theorem selection_probability (p_husband p_wife : ℚ) (h1 : p_husband = 1/7) (h2 : p_wife = 1/5) :
  (p_husband * (1 - p_wife) + p_wife * (1 - p_husband)) = 2/7 :=
by
  sorry

end selection_probability_l6_6391


namespace value_of_x_y_squared_l6_6532

theorem value_of_x_y_squared (x y : ℝ) (h1 : x + y = 6) (h2 : x * y = 5) : (x - y)^2 = 16 :=
by
  sorry

end value_of_x_y_squared_l6_6532


namespace uniformly_integrable_conditional_expectation_l6_6646

open MeasureTheory

variables {α : Type*} {A : Type*} [MeasurableSpace α] {μ : MeasureTheory.Measure α}
variables {ξ : A → α → ℝ} {F G : MeasurableSpace α}
variables [μ : MeasureTheory.Measure α]

noncomputable def is_uniformly_integrable_family (ξ : A → α → ℝ) (μ : MeasureTheory.Measure α) :=
  ∀ ε > 0, ∃ δ > 0, ∀ (B : Set α), μ B < δ → sup (λ α, ∫ x in B, |ξ α x| ∂μ) < ε

theorem uniformly_integrable_conditional_expectation 
  (h : is_uniformly_integrable_family ξ μ) :
  is_uniformly_integrable_family (λ α, @MeasureTheory.condexp α _ F μ (ξ α)) μ :=
  sorry

end uniformly_integrable_conditional_expectation_l6_6646


namespace largest_prime_difference_sum_140_l6_6747

/-- 
  The stronger Goldbach conjecture states that any even integer 
  greater than 7 can be written as the sum of two different prime numbers. 
  Prove that the largest possible difference between two prime numbers 
  that sum to 140 is 112.
-/
theorem largest_prime_difference_sum_140 :
  ∃ p q : ℕ, p ≠ q ∧ nat.prime p ∧ nat.prime q ∧ p + q = 140 ∧ (q - p) = 112 :=
sorry

end largest_prime_difference_sum_140_l6_6747


namespace recurring_decimal_to_fraction_l6_6046

theorem recurring_decimal_to_fraction
  (h : (2:ℚ) + 3 * (2 / 99) = 2.06) :
  (2:ℚ) + 0.\overline{06} = (68 / 33) :=
by
  -- Given: 0.\overline{02} = 2 / 99
  have h0 : (0.\overline{02} : ℚ) = 2 / 99 := by sorry

  -- 0.\overline{06} = 3 * 0.\overline{02}
  have h1 : (0.\overline{06} : ℚ) = 3 * (0.\overline{02} : ℚ) :=
    by rw [← h0]; sorry

  -- Hence, 0.\overline{06} = 6 / 99 = 2 / 33
  have h2 : (0.\overline{06} : ℚ) = 2 / 33 :=
    by sorry

  -- Therefore, 2.\overline{06} = 2 + 0.\overline{06} = 2 + 2 / 33 = 68 / 33
  show (2:ℚ) + (0.\overline{06} : ℚ) = 68 / 33
    by sorry

end recurring_decimal_to_fraction_l6_6046


namespace plains_routes_count_l6_6195

-- Defining the total number of cities and the number of cities in each region
def total_cities : Nat := 100
def mountainous_cities : Nat := 30
def plains_cities : Nat := total_cities - mountainous_cities

-- Defining the number of routes established each year and over three years
def routes_per_year : Nat := 50
def total_routes : Nat := routes_per_year * 3

-- Defining the number of routes connecting pairs of mountainous cities
def mountainous_routes : Nat := 21

-- The statement to prove the number of routes connecting pairs of plains cities
theorem plains_routes_count :
  plains_cities = 70 →
  total_routes = 150 →
  mountainous_routes = 21 →
  3 * mountainous_cities - 2 * mountainous_routes = 48 →
  3 * plains_cities - 48 = 162 →
  81 = 81 := sorry

end plains_routes_count_l6_6195


namespace find_positive_n_unique_solution_l6_6899

theorem find_positive_n_unique_solution (n : ℝ) :
  (∃ x : ℝ, 4 * x^2 + n * x + 4 = 0) → n = 8 :=
by
  intro h
  sorry

end find_positive_n_unique_solution_l6_6899


namespace find_FR_l6_6823

-- Definitions for the conditions
def θ := 40 * Real.pi / 180
def μ := 0.40
def sinθ := Real.sin θ
def cosθ := Real.cos θ

-- Statement of the proof
theorem find_FR 
  (F R : ℝ)
  (hR : R = Real.sin θ + μ * Real.cos θ)
  : F / R = 1.239 :=
  sorry

end find_FR_l6_6823


namespace area_ratio_trapezoid_l6_6630

-- Define the geometric setup and areas
variables (AB CD : ℝ) (ABCD EAB : ℝ)
def trapezoid := AB = 10 ∧ CD = 25
def ratio_of_areas := EAB / (ABCD - EAB)

-- The theorem we need to prove
theorem area_ratio_trapezoid (h : trapezoid AB CD) : EAB / ABCD = 4 / 21 :=
by
  sorry

end area_ratio_trapezoid_l6_6630


namespace billard_table_angle_range_l6_6815

noncomputable theory
open Real

theorem billard_table_angle_range (θ : ℝ) (h : ∃ (P Q : Point) (k : ℝ),
  P ∈ midpoint A B ∧ 
  Q ∈ segment B C ∧ 
  0 < k < 1 ∧ 
  \frac{BQ}{BP} = k ∧
  \frac{BP}{BQ} = \frac{\sqrt{3}}{2} * cot θ - \frac{1}{2}) :
  arctan (3 * sqrt 3 / 10) < θ ∧ θ < arctan (3 * sqrt 3 / 8) :=
sorry

end billard_table_angle_range_l6_6815


namespace identify_irrational_among_options_l6_6857

def is_irrational (x : ℝ) : Prop := ¬ ∃ (a b : ℤ), b ≠ 0 ∧ x = a / b

theorem identify_irrational_among_options :
  ∃ x, (x = 3.142 ∨ x = sqrt 4 ∨ x = 22 / 7 ∨ x = real.pi) ∧ is_irrational x ↔ x = real.pi :=
by
  sorry

end identify_irrational_among_options_l6_6857


namespace maximum_capacity_of_theater_l6_6435

noncomputable def pricePerTicket : ℝ := 8.0
noncomputable def ticketsSold : ℕ := 24
noncomputable def lostAmount : ℝ := 208

theorem maximum_capacity_of_theater : 
  ∃ C : ℕ, C = ticketsSold + (lostAmount / pricePerTicket) ∧ C = 50 :=
by
  use ticketsSold + (lostAmount / pricePerTicket)
  split
  · rfl
  · sorry

end maximum_capacity_of_theater_l6_6435


namespace projection_is_correct_l6_6903

def u : ℝ × ℝ := (3, -4)
def v : ℝ × ℝ := (-1, 2)
def b : ℝ × ℝ := (1, 3)
def proj (u v : ℝ × ℝ) : ℝ × ℝ := ((u.1 * v.1 + u.2 * v.2) / (v.1 * v.1 + v.2 * v.2)) * v

theorem projection_is_correct : proj (u.1 + v.1, u.2 + v.2) b = (-0.4, -1.2) :=
by
  sorry

end projection_is_correct_l6_6903


namespace plains_routes_count_l6_6175

theorem plains_routes_count (total_cities mountainous_cities plains_cities total_routes routes_mountainous_pairs: ℕ) :
  total_cities = 100 →
  mountainous_cities = 30 →
  plains_cities = 70 →
  total_routes = 150 →
  routes_mountainous_pairs = 21 →
  let endpoints_mountainous := mountainous_cities * 3 in
  let endpoints_mountainous_pairs := routes_mountainous_pairs * 2 in
  let endpoints_mountainous_plains := endpoints_mountainous - endpoints_mountainous_pairs in
  let endpoints_plains := plains_cities * 3 in
  let routes_mountainous_plains := endpoints_mountainous_plains in
  let endpoints_plains_pairs := endpoints_plains - routes_mountainous_plains in
  let routes_plains_pairs := endpoints_plains_pairs / 2 in
  routes_plains_pairs = 81 :=
by
  intros h1 h2 h3 h4 h5
  dsimp
  rw [h1, h2, h3, h4, h5]
  sorry

end plains_routes_count_l6_6175


namespace first_1963_digits_all_zero_l6_6077

-- Definitions for conditions
def a : Real := Real.sqrt 26 + 5
def b : Real := 5 - Real.sqrt 26
def N : Int := Int.floor ((Real.sqrt 26 + 5) ^ 1963)

-- Statement of the proof problem
theorem first_1963_digits_all_zero :
  0 < 1963 → -- Ensures that we are looking at a positive number of digits
  (a ^ 1963) - N = -((b ^ 1963)) → -- Expresses that the digit difference is due to b ^ 1963, which is very small
  ∀ k : Nat, (1 ≤ k ∧ k ≤ 1963) → Real.frac (a ^ 1963) * (10 ^ k) < 1 :=
by
  intros
  sorry

end first_1963_digits_all_zero_l6_6077


namespace tan_a_necessary_condition_tan_a_not_sufficient_condition_l6_6588

theorem tan_a_necessary_condition (a : ℝ) : (tan a = 1) ↔ (∃ k : ℤ, a = k * π + π / 4) :=
sorry

theorem tan_a_not_sufficient_condition (a : ℝ) : (a = π / 4) → (tan a = 1) :=
sorry

end tan_a_necessary_condition_tan_a_not_sufficient_condition_l6_6588


namespace parabola_and_line_sum_l6_6572

theorem parabola_and_line_sum (A B F : ℝ × ℝ)
  (h_parabola : ∀ x y : ℝ, (y^2 = 4 * x) ↔ (x, y) = A ∨ (x, y) = B)
  (h_line : ∀ x y : ℝ, (2 * x + y - 4 = 0) ↔ (x, y) = A ∨ (x, y) = B)
  (h_focus : F = (1, 0))
  : |F - A| + |F - B| = 7 := 
sorry

end parabola_and_line_sum_l6_6572


namespace sum_floored_fractions_l6_6669

theorem sum_floored_fractions (N : ℕ) (h : N > 0) : 
  N = ∑' n, ⌊N / (2^n)⌋ := 
sorry

end sum_floored_fractions_l6_6669


namespace num_valid_digits_l6_6425

def is_divisible_by_4 (n : ℕ) : Prop := n % 4 = 0

def valid_digits : Finset ℕ := {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}

theorem num_valid_digits : 
  (valid_digits.filter (λ N, is_divisible_by_4 (64 + N))).card = 5 :=
by
  sorry

end num_valid_digits_l6_6425


namespace average_snack_sales_per_ticket_l6_6748

theorem average_snack_sales_per_ticket :
  let cracker_price := (3 : ℝ) * 2.25
  let beverage_price := (4 : ℝ) * 1.50
  let chocolate_price := (4 : ℝ) * 1.00
  let total_sales := cracker_price + beverage_price + chocolate_price
  let movie_tickets := 6
  (total_sales / movie_tickets = 2.79) :=
by
  let cracker_price := (3 : ℝ) * 2.25
  let beverage_price := (4 : ℝ) * 1.50
  let chocolate_price := (4 : ℝ) * 1.00
  let total_sales := cracker_price + beverage_price + chocolate_price
  let movie_tickets := 6
  show total_sales / movie_tickets = 2.79
  sorry

end average_snack_sales_per_ticket_l6_6748


namespace average_of_first_12_l6_6697

theorem average_of_first_12 (avg_25 : ℝ) (avg_last_12 : ℝ) (res13 : ℝ) (sum_25 : ℝ) (sum_last_12 : ℝ) (sum_first_12 : ℝ) :
  avg_25 = 50 → avg_last_12 = 17 → res13 = 878 →
  sum_25 = avg_25 * 25 → sum_last_12 = avg_last_12 * 12 → sum_25 = sum_first_12 + res13 + sum_last_12 →
  sum_first_12 / 12 = 14 :=
by
  intros h_avg25 h_avgLast12 h_res13 h_sum25 h_sumLast12 h_eq.
  sorry

end average_of_first_12_l6_6697


namespace problem_solution_l6_6613

namespace MathProof

-- Definitions of our events and probabilities

structure BallPocket := 
  (ball_count : ℕ)
  (red_balls : Finset ℕ)
  (blue_balls : Finset ℕ)

-- Define the specific setup of the problem
def pocket : BallPocket := {
  ball_count := 8,
  red_balls := {1, 2},
  blue_balls := {1, 2, 3, 4, 5, 6}
}

def event_A (p : BallPocket) : Set ℕ := p.red_balls
def event_B (p : BallPocket) : Set ℕ := {ball ∈ Finset.range (p.ball_count + 1) | ball % 2 = 0}
def event_C (p : BallPocket) : Set ℕ := {ball ∈ Finset.range (p.ball_count + 1) | ball % 3 = 0}

-- To be used for independence and mutual exclusivity
def probability (p : BallPocket) (event : Set ℕ) : ℚ := (Finset.card (event ∩ (p.red_balls ∪ p.blue_balls)).val : ℚ) / (p.ball_count : ℚ)

theorem problem_solution :
  (event_A pocket ∩ event_C pocket = ∅) ∧
  (probability pocket (event_A pocket ∩ event_B pocket) = probability pocket (event_A pocket) * probability pocket (event_B pocket)) ∧
  (probability pocket (event_B pocket ∩ event_C pocket) = probability pocket (event_B pocket) * probability pocket (event_C pocket)) := 
  sorry

end MathProof

end problem_solution_l6_6613


namespace largest_possible_perimeter_l6_6917

-- Definitions related to the conditions
def is_regular_polygon (n : ℕ) := n ≥ 3

def interior_angle (n : ℕ) : ℝ := 180 * (n - 2) / n

def sum_interior_angles_at_point (angles : List ℝ) : ℝ := angles.sum

-- Main theorem definition
theorem largest_possible_perimeter :
  ∃ (polygons : List ℕ), 
  length polygons = 4 ∧ 
  (∃ n_i, (polygons.filter (λ n, n = n_i)).length ≥ 2) ∧ 
  ∀ n ∈ polygons, is_regular_polygon n ∧ side_length = 1 ∧
  sum_interior_angles_at_point (polygons.map interior_angle) = 360 ∧
  perimeter polygons = 12 :=
sorry

end largest_possible_perimeter_l6_6917


namespace triangle_ABC_AB_proof_l6_6997

noncomputable def triangle_ABC_AB : ℝ := 
  let A : ℝ := 90
  let tan_C : ℝ := 3
  let AC : ℝ := 150
  let AB : ℝ := 45 * Real.sqrt 10
  AB

theorem triangle_ABC_AB_proof :
  ∀ (AC : ℝ) (tan_C : ℝ) (angle_A : ℝ), 
  angle_A = 90 → tan_C = 3 → AC = 150 → triangle_ABC_AB = 45 * Real.sqrt 10 :=
by
  intros AC tan_C angle_A hA ht hAC
  rw [triangle_ABC_AB]
  exact rfl

end triangle_ABC_AB_proof_l6_6997


namespace investment_rate_l6_6992

theorem investment_rate (r : ℝ) (A : ℝ) (income_diff : ℝ) (total_invested : ℝ) (eight_percent_invested : ℝ) :
  total_invested = 2000 → 
  eight_percent_invested = 750 → 
  income_diff = 65 → 
  A = total_invested - eight_percent_invested → 
  (A * r) - (eight_percent_invested * 0.08) = income_diff → 
  r = 0.1 :=
by
  intros h_total h_eight h_income_diff h_A h_income_eq
  sorry

end investment_rate_l6_6992


namespace parabola_count_l6_6524

theorem parabola_count :
  let S := {-2, -1, 0, 1, 2, 3}
  ∃ a b c : ℤ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ∈ S ∧ b ∈ S ∧ c = 0 ∧ ( ∀ x : ℝ, a * x^2 + b * x + c = 0 → x = 0 ∧ 0 < -(b^2) / (4 * a) ) → 
  (∃ y z w : ℤ, {y, z, w} = S ∧ w = 0 ∧ y ∈ {-2, -1} ∧ z ∈ {1, 2, 3} ∧ 6) :=
by
  sorry

end parabola_count_l6_6524


namespace more_black_females_than_males_l6_6832

theorem more_black_females_than_males 
  (total_pigeons : ℕ)
  (half_black : total_pigeons / 2 = 35)
  (percent_male : 20)
  (black_pigeons : total_pigeons / 2 = 35)
  (black_male_pigeons : black_pigeons * percent_male / 100 = 7) :
  (black_pigeons - black_male_pigeons) - black_male_pigeons = 21 :=
by
  sorry

end more_black_females_than_males_l6_6832


namespace tan_is_increasing_in_interval_l6_6459

theorem tan_is_increasing_in_interval : 
  ∀ x₁ x₂ : ℝ, (π / 2) < x₁ ∧ x₁ < x₂ ∧ x₂ < π → tan x₁ < tan x₂ :=
by sorry

end tan_is_increasing_in_interval_l6_6459


namespace find_cost_price_l6_6834

noncomputable def original_cost_price (C S C_new S_new : ℝ) : Prop :=
  S = 1.25 * C ∧
  C_new = 0.80 * C ∧
  S_new = 1.25 * C - 10.50 ∧
  S_new = 1.04 * C

theorem find_cost_price (C S C_new S_new : ℝ) :
  original_cost_price C S C_new S_new → C = 50 :=
by
  sorry

end find_cost_price_l6_6834


namespace find_area_of_fourth_rectangle_l6_6828

theorem find_area_of_fourth_rectangle (a b c d : ℕ) 
  (h1 : a + b + c + d = 2 * (a + b)) 
  (h2 : a = 12) 
  (h3 : b = 27) 
  (h4 : c = 18) 
  (h5 : d = 4 * (a + b) - a - b - c) : 
  d = 27 :=
begin
  sorry
end

end find_area_of_fourth_rectangle_l6_6828


namespace trigonometric_expression_value_l6_6525

theorem trigonometric_expression_value (θ : ℝ) (h : sin θ + (sin θ)^2 = 1) :
    3 * (cos θ)^2 + (cos θ)^4 - 2 * sin θ + 1 = 2 :=
sorry

end trigonometric_expression_value_l6_6525


namespace MongePoint_eq_Orthocenter_l6_6294

noncomputable section

-- Define an orthocentric tetrahedron and the concept of Monge point and orthocenter
structure OrthocentricTetrahedron where
  A B C D : Point3D
  edges_perpendicular : ∀ P Q R S : Point3D, (P = A ∨ P = B) → (Q = C ∨ Q = D) → 
    (R = A ∨ R = B) → (S = C ∨ S = D) → (P ≠ Q → R ≠ S → perp (P - Q) (R - S))

def MongePoint (T : OrthocentricTetrahedron) : Point3D :=
  sorry -- Some definition for Monge point based on the given orthocentric tetrahedron

def Orthocenter (T : OrthocentricTetrahedron) : Point3D :=
  sorry -- Some definition for Orthocenter based on the given orthocentric tetrahedron

-- The problem statement to prove in Lean
theorem MongePoint_eq_Orthocenter (T : OrthocentricTetrahedron) : MongePoint T = Orthocenter T := 
  sorry

end MongePoint_eq_Orthocenter_l6_6294


namespace routes_between_plains_cities_correct_l6_6180

noncomputable def number_of_routes_connecting_pairs_of_plains_cities
    (total_cities : ℕ)
    (mountainous_cities : ℕ)
    (plains_cities : ℕ)
    (total_routes : ℕ)
    (routes_between_mountainous_cities : ℕ) : ℕ :=
let mountainous_city_endpoints := mountainous_cities * 3 in
let routes_between_mountainous_cities_endpoints := routes_between_mountainous_cities * 2 in
let mountainous_to_plains_routes_endpoints := mountainous_city_endpoints - routes_between_mountainous_cities_endpoints in
let plains_city_endpoints := plains_cities * 3 in
let plains_city_to_mountainous_city_routes_endpoints := mountainous_to_plains_routes_endpoints in
let endpoints_fully_in_plains_cities := plains_city_endpoints - plains_city_to_mountainous_city_routes_endpoints in
endpoints_fully_in_plains_cities / 2

theorem routes_between_plains_cities_correct :
    number_of_routes_connecting_pairs_of_plains_cities 100 30 70 150 21 = 81 := by
    sorry

end routes_between_plains_cities_correct_l6_6180


namespace proof_problem_l6_6656

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := ln (1 + a * x) + b * x
noncomputable def g (x : ℝ) (a b : ℝ) : ℝ := f x a b - b * x^2

def problem_1_monotonic_intervals (h₁ : f x 1 (-1) = ln (1 + x) - x) : Prop :=
  -- Monotonic intervals of f(x)
  ∀ x, f x 1 (-1) = ln (1 + x) - x → 
      (monotonic_increasing x (-1, 0) ∧ 
       monotonic_decreasing x (0, +∞))

def problem_2_tangent_condition (h₂ : g 1 2 (-3) = ln 3) (h₃ : deriv (g x 2 (-3)) 1 = 11/3) : Prop :=
  -- Conditions to find values of a and b
  a = 2 ∧ b = -3

def problem_3_inequality (k : ℝ) (h₄ : k ≤ 3) : Prop :=
  -- Range of real numbers k such that g(x) > k (x^2 - x) for x ∈ (0,+∞)
  ∀ x, x ∈ (0, +∞) → g(x, 2, -3) - k * (x^2 - x) > 0

theorem proof_problem (x : ℝ) (a b k : ℝ) :
  problem_1_monotonic_intervals 
  ∧ problem_2_tangent_condition 
  ∧ problem_3_inequality k := sorry

end proof_problem_l6_6656


namespace students_not_in_any_activity_l6_6617

def total_students : ℕ := 1500
def students_chorus : ℕ := 420
def students_band : ℕ := 780
def students_chorus_and_band : ℕ := 150
def students_drama : ℕ := 300
def students_drama_and_other : ℕ := 50

theorem students_not_in_any_activity :
  total_students - ((students_chorus + students_band - students_chorus_and_band) + (students_drama - students_drama_and_other)) = 200 :=
by
  sorry

end students_not_in_any_activity_l6_6617


namespace distance_after_100_moves_l6_6664

def movement_series (n : ℕ) : ℤ :=
  if n % 2 = 1 then ↑n else -↑n

def resultant_position (N : ℕ) : ℤ :=
  (finset.range N).sum movement_series

def distance_from_origin (N : ℕ) : ℤ :=
  abs (resultant_position N)

theorem distance_after_100_moves : distance_from_origin 100 = 50 := 
by
  sorry

end distance_after_100_moves_l6_6664


namespace sum_ages_l6_6354

theorem sum_ages (a b c : ℕ) (h1 : a = b) (h2 : a * b * c = 72) : a + b + c = 14 := 
by 
  sorry

end sum_ages_l6_6354


namespace predicted_height_at_age_10_l6_6434

-- Define the regression model as a function
def regression_model (x : ℝ) : ℝ := 7.19 * x + 73.93

-- Assert the predicted height at age 10
theorem predicted_height_at_age_10 : abs (regression_model 10 - 145.83) < 0.01 := 
by
  -- Here, we would prove the calculation steps
  sorry

end predicted_height_at_age_10_l6_6434


namespace g_9_eq_64_l6_6717

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g(x + y) = g(x) * g(y)

axiom g_3_eq_4 : g(3) = 4

theorem g_9_eq_64 : g(9) = 64 := by
  sorry

end g_9_eq_64_l6_6717


namespace proof_equivalent_problem_l6_6253

variables (a b c : ℝ)
-- Conditions
axiom h1 : a < b
axiom h2 : b < 0
axiom h3 : c > 0

theorem proof_equivalent_problem :
  (a * c < b * c) ∧ (a + b + c < b + c) ∧ (c / a > 1) :=
by
  sorry

end proof_equivalent_problem_l6_6253


namespace solve_eq1_solve_eq2_l6_6305

theorem solve_eq1 (x : ℝ) : x^2 - 6*x - 7 = 0 → x = 7 ∨ x = -1 :=
by
  sorry

theorem solve_eq2 (x : ℝ) : 3*x^2 - 1 = 2*x → x = 1 ∨ x = -1/3 :=
by
  sorry

end solve_eq1_solve_eq2_l6_6305


namespace total_barking_dogs_eq_l6_6410

-- Definitions
def initial_barking_dogs : ℕ := 30
def additional_barking_dogs : ℕ := 10

-- Theorem to prove the total number of barking dogs
theorem total_barking_dogs_eq :
  initial_barking_dogs + additional_barking_dogs = 40 :=
by
  sorry

end total_barking_dogs_eq_l6_6410


namespace rachel_brought_16_brownies_l6_6673

def total_brownies : ℕ := 40
def brownies_left_at_home : ℕ := 24

def brownies_brought_to_school : ℕ :=
  total_brownies - brownies_left_at_home

theorem rachel_brought_16_brownies :
  brownies_brought_to_school = 16 :=
by
  sorry

end rachel_brought_16_brownies_l6_6673


namespace second_worker_time_DE_l6_6777

def time_spent_paving_section_DE 
(v : ℝ) -- speed of the first worker
(d_abc d_cdef : ℝ) -- distances of sections A-B-C and A-D-E-F-C respectively
(h1 : 0 < v) -- speed must be positive
(h2 : 0 < d_abc) -- distance A-B-C must be positive
(h3 : 0 < d_cdef) -- distance A-D-E-F-C must be positive
(h4 : 1.2 * v * 9 = d_cdef) -- total distance covered by the second worker
(h5 : v * 9 = d_abc) -- total distance covered by the first worker
: ℝ := 
  let de_distance := (d_cdef - d_abc) / 2 in -- distance DE is the excess distance in the loop divided by 2
  let de_speed   := 1.2 * v in
  LET total_time := (de_distance / de_speed) in
  total_time * 60 -- converting hours to minutes

theorem second_worker_time_DE
(v: ℝ) 
(d_abc d_cdef: ℝ) 
(h1: 0 < v) 
(h2: 0 < d_abc) 
(h3: 0 < d_cdef) 
(h4: 1.2 * v * 9 = d_cdef) 
(h5: v * 9 = d_abc) 
: 
time_spent_paving_section_DE v d_abc d_cdef h1 h2 h3 h4 h5 = 45 := 
sorry

end second_worker_time_DE_l6_6777


namespace individual_max_food_l6_6728

/-- Given a minimum number of guests and a total amount of food consumed,
    we want to find the maximum amount of food an individual guest could have consumed. -/
def total_food : ℝ := 319
def min_guests : ℝ := 160
def max_food_per_guest : ℝ := 1.99

theorem individual_max_food :
  total_food / min_guests <= max_food_per_guest := by
  sorry

end individual_max_food_l6_6728


namespace plains_routes_l6_6208

theorem plains_routes 
  (total_cities : ℕ)
  (mountainous_cities : ℕ)
  (plains_cities : ℕ)
  (total_routes : ℕ)
  (mountainous_routes : ℕ)
  (num_pairs_with_mount_to_mount : ℕ)
  (routes_per_year : ℕ)
  (years : ℕ)
  (mountainous_roots_connections : ℕ)
  : (mountainous_cities = 30) →
    (plains_cities = 70) →
    (total_cities = mountainous_cities + plains_cities) →
    (routes_per_year = 50) →
    (years = 3) →
    (total_routes = routes_per_year * years) →
    (mountainous_routes = num_pairs_with_mount_to_mount * 2) →
    (num_pairs_with_mount_to_mount = 21) →
    let num_endpoints_per_city_route = 2 in
    let mountainous_city_endpoints = mountainous_cities * 3 in
    let mountainous_endpoints = mountainous_routes in
    let mountain_to_plains_endpoints = mountainous_city_endpoints - mountainous_endpoints in
    let total_endpoints = total_routes * num_endpoints_per_city_route in
    let plains_city_endpoints = plains_cities * 3 in
    let routes_between_plain_and_mountain = mountain_to_plains_endpoints in
    let plain_to_plain_endpoints = plains_city_endpoints - routes_between_plain_and_mountain in
    let plain_to_plain_routes = plain_to_plain_endpoints / 2 in
    plain_to_plain_routes = 81 :=
sorry

end plains_routes_l6_6208


namespace repeating_decimal_fraction_l6_6066

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l6_6066


namespace product_of_solutions_l6_6250

theorem product_of_solutions :
  ∃ (x y : ℤ → ℤ), (∀ n, |x n - 5| = |y n - 11| ∧ |x n - 11| = 3 * |y n - 5|) →
  (∏ (n : ℕ) in finset.range 3, (x n * y n)) = 72765 := 
sorry

end product_of_solutions_l6_6250


namespace plains_routes_count_l6_6179

theorem plains_routes_count (total_cities mountainous_cities plains_cities total_routes routes_mountainous_pairs: ℕ) :
  total_cities = 100 →
  mountainous_cities = 30 →
  plains_cities = 70 →
  total_routes = 150 →
  routes_mountainous_pairs = 21 →
  let endpoints_mountainous := mountainous_cities * 3 in
  let endpoints_mountainous_pairs := routes_mountainous_pairs * 2 in
  let endpoints_mountainous_plains := endpoints_mountainous - endpoints_mountainous_pairs in
  let endpoints_plains := plains_cities * 3 in
  let routes_mountainous_plains := endpoints_mountainous_plains in
  let endpoints_plains_pairs := endpoints_plains - routes_mountainous_plains in
  let routes_plains_pairs := endpoints_plains_pairs / 2 in
  routes_plains_pairs = 81 :=
by
  intros h1 h2 h3 h4 h5
  dsimp
  rw [h1, h2, h3, h4, h5]
  sorry

end plains_routes_count_l6_6179


namespace shifted_parabola_expression_l6_6726

def original_parabola (x : ℝ) : ℝ := 3 * x^2

def left_shift (f : ℝ → ℝ) (shift : ℝ) (x : ℝ) : ℝ := f (x + shift)

def down_shift (f : ℝ → ℝ) (shift : ℝ) (x : ℝ) : ℝ := f x - shift

theorem shifted_parabola_expression :
  (down_shift (left_shift original_parabola 1) 4) = (λ x, 3 * (x + 1)^2 - 4) :=
by 
  sorry

end shifted_parabola_expression_l6_6726


namespace least_value_of_f_l6_6967

def f (x : ℝ) : ℝ := (1/2) * x^2 + 3 * x + 4

theorem least_value_of_f :
  ∃ m, (∀ x : ℝ, f x ≥ m) ∧ m = -1/2 := 
sorry

end least_value_of_f_l6_6967


namespace rhombus_has_perpendicular_diagonals_and_rectangle_not_l6_6458

-- Definitions based on conditions (a))
def rhombus (sides_equal : Prop) (diagonals_bisect : Prop) (diagonals_perpendicular : Prop) : Prop :=
  sides_equal ∧ diagonals_bisect ∧ diagonals_perpendicular

def rectangle (sides_equal : Prop) (diagonals_bisect : Prop) (diagonals_equal : Prop) : Prop :=
  sides_equal ∧ diagonals_bisect ∧ diagonals_equal

-- Theorem to prove (c))
theorem rhombus_has_perpendicular_diagonals_and_rectangle_not 
  (rhombus_sides_equal rhombus_diagonals_bisect rhombus_diagonals_perpendicular : Prop)
  (rectangle_sides_equal rectangle_diagonals_bisect rectangle_diagonals_equal : Prop) :
  rhombus rhombus_sides_equal rhombus_diagonals_bisect rhombus_diagonals_perpendicular → 
  rectangle rectangle_sides_equal rectangle_diagonals_bisect rectangle_diagonals_equal → 
  rhombus_diagonals_perpendicular ∧ ¬(rectangle (rectangle_sides_equal) (rectangle_diagonals_bisect) (rhombus_diagonals_perpendicular)) :=
sorry

end rhombus_has_perpendicular_diagonals_and_rectangle_not_l6_6458


namespace recurring_decimal_to_fraction_l6_6027

theorem recurring_decimal_to_fraction
  (h : 0.\overline{02} = (2 : ℝ) / 99) :
  2.\overline{06} = 68 / 33 := by
  sorry

end recurring_decimal_to_fraction_l6_6027


namespace hypotenuse_length_l6_6735

def triangle_hypotenuse (x : ℝ) (h : ℝ) : Prop :=
  (3 * x - 3)^2 + x^2 = h^2 ∧
  (1 / 2) * x * (3 * x - 3) = 72

theorem hypotenuse_length :
  ∃ (x h : ℝ), triangle_hypotenuse x h ∧ h = Real.sqrt 505 :=
by
  sorry

end hypotenuse_length_l6_6735


namespace solvable_eq_l6_6503

theorem solvable_eq (x : ℝ) :
    Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6 →
    (x = 2 ∨ x = -2) :=
by
  sorry

end solvable_eq_l6_6503


namespace find_a_plus_b_l6_6255

theorem find_a_plus_b (a b : ℝ) 
  (h1 : ∀ x : ℝ, g (f x) = 3 * x + 4)
  (h2 : ∀ x : ℝ, f x = a * x + b)
  (h3 : ∀ x : ℝ, g x = 2 * x - 5) :
  a + b = 6 := sorry

end find_a_plus_b_l6_6255


namespace abc_sum_l6_6592

theorem abc_sum (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a + b + c = 13 * Real.sqrt 6 := 
sorry

end abc_sum_l6_6592


namespace polynomial_factorization_l6_6377

open Polynomial

theorem polynomial_factorization :
  (X ^ 15 + X ^ 10 + X ^ 5 + 1) =
    (X ^ 3 + X ^ 2 + 1) * 
    (X ^ 12 - X ^ 11 + X ^ 9 - X ^ 8 + X ^ 6 - X ^ 5 + X ^ 4 + X ^ 3 + X ^ 2 + X + 1) :=
by
  sorry

end polynomial_factorization_l6_6377


namespace safe_to_drive_in_eight_hours_l6_6493

noncomputable def alcohol_content_after_hours (initial_content : ℝ) (decay_rate : ℝ) (hours : ℕ) : ℝ :=
  initial_content * (decay_rate ^ hours)

theorem safe_to_drive_in_eight_hours :
  ∀ (initial_content : ℝ) (decay_rate : ℝ),
    initial_content = 100 → decay_rate = 0.8 →
    ∃ (hours : ℕ), hours ≥ 8 ∧ alcohol_content_after_hours initial_content decay_rate hours < 20 := 
by 
  intros initial_content decay_rate h1 h2 
  use 8 
  split
  · exact le_rfl
  · sorry

end safe_to_drive_in_eight_hours_l6_6493


namespace count_valid_numbers_l6_6583

def is_valid_digit (n : ℕ) : Prop :=
  0 ≤ n ∧ n ≤ 9

def valid_number (a b c d: ℕ) : Prop :=
  (is_valid_digit a) ∧ 
  (is_valid_digit b) ∧ 
  (is_valid_digit c) ∧ 
  (is_valid_digit d) ∧
  (b = (a + c) / 2) ∧ 
  (d = 2 * a)

theorem count_valid_numbers : 
  (cardinal.mk (Σ' (a b c d : ℕ), valid_number a b c d) : ℕ) = 8 :=
  sorry

end count_valid_numbers_l6_6583


namespace part1_part2_l6_6564

-- Definition of the function f(x)
def f (k x : ℝ) := (k - x) * Real.exp x - x - 3

-- Definition of the derivative of f(x) when k = 0
def f_deriv (x : ℝ) := (-1 - x) * Real.exp x - 1

-- The first part of the problem: Tangent line equation at (0, -3) when k = 0
theorem part1
  (k : ℝ)
  (hx0 : x = 0)
  (hk0 : k = 0)
  (hfx0 : f k 0 = -3) -- f(0) = -3 when k = 0
  : tangent_line_eq x 0 (-3) = -2 * x - 3 := 
sorry

-- The second part of the problem: Maximum integer k where f(x) < 0 for any x > 0
theorem part2
  (hfx : ∀ x > 0, f k x < 0)
  : ∃ k_max : ℤ, (∀ k : ℤ, k < k_max -> f k x < 0) ∧ k_max = 2 := 
sorry

end part1_part2_l6_6564


namespace product_of_roots_eq_neg25_over_2_l6_6512

theorem product_of_roots_eq_neg25_over_2 :
  ∀ x : ℝ, quadratic_roots_product 24 66 (-300) = -25 / 2 :=
by
  intro x
  -- Define the quadratic_roots_product function assuming the condition
  def quadratic_roots_product (a b c : ℝ) : ℝ :=
    (λ (r1 r2 : ℝ), r1 * r2) (complex.roots (polynomial.map_ring_hom (algebra_map ℝ ℂ) (a * X ^ 2 + b * X + c)))
  sorry

end product_of_roots_eq_neg25_over_2_l6_6512


namespace arithmetic_sequence_sum_l6_6560

variable {S : ℕ → ℕ}

theorem arithmetic_sequence_sum (h1 : S 3 = 15) (h2 : S 9 = 153) : S 6 = 66 :=
sorry

end arithmetic_sequence_sum_l6_6560


namespace sqrt_nested_expression_l6_6472

theorem sqrt_nested_expression : sqrt (25 * sqrt (25 * sqrt 25)) = 5 * sqrt (5 * sqrt 5) :=
  sorry

end sqrt_nested_expression_l6_6472


namespace Amanda_needs_12_more_marbles_l6_6856

theorem Amanda_needs_12_more_marbles (K A M : ℕ)
  (h1 : M = 5 * K)
  (h2 : M = 85)
  (h3 : M = A + 63) :
  A + 12 = 2 * K := 
sorry

end Amanda_needs_12_more_marbles_l6_6856


namespace jacqueline_erasers_l6_6641

def num_boxes : ℕ := 4
def erasers_per_box : ℕ := 10
def total_erasers : ℕ := num_boxes * erasers_per_box

theorem jacqueline_erasers : total_erasers = 40 := by
  sorry

end jacqueline_erasers_l6_6641


namespace plains_routes_l6_6206

theorem plains_routes 
  (total_cities : ℕ)
  (mountainous_cities : ℕ)
  (plains_cities : ℕ)
  (total_routes : ℕ)
  (mountainous_routes : ℕ)
  (num_pairs_with_mount_to_mount : ℕ)
  (routes_per_year : ℕ)
  (years : ℕ)
  (mountainous_roots_connections : ℕ)
  : (mountainous_cities = 30) →
    (plains_cities = 70) →
    (total_cities = mountainous_cities + plains_cities) →
    (routes_per_year = 50) →
    (years = 3) →
    (total_routes = routes_per_year * years) →
    (mountainous_routes = num_pairs_with_mount_to_mount * 2) →
    (num_pairs_with_mount_to_mount = 21) →
    let num_endpoints_per_city_route = 2 in
    let mountainous_city_endpoints = mountainous_cities * 3 in
    let mountainous_endpoints = mountainous_routes in
    let mountain_to_plains_endpoints = mountainous_city_endpoints - mountainous_endpoints in
    let total_endpoints = total_routes * num_endpoints_per_city_route in
    let plains_city_endpoints = plains_cities * 3 in
    let routes_between_plain_and_mountain = mountain_to_plains_endpoints in
    let plain_to_plain_endpoints = plains_city_endpoints - routes_between_plain_and_mountain in
    let plain_to_plain_routes = plain_to_plain_endpoints / 2 in
    plain_to_plain_routes = 81 :=
sorry

end plains_routes_l6_6206


namespace derivative_at_two_l6_6370

noncomputable def f (x a b : ℝ) : ℝ := a * Real.log x + b / x

theorem derivative_at_two (a b : ℝ) (h1 : f 1 a b = -2) (h2 : ∀ x, f x a b ≤ f 1 a b) :
  (a = -2 ∧ b = -2) → deriv (λ x, f x a b) 2 = -1/2 :=
begin
  intros hab,
  obtain ⟨ha, hb⟩ := hab,
  -- omitted steps for clarity
  exact sorry
end

end derivative_at_two_l6_6370


namespace routes_between_plains_cities_correct_l6_6181

noncomputable def number_of_routes_connecting_pairs_of_plains_cities
    (total_cities : ℕ)
    (mountainous_cities : ℕ)
    (plains_cities : ℕ)
    (total_routes : ℕ)
    (routes_between_mountainous_cities : ℕ) : ℕ :=
let mountainous_city_endpoints := mountainous_cities * 3 in
let routes_between_mountainous_cities_endpoints := routes_between_mountainous_cities * 2 in
let mountainous_to_plains_routes_endpoints := mountainous_city_endpoints - routes_between_mountainous_cities_endpoints in
let plains_city_endpoints := plains_cities * 3 in
let plains_city_to_mountainous_city_routes_endpoints := mountainous_to_plains_routes_endpoints in
let endpoints_fully_in_plains_cities := plains_city_endpoints - plains_city_to_mountainous_city_routes_endpoints in
endpoints_fully_in_plains_cities / 2

theorem routes_between_plains_cities_correct :
    number_of_routes_connecting_pairs_of_plains_cities 100 30 70 150 21 = 81 := by
    sorry

end routes_between_plains_cities_correct_l6_6181


namespace fraction_shaded_l6_6337

-- Define relevant elements
def quilt : ℕ := 9
def rows : ℕ := 3
def shaded_rows : ℕ := 1
def shaded_fraction := shaded_rows / rows

-- We are to prove the fraction of the quilt that is shaded
theorem fraction_shaded (h : quilt = 3 * 3) : shaded_fraction = 1 / 3 :=
by
  -- Proof goes here
  sorry

end fraction_shaded_l6_6337


namespace necessary_but_not_sufficient_condition_l6_6460

theorem necessary_but_not_sufficient_condition (x : ℝ) : |x - 1| < 2 → -3 < x ∧ x < 3 :=
by
  sorry

end necessary_but_not_sufficient_condition_l6_6460


namespace range_of_m_le_neg3_l6_6135

theorem range_of_m_le_neg3 (m : ℝ) : (∀ x ∈ set.Icc (0 : ℝ) 1, x^2 - 4 * x ≥ m) → (m ≤ -3) := by
  sorry

end range_of_m_le_neg3_l6_6135


namespace system_of_equations_solution_l6_6547

theorem system_of_equations_solution (a b x y : ℝ) 
  (h1 : x = 1) 
  (h2 : y = 2)
  (h3 : a * x + y = -1)
  (h4 : 2 * x - b * y = 0) : 
  a + b = -2 := 
sorry

end system_of_equations_solution_l6_6547


namespace student_A_more_stable_l6_6776

-- Define the variances for students A and B
def variance_A : ℝ := 0.05
def variance_B : ℝ := 0.06

-- The theorem to prove that student A has more stable performance
theorem student_A_more_stable : variance_A < variance_B :=
by {
  -- proof goes here
  sorry
}

end student_A_more_stable_l6_6776


namespace function_decreasing_interval_l6_6731

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 1

theorem function_decreasing_interval :
  ∃ I : Set ℝ, I = (Set.Ioo 0 2) ∧ ∀ x ∈ I, deriv f x < 0 :=
by
  sorry

end function_decreasing_interval_l6_6731


namespace hamburgers_left_over_l6_6844

-- Define the conditions as constants
def hamburgers_made : ℕ := 9
def hamburgers_served : ℕ := 3

-- Prove that the number of hamburgers left over is 6
theorem hamburgers_left_over : hamburgers_made - hamburgers_served = 6 := 
by
  sorry

end hamburgers_left_over_l6_6844


namespace pentagon_area_l6_6437

noncomputable def area_of_pentagon := 865

theorem pentagon_area :
  let sides := {17, 23, 18, 30, 34} in
  ∃ a b c d e : ℕ,
    sides = {a, b, c, d, e} ∧
    (∃ r s : ℕ, r^2 + s^2 = 30^2 ∧ (r = b - d ∨ r = d - b) ∧ (s = c - a ∨ s = a - c) ∧
    a * c - 1/2 * r * s = area_of_pentagon)
:= sorry

end pentagon_area_l6_6437


namespace abc_sum_eq_11sqrt6_l6_6589

theorem abc_sum_eq_11sqrt6 (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108) :
  a + b + c = 11 * Real.sqrt 6 :=
sorry

end abc_sum_eq_11sqrt6_l6_6589


namespace well_filled_subsets_count_l6_6835

def well_filled (S : Set ℕ) : Prop :=
  S.Nonempty ∧ ∀ m ∈ S, S.count(fun n => n < m) < m / 2

def problem_statement : Prop :=
  (Finset.filter well_filled (Finset.powerset (Finset.range 43))).card = (Nat.choose 43 21) - 1

theorem well_filled_subsets_count : problem_statement := 
by 
  sorry

end well_filled_subsets_count_l6_6835


namespace vertex_in_second_quadrant_l6_6994

-- Theorems and properties regarding quadratic functions and their roots.
theorem vertex_in_second_quadrant (c : ℝ) (h : 4 + 4 * c < 0) : 
  (1:ℝ) * -1^2 + 2 * -1 - c > 0 :=
sorry

end vertex_in_second_quadrant_l6_6994


namespace beautiful_fold_probs_is_half_l6_6661

noncomputable def beautiful_fold_probability (square : set (ℝ × ℝ)) (F : ℝ × ℝ) : ℝ :=
  if F ∈ square then 0.5 else 0

theorem beautiful_fold_probs_is_half (square : set (ℝ × ℝ)) (F : ℝ × ℝ) (h1 : F ∈ square) :
  beautiful_fold_probability square F = 0.5 :=
sorry

end beautiful_fold_probs_is_half_l6_6661


namespace complex_conjugate_of_z_l6_6625

def z : ℂ := 3 + 4 * complex.I

theorem complex_conjugate_of_z : complex.conj z = 3 - 4 * complex.I :=
by
  -- The proof is omitted here.
  sorry

end complex_conjugate_of_z_l6_6625


namespace second_square_area_l6_6218

theorem second_square_area
  (leg : ℝ)
  (first_square_area_h : (1/2 * leg) ^ 2 = 289)
  (isosceles_right_triangle_h : true) :
  let hypotenuse := leg * Real.sqrt 2 in
  let second_square_side := hypotenuse / 2 in
  second_square_side ^ 2 = 578 :=
by
  sorry

end second_square_area_l6_6218


namespace range_of_a_l6_6609

-- Given conditions and definitions
def b := 2
def B := 45 * (Real.pi / 180)  -- Converting degrees to radians for Lean usage
def sine_rule (a b : ℝ) (sin_A sin_B : ℝ) : Prop := (a / sin_A) = (b / sin_B)

-- The proof problem statement
theorem range_of_a (A C : ℝ) (sin_A : ℝ)
  (h1 : C = Real.pi - (B + A))
  (h2 : sine_rule a 2 sin_A Real.sqrt_two)
  (h3 : 0 < A ∧ A < Real.pi)
  (h4 : B + A < Real.pi)
  (h5 : Real.sqrt_two / 2 < sin_A ∧ sin_A < 1) :
  2 < a ∧ a < 2 * Real.sqrt_two :=
sorry

end range_of_a_l6_6609


namespace collinear_points_l6_6352

-- Given conditions from the problem
variables {Γ1 Γ2 : Type*} [Circle Γ1] [Circle Γ2]
variables (A B : Point)
variables (P Q : Point)
variables (X : Point) 
variables (C : Point)
noncomputable def is_reflection (B C : Point) (PQ : Line) : Prop := sorry
noncomputable def is_circumcircle (APQ : Triangle) (Circ : Circle) : Prop := sorry
noncomputable def reflect_point (B : Point) (PQ : Line) : Point := sorry

-- Define tangents
def is_tangent (P : Point) (Γ : Circle) : Prop := sorry

-- The main theorem statement
theorem collinear_points 
  (h1 : intersect_at_two_points Γ1 Γ2 A B) 
  (h2 : is_tangent P Γ1) 
  (h3 : is_tangent Q Γ2) 
  (h4 : ∃ Circ, is_circumcircle (triangle A P Q) Circ ∧ 
                tangent_to_circumcircle Circ P X ∧ 
                tangent_to_circumcircle Circ Q X) 
  (h5 : C = reflect_point B (line_through PQ))
:
  collinear A C X :=
sorry

end collinear_points_l6_6352


namespace range_f_pos_l6_6265

noncomputable def f : ℝ → ℝ :=
λ x, if x > 0 then Real.log x else if x < 0 then -Real.log (-x) else 0

-- Prove that the range of x for which f(x) > 0 is (-1, 0) ∪ (1, +∞)
theorem range_f_pos : { x : ℝ | f x > 0 } = set.Ioo (-1) 0 ∪ set.Ioi 1 :=
by
  sorry

end range_f_pos_l6_6265


namespace total_distance_traveled_in_12_hours_l6_6399

theorem total_distance_traveled_in_12_hours 
  (initial_distance : ℕ := 40) 
  (speed_increment : ℕ := 2) 
  (hours : ℕ := 12) 
  : (finset.range hours).sum (λ n, initial_distance + n * speed_increment) = 600 := 
sorry

end total_distance_traveled_in_12_hours_l6_6399


namespace hexagon_diagonal_inequality_l6_6492

theorem hexagon_diagonal_inequality :
  ¬ ∃ (hexagon : Finset (ℝ × ℝ)), hexagon.card = 6 ∧  -- Hexagon with 6 vertices
    (∃ d : ℝ, ∃ unique_diag : (ℝ × ℝ) × (ℝ × ℝ),
      ∀ (v1 v2 v3 v4 v5 v6 : ℝ × ℝ),
        v1 ∈ hexagon ∧ v2 ∈ hexagon ∧ v3 ∈ hexagon ∧
        v4 ∈ hexagon ∧ v5 ∈ hexagon ∧ v6 ∈ hexagon → 
        (Finset.card (Finset.image (λ (pr : (ℝ × ℝ) × (ℝ × ℝ)),
          if pr ≠ unique_diag then dist pr.fst pr.snd else 0) 
          (hexagon.powerset.image (λ s, (s.any_choice, (hexagon \ s).any_choice)))) = 8 ∧ 
        Finset.card (hexagon.powerset.image (λ s, dist s.any_choice (hexagon \ s).any_choice)) = 9 ) 
sorry

end hexagon_diagonal_inequality_l6_6492


namespace c_range_l6_6929

theorem c_range (c : ℝ) (h : c > 0)
    (p : Prop := ∀ x : ℝ, c^x ≤ c^(x + 1))
    (q : Prop := ∀ x : ℝ, x + |x - 2 * c| > 1) :
    ( ( p ∧ ¬ q → c ∈ Ioc 0 (1/2) ) ∧ ( ¬ p ∧ q → c ∈ Ici 1 ) ) :=
sorry

end c_range_l6_6929


namespace repeating_decimal_fraction_l6_6072

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l6_6072


namespace hannah_sweatshirts_l6_6581

theorem hannah_sweatshirts (S : ℕ) (h1 : 15 * S + 2 * 10 = 65) : S = 3 := 
by
  sorry

end hannah_sweatshirts_l6_6581


namespace problem_parts_l6_6793

theorem problem_parts (a b : Type) :
  (0 ∈ ({0} : Set Nat)) ∧ 
  (∅ ⊆ ({0} : Set Nat)) ∧ 
  ¬ ({(0, 1)} : Set (Nat × Nat) ⊆ ({(0, 1)} : Set (Nat × Nat))) ∧ 
  ({(a, b)} : Set (Prod a b)) = ({(b, a)} : Set (Prod a b)) :=
by
  sorry

end problem_parts_l6_6793


namespace vector_addition_correct_l6_6167

variables (a b : ℝ × ℝ)
def vector_a : ℝ × ℝ := (2, 3)
def vector_b : ℝ × ℝ := (-1, 2)

theorem vector_addition_correct : vector_a + vector_b = (1, 5) :=
by
  -- Assume a and b are vectors in 2D space
  have a := vector_a
  have b := vector_b
  -- By definition of vector addition
  sorry

end vector_addition_correct_l6_6167


namespace factorial_inequality_l6_6807

theorem factorial_inequality (n : ℕ) : 2^n * n! < (n+1)^n :=
by
  sorry

end factorial_inequality_l6_6807


namespace periodic_decimal_to_fraction_l6_6012

theorem periodic_decimal_to_fraction
  (h : ∀ n : ℕ, 0.<digit>02 n / 99) :
  2.0<digit>06 = 68 / 33 :=
sorry

end periodic_decimal_to_fraction_l6_6012


namespace candidate_percentage_l6_6816

theorem candidate_percentage (P : ℕ) (total_votes : ℕ) (vote_diff : ℕ)
  (h1 : total_votes = 7000)
  (h2 : vote_diff = 2100)
  (h3 : (P * total_votes / 100) + (P * total_votes / 100) + vote_diff = total_votes) :
  P = 35 :=
by
  sorry

end candidate_percentage_l6_6816


namespace routes_between_plains_cities_correct_l6_6182

noncomputable def number_of_routes_connecting_pairs_of_plains_cities
    (total_cities : ℕ)
    (mountainous_cities : ℕ)
    (plains_cities : ℕ)
    (total_routes : ℕ)
    (routes_between_mountainous_cities : ℕ) : ℕ :=
let mountainous_city_endpoints := mountainous_cities * 3 in
let routes_between_mountainous_cities_endpoints := routes_between_mountainous_cities * 2 in
let mountainous_to_plains_routes_endpoints := mountainous_city_endpoints - routes_between_mountainous_cities_endpoints in
let plains_city_endpoints := plains_cities * 3 in
let plains_city_to_mountainous_city_routes_endpoints := mountainous_to_plains_routes_endpoints in
let endpoints_fully_in_plains_cities := plains_city_endpoints - plains_city_to_mountainous_city_routes_endpoints in
endpoints_fully_in_plains_cities / 2

theorem routes_between_plains_cities_correct :
    number_of_routes_connecting_pairs_of_plains_cities 100 30 70 150 21 = 81 := by
    sorry

end routes_between_plains_cities_correct_l6_6182


namespace area_of_shaded_region_l6_6842

theorem area_of_shaded_region :
  let length := 12  -- length of the rectangle in cm
  let width := 8    -- width of the rectangle in cm
  let radius := 4   -- radius of each quarter circle in cm
  let area_rect := length * width -- area of the rectangle
  let area_circle := π * radius^2 -- area of one full circle
  in area_rect - area_circle = 96 - 16 * π := 
  sorry

end area_of_shaded_region_l6_6842


namespace geometric_sequence_q_and_an_l6_6971

theorem geometric_sequence_q_and_an
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geom : ∀ n, a (n + 1) = a n * q)
  (h_pos : q > 0)
  (h2_eq : a 2 = 1)
  (h2_h6_eq_9h4 : a 2 * a 6 = 9 * a 4) :
  q = 3 ∧ ∀ n, a n = 3^(n - 2) := by
sorry

end geometric_sequence_q_and_an_l6_6971


namespace cannot_move_reach_goal_l6_6303

structure Point :=
(x : ℤ)
(y : ℤ)

def area (p1 p2 p3 : Point) : ℚ :=
  (1 / 2 : ℚ) * abs (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))

noncomputable def isTriangleAreaPreserved (initPos finalPos : Point) (helper1Init helper1Final helper2Init helper2Final : Point) : Prop :=
  area initPos helper1Init helper2Init = area finalPos helper1Final helper2Final

theorem cannot_move_reach_goal :
  ¬ ∃ (r₀ r₁ : Point) (a₀ a₁ : Point) (s₀ s₁ : Point),
    r₀ = ⟨0, 0⟩ ∧ r₁ = ⟨2, 2⟩ ∧
    a₀ = ⟨0, 1⟩ ∧ a₁ = ⟨0, 1⟩ ∧
    s₀ = ⟨1, 0⟩ ∧ s₁ = ⟨1, 0⟩ ∧
    isTriangleAreaPreserved r₀ r₁ a₀ a₁ s₀ s₁ :=
by sorry

end cannot_move_reach_goal_l6_6303


namespace number_of_divisors_with_2310_factors_l6_6471

theorem number_of_divisors_with_2310_factors :
  let n := 2310
  let n_pow := n ^ n
  (∃ div_count : Nat, div_count = ∏ x in {1, 2, 4, 6, 10}.to_finset, x ∧ (div_count = n)) →
  (finset.card { d : Nat // (∃ (a b c d e : ℕ), d = (2^a) * (3^b) * (5^c) * (7^d) * (11^e) ∧ a, b, c, d, e ≤ n ∧ (a + 1) * (b + 1) * (c + 1) * (d + 1) * (e + 1) = n)} = 120) :=
by sorry

end number_of_divisors_with_2310_factors_l6_6471


namespace g_at_9_l6_6719

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g_at_3 : g 3 = 4

theorem g_at_9 : g 9 = 64 :=
by
  sorry

end g_at_9_l6_6719


namespace length_more_than_breadth_l6_6327

theorem length_more_than_breadth
  (b x : ℝ)
  (h1 : b + x = 60)
  (h2 : 4 * b + 2 * x = 200) :
  x = 20 :=
by
  sorry

end length_more_than_breadth_l6_6327


namespace distance_to_fountain_l6_6281

def total_distance : ℝ := 120
def number_of_trips : ℝ := 4
def distance_per_trip : ℝ := total_distance / number_of_trips

theorem distance_to_fountain : distance_per_trip = 30 := by
  sorry

end distance_to_fountain_l6_6281


namespace recurring_decimal_reduced_fraction_l6_6059

noncomputable def recurring_decimal_as_fraction : Prop := 
  ∀ (x y : ℚ), (x = 2.06) ∧ (y = 0.02) → y = 2 / 99 → x = 68 / 33

theorem recurring_decimal_reduced_fraction (x y : ℚ) 
  (h1 : x = 2 + 0.06) (h2 : y = 0.02) (h3 : y = 2 / 99) : 
  x = 68 / 33 := 
begin
  -- Proof here
  sorry
end

end recurring_decimal_reduced_fraction_l6_6059


namespace exterior_angle_bisector_ratio_l6_6998

theorem exterior_angle_bisector_ratio (XYZ : Triangle)
  (XZ ZY : ℝ) (h_ratio : XZ / ZY = 2 / 5)
  (Q : Point) (h_bisector : Z_ext_angle_bisector XYZ Z Q) :
  (segment_length QY / segment_length YX) = 5 / 2 := by sorry

end exterior_angle_bisector_ratio_l6_6998


namespace inequality_f1_lt_2ef2_l6_6101

noncomputable def problem_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) :=
  ∀ x : ℝ, x ∈ Set.Ici 0 → (x + 1) * f x + x * f' x ≥ 0

theorem inequality_f1_lt_2ef2 
  (f : ℝ → ℝ) 
  (f' : ℝ → ℝ) 
  (h_deriv : problem_inequality f f') : 
  f 1 < 2 * Real.exp 1 * (f 2) := 
begin 
  sorry 
end

end inequality_f1_lt_2ef2_l6_6101


namespace min_4x_y_l6_6633

variables {A B C D E M N : Type}
variables (x y : ℝ)
variables [has_zero B] [has_zero C]
variables [has_add B] [has_add C]
variables [has_smul ℝ B] [has_smul ℝ C]
variables [linear_map ℝ B A] [linear_map ℝ C A]

/-- In triangle ABC, with midpoint E of median AD, points M and N on sides AB and AC respectively, 
    if vectors AM and AN are scaled versions of AB and AC, respectively,
    the minimum value of 4x + y is 9/4. --/
theorem min_4x_y (h1 : A = (B + C) / 2) (h2 : x * B = A - M) (h3 : y * C = A - N)
  (h4 : ∃ (λ : ℝ), λ * (M - A + (B - E)) = N - A + (C - E)) :
  ∃ x y, 4 * x + y = 9 / 4 :=
sorry

end min_4x_y_l6_6633


namespace recurring_decimal_to_fraction_l6_6029

theorem recurring_decimal_to_fraction
  (h : 0.\overline{02} = (2 : ℝ) / 99) :
  2.\overline{06} = 68 / 33 := by
  sorry

end recurring_decimal_to_fraction_l6_6029


namespace point_returns_to_original_after_seven_steps_l6_6940

-- Define a structure for a triangle and a point inside it
structure Triangle :=
  (A B C : Point)

structure Point :=
  (x y : ℝ)

-- Given a triangle and a point inside it
variable (ABC : Triangle)
variable (M : Point)

-- Define the set of movements and the intersection points
def move_parallel_to_BC (M : Point) (ABC : Triangle) : Point := sorry
def move_parallel_to_AB (M : Point) (ABC : Triangle) : Point := sorry
def move_parallel_to_AC (M : Point) (ABC : Triangle) : Point := sorry

-- Function to perform the stepwise movement through 7 steps
def move_M_seven_times (M : Point) (ABC : Triangle) : Point :=
  let M1 := move_parallel_to_BC M ABC
  let M2 := move_parallel_to_AB M1 ABC 
  let M3 := move_parallel_to_AC M2 ABC
  let M4 := move_parallel_to_BC M3 ABC
  let M5 := move_parallel_to_AB M4 ABC
  let M6 := move_parallel_to_AC M5 ABC
  let M7 := move_parallel_to_BC M6 ABC
  M7

-- The theorem stating that after 7 steps, point M returns to its original position
theorem point_returns_to_original_after_seven_steps :
  move_M_seven_times M ABC = M := sorry

end point_returns_to_original_after_seven_steps_l6_6940


namespace max_a_for_f_l6_6523

theorem max_a_for_f :
  ∀ (a : ℝ), (∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → |a * x^2 - a * x + 1| ≤ 1) → a ≤ 8 :=
sorry

end max_a_for_f_l6_6523


namespace range_of_a_l6_6526

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > a → x^2 > 2 * x) → (a ≥ 2) :=
begin
  sorry
end

end range_of_a_l6_6526


namespace largest_third_altitude_of_scalene_triangle_l6_6775

theorem largest_third_altitude_of_scalene_triangle (DEF : Triangle) 
  (altitude_DE : DEF.altitude DE = 6)
  (altitude_DF : DEF.altitude DF = 18)
  (h_scalene : DEF.angles.all_unique) :
  ∃ k : ℕ, (DEF.altitude EF = k) ∧ ∀ q : ℕ, DEF.altitude EF = q → k ≥ q :=
sorry

end largest_third_altitude_of_scalene_triangle_l6_6775


namespace sarah_meals_count_l6_6679

theorem sarah_meals_count :
  let main_courses := 4
  let sides := 3
  let drinks := 2
  let desserts := 2
  main_courses * sides * drinks * desserts = 48 := 
by
  let main_courses := 4
  let sides := 3
  let drinks := 2
  let desserts := 2
  calc
    4 * 3 * 2 * 2 = 48 := sorry

end sarah_meals_count_l6_6679


namespace fixed_point_HN_through_l6_6958

-- Define the conditions and entities
def ellipse (x y : ℝ) : Prop := (x^2 / 3) + (y^2 / 4) = 1

def pointA : ℝ × ℝ := (0, -2)
def pointB : ℝ × ℝ := (3/2, -1)

-- Main statement of the problem
theorem fixed_point_HN_through (P : ℝ × ℝ) (P1 : P = (1, -2)) 
    (H : ℝ × ℝ → ℝ × ℝ → ℝ × ℝ) : 
  ellipse (fst pointA) (snd pointA) →
  ellipse (fst pointB) (snd pointB) →
  (∀ M N : ℝ × ℝ, (ellipse (fst M) (snd M) ∧ ellipse (fst N) (snd N)) ↔ 
    ((P = (1, -2)) ∧ 
    (H M T = T)) → 
  ∃ K : ℝ × ℝ, K = (0, -2) ∧ collinear K H N :=
sorry

end fixed_point_HN_through_l6_6958


namespace polynomial_self_composition_l6_6891

theorem polynomial_self_composition {p : Polynomial ℝ} {n : ℕ} (hn : 0 < n) :
  (∀ x, p.eval (p.eval x) = (p.eval x) ^ n) ↔ p = Polynomial.X ^ n :=
by sorry

end polynomial_self_composition_l6_6891


namespace smallest_n_which_contains_643_l6_6703

theorem smallest_n_which_contains_643 (m n : ℕ) (h_rel_prime : Nat.coprime m n) (h_cond : m < n) (h_contains_643 : ∃ a b c, (6::4::3::a) ∈ (n::b::c)) :
  n = 358 :=
sorry

end smallest_n_which_contains_643_l6_6703


namespace probability_same_number_on_five_dice_l6_6908

theorem probability_same_number_on_five_dice : 
  (5 : ℕ) → (6 : ℕ) → ℚ := sorry

end probability_same_number_on_five_dice_l6_6908


namespace floor_abs_sum_l6_6887

theorem floor_abs_sum (x : ℝ) (h : x = -7.3) : 
    (⌊|x|⌋ + |⌊x⌋|) = 15 := 
by
  rw h
  rw abs_of_neg
  rw floor_neg
  simp
  sorry   -- Placeholder for additional proof steps

end floor_abs_sum_l6_6887


namespace water_fee_17tons_maximize_first_tier_households_l6_6817

def water_fee (usage : ℕ) : ℝ :=
  if usage ≤ 12 then usage * 4 else if usage ≤ 16 then (12 * 4) + (usage - 12) * 5 else (12 * 4) + (4 * 5) + (usage - 16) * 7

-- Statement for part (1)
theorem water_fee_17tons : water_fee 17 = 75 :=
  by sorry

-- Sample water usage data for part (2) and part (3)
def sample_data : list ℕ := [7, 8, 8, 9, 10, 11, 13, 14, 15, 20]

-- Check if a household uses water in the second tier
def is_second_tier (usage : ℕ) : bool :=
  12 < usage ∧ usage ≤ 16

-- Get the number of households in the second tier
def count_second_tier (data : list ℕ) : ℕ :=
  data.countp is_second_tier

-- Binomial distribution to maximize the number of first-tier households
def binomial_distribution (n k : ℕ) (p : ℚ) : ℚ :=
  (nat.choose n k : ℚ) * p ^ k * (1 - p) ^ (n - k)

-- Statement for part (3)
theorem maximize_first_tier_households (n : ℕ) (p : ℚ) : ∃ k : ℕ, binomial_distribution n k (3/5) ∧ k = 6 :=
  by sorry

end water_fee_17tons_maximize_first_tier_households_l6_6817


namespace double_inequality_l6_6404

variable (a b c : ℝ)

theorem double_inequality 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (hab : a + b ≤ 1) (hbc : b + c ≤ 1) (hca : c + a ≤ 1) :
  a^2 + b^2 + c^2 ≤ a + b + c - a * b - b * c - c * a ∧ 
  a + b + c - a * b - b * c - c * a ≤ 1 / 2 * (1 + a^2 + b^2 + c^2) := 
sorry

end double_inequality_l6_6404


namespace range_of_a_l6_6569

noncomputable def f (x : ℝ) : ℝ := (Real.exp x + Real.exp (-x)) * (2 * x^2 + 1)

theorem range_of_a (a : ℝ) :
  (∀ x ∈ set.Icc (3 : ℝ) 4, f (a * x + 1) ≤ f (x - 2)) ↔ -2 / 3 ≤ a ∧ a ≤ 0 := by
  sorry

end range_of_a_l6_6569


namespace unique_real_solution_eq_sqrt3_l6_6086

theorem unique_real_solution_eq_sqrt3
  (a : ℝ)
  (h : ∀ x : ℝ, x² + a * |x| + a² - 3 = 0 → x = 0) :
  a = sqrt 3 :=
by
  sorry

end unique_real_solution_eq_sqrt3_l6_6086


namespace investment_amount_l6_6798

variable (I : ℝ) (R : ℝ) (P : ℝ) (T : ℝ)

-- Conditions
def monthly_interest_payment (I : ℝ) := I = 216
def annual_interest_rate (R : ℝ) := R = 0.09
def time_in_months (T : ℝ) := T = 1
def monthly_rate (R : ℝ) := R / 12

-- Problem Statement
theorem investment_amount (I : ℝ) (R : ℝ) (P : ℝ) (T : ℝ) :
  monthly_interest_payment I → annual_interest_rate R → time_in_months T →
  P = I / (monthly_rate R * T) → P = 28800 :=
by
  intros hI hR hT hP
  sorry

end investment_amount_l6_6798


namespace repeating_decimal_fraction_l6_6071

theorem repeating_decimal_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by
  sorry

end repeating_decimal_fraction_l6_6071


namespace zero_of_f_l6_6752

noncomputable def f (x : ℝ) : ℝ := (x - 1) / (x + 1)

theorem zero_of_f :
  ∃ x : ℝ, f x = 0 ↔ x = 1 :=
by
  sorry

end zero_of_f_l6_6752


namespace initial_kittens_l6_6439

theorem initial_kittens (x : ℕ) (h : x + 3 = 9) : x = 6 :=
by {
  sorry
}

end initial_kittens_l6_6439


namespace double_increase_divide_l6_6989

theorem double_increase_divide (x : ℤ) (h : (2 * x + 7) / 5 = 17) : x = 39 := by
  sorry

end double_increase_divide_l6_6989


namespace problem1_problem2_l6_6570

def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x)
def g (x a : ℝ) : ℝ := a - 4 * Real.sqrt 3 * Real.sin x
def h (x : ℝ) : ℝ := f x + g x 0

theorem problem1 :
  (∀ x : ℝ, h x ≤ 5) ∧ (∃ x : ℝ, h x = 5) ∧
  (∀ x : ℝ, h x ≥ -4 - 4 * Real.sqrt 3) ∧ (∃ x : ℝ, h x = -4 - 4 * Real.sqrt 3) :=
sorry

theorem problem2 (a : ℝ) (n m : ℝ) :
  (∀ x ∈ Set.Icc n m, f x ≥ g x a) ∧ (m - n = 5 * Real.pi / 3) → a = -7 :=
sorry

end problem1_problem2_l6_6570


namespace parallel_lines_lambda_l6_6576

theorem parallel_lines_lambda (λ : ℝ) :
  ((λ - 1) * λ - 2 = 0) → λ = -1 :=
by
  intros h
  have : λ^2 - λ - 2 = 0 := by sorry
  have : λ = -1 ∨ λ = 2 := by sorry
  show λ = -1
    from sorry

end parallel_lines_lambda_l6_6576


namespace periodic_decimal_to_fraction_l6_6007

theorem periodic_decimal_to_fraction
  (h : ∀ n : ℕ, 0.<digit>02 n / 99) :
  2.0<digit>06 = 68 / 33 :=
sorry

end periodic_decimal_to_fraction_l6_6007


namespace loss_of_450_is_negative_450_l6_6157

-- Define the concept of profit and loss based on given conditions.
def profit (x : Int) := x
def loss (x : Int) := -x

-- The mathematical statement:
theorem loss_of_450_is_negative_450 :
  (profit 1000 = 1000) → (loss 450 = -450) :=
by
  intro h
  sorry

end loss_of_450_is_negative_450_l6_6157


namespace slope_AB_is_2_l6_6156

def point := (ℝ × ℝ)

def A : point := (1, 2)
def B : point := (3, 6)

def slope (P Q : point) : ℝ := 
  (Q.2 - P.2) / (Q.1 - P.1)

theorem slope_AB_is_2 : slope A B = 2 := 
  sorry

end slope_AB_is_2_l6_6156


namespace plane_equation_exists_l6_6896

def point1 : ℝ × ℝ × ℝ := (2, -1, 0)
def point2 : ℝ × ℝ × ℝ := (0, 3, 2)
def normal_of_given_plane : ℝ × ℝ × ℝ := (2, -1, 4)
def normal_of_desired_plane : ℝ × ℝ × ℝ := (9, -2, -3)

theorem plane_equation_exists :
  ∃ A B C D : ℤ, (A, B, C) = (9, -2, -3) ∧ D = -20 ∧
    (∀ (x y z : ℝ), A*x + B*y + C*z + D = 0) ∧
    A > 0 ∧ Int.gcd A (Int.gcd B (Int.gcd C D)) = 1 :=
by
  use 9, -2, -3, -20
  split; try { simp }
  intro x y z
  linarith
  simp only [lt_self_iff_false, not_false_iff, zero_lt_bit0, zero_lt_one, true_and, int.coe_nat_pos]
  sorry -- This is to skip the proof part

end plane_equation_exists_l6_6896


namespace find_number_l6_6696

variable (x : ℕ)

theorem find_number (h : (10 + 20 + x) / 3 = ((10 + 40 + 25) / 3) + 5) : x = 60 :=
by
  sorry

end find_number_l6_6696


namespace g_9_eq_64_l6_6714

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g(x + y) = g(x) * g(y)

axiom g_3_eq_4 : g(3) = 4

theorem g_9_eq_64 : g(9) = 64 := by
  sorry

end g_9_eq_64_l6_6714


namespace max_n_sum_negative_l6_6623

theorem max_n_sum_negative (a : ℕ → ℤ) (S : ℕ → ℤ) 
  (h1 : a 10 < 0) 
  (h2 : a 11 > 0) 
  (h3 : a 11 > abs (a 10)) 
  (hS : ∀ n, S n = (n * (2 * (a 1) + (n - 1) * (a 2))) / 2) :
  max {n : ℕ | S n < 0} = 19 :=
sorry

end max_n_sum_negative_l6_6623


namespace problem1_sol_l6_6741

noncomputable def problem1 :=
  let total_people := 200
  let avg_feelings_total := 70
  let female_total := 100
  let a := 30 -- derived from 2a + (70 - a) = 100
  let chi_square := 200 * (70 * 40 - 30 * 60) ^ 2 / (130 * 70 * 100 * 100)
  let k_95 := 3.841 -- critical value for 95% confidence
  let p_xi_2 := (1 / 3)
  let p_xi_3 := (1 / 2)
  let p_xi_4 := (1 / 6)
  let exi := (2 * (1 / 3)) + (3 * (1 / 2)) + (4 * (1 / 6))
  chi_square < k_95 ∧ exi = 17 / 6

theorem problem1_sol : problem1 :=
  by {
    sorry
  }

end problem1_sol_l6_6741


namespace geometric_sequence_y_l6_6954

theorem geometric_sequence_y (x y z : ℝ) (h1 : 1 ≠ 0) (h2 : x ≠ 0) (h3 : y ≠ 0) (h4 : z ≠ 0) (h5 : 9 ≠ 0)
  (h_seq : ∀ a b c d e : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0 ∧ a * e = b * d ∧ b * d = c^2) →
           (a, b, c, d, e) = (1, x, y, z, 9)) :
  y = 3 :=
sorry

end geometric_sequence_y_l6_6954


namespace find_g_9_l6_6713

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_of_3 : g 3 = 4

theorem find_g_9 : g 9 = 64 := by
  sorry

end find_g_9_l6_6713


namespace square_increasing_on_positive_reals_l6_6671

noncomputable def is_increasing_on (f : ℝ → ℝ) (I : set ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ∈ I → x₂ ∈ I → x₁ < x₂ → f x₁ < f x₂

theorem square_increasing_on_positive_reals :
  is_increasing_on (λ x, x^2) (set.Ioi 0) :=
by
  sorry

end square_increasing_on_positive_reals_l6_6671


namespace recurring_decimal_reduced_fraction_l6_6063

noncomputable def recurring_decimal_as_fraction : Prop := 
  ∀ (x y : ℚ), (x = 2.06) ∧ (y = 0.02) → y = 2 / 99 → x = 68 / 33

theorem recurring_decimal_reduced_fraction (x y : ℚ) 
  (h1 : x = 2 + 0.06) (h2 : y = 0.02) (h3 : y = 2 / 99) : 
  x = 68 / 33 := 
begin
  -- Proof here
  sorry
end

end recurring_decimal_reduced_fraction_l6_6063


namespace leak_empty_time_l6_6396

theorem leak_empty_time (P L : ℝ) (h1 : P = 1 / 6) (h2 : P - L = 1 / 12) : 1 / L = 12 :=
by
  -- Proof to be provided
  sorry

end leak_empty_time_l6_6396


namespace repeating_decimal_to_fraction_l6_6003

theorem repeating_decimal_to_fraction (h : 0.\overline{02} = 2 / 99) : 
  2.\overline{06} = 68 / 33 := by
  sorry

end repeating_decimal_to_fraction_l6_6003


namespace smallest_good_number_l6_6859

def is_good_number (n : ℕ) : Prop :=
  (∀ (d : List ℕ), (d = (List.filter (λ x, n % x = 0) (List.range (n+1))) ∧ d.length = 8 ∧ d.sum = 3240))

theorem smallest_good_number : ∃ n : ℕ, is_good_number n ∧ ∀ m : ℕ, is_good_number m → n ≤ m :=
  ⟨1614, 
  begin
    sorry
  end⟩

end smallest_good_number_l6_6859


namespace coefficient_of_x4_in_expansion_l6_6361

theorem coefficient_of_x4_in_expansion :
  (∑ k in Finset.range (8 + 1), (Nat.choose 8 k) * (x : ℝ)^(8 - k) * (3 * Real.sqrt 2)^k).coeff 4 = 22680 :=
by
  sorry

end coefficient_of_x4_in_expansion_l6_6361


namespace relationship_between_c_and_d_l6_6596

noncomputable def c : ℝ := Real.log 400 / Real.log 4
noncomputable def d : ℝ := Real.log 20 / Real.log 2

theorem relationship_between_c_and_d : c = d := by
  sorry

end relationship_between_c_and_d_l6_6596


namespace option_a_is_false_option_b_is_false_option_c_is_true_option_d_is_true_l6_6099

open real

-- Definitions
def circle (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def line (x y m : ℝ) : Prop := m * x + x + 2 * y - 1 + m = 0

-- Proof statements
theorem option_a_is_false : ¬ ∃ (x y : ℝ), circle x y ∧ (∃ m : ℝ, m = 0 ∧ dist (x, y) (m * x + x + 2 * y - 1 + m = 0) = 1) :=
sorry

theorem option_b_is_false : ¬ ∃ m : ℝ, ∀ x y : ℝ, ¬ (circle x y ∧ line x y m) :=
sorry

theorem option_c_is_true (a : ℝ) : ((a = 8) ↔ ∀ x y : ℝ, circle x y →
  ((x - 1)^2 + (y + 4)^2 = 17 - a) ∧
  (∃ (m : ℝ), m * x + x + 2 * y - 1 + m = 0)) :=
sorry

theorem option_d_is_true : ∀ x y : ℝ, (line x y 1) → (circle x y → (x^2 + (y - 2)^2 = 4)) :=
sorry

end option_a_is_false_option_b_is_false_option_c_is_true_option_d_is_true_l6_6099


namespace area_inside_C_but_outside_A_and_B_l6_6872

def radius_A := 1
def radius_B := 1
def radius_C := 2
def tangency_AB := true
def tangency_AC_non_midpoint := true

theorem area_inside_C_but_outside_A_and_B :
  let areaC := π * (radius_C ^ 2)
  let areaA := π * (radius_A ^ 2)
  let areaB := π * (radius_B ^ 2)
  let overlapping_area := 2 * (π * (radius_A ^ 2) / 2) -- approximation
  areaC - overlapping_area = 3 * π - 2 :=
by
  sorry

end area_inside_C_but_outside_A_and_B_l6_6872


namespace solution_exists_unique_n_l6_6905

theorem solution_exists_unique_n (n : ℕ) : 
  (∀ m : ℕ, (10 * m > 120) ∨ ∃ k1 k2 k3 : ℕ, 10 * k1 + n * k2 + (n + 1) * k3 = 120) = false → 
  n = 16 := by sorry

end solution_exists_unique_n_l6_6905


namespace total_fruits_is_43_apple_to_pear_ratio_is_24_to_19_l6_6642

def keith_pears : ℕ := 6
def keith_apples : ℕ := 4
def jason_pears : ℕ := 9
def jason_apples : ℕ := 8
def joan_pears : ℕ := 4
def joan_apples : ℕ := 12

def total_pears : ℕ := keith_pears + jason_pears + joan_pears
def total_apples : ℕ := keith_apples + jason_apples + joan_apples
def total_fruits : ℕ := total_pears + total_apples
def apple_to_pear_ratio : ℚ := total_apples / total_pears

theorem total_fruits_is_43 : total_fruits = 43 := by
  sorry

theorem apple_to_pear_ratio_is_24_to_19 : apple_to_pear_ratio = 24/19 := by
  sorry

end total_fruits_is_43_apple_to_pear_ratio_is_24_to_19_l6_6642


namespace unique_solution_quadratic_l6_6901

theorem unique_solution_quadratic (n : ℕ) : (∀ x : ℝ, 4 * x^2 + n * x + 4 = 0) → n = 8 :=
by
  intros h
  sorry

end unique_solution_quadratic_l6_6901


namespace fruit_basket_combinations_l6_6586

theorem fruit_basket_combinations (apples oranges : ℕ) (ha : apples = 6) (ho : oranges = 12) : 
  (∃ (baskets : ℕ), 
    (∀ a, 1 ≤ a ∧ a ≤ apples → ∃ b, 2 ≤ b ∧ b ≤ oranges ∧ baskets = a * b) ∧ baskets = 66) :=
by {
  sorry
}

end fruit_basket_combinations_l6_6586


namespace probability_sum_seven_two_dice_l6_6788

theorem probability_sum_seven_two_dice :
  let outcomes := { (x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} }
  let favorable := { (x, y) ∈ outcomes | x + y = 7 }
  ∑ x in favorable, 1 / ∑ xy in outcomes, 1 = 1 / 6 := 
by
  let outcomes := { (x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6} }
  let favorable := { (x, y) ∈ outcomes | x + y = 7 }
  have favorable_count : favorable.card = 6 := sorry
  have outcomes_count : outcomes.card = 36 := sorry
  have probability_eq : favorable_count / outcomes_count = 1 / 6 := by 
    rw [favorable_count, outcomes_count]
    norm_num
  show probability_eq = 1 / 6 from probability_eq

end probability_sum_seven_two_dice_l6_6788


namespace cost_distribution_correct_l6_6210

-- Define the vertices of the rhombus and their properties
variable (A B C D : Type)
variable [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]

-- Define the distances and the properties of the rhombus
variable (side_length : ℝ)
variable (angle : ℝ)
variable (long_diagonal : ℝ)

-- The conditions
axiom side_length_is_15 : side_length = 15
axiom angle_is_60_degrees : angle = π / 3
axiom long_diagonal_condition : long_diagonal = 2 * side_length * cos(π / 6)

-- The cost distribution ratios
def cost_distribution_ratio (A B C D : Type) [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D] : ℝ × ℝ × ℝ × ℝ :=
  if side_length_is_15 ∧ angle_is_60_degrees ∧ long_diagonal_condition then
    ((22.7 / 100), (27.3 / 100), (22.7 / 100), (27.3 / 100))
  else
    (0, 0, 0, 0)

-- We need to prove that the cost distribution ratio is correctly calculated
theorem cost_distribution_correct : 
  cost_distribution_ratio A B C D = (22.7 / 100, 27.3 / 100, 22.7 / 100, 27.3 / 100) :=
sorry

end cost_distribution_correct_l6_6210


namespace digits_10s_place_repetition_l6_6690

theorem digits_10s_place_repetition (n : ℕ) (hn : n ≥ 18) :
  (∃ m, ∀ k < 3, (3 : ℕ)^n % 1000 = (3 : ℕ)^(n + k) % 1000) ∧
  (∃ m > 3, ∀ k < m, (3 : ℕ)^n % 1000 = (3 : ℕ)^(n + k) % 1000) :=
by
  -- Implementation of the proof should use repeated properties of modular arithmetic, and
  -- specifically considering cycles in the powers of 3 modulo 1000.
  sorry

end digits_10s_place_repetition_l6_6690


namespace shaded_region_area_l6_6840

def Point := (ℝ × ℝ)

structure Rectangle :=
(M N P Q : Point)
(MP_length : 0 < dist M P)
(MN_length : dist M N = 2 * dist M P)
(P_coordinates : P = (0, 0))
(Q_coordinates : Q = (4, 0))
(N_coordinates : N = (4, 2))
(M_coordinates : M = (0, 2))

structure Extensions :=
(S : Point)
(R : Point)
(S_coordinates : S = (-2, 2))
(R_coordinates : R = (4, -2))

def area_of_shaded_region (rect : Rectangle) (ext : Extensions) : ℝ :=
1

theorem shaded_region_area (rect : Rectangle) (ext : Extensions) :
  area_of_shaded_region rect ext = 1 :=
sorry

end shaded_region_area_l6_6840


namespace least_cost_between_65_and_80_pounds_l6_6429

def type_a_price_per_pound : ℕ → ℝ
| 5 := 13.85 / 5
| 10 := 20.40 / 10
| 25 := 32.25 / 25
| _ := sorry

def type_b_price_per_pound : ℕ → ℝ
| 7 := 18.65 / 7
| 15 := 28.50 / 15
| 30 := 43.75 / 30
| _ := sorry

def type_c_price_per_pound : ℕ → ℝ
| 10 := 22.00 / 10
| 20 := 36.20 / 20
| 50 := 66.50 / 50
| _ := sorry

def price_of_combination (a : ℕ) (b : ℕ) (c : ℕ) : ℝ :=
  (if a = 25 then 32.25 else 0) + 
  (if b = 30 then 43.75 else 0) + 
  (if c = 50 then 66.50 else 0)

def applicable_discount (total_weight : ℕ) (total_cost : ℝ) : ℝ :=
  if total_weight >= 70 then total_cost * 0.95 else total_cost

theorem least_cost_between_65_and_80_pounds : ∃ (a b c : ℕ), 
  (65 ≤ a + b + c ∧ a + b + c ≤ 80) ∧ 
  applicable_discount (a + b + c) (price_of_combination a b c) = 93.8125 :=
by sorry

end least_cost_between_65_and_80_pounds_l6_6429


namespace coplanar_iff_m_eq_neg_8_l6_6251

variable {V : Type} [AddCommGroup V] [Module ℝ V]
variable (A B C D : V)
variable (m : ℝ)

theorem coplanar_iff_m_eq_neg_8 
  (h : 4 • A - 3 • B + 7 • C + m • D = 0) : m = -8 ↔ ∃ a b c d : ℝ, a + b + c + d = 0 ∧ a • A + b • B + c • C + d • D = 0 :=
by
  sorry

end coplanar_iff_m_eq_neg_8_l6_6251


namespace coeff_sum_eq_32_l6_6166

theorem coeff_sum_eq_32 (n : ℕ) (h : (2 : ℕ)^n = 32) : n = 5 :=
sorry

end coeff_sum_eq_32_l6_6166


namespace Todd_time_correct_l6_6470

theorem Todd_time_correct :
  let Brian_time := 96
  let Todd_time := Brian_time - 8
  Todd_time = 88 :=
by
  let Brian_time := 96
  let Todd_time := Brian_time - 8
  sorry

end Todd_time_correct_l6_6470


namespace average_postcards_per_day_l6_6275

theorem average_postcards_per_day 
  (a : ℕ) (d : ℕ) (n : ℕ)
  (h_a : a = 10)
  (h_d : d = 12)
  (h_n : n = 7) :
  (∑ i in finset.range n, (a + i * d)) / n = 46 :=
by 
  sorry

end average_postcards_per_day_l6_6275


namespace find_missing_edge_l6_6706

-- Define the known parameters
def volume : ℕ := 80
def edge1 : ℕ := 2
def edge3 : ℕ := 8

-- Define the missing edge
def missing_edge : ℕ := 5

-- State the problem
theorem find_missing_edge (volume : ℕ) (edge1 : ℕ) (edge3 : ℕ) (missing_edge : ℕ) :
  volume = edge1 * missing_edge * edge3 →
  missing_edge = 5 :=
by
  sorry

end find_missing_edge_l6_6706


namespace permutation_property_P_more_l6_6653

noncomputable def has_property_P {α : Type*} [Fintype α] [LinearOrder α]
  (n : ℕ) (x : List α) : Prop :=
∃ i, i < (x.length - 1) ∧ (|x.nth_le i _ - x.nth_le (i + 1) _| = n)

/--
For any positive integer \( n \), the number of permutations with property \( P_n \)
is greater than the number of permutations without property \( P_n \).
-/
theorem permutation_property_P_more {α : Type*} [Fintype α] [LinearOrder α]
  (n : ℕ) (hpos : 0 < n) :
  let S := Finset.univ.permutations;
  let with_P := S.filter (has_property_P n);
  let without_P := S.filter (λ x, ¬(has_property_P n x));
  with_P.card > without_P.card :=
sorry

end permutation_property_P_more_l6_6653


namespace max_mn_sq_l6_6485

theorem max_mn_sq {m n : ℤ} (h1: 1 ≤ m ∧ m ≤ 2005) (h2: 1 ≤ n ∧ n ≤ 2005) 
(h3: (n^2 + 2*m*n - 2*m^2)^2 = 1): m^2 + n^2 ≤ 702036 :=
sorry

end max_mn_sq_l6_6485


namespace find_range_x_l6_6948

variable {f : ℝ → ℝ}

-- Given conditions
def even_function (f : ℝ → ℝ) := ∀ x, f x = f (-x)
def monotone_decreasing (f : ℝ → ℝ) := ∀ x y, 0 ≤ x → x ≤ y → f y ≤ f x
def f_two_eq_zero := f 2 = 0

-- Define the range problem
theorem find_range_x (h_even : even_function f)
  (h_monotone: monotone_decreasing f) (h_f_two: f_two_eq_zero) :
  {x : ℝ | x * f (x - 1) > 0} = (Set.Ioo (-∞) (-1)) ∪ (Set.Ioo 0 3) := 
  sorry

end find_range_x_l6_6948


namespace length_more_than_breadth_l6_6330

theorem length_more_than_breadth (b x : ℕ) 
  (h1 : 60 = b + x) 
  (h2 : 4 * b + 2 * x = 200) : x = 20 :=
by {
  sorry
}

end length_more_than_breadth_l6_6330


namespace repeating_decimal_as_fraction_l6_6041

theorem repeating_decimal_as_fraction : (0.\overline{02} = 2 / 99) → (2.\overline{06} = 68 / 33) :=
by
  sorry

end repeating_decimal_as_fraction_l6_6041


namespace plains_routes_count_l6_6197

-- Defining the total number of cities and the number of cities in each region
def total_cities : Nat := 100
def mountainous_cities : Nat := 30
def plains_cities : Nat := total_cities - mountainous_cities

-- Defining the number of routes established each year and over three years
def routes_per_year : Nat := 50
def total_routes : Nat := routes_per_year * 3

-- Defining the number of routes connecting pairs of mountainous cities
def mountainous_routes : Nat := 21

-- The statement to prove the number of routes connecting pairs of plains cities
theorem plains_routes_count :
  plains_cities = 70 →
  total_routes = 150 →
  mountainous_routes = 21 →
  3 * mountainous_cities - 2 * mountainous_routes = 48 →
  3 * plains_cities - 48 = 162 →
  81 = 81 := sorry

end plains_routes_count_l6_6197


namespace find_X_l6_6371

theorem find_X :
  let N := 90
  let X := (1 / 15) * N - (1 / 2 * 1 / 3 * 1 / 5 * N)
  X = 3 := by
  sorry

end find_X_l6_6371


namespace necessary_but_not_sufficient_l6_6910

-- Define propositions p and q
def p (f : ℝ → ℝ) (x0 : ℝ) : Prop := deriv f x0 = 0
def q (f : ℝ → ℝ) (x0 : ℝ) : Prop := ∀ x, (x = x0 → local_min f x0 ∨ local_max f x0)

-- The statement to prove
theorem necessary_but_not_sufficient (f : ℝ → ℝ) (x0 : ℝ) [differentiable ℝ f] :
  (p f x0 → q f x0) ∧ ¬ (q f x0 → p f x0) :=
by
  sorry

end necessary_but_not_sufficient_l6_6910


namespace periodic_decimal_to_fraction_l6_6011

theorem periodic_decimal_to_fraction
  (h : ∀ n : ℕ, 0.<digit>02 n / 99) :
  2.0<digit>06 = 68 / 33 :=
sorry

end periodic_decimal_to_fraction_l6_6011


namespace polynomial_factorization_l6_6379

theorem polynomial_factorization :
  (x : ℤ[X]) →
  x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) :=
by
  intros x
  sorry

end polynomial_factorization_l6_6379


namespace probability_to_left_of_y_axis_l6_6290

-- Define the vertices of the parallelogram
structure Point :=
  (x : ℝ)
  (y : ℝ)

noncomputable def A : Point := ⟨3, 4⟩
noncomputable def B : Point := ⟨-2, 1⟩
noncomputable def C : Point := ⟨-5, -2⟩
noncomputable def D : Point := ⟨0, 1⟩

-- Define the parallelogram
structure Parallelogram :=
  (A B C D : Point)

noncomputable def ABCD : Parallelogram := ⟨A, B, C, D⟩

-- Lean statement for the proof problem
theorem probability_to_left_of_y_axis (P : Parallelogram) : 
  (P = ABCD) → (probability_to_left_of_y_axis P = 1 / 2) :=
by 
  sorry

end probability_to_left_of_y_axis_l6_6290


namespace evaluate_integral_l6_6494

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := 3 * x^2 + k

theorem evaluate_integral (k : ℝ) :
  (∫ x in 0..2, f(k, x)) = 10 → k = 1 :=
by
  intro h
  sorry

end evaluate_integral_l6_6494


namespace affine_parallel_l6_6299

-- Define an affine transformation T
def affine_transformation (A : ℝ → ℝ → ℝ → ℝ) (b : ℝ × ℝ) (x : ℝ × ℝ) : ℝ × ℝ :=
  (A x.1 x.2 + b.1, A x.2 x.1 + b.2)

-- Assume L1 and L2 are parallel lines in ℝ²
def parallel_lines (L1 L2 : ℝ → ℝ × ℝ) : Prop :=
  ∀ t1 t2, L1 t1 = L2 t2

-- The theorem statement
theorem affine_parallel {A : ℝ → ℝ → ℝ → ℝ} {b : ℝ × ℝ} {L1 L2 : ℝ → ℝ × ℝ} :
  parallel_lines L1 L2 →
  parallel_lines (affine_transformation A b ∘ L1) (affine_transformation A b ∘ L2) :=
by
  intros
  sorry

end affine_parallel_l6_6299


namespace triangle_BCM_is_isosceles_l6_6627

-- Definitions for the geometric setup.
variables {Point : Type} {Line : Type} {Circle : Type}
variables A B C D P S T M : Point
variables AD BC : Line
variables ABP CDP : Circle

-- Conditions given in the problem.
axiom parallel_AD_BC : AD ∥ BC
axiom acute_angle_A : ∠ A < π / 2
axiom acute_angle_D : ∠ D < π / 2
axiom diagonals_intersect_P : ∃ Q : Point, Q = P
axiom circumcircle_ABP_S : S ∈ ABP ∧ S ∈ AD
axiom circumcircle_CDP_T : T ∈ CDP ∧ T ∈ AD
axiom midpoint_M_ST : M = midpoint S T

-- The goal to prove.
theorem triangle_BCM_is_isosceles :
  is_isosceles (triangle B C M) :=
sorry

end triangle_BCM_is_isosceles_l6_6627


namespace decimal_to_fraction_l6_6017

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l6_6017


namespace repeating_decimal_to_fraction_l6_6002

theorem repeating_decimal_to_fraction (h : 0.\overline{02} = 2 / 99) : 
  2.\overline{06} = 68 / 33 := by
  sorry

end repeating_decimal_to_fraction_l6_6002


namespace periodic_decimal_to_fraction_l6_6006

theorem periodic_decimal_to_fraction
  (h : ∀ n : ℕ, 0.<digit>02 n / 99) :
  2.0<digit>06 = 68 / 33 :=
sorry

end periodic_decimal_to_fraction_l6_6006


namespace sin2alpha_plus_2cos2alpha_eq_neg2_l6_6159

theorem sin2alpha_plus_2cos2alpha_eq_neg2 (α : ℝ) (h : sin α = -2 * cos α) : 
  sin (2 * α) + 2 * cos (2 * α) = -2 := 
by
  sorry

end sin2alpha_plus_2cos2alpha_eq_neg2_l6_6159


namespace weeks_to_save_remaining_l6_6272

-- Assuming the conditions
def cost_of_shirt : ℝ := 3
def amount_saved : ℝ := 1.5
def saving_per_week : ℝ := 0.5

-- The proof goal
theorem weeks_to_save_remaining (cost_of_shirt amount_saved saving_per_week : ℝ) :
  cost_of_shirt = 3 ∧ amount_saved = 1.5 ∧ saving_per_week = 0.5 →
  ((cost_of_shirt - amount_saved) / saving_per_week) = 3 := by
  sorry

end weeks_to_save_remaining_l6_6272


namespace exists_distinct_positive_integers_l6_6915

def D (n : ℕ) : Set ℤ :=
  {d | ∃ a b : ℕ, a * b = n ∧ a > b ∧ a > 0 ∧ b > 0 ∧ d = a - b}

theorem exists_distinct_positive_integers (k : ℕ) (hk : k > 1) :
  ∃ n : Fin k → ℕ,
    (∀ i : Fin k, (n i) > 1) ∧ 
    (Function.Injective n) ∧ 
    (Set.card {i | ∃ j, j ≠ i ∧ n i ∈ D (n j)} ≥ 2) := 
sorry

end exists_distinct_positive_integers_l6_6915


namespace sequence_a_n_final_expression_l6_6939

noncomputable def a : ℕ → ℚ
| 0       := 3 / 2  -- base case for the first element
| (n + 1) := 2 - 1 / 2^(n + 1)

def S (n : ℕ) : ℚ := (finset.range (n + 1)).sum a

theorem sequence_a_n (n : ℕ) : S n + a n = 2 * n + 1 := by
  induction n with
  | zero => 
    -- base case
    sorry
  | succ n ih =>
    -- induction step
    sorry

theorem final_expression (n : ℕ) : a n = 2 - 1 / 2^n := by
  induction n with
  | zero =>
    -- base case
    sorry
  | succ n ih =>
    -- induction step
    sorry

end sequence_a_n_final_expression_l6_6939


namespace toothpick_problem_solve_l6_6091

noncomputable def removing_minimum_toothpicks (t : Type) (toothpicks_used : ℕ) (upward_triangles : ℕ) (downward_triangles : ℕ) : Prop :=
  upward_triangles + downward_triangles = 25 ∧
  toothpicks_used = 45 ∧
  ∃ removed_toothpicks, removed_toothpicks = 15 ∧ (∀ upward_triangles downward_triangles, (upward_triangles = 0 ∧ downward_triangles = 0))

theorem toothpick_problem_solve : removing_minimum_toothpicks ℕ 45 15 10 :=
begin
  sorry
end

end toothpick_problem_solve_l6_6091


namespace abc_sum_l6_6594

theorem abc_sum (a b c : ℝ) (h1 : a * b = 36) (h2 : a * c = 72) (h3 : b * c = 108)
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : a + b + c = 13 * Real.sqrt 6 := 
sorry

end abc_sum_l6_6594


namespace percent_covered_by_larger_triangles_l6_6839

-- Define the number of small triangles in one large hexagon
def total_small_triangles := 16

-- Define the number of small triangles that are part of the larger triangles within one hexagon
def small_triangles_in_larger_triangles := 9

-- Calculate the fraction of the area of the hexagon covered by larger triangles
def fraction_covered_by_larger_triangles := 
  small_triangles_in_larger_triangles / total_small_triangles

-- Define the expected result as a fraction of the total area
def expected_fraction := 56 / 100

-- The proof problem in Lean 4 statement:
theorem percent_covered_by_larger_triangles
  (h1 : fraction_covered_by_larger_triangles = 9 / 16) :
  fraction_covered_by_larger_triangles = expected_fraction :=
  by
    sorry

end percent_covered_by_larger_triangles_l6_6839


namespace problem_probability_ao_drawn_second_l6_6758

def is_ao_drawn_second (pair : ℕ × ℕ) : Bool :=
  pair.snd = 3

def random_pairs : List (ℕ × ℕ) := [
  (1, 3), (2, 4), (1, 2), (3, 2), (4, 3), (1, 4), (2, 4), (3, 2), (3, 1), (2, 1), 
  (2, 3), (1, 3), (3, 2), (2, 1), (2, 4), (4, 2), (1, 3), (3, 2), (2, 1), (3, 4)
]

def count_ao_drawn_second : ℕ :=
  (random_pairs.filter is_ao_drawn_second).length

def probability_ao_drawn_second : ℚ :=
  count_ao_drawn_second / random_pairs.length

theorem problem_probability_ao_drawn_second :
  probability_ao_drawn_second = 1 / 4 :=
by
  sorry

end problem_probability_ao_drawn_second_l6_6758


namespace range_of_m_decreasing_l6_6522

theorem range_of_m_decreasing (m : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → (m - 3) * x₁ + 5 > (m - 3) * x₂ + 5) ↔ m < 3 :=
by
  sorry

end range_of_m_decreasing_l6_6522


namespace range_of_x_l6_6606

variable (a x : ℝ)

theorem range_of_x :
  (∃ a ∈ Set.Icc 2 4, a * x ^ 2 + (a - 3) * x - 3 > 0) →
  x ∈ Set.Iio (-1) ∪ Set.Ioi (3 / 4) :=
by
  sorry

end range_of_x_l6_6606


namespace selection_methods_l6_6934

theorem selection_methods (students : Finset ℕ) (A B : ℕ) (hA : A ∈ students) (hB : B ∈ students) (h_size : students.card = 6) :
  (∃ (s : Finset ℕ), s.subset students ∧ s.card = 3 ∧ (A ∈ s ∨ B ∈ s) ∧ (s ≠ {A, B} ∧ (∃ x, x ∈ students ∧ x ∉ {A, B} ∧ x ∈ s)) ∧ (∀ x ∈ s, x ≠ A ∧ x ≠ B) ∧ (∃ ss, ss.subset students \ s ∧ ss.card = 3)) ↔ 96 := sorry

end selection_methods_l6_6934


namespace coefficient_of_x4_in_expansion_l6_6894

theorem coefficient_of_x4_in_expansion :
  let expansion_term (r : ℕ) := (Nat.choose 6 r) * 2 ^ (6 - r) * x^r
  let coeff_x4 := expansion_term 4
  let coeff_x3 := expansion_term 3
  coeff_x4 - coeff_x3 * x = -100 := by
  sorry

end coefficient_of_x4_in_expansion_l6_6894


namespace circumscribed_inscribed_inequality_l6_6808

theorem circumscribed_inscribed_inequality {A B C : Point} (triangleABC : triangle A B C) :
    (let R := circumradius A B C in
    let r := inradius A B C in
    R ≥ 2 * r ∧ (R = 2 * r ↔ is_equilateral_triangle A B C)) :=
by sorry

end circumscribed_inscribed_inequality_l6_6808


namespace prod_eq_zero_count_l6_6521

theorem prod_eq_zero_count :
  (∑ n in finset.range 1000, if (∃ k, (2 * k + 1) * 3 = n + 1) then 1 else 0) = 167 := 
sorry

end prod_eq_zero_count_l6_6521


namespace atomic_weight_of_bromine_l6_6737

theorem atomic_weight_of_bromine (molecular_weight : ℝ) 
    (weight_N : ℝ) (weight_H : ℝ) : 
    molecular_weight = 98 →
    weight_N = 14.01 →
    weight_H = 1.008 →
    let total_weight_NH := (1 * weight_N) + (4 * weight_H)
    in molecular_weight = total_weight_NH + 79.958 :=
by
  intros
  sorry

end atomic_weight_of_bromine_l6_6737


namespace fixed_point_l6_6324

theorem fixed_point (a : ℝ) (h_pos : a > 0) (h_neq_one : a ≠ 1) : 
  (a ^ (-2 + 2) - 3 = -2) :=
by
  sorry

end fixed_point_l6_6324


namespace speed_of_man_l6_6850

open Real Int

/-- 
  A train 110 m long is running with a speed of 40 km/h.
  The train passes a man who is running at a certain speed
  in the direction opposite to that in which the train is going.
  The train takes 9 seconds to pass the man.
  This theorem proves that the speed of the man is 3.992 km/h.
-/
theorem speed_of_man (T_length : ℝ) (T_speed : ℝ) (t_pass : ℝ) (M_speed : ℝ) : 
  T_length = 110 → T_speed = 40 → t_pass = 9 → M_speed = 3.992 :=
by
  intro h1 h2 h3
  sorry

end speed_of_man_l6_6850


namespace only_solutions_mod_n_l6_6074

theorem only_solutions_mod_n (n : ℕ) : (∀ k : ℤ, ∃ a : ℤ, (a^3 + a - k) % (n : ℤ) = 0) ↔ (∃ k : ℕ, n = 3 ^ k) := 
sorry

end only_solutions_mod_n_l6_6074


namespace base_b_representation_l6_6160

theorem base_b_representation (b : ℕ) : (2 * b + 9)^2 = 7 * b^2 + 3 * b + 4 → b = 14 := 
sorry

end base_b_representation_l6_6160


namespace decimal_to_fraction_l6_6023

theorem decimal_to_fraction (h : 0.02 = 2 / 99) : 2.06 = 68 / 33 :=
by sorry

end decimal_to_fraction_l6_6023


namespace g_9_eq_64_l6_6715

noncomputable def g : ℝ → ℝ := sorry

axiom g_add (x y : ℝ) : g(x + y) = g(x) * g(y)

axiom g_3_eq_4 : g(3) = 4

theorem g_9_eq_64 : g(9) = 64 := by
  sorry

end g_9_eq_64_l6_6715


namespace part1_i_part1_ii_part2_l6_6826

def f : ℕ → ℕ
| 1        := 1
| 2        := 2
| (n + 2) := f(n + 2 - f(n + 1)) + f(n + 1 - f(n))

theorem part1_i : ∀ n, 0 ≤ f(n + 1) - f(n) ∧ f(n + 1) - f(n) ≤ 1 := by
  sorry

theorem part1_ii : ∀ n, f(n) % 2 = 1 → f(n + 1) = f(n) + 1 := by
  sorry

theorem part2 : ∃ n, f(n) = 2^10 + 1 := by
  sorry

end part1_i_part1_ii_part2_l6_6826


namespace dan_initial_amount_l6_6481

variables (initial_amount spent_amount remaining_amount : ℝ)

theorem dan_initial_amount (h1 : spent_amount = 1) (h2 : remaining_amount = 2) : initial_amount = spent_amount + remaining_amount := by
  sorry

end dan_initial_amount_l6_6481


namespace range_of_a_for_local_maximum_l6_6126

noncomputable def f' (a x : ℝ) := a * (x + 1) * (x - a)

theorem range_of_a_for_local_maximum {a : ℝ} (hf_max : ∀ x : ℝ, f' a x = 0 → ∀ y : ℝ, y ≠ x → f' a y ≤ f' a x) :
  -1 < a ∧ a < 0 :=
sorry

end range_of_a_for_local_maximum_l6_6126


namespace polynomial_factorization_l6_6383

theorem polynomial_factorization : 
  (x : ℤ) → (x^15 + x^10 + x^5 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1)) := 
by
  sorry

end polynomial_factorization_l6_6383


namespace problem_l6_6129

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := a * x^2 - 4 * x + 2
noncomputable def g (x : ℝ) (f: ℝ -> ℝ) : ℝ := (1/3)^(f x)

theorem problem 
  (a : ℝ) 
  (h1 : ∃ x, f x a = f 2 a):
  (f 2 a = - 2 * a + 2) ->
  (f 2 1 = -2) ->
  (∀ x, (1/3)^(f x 1) ∈ set.Icc 0 9): 
  sorry

end problem_l6_6129


namespace part1_a_n_part1_b_n_part2_T_n_l6_6943

noncomputable def a_n (n : ℕ) : ℤ := 2 * n - 1
noncomputable def b_n (n : ℕ) : ℤ := 2 ^ n
noncomputable def c_n (n : ℕ) : ℚ := (2 * n - 1) / (2 ^ n : ℚ)
noncomputable def T_n (n : ℕ) : ℚ := 3 - (2 * n + 3) / (2 ^ n : ℚ)

-- Given conditions
axiom a2_a3_a4 : a_n 2 + a_n 3 + a_n 4 = 15
axiom a4_a6 : a_n 4 + a_n 6 = 18
axiom Sn_definition (n : ℕ) : 2 * b_n n - 2 = (list.sum (list.range (n+1)).map b_n : ℤ)

-- Prove that
theorem part1_a_n : ∀ n : ℕ, a_n n = 2 n - 1 := sorry
theorem part1_b_n : ∀ n : ℕ, b_n n = 2 ^ n := sorry
theorem part2_T_n : ∀ n : ℕ, T_n n = 3 - (2 * n + 3) / (2 ^ n : ℚ) := sorry

end part1_a_n_part1_b_n_part2_T_n_l6_6943


namespace skateboard_weight_is_18_l6_6692

def weight_of_canoe : Nat := 45
def weight_of_four_canoes := 4 * weight_of_canoe
def weight_of_ten_skateboards := weight_of_four_canoes
def weight_of_one_skateboard := weight_of_ten_skateboards / 10

theorem skateboard_weight_is_18 : weight_of_one_skateboard = 18 := by
  sorry

end skateboard_weight_is_18_l6_6692


namespace range_of_a_min_value_of_a_l6_6926

variable (f : ℝ → ℝ) (a x : ℝ)

-- Part 1
theorem range_of_a (f_def : ∀ x, f x = abs (x - a)) 
  (h₁ : ∀ x, 1 ≤ x ∧ x ≤ 3 → f x ≤ 3) : 0 ≤ a ∧ a ≤ 4 :=
sorry

-- Part 2
theorem min_value_of_a (f_def : ∀ x, f x = abs (x - a)) 
  (h₂ : ∀ x, f (x - a) + f (x + a) ≥ 1 - a) : a ≥ 1/3 :=
sorry

end range_of_a_min_value_of_a_l6_6926


namespace range_of_m_l6_6951

theorem range_of_m (m : ℝ) (f g : ℝ → ℝ) :
  (∀ x1 ∈ set.Icc (-1:ℝ) 2, ∃ x2 ∈ set.Icc (0:ℝ) 3, f x1 = g x2) →
  (∀ x, f x = x^2 + m) →
  (∀ x, g x = 2^x - m) →
  m ∈ set.Icc (1/2:ℝ) 2 :=
by
  intro h1 h2 h3
  sorry

end range_of_m_l6_6951


namespace find_g_9_l6_6711

noncomputable def g : ℝ → ℝ := sorry

axiom functional_equation : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g_of_3 : g 3 = 4

theorem find_g_9 : g 9 = 64 := by
  sorry

end find_g_9_l6_6711


namespace f_of_g_of_neg3_l6_6598

def f (x : ℝ) : ℝ := 4 - Real.sqrt x
def g (x : ℝ) : ℝ := 3 * x + 3 * x^2

theorem f_of_g_of_neg3 : f (g (-3)) = 4 - 3 * Real.sqrt 2 :=
by
  sorry

end f_of_g_of_neg3_l6_6598


namespace max_time_digit_sum_l6_6426

-- Define the conditions
def is_valid_time (h m : ℕ) : Prop :=
  (0 ≤ h ∧ h < 24) ∧ (0 ≤ m ∧ m < 60)

-- Define the function to calculate the sum of the digits of a number
def digit_sum (n : ℕ) : ℕ :=
  n % 10 + n / 10

-- Define the function to calculate the sum of digits in the time display
def time_digit_sum (h m : ℕ) : ℕ :=
  digit_sum h + digit_sum m

-- The theorem to prove
theorem max_time_digit_sum : ∀ (h m : ℕ),
  is_valid_time h m → time_digit_sum h m ≤ 24 :=
by {
  sorry
}

end max_time_digit_sum_l6_6426


namespace exists_point_on_line_equidistant_l6_6779

variable {x1 y1 x2 y2 m c: ℝ}

-- Define the points A and B
def A : ℝ × ℝ := (x1, y1)
def B : ℝ × ℝ := (x2, y2)

-- Define the given line
def line (x : ℝ) : ℝ := m * x + c

-- Statement of the problem
theorem exists_point_on_line_equidistant (h_neq: x1 ≠ x2 ∨ y1 ≠ y2): ∃ x y: ℝ, (y = line x) ∧ (dist (x, y) A = dist (x, y) B) :=
by {
  sorry
}

end exists_point_on_line_equidistant_l6_6779


namespace number_of_scoops_l6_6663

/-- Pierre gets 3 scoops of ice cream given the conditions described -/
theorem number_of_scoops (P : ℕ) (cost_per_scoop total_bill : ℝ) (mom_scoops : ℕ)
  (h1 : cost_per_scoop = 2) 
  (h2 : mom_scoops = 4) 
  (h3 : total_bill = 14) 
  (h4 : cost_per_scoop * P + cost_per_scoop * mom_scoops = total_bill) :
  P = 3 :=
by
  sorry

end number_of_scoops_l6_6663


namespace three_boys_in_shop_at_same_time_l6_6810

-- Definitions for the problem conditions
def boys : Type := Fin 7  -- Representing the 7 boys
def visits : Type := Fin 3  -- Each boy makes 3 visits

-- A structure representing a visit by a boy
structure Visit := (boy : boys) (visit_num : visits)

-- Meeting condition: Every pair of boys meets at the shop
def meets_at_shop (v1 v2 : Visit) : Prop :=
  v1.boy ≠ v2.boy  -- Ensure it's not the same boy (since we assume each pair meets)

-- The theorem to be proven
theorem three_boys_in_shop_at_same_time :
  ∃ (v1 v2 v3 : Visit), v1.boy ≠ v2.boy ∧ v2.boy ≠ v3.boy ∧ v1.boy ≠ v3.boy :=
sorry

end three_boys_in_shop_at_same_time_l6_6810


namespace solvable_eq_l6_6501

theorem solvable_eq (x : ℝ) :
    Real.sqrt ((3 + Real.sqrt 5) ^ x) + Real.sqrt ((3 - Real.sqrt 5) ^ x) = 6 →
    (x = 2 ∨ x = -2) :=
by
  sorry

end solvable_eq_l6_6501


namespace perpendicular_bisector_lengths_l6_6267

variables {a b c t_a t_b t_c : ℝ}

-- Conditions
axiom triangle_sides (hab : a > b) (hbc : b > c) (hc0 : c > 0) (hta : t_a > 0) (htb : t_b > 0) (htc : t_c > 0)

-- Assuming the segments of the perpendicular bisectors fall within the triangle sides
axiom length_ta_tb_tc (hta : t_a > t_b) (htc : t_c > t_b)

-- Proof Problem
theorem perpendicular_bisector_lengths (hab : a > b) (hbc : b > c) (hc0 : c > 0) : t_a > t_b ∧ t_c > t_b ∧ (∃ (a' : ℝ), a' > b ∧ t_a = t_c) :=
sorry

end perpendicular_bisector_lengths_l6_6267


namespace problem_statement_l6_6473

-- Definitions based on conditions
def cos_45_deg : ℝ := real.cos (real.pi / 4)
def negative_half_inv_sq : ℝ := ((-1/2) : ℝ) ^ (-2)
def sqrt_8 : ℝ := real.sqrt 8
def negative_one_pow_2023 : ℤ := (-1) ^ 2023
def two_thousand_twenty_three_minus_pi_pow_0 : ℝ := (2023 - real.pi) ^ 0

-- Prove the original expression simplifies correctly
theorem problem_statement :
  4 * cos_45_deg +
  negative_half_inv_sq -
  sqrt_8 +
  negative_one_pow_2023 +
  two_thousand_twenty_three_minus_pi_pow_0 = 4 := by
  -- Conditions from the problem
  have h1 : cos_45_deg = real.sqrt 2 / 2 := by sorry
  have h2 : negative_half_inv_sq = 4 := by sorry
  have h3 : sqrt_8 = 2 * real.sqrt 2 := by sorry
  have h4 : negative_one_pow_2023 = -1 := by sorry
  have h5 : two_thousand_twenty_three_minus_pi_pow_0 = 1 := by sorry
  -- Combine the conditions to reach the conclusion
  rw [h1, h2, h3, h4, h5],
  sorry

end problem_statement_l6_6473


namespace recurring_decimal_to_fraction_l6_6024

theorem recurring_decimal_to_fraction
  (h : 0.\overline{02} = (2 : ℝ) / 99) :
  2.\overline{06} = 68 / 33 := by
  sorry

end recurring_decimal_to_fraction_l6_6024


namespace proposition_1_proposition_2_proposition_3_proposition_4_l6_6128

noncomputable def f (x : ℝ) : ℝ := cos (2 * x - π / 3) + cos (2 * x + π / 6)

theorem proposition_1 : ∃ x : ℝ, f x = sqrt 2 := sorry

theorem proposition_2 : ∀ (x : ℝ), f (x + π) = f x := sorry

theorem proposition_3 : ∀ (x : ℝ), x ∈ Ioo (π / 24) (13 * π / 24) → (f' x < 0) := sorry

theorem proposition_4 : ∀ (x : ℝ), f (x - π / 24) = sqrt 2 * cos (2 * x) := sorry

end proposition_1_proposition_2_proposition_3_proposition_4_l6_6128


namespace problem_function_nonnegative_iff_l6_6914

theorem problem_function_nonnegative_iff (a : ℝ) :
  (∀ x : ℝ, -1 ≤ x ∧ x ≤ 1 → a * x^3 - 3 * x + 1 ≥ 0) ↔ a = 4 :=
by
  split
  sorry -- Proof that the function is non-negative for all x implies a = 4.
  sorry -- Proof that a = 4 implies the function is non-negative for all x.

end problem_function_nonnegative_iff_l6_6914


namespace correct_classification_of_random_events_l6_6491

def is_non_random_event (P : Prop) : Prop := ∀ω, P
def is_random_event (P : Prop) : Prop := ∃ω, P ω
def is_impossible_event (P : Prop) : Prop := ∀ω, ¬P ω

variables (E1 E2 E3 E4 : Prop)

-- condition statements
axiom E1_non_random : is_non_random_event E1
axiom E2_random : is_random_event E2
axiom E3_impossible : is_impossible_event E3
axiom E4_random : is_random_event E4

-- proof problem statement
theorem correct_classification_of_random_events : 
  (E2 ∧ E4) = true :=
by
  exact true.intro

end correct_classification_of_random_events_l6_6491


namespace cone_height_l6_6123

theorem cone_height (r : ℝ) (θ : ℝ) (h : ℝ)
  (hr : r = 1)
  (hθ : θ = (2 / 3) * Real.pi)
  (h_eq : h = 2 * Real.sqrt 2) :
  ∃ l : ℝ, l = 3 ∧ h = Real.sqrt (l^2 - r^2) :=
by
  sorry

end cone_height_l6_6123


namespace problem1_problem2_l6_6306

-- Problem 1: Simplify (x + 2) * (-3x + 4)
theorem problem1 (x : ℝ) : (x + 2) * (-3 * x + 4) = -3 * x^2 - 2 * x + 8 := 
sorry

-- Problem 2: Simplify 1/5 * (1/5)^0 ÷ (-1/5)^-2
theorem problem2 : (1/5) * ((1/5)^0) / ((-1/5)^-2) = 1/125 := 
sorry

end problem1_problem2_l6_6306


namespace trajectory_equation_l6_6949

noncomputable def circle1_center := (-3, 0)
noncomputable def circle2_center := (3, 0)

def circle1 (x y : ℝ) : Prop := (x + 3)^2 + y^2 = 1
def circle2 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 81

def is_tangent_internally (x y : ℝ) : Prop := 
  ∃ (P : ℝ × ℝ), circle1 P.1 P.2 ∧ circle2 P.1 P.2

theorem trajectory_equation :
  ∀ (x y : ℝ), is_tangent_internally x y → (x^2 / 16 + y^2 / 7 = 1) :=
sorry

end trajectory_equation_l6_6949


namespace regular_tetrahedron_has_five_spheres_tetrahedron_with_five_spheres_is_regular_l6_6292

structure Tetrahedron where
  edges_equal : ∀ (e1 e2 : ℝ), e1 = e2

structure Sphere where
  tangent_to_faces : ∀ (tetra : Tetrahedron) (face1 face2 face3 face4 : ℕ), 
    Bool

theorem regular_tetrahedron_has_five_spheres (tetra : Tetrahedron) :
  ∃ (S : Sphere) (S1 S2 S3 S4 : Sphere), 
    S.tangent_to_faces tetra 0 1 2 3 ∧
    S1.tangent_to_faces tetra 1 2 3 0 ∧
    S2.tangent_to_faces tetra 2 3 0 1 ∧
    S3.tangent_to_faces tetra 3 0 1 2 ∧
    S4.tangent_to_faces tetra 0 1 2 3 := 
  sorry

theorem tetrahedron_with_five_spheres_is_regular (S S1 S2 S3 S4 : Sphere) :
  (∀ (tetra : Tetrahedron), 
     S.tangent_to_faces tetra 0 1 2 3 ∧
     S1.tangent_to_faces tetra 1 2 3 0 ∧
     S2.tangent_to_faces tetra 2 3 0 1 ∧
     S3.tangent_to_faces tetra 3 0 1 2 ∧
     S4.tangent_to_faces tetra 0 1 2 3) → 
  ∃ (tetra : Tetrahedron), tetra.edges_equal := 
  sorry

end regular_tetrahedron_has_five_spheres_tetrahedron_with_five_spheres_is_regular_l6_6292


namespace smallest_positive_angle_l6_6122

-- Define the conditions
def point : ℝ × ℝ := (sqrt 3 / 2, -1 / 2)

-- Define the question to prove
theorem smallest_positive_angle (α : ℝ) : 
  point = (sqrt 3 / 2, -1 / 2) → 
  α = 11 * π / 6  :=
sorry

end smallest_positive_angle_l6_6122


namespace moles_of_beryllium_hydroxide_formed_l6_6898

def balanced_equation : Prop := 
  ∀ (Be2C H2O BeOH2 CH4 : Type) 
    (n_Be2C n_H2O n_BeOH2 n_CH4 : ℝ), 
    (n_Be2C = 1) ∧ (n_H2O = 4) → 
    (2 * n_BeOH2 = 2) ∧ (1 * n_CH4 = 1)

theorem moles_of_beryllium_hydroxide_formed (Be2C H2O BeOH2 CH4 : Type) 
  (n_Be2C n_H2O n_BeOH2 n_CH4 : ℝ) 
  (h : balanced_equation Be2C H2O BeOH2 CH4) : 
  (n_Be2C = 1) ∧ (n_H2O = 4) → (n_BeOH2 = 2) :=
by 
  intro hp 
  have hp1 := hp.1 
  have hp2 := hp.2 
  have balanced := h Be2C H2O BeOH2 CH4 n_Be2C n_H2O n_BeOH2 n_CH4 
  exact balanced hp1 hp2 sorry

end moles_of_beryllium_hydroxide_formed_l6_6898


namespace concurrency_of_lines_l6_6773

-- Geometric entities
variable {Γ1 Γ2 : Circle}
variable {S O : Point}
variable {A B T : Point}

-- Tangency and configuration conditions
axiom tangent_at_S : Tangent Γ1 Γ2 S
axiom Γ1_inside_Γ2 : Inside Γ1 Γ2
axiom O_center_Γ1 : Center Γ1 O
axiom AB_chord_Γ2 : Chord Γ2 A B
axiom T_tangent_Γ1_AB : Tangent Γ1 (Line A B) T

-- Statement to prove concurrency
theorem concurrency_of_lines :
  Concurrent (Line A O) (Perpendicular (Line A B) B) (Perpendicular (Line S T) S) :=
sorry

end concurrency_of_lines_l6_6773


namespace periodic_decimal_to_fraction_l6_6013

theorem periodic_decimal_to_fraction
  (h : ∀ n : ℕ, 0.<digit>02 n / 99) :
  2.0<digit>06 = 68 / 33 :=
sorry

end periodic_decimal_to_fraction_l6_6013


namespace interest_groups_ranges_l6_6345

variable (A B C : Finset ℕ)

-- Given conditions
axiom card_A : A.card = 5
axiom card_B : B.card = 4
axiom card_C : C.card = 7
axiom card_A_inter_B : (A ∩ B).card = 3
axiom card_A_inter_B_inter_C : (A ∩ B ∩ C).card = 2

-- Mathematical statement to be proved
theorem interest_groups_ranges :
  2 ≤ ((A ∪ B) ∩ C).card ∧ ((A ∪ B) ∩ C).card ≤ 5 ∧
  8 ≤ (A ∪ B ∪ C).card ∧ (A ∪ B ∪ C).card ≤ 11 := by
  sorry

end interest_groups_ranges_l6_6345


namespace angle_AOB_in_tangent_triangle_l6_6764

-- Statement of the problem
theorem angle_AOB_in_tangent_triangle (P A B O: Type) 
  [Is_Triangle PAB] [TangentToCircle P A B O] 
  (h_angle_APB : ∠ APB = 50°) :
  ∠ AOB = 130° :=
sorry

end angle_AOB_in_tangent_triangle_l6_6764


namespace min_ones_in_11x11_l6_6226

open Matrix

def is_odd (n : ℕ) : Prop := n % 2 = 1

def four_cell_sum (m : Matrix (Fin 11) (Fin 11) ℕ) (i j : Fin 10) : ℕ :=
  m i j + m (i+1) j + m i (j+1) + m (i+1) (j+1)

theorem min_ones_in_11x11 (m : Matrix (Fin 11) (Fin 11) ℕ) (h : ∀ i j : Fin 10, is_odd (four_cell_sum m i j)) :
  ∑ i : Fin 11, ∑ j : Fin 11, m i j ≥ 25 :=
begin
  sorry
end

end min_ones_in_11x11_l6_6226


namespace problem_statement_l6_6920

structure Vector2D where
  x : Float
  y : Float

noncomputable def f (x : Float) : Float := 1 + 2 * Math.sin (2 * x + Float.pi / 6)

theorem problem_statement (x y A b c : Float) (m n : Vector2D)
  (h_m : m = ⟨2 * Math.cos x + 2 * Float.sqrt 3 * Math.sin x, 1⟩)
  (h_n : n = ⟨Math.cos x, -y⟩)
  (h_perp : m.x * n.x + m.y * n.y = 0)
  (h_fA : f (A / 2) = 3)
  (h_a : 2 = 2)
  (h_bc_sum : b + c = 4) :
  y = f x ∧ (A = Float.pi / 3) ∧ (b * c = 4) ∧ (1 / 2 * b * c * Math.sin A = Float.sqrt 3) := by
  sorry

end problem_statement_l6_6920


namespace polynomial_prime_is_11_l6_6149

def P (a : ℕ) : ℕ := a^4 - 4 * a^3 + 15 * a^2 - 30 * a + 27

theorem polynomial_prime_is_11 (a : ℕ) (hp : Nat.Prime (P a)) : P a = 11 := 
by {
  sorry
}

end polynomial_prime_is_11_l6_6149


namespace find_value_l6_6555

-- Define the theorem with the given conditions and the expected result
theorem find_value (a b c : ℕ) (h_pos : a > 0 ∧ b > 0 ∧ c > 0) (h_eq : a^2 * b + b^2 * c + a * c^2 + a + b + c = 2 * (a * b + b * c + a * c)) :
  c^2017 / (a^2016 + b^2018) = 1 / 2 :=
sorry

end find_value_l6_6555


namespace sticks_can_form_13_triangles_l6_6928

-- Define the problem conditions and conclusions.
theorem sticks_can_form_13_triangles :
  ∃ (a : ℝ) (sticks : list ℝ), 
    (length sticks = 12) ∧ 
    (∀ (x ∈ sticks), x = 13 * a) ∧ 
    ∃ (pieces : list ℝ), 
      (length pieces = 39 + 52 + 65) ∧ 
      ((∀ (y ∈ pieces, y = 3 * a) ↔ (length (filter (λ b, b = 3 * a) pieces)) = 39) ∧
       (∀ (z ∈ pieces, z = 4 * a) ↔ (length (filter (λ c, c = 4 * a) pieces)) = 52) ∧
       (∀ (w ∈ pieces, w = 5 * a) ↔ (length (filter (λ d, d = 5 * a) pieces)) = 65)) :=
begin
  sorry
end

end sticks_can_form_13_triangles_l6_6928


namespace max_points_l6_6220

noncomputable def points (K C Y : ℕ) : ℕ :=
  K + 2 * K * C + 3 * C * Y

theorem max_points :
  ∃ K C Y : ℕ, K + C + Y = 15 ∧ points K C Y = 168 :=
begin
  use [0, 7, 8],
  split,
  {
    -- K + C + Y = 15
    calc
      0 + 7 + 8 = 15 : by simp
  },
  {
    -- points K C Y = 168
    unfold points,
    calc
      0 + 2 * 0 * 7 + 3 * 7 * 8 = 168 : by norm_num
  }
end

end max_points_l6_6220


namespace expected_value_sum_of_marbles_l6_6147

theorem expected_value_sum_of_marbles : 
  let marbles := {1, 2, 3, 4, 5, 6, 7} in
  let combinations := (marbles : Finset ℕ).powerset.filter (λ s, s.card = 3) in
  let sum_combinations := combinations.sum (λ s, s.sum id) in
  let num_combinations := combinations.card in
  (sum_combinations : ℚ) / num_combinations = 12 := 
by
  sorry

end expected_value_sum_of_marbles_l6_6147


namespace g_at_9_l6_6720

noncomputable def g : ℝ → ℝ := sorry

axiom functional_eq (x y : ℝ) : g (x + y) = g x * g y
axiom g_at_3 : g 3 = 4

theorem g_at_9 : g 9 = 64 :=
by
  sorry

end g_at_9_l6_6720


namespace range_distance_from_origin_l6_6291

theorem range_distance_from_origin (x y : ℝ) (h₁ : 4 * x + 3 * y = 0) (h₂ : -14 ≤ x - y ∧ x - y ≤ 7) :
    ∃ d : ℝ, ∀ (P : ℝ × ℝ), P = (x, y) → 0 ≤ d ∧ d ≤ 10 ∧ d = real.sqrt (x^2 + y^2) :=
by 
  use (real.sqrt (x^2 + y^2))
  sorry

end range_distance_from_origin_l6_6291
