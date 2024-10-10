import Mathlib

namespace fraction_division_and_addition_l3779_377996

theorem fraction_division_and_addition : (3 / 7 : ℚ) / 4 + 1 / 28 = 1 / 7 := by
  sorry

end fraction_division_and_addition_l3779_377996


namespace f_increasing_implies_a_geq_two_l3779_377932

/-- The function f(x) = x^2 - 4x + 3 -/
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

/-- The theorem stating that if f(x+a) is increasing on [0, +∞), then a ≥ 2 -/
theorem f_increasing_implies_a_geq_two (a : ℝ) :
  (∀ x y : ℝ, 0 ≤ x ∧ x < y → f (x + a) < f (y + a)) →
  a ∈ Set.Ici 2 :=
sorry

end f_increasing_implies_a_geq_two_l3779_377932


namespace polar_equation_pi_over_four_is_line_l3779_377985

/-- The set of points (x, y) satisfying the polar equation θ = π/4 forms a line in the Cartesian plane. -/
theorem polar_equation_pi_over_four_is_line :
  ∀ (x y : ℝ), (∃ (r : ℝ), x = r * Real.cos (π / 4) ∧ y = r * Real.sin (π / 4)) ↔
  ∃ (m b : ℝ), y = m * x + b ∧ m = 1 := by
  sorry

end polar_equation_pi_over_four_is_line_l3779_377985


namespace sum_of_solutions_equals_sixteen_l3779_377908

theorem sum_of_solutions_equals_sixteen :
  let f : ℝ → ℝ := λ x => Real.sqrt x + Real.sqrt (9 / x) + Real.sqrt (x + 9 / x)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 9 ∧ f x₂ = 9 ∧ x₁ + x₂ = 16 ∧
  ∀ (x : ℝ), f x = 9 → x = x₁ ∨ x = x₂ :=
by sorry


end sum_of_solutions_equals_sixteen_l3779_377908


namespace line_segment_product_l3779_377973

/-- Given four points A, B, C, D on a line in this order, prove that AB · CD + AD · BC = 1000 -/
theorem line_segment_product (A B C D : ℝ) : 
  (A < B) → (B < C) → (C < D) →  -- Points are in order on the line
  (C - A = 25) →  -- AC = 25
  (D - B = 40) →  -- BD = 40
  (D - A = 57) →  -- AD = 57
  (B - A) * (D - C) + (D - A) * (C - B) = 1000 := by
  sorry

end line_segment_product_l3779_377973


namespace cookies_in_blue_tin_l3779_377906

/-- Proves that the fraction of cookies in the blue tin is 8/27 -/
theorem cookies_in_blue_tin
  (total_cookies : ℚ)
  (blue_green_fraction : ℚ)
  (red_fraction : ℚ)
  (green_fraction_of_blue_green : ℚ)
  (h1 : blue_green_fraction = 2 / 3)
  (h2 : red_fraction = 1 - blue_green_fraction)
  (h3 : green_fraction_of_blue_green = 5 / 9)
  : (blue_green_fraction * (1 - green_fraction_of_blue_green)) = 8 / 27 := by
  sorry


end cookies_in_blue_tin_l3779_377906


namespace coefficient_of_x3y2z5_in_expansion_l3779_377939

/-- The coefficient of x^3y^2z^5 in the expansion of (2x+y+z)^10 -/
def coefficient : ℕ := 20160

/-- The exponent of the trinomial expression -/
def exponent : ℕ := 10

/-- Theorem stating that the coefficient of x^3y^2z^5 in (2x+y+z)^10 is 20160 -/
theorem coefficient_of_x3y2z5_in_expansion : 
  coefficient = (2^3 : ℕ) * Nat.choose exponent 3 * Nat.choose (exponent - 3) 2 * Nat.choose ((exponent - 3) - 2) 5 := by
  sorry

end coefficient_of_x3y2z5_in_expansion_l3779_377939


namespace parrots_per_cage_l3779_377969

theorem parrots_per_cage (num_cages : ℕ) (total_birds : ℕ) : 
  num_cages = 9 →
  total_birds = 36 →
  (∃ (parrots_per_cage : ℕ), 
    parrots_per_cage * num_cages * 2 = total_birds ∧ 
    parrots_per_cage = 2) :=
by
  sorry

end parrots_per_cage_l3779_377969


namespace sum_of_i_powers_l3779_377921

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem sum_of_i_powers : i^11 + i^16 + i^21 + i^26 + i^31 + i^36 = 1 := by
  sorry

end sum_of_i_powers_l3779_377921


namespace combination_equation_solution_l3779_377988

def binomial (n k : ℕ) : ℕ := sorry

theorem combination_equation_solution :
  ∀ x : ℕ, (binomial 28 x = binomial 28 (3*x - 8)) ↔ (x = 4 ∨ x = 9) :=
sorry

end combination_equation_solution_l3779_377988


namespace ratio_proportion_problem_l3779_377954

theorem ratio_proportion_problem (x : ℝ) :
  (2975.75 / 7873.125 = 12594.5 / x) → x = 33333.75 := by
  sorry

end ratio_proportion_problem_l3779_377954


namespace seating_arrangements_l3779_377938

/-- The number of ways to arrange n people in a row -/
def totalArrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n people in a row where a group of k people must sit together -/
def groupedArrangements (n k : ℕ) : ℕ := 
  Nat.factorial (n - k + 1) * Nat.factorial k

/-- The number of people to be seated -/
def totalPeople : ℕ := 10

/-- The number of people in the group that can't sit together -/
def groupSize : ℕ := 4

theorem seating_arrangements :
  totalArrangements totalPeople - groupedArrangements totalPeople groupSize = 3507840 := by
  sorry

end seating_arrangements_l3779_377938


namespace square_difference_and_product_l3779_377918

theorem square_difference_and_product (x y : ℝ) 
  (h1 : (x + y)^2 = 81)
  (h2 : x * y = 15) : 
  (x - y)^2 = 21 ∧ (x + y) * (x - y) = Real.sqrt 1701 := by
  sorry

end square_difference_and_product_l3779_377918


namespace intersection_of_M_and_N_l3779_377958

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by
  sorry

end intersection_of_M_and_N_l3779_377958


namespace regular_dinosaur_weight_is_800_l3779_377990

/-- The weight of a regular dinosaur in pounds -/
def regular_dinosaur_weight : ℝ := sorry

/-- The weight of Barney the dinosaur in pounds -/
def barney_weight : ℝ := 5 * regular_dinosaur_weight + 1500

/-- The total weight of Barney and five regular dinosaurs in pounds -/
def total_weight : ℝ := 9500

/-- Theorem stating that each regular dinosaur weighs 800 pounds -/
theorem regular_dinosaur_weight_is_800 : regular_dinosaur_weight = 800 :=
by
  sorry

end regular_dinosaur_weight_is_800_l3779_377990


namespace worker_completion_time_l3779_377922

/-- Given two workers A and B, proves that A can complete a job in 14 days 
    when A and B together can complete the job in 10 days, 
    and B alone can complete the job in 35 days. -/
theorem worker_completion_time 
  (joint_completion_time : ℝ) 
  (b_alone_completion_time : ℝ) 
  (h1 : joint_completion_time = 10) 
  (h2 : b_alone_completion_time = 35) : 
  ∃ (a_alone_completion_time : ℝ), 
    a_alone_completion_time = 14 ∧ 
    (1 / a_alone_completion_time + 1 / b_alone_completion_time = 1 / joint_completion_time) :=
by sorry

end worker_completion_time_l3779_377922


namespace arithmetic_sequence_a2_l3779_377971

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_a2 (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 3 + a 5 = 24 →
  a 7 - a 3 = 24 →
  a 2 = 0 := by
sorry

end arithmetic_sequence_a2_l3779_377971


namespace lcm_from_product_and_gcd_l3779_377928

theorem lcm_from_product_and_gcd (a b : ℕ+) 
  (h_product : a * b = 17820)
  (h_gcd : Nat.gcd a b = 12) :
  Nat.lcm a b = 1485 := by
  sorry

end lcm_from_product_and_gcd_l3779_377928


namespace percentage_commutativity_l3779_377924

theorem percentage_commutativity (x : ℝ) (h : (30 / 100) * (40 / 100) * x = 60) :
  (40 / 100) * (30 / 100) * x = 60 := by
  sorry

end percentage_commutativity_l3779_377924


namespace negation_of_universal_statement_l3779_377975

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, 2^x + x^2 > 0) ↔ (∃ x : ℝ, 2^x + x^2 ≤ 0) := by
  sorry

end negation_of_universal_statement_l3779_377975


namespace optimal_ships_l3779_377948

/-- Revenue function -/
def R (x : ℕ) : ℚ := 3700 * x + 45 * x^2 - 10 * x^3

/-- Cost function -/
def C (x : ℕ) : ℚ := 460 * x + 5000

/-- Profit function -/
def P (x : ℕ) : ℚ := R x - C x

/-- The maximum number of ships that can be built annually -/
def max_capacity : ℕ := 20

/-- Theorem: The number of ships that maximizes annual profit is 12 -/
theorem optimal_ships :
  ∃ (x : ℕ), x ≤ max_capacity ∧ x > 0 ∧
  ∀ (y : ℕ), y ≤ max_capacity ∧ y > 0 → P x ≥ P y ∧
  x = 12 :=
sorry

end optimal_ships_l3779_377948


namespace consecutive_biology_majors_probability_l3779_377942

/-- The number of people sitting at the round table -/
def total_people : ℕ := 10

/-- The number of biology majors -/
def biology_majors : ℕ := 4

/-- The number of math majors -/
def math_majors : ℕ := 4

/-- The number of physics majors -/
def physics_majors : ℕ := 2

/-- The probability of all biology majors sitting in consecutive seats -/
def consecutive_biology_prob : ℚ := 2/3

theorem consecutive_biology_majors_probability :
  let total_arrangements := Nat.factorial (total_people - 1)
  let favorable_arrangements := (total_people - biology_majors) * Nat.factorial (total_people - biology_majors - 1)
  (favorable_arrangements : ℚ) / total_arrangements = consecutive_biology_prob :=
sorry

end consecutive_biology_majors_probability_l3779_377942


namespace arithmetic_geometric_sequence_l3779_377961

theorem arithmetic_geometric_sequence (a : ℕ → ℝ) (d : ℝ) :
  a 1 = 2 ∧ 
  d ≠ 0 ∧ 
  (∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  (∃ r : ℝ, r ≠ 0 ∧ a 3 = r * a 1 ∧ a 11 = r * a 3) →
  (∃ r : ℝ, r = 4 ∧ a 3 = r * a 1 ∧ a 11 = r * a 3) :=
by
  sorry

end arithmetic_geometric_sequence_l3779_377961


namespace expression_equals_one_l3779_377957

theorem expression_equals_one :
  (π + 2023) ^ 0 + 2 * Real.sin (π / 4) - (1 / 2)⁻¹ + |Real.sqrt 2 - 2| = 1 := by
  sorry

end expression_equals_one_l3779_377957


namespace triangular_pyramid_surface_area_l3779_377914

/-- A triangular pyramid with given base and side areas -/
structure TriangularPyramid where
  base_area : ℝ
  side_area : ℝ

/-- The surface area of a triangular pyramid -/
def surface_area (tp : TriangularPyramid) : ℝ :=
  tp.base_area + 3 * tp.side_area

/-- Theorem: The surface area of a triangular pyramid with base area 3 and side area 6 is 21 -/
theorem triangular_pyramid_surface_area :
  ∃ (tp : TriangularPyramid), tp.base_area = 3 ∧ tp.side_area = 6 ∧ surface_area tp = 21 := by
  sorry

end triangular_pyramid_surface_area_l3779_377914


namespace square_root_problem_l3779_377978

theorem square_root_problem (x y : ℝ) 
  (h1 : (5 * x - 1).sqrt = 3)
  (h2 : (4 * x + 2 * y + 1)^(1/3) = 1) :
  (4 * x - 2 * y).sqrt = 4 ∨ (4 * x - 2 * y).sqrt = -4 := by
sorry

end square_root_problem_l3779_377978


namespace h_function_iff_strictly_increasing_l3779_377943

/-- A function is an "H function" if for any two distinct real numbers x₁ and x₂,
    it satisfies x₁f(x₁) + x₂f(x₂) > x₁f(x₂) + x₂f(x₁) -/
def is_h_function (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁

/-- A function is strictly increasing if for any two real numbers x₁ < x₂,
    we have f(x₁) < f(x₂) -/
def strictly_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem h_function_iff_strictly_increasing (f : ℝ → ℝ) :
  is_h_function f ↔ strictly_increasing f :=
sorry

end h_function_iff_strictly_increasing_l3779_377943


namespace line_intersection_intersection_point_l3779_377979

/-- Two lines intersect at a unique point -/
theorem line_intersection (s t : ℝ) : ∃! (p : ℝ × ℝ), 
  (∃ s, p = (1 + 3*s, 2 - 7*s)) ∧ 
  (∃ t, p = (-5 + 5*t, 3 - 8*t)) :=
by sorry

/-- The intersection point of the two lines is (7, -12) -/
theorem intersection_point : 
  ∃ (s t : ℝ), (1 + 3*s, 2 - 7*s) = (-5 + 5*t, 3 - 8*t) ∧ 
                (1 + 3*s, 2 - 7*s) = (7, -12) :=
by sorry

end line_intersection_intersection_point_l3779_377979


namespace product_of_real_parts_complex_equation_l3779_377927

theorem product_of_real_parts_complex_equation : ∃ (z₁ z₂ : ℂ),
  (z₁^2 - 2*z₁ = Complex.I) ∧
  (z₂^2 - 2*z₂ = Complex.I) ∧
  (z₁ ≠ z₂) ∧
  (z₁.re * z₂.re = (1 - Real.sqrt 2) / 2) := by
sorry

end product_of_real_parts_complex_equation_l3779_377927


namespace five_people_booth_arrangements_l3779_377930

/-- The number of ways to arrange n people in a booth with at most k people on each side -/
def boothArrangements (n k : ℕ) : ℕ := sorry

/-- The number of ways to arrange 5 people in a booth with at most 3 people on each side -/
theorem five_people_booth_arrangements :
  boothArrangements 5 3 = 240 := by sorry

end five_people_booth_arrangements_l3779_377930


namespace certain_number_proof_l3779_377989

theorem certain_number_proof (x : ℝ) (h : 5 * x - 28 = 232) : x = 52 := by
  sorry

end certain_number_proof_l3779_377989


namespace homework_time_is_48_minutes_l3779_377934

def math_problems : ℕ := 15
def social_studies_problems : ℕ := 6
def science_problems : ℕ := 10

def math_time_per_problem : ℚ := 2
def social_studies_time_per_problem : ℚ := 1/2
def science_time_per_problem : ℚ := 3/2

def total_homework_time : ℚ :=
  math_problems * math_time_per_problem +
  social_studies_problems * social_studies_time_per_problem +
  science_problems * science_time_per_problem

theorem homework_time_is_48_minutes :
  total_homework_time = 48 := by sorry

end homework_time_is_48_minutes_l3779_377934


namespace complex_equation_solution_l3779_377949

theorem complex_equation_solution : ∃ z : ℂ, (z + 2) * (1 + Complex.I ^ 3) = 2 * Complex.I ∧ z = -1 + Complex.I := by
  sorry

end complex_equation_solution_l3779_377949


namespace no_valid_pairs_l3779_377940

theorem no_valid_pairs : ¬∃ (M N K : ℕ), 
  M > 0 ∧ N > 0 ∧ 
  (M : ℚ) / 5 = 5 / (N : ℚ) ∧ 
  M = 2 * K := by
  sorry

end no_valid_pairs_l3779_377940


namespace sum_Q_mod_500_l3779_377955

/-- The set of distinct remainders when 3^k is divided by 500, for 0 ≤ k < 200 -/
def Q : Finset ℕ :=
  (Finset.range 200).image (fun k => (3^k : ℕ) % 500)

/-- The sum of all elements in Q -/
def sum_Q : ℕ := Q.sum id

/-- The theorem to prove -/
theorem sum_Q_mod_500 :
  sum_Q % 500 = (Finset.range 200).sum (fun k => (3^k : ℕ) % 500) % 500 := by
  sorry

end sum_Q_mod_500_l3779_377955


namespace max_product_sum_2020_l3779_377912

theorem max_product_sum_2020 (n : ℕ) (as : List ℕ) :
  n ≥ 1 →
  as.length = n →
  as.sum = 2020 →
  as.prod ≤ 2^2 * 3^672 :=
by sorry

end max_product_sum_2020_l3779_377912


namespace problem_statement_l3779_377937

theorem problem_statement (x : ℝ) (h : x + 1/x = 2) : x^5 - 5*x^3 + 6*x = 2 := by
  sorry

end problem_statement_l3779_377937


namespace product_xyz_l3779_377904

theorem product_xyz (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_xy : x * y = 27 * Real.rpow 3 (1/3))
  (h_xz : x * z = 45 * Real.rpow 3 (1/3))
  (h_yz : y * z = 18 * Real.rpow 3 (1/3))
  (h_x_2y : x = 2 * y) : 
  x * y * z = 108 * Real.sqrt 3 := by
sorry

end product_xyz_l3779_377904


namespace complex_square_one_plus_i_l3779_377953

theorem complex_square_one_plus_i : 
  (Complex.I + 1) ^ 2 = 2 * Complex.I :=
by sorry

end complex_square_one_plus_i_l3779_377953


namespace population_scientific_notation_l3779_377947

def population : ℝ := 1370000000

theorem population_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), population = a * (10 : ℝ) ^ n ∧ 1 ≤ |a| ∧ |a| < 10 ∧ a = 1.37 ∧ n = 9 := by
  sorry

end population_scientific_notation_l3779_377947


namespace bounded_sequence_convergence_l3779_377936

def is_bounded (s : ℕ → ℝ) : Prop :=
  ∃ M : ℝ, ∀ n : ℕ, |s n| ≤ M

theorem bounded_sequence_convergence
  (a : ℕ → ℝ)
  (h_rec : ∀ n : ℕ, a (n + 1) = 3 * a n - 4)
  (h_bounded : is_bounded a) :
  ∀ n : ℕ, a n = 2 :=
sorry

end bounded_sequence_convergence_l3779_377936


namespace solve_for_y_l3779_377926

theorem solve_for_y (x y : ℝ) (h1 : 3 * x + 2 = 2) (h2 : y - x = 2) : y = 2 := by
  sorry

end solve_for_y_l3779_377926


namespace statement_equivalence_l3779_377982

theorem statement_equivalence (P Q : Prop) :
  (Q → ¬P) ↔ (P → ¬Q) := by sorry

end statement_equivalence_l3779_377982


namespace quadratic_roots_relation_l3779_377992

theorem quadratic_roots_relation (A B C : ℝ) (r s : ℝ) (p q : ℝ) :
  (A * r^2 + B * r + C = 0) →
  (A * s^2 + B * s + C = 0) →
  ((r + 3)^2 + p * (r + 3) + q = 0) →
  ((s + 3)^2 + p * (s + 3) + q = 0) →
  (A ≠ 0) →
  (p = B / A - 6) := by
sorry

end quadratic_roots_relation_l3779_377992


namespace min_value_theorem_l3779_377963

theorem min_value_theorem (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1 / (b * (a - b)) ≥ 4 ∧
  (a^2 + 1 / (b * (a - b)) = 4 ↔ a = Real.sqrt 2 ∧ b = Real.sqrt 2 / 2) :=
by sorry

end min_value_theorem_l3779_377963


namespace quadratic_equation_solution_l3779_377960

theorem quadratic_equation_solution : 
  ∃! x : ℝ, x^2 + 6*x + 8 = -(x + 4)*(x + 6) :=
by
  use -4
  sorry

end quadratic_equation_solution_l3779_377960


namespace nine_sided_polygon_diagonals_l3779_377910

/-- The number of diagonals in a regular polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n.choose 2) - n

/-- Theorem: A regular nine-sided polygon contains 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by
  sorry

end nine_sided_polygon_diagonals_l3779_377910


namespace sum_of_combinations_l3779_377945

theorem sum_of_combinations : (Nat.choose 4 4) + (Nat.choose 5 4) + (Nat.choose 6 4) + 
  (Nat.choose 7 4) + (Nat.choose 8 4) + (Nat.choose 9 4) + (Nat.choose 10 4) = 462 := by
  sorry

end sum_of_combinations_l3779_377945


namespace intersection_point_solution_l3779_377964

/-- Given two lines y = x + b and y = ax + 2 that intersect at point (3, -1),
    prove that the solution to (a - 1)x = b - 2 is x = 3. -/
theorem intersection_point_solution (a b : ℝ) :
  (3 + b = 3 * a + 2) →  -- Intersection point condition
  (-1 = 3 + b) →         -- y-coordinate of intersection point
  ((a - 1) * 3 = b - 2)  -- Solution x = 3 satisfies the equation
  := by sorry

end intersection_point_solution_l3779_377964


namespace vector_orthogonality_l3779_377986

-- Define the vectors
def a (x : ℝ) : Fin 2 → ℝ := ![x, 2]
def b : Fin 2 → ℝ := ![1, -1]

-- Define the orthogonality condition
def orthogonal (v w : Fin 2 → ℝ) : Prop :=
  (v 0) * (w 0) + (v 1) * (w 1) = 0

-- Theorem statement
theorem vector_orthogonality (x : ℝ) :
  orthogonal (λ i => a x i - b i) b → x = 4 := by
  sorry

end vector_orthogonality_l3779_377986


namespace vector_difference_magnitude_l3779_377923

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![2, 1]
def b : Fin 2 → ℝ := ![-2, 4]

-- State the theorem
theorem vector_difference_magnitude : ‖a - b‖ = 5 := by
  sorry

end vector_difference_magnitude_l3779_377923


namespace absolute_value_equation_solution_l3779_377941

theorem absolute_value_equation_solution :
  {x : ℝ | |2007*x - 2007| = 2007} = {0, 2} := by sorry

end absolute_value_equation_solution_l3779_377941


namespace cost_of_mangos_rice_flour_l3779_377994

/-- The cost of mangos, rice, and flour given certain price relationships -/
theorem cost_of_mangos_rice_flour (mango_cost rice_cost flour_cost : ℝ) 
  (h1 : 10 * mango_cost = 24 * rice_cost)
  (h2 : flour_cost = 2 * rice_cost)
  (h3 : flour_cost = 21) : 
  4 * mango_cost + 3 * rice_cost + 5 * flour_cost = 237.3 := by
  sorry

end cost_of_mangos_rice_flour_l3779_377994


namespace average_star_rating_l3779_377902

def total_reviews : ℕ := 18
def five_star_reviews : ℕ := 6
def four_star_reviews : ℕ := 7
def three_star_reviews : ℕ := 4
def two_star_reviews : ℕ := 1

def total_star_points : ℕ := 5 * five_star_reviews + 4 * four_star_reviews + 3 * three_star_reviews + 2 * two_star_reviews

theorem average_star_rating :
  (total_star_points : ℚ) / total_reviews = 4 := by sorry

end average_star_rating_l3779_377902


namespace min_treasures_is_15_l3779_377916

/-- Represents the number of palm trees with signs -/
def total_trees : ℕ := 30

/-- Represents the number of signs saying "Exactly under 15 signs a treasure is buried" -/
def signs_15 : ℕ := 15

/-- Represents the number of signs saying "Exactly under 8 signs a treasure is buried" -/
def signs_8 : ℕ := 8

/-- Represents the number of signs saying "Exactly under 4 signs a treasure is buried" -/
def signs_4 : ℕ := 4

/-- Represents the number of signs saying "Exactly under 3 signs a treasure is buried" -/
def signs_3 : ℕ := 3

/-- Predicate that checks if a given number of treasures satisfies all conditions -/
def satisfies_conditions (n : ℕ) : Prop :=
  n ≤ total_trees ∧
  (n ≠ 15 ∨ signs_15 = total_trees - n) ∧
  (n ≠ 8 ∨ signs_8 = total_trees - n) ∧
  (n ≠ 4 ∨ signs_4 = total_trees - n) ∧
  (n ≠ 3 ∨ signs_3 = total_trees - n)

/-- Theorem stating that the minimum number of signs under which treasures can be buried is 15 -/
theorem min_treasures_is_15 :
  ∃ (n : ℕ), n = 15 ∧ satisfies_conditions n ∧ ∀ (m : ℕ), m < n → ¬satisfies_conditions m :=
by sorry

end min_treasures_is_15_l3779_377916


namespace union_with_complement_l3779_377999

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2, 3}
def B : Set Nat := {2, 4}

theorem union_with_complement : A ∪ (U \ B) = {1, 2, 3, 5} := by
  sorry

end union_with_complement_l3779_377999


namespace complex_equation_sum_l3779_377980

theorem complex_equation_sum (z : ℂ) (a b : ℝ) : 
  z = a + b * I → z * (1 + I^3) = 2 + I → a + b = 2 := by
  sorry

end complex_equation_sum_l3779_377980


namespace function_zero_set_empty_l3779_377905

theorem function_zero_set_empty (f : ℝ → ℝ) (h : ∀ x : ℝ, f x + 3 * f (1 - x) = x^2) :
  {x : ℝ | f x = 0} = ∅ := by
  sorry

end function_zero_set_empty_l3779_377905


namespace seaweed_harvest_l3779_377925

theorem seaweed_harvest (total : ℝ) :
  (0.5 * total ≥ 0) →                    -- 50% used for starting fires
  (0.25 * (0.5 * total) ≥ 0) →           -- 25% of remaining for human consumption
  (0.75 * (0.5 * total) = 150) →         -- 75% of remaining (150 pounds) fed to livestock
  (total = 400) :=
by sorry

end seaweed_harvest_l3779_377925


namespace complement_intersection_A_B_l3779_377929

def A : Set ℝ := {x | |x - 1| ≤ 1}
def B : Set ℝ := {y | ∃ x, y = -x^2 ∧ -Real.sqrt 2 ≤ x ∧ x < 1}

theorem complement_intersection_A_B : 
  (A ∩ B)ᶜ = {x : ℝ | x ≠ 0} := by sorry

end complement_intersection_A_B_l3779_377929


namespace tate_education_ratio_l3779_377993

/-- Represents the duration of Tate's education -/
structure TateEducation where
  normalHighSchool : ℕ  -- Normal duration of high school
  tateHighSchool : ℕ    -- Tate's actual high school duration
  higherEd : ℕ          -- Duration of bachelor's degree and Ph.D.
  totalYears : ℕ        -- Total years spent in education

/-- Conditions of Tate's education -/
def validTateEducation (e : TateEducation) : Prop :=
  e.tateHighSchool = e.normalHighSchool - 1 ∧
  e.higherEd = e.tateHighSchool * (e.higherEd / e.tateHighSchool) ∧
  e.totalYears = e.tateHighSchool + e.higherEd ∧
  e.totalYears = 12

/-- The theorem to be proved -/
theorem tate_education_ratio (e : TateEducation) 
  (h : validTateEducation e) : 
  e.higherEd / e.tateHighSchool = 3 := by
  sorry

#check tate_education_ratio

end tate_education_ratio_l3779_377993


namespace smallest_three_digit_geometric_sequence_l3779_377901

/-- Checks if a number is a three-digit integer -/
def isThreeDigit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

/-- Extracts the hundreds digit of a three-digit number -/
def hundredsDigit (n : ℕ) : ℕ := n / 100

/-- Extracts the tens digit of a three-digit number -/
def tensDigit (n : ℕ) : ℕ := (n / 10) % 10

/-- Extracts the ones digit of a three-digit number -/
def onesDigit (n : ℕ) : ℕ := n % 10

/-- Checks if the digits of a three-digit number are distinct -/
def hasDistinctDigits (n : ℕ) : Prop :=
  let h := hundredsDigit n
  let t := tensDigit n
  let o := onesDigit n
  h ≠ t ∧ t ≠ o ∧ h ≠ o

/-- Checks if the digits of a three-digit number form a geometric sequence -/
def formsGeometricSequence (n : ℕ) : Prop :=
  let h := hundredsDigit n
  let t := tensDigit n
  let o := onesDigit n
  ∃ r : ℚ, r > 1 ∧ t = h * r ∧ o = t * r

theorem smallest_three_digit_geometric_sequence :
  ∀ n : ℕ, isThreeDigit n → hasDistinctDigits n → formsGeometricSequence n → n ≥ 248 :=
sorry

end smallest_three_digit_geometric_sequence_l3779_377901


namespace bouncy_ball_difference_l3779_377997

-- Define the given quantities
def red_packs : ℕ := 12
def yellow_packs : ℕ := 9
def balls_per_red_pack : ℕ := 24
def balls_per_yellow_pack : ℕ := 20

-- Define the theorem
theorem bouncy_ball_difference :
  red_packs * balls_per_red_pack - yellow_packs * balls_per_yellow_pack = 108 := by
  sorry

end bouncy_ball_difference_l3779_377997


namespace collinear_points_sum_l3779_377995

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), p2 = p1 + t • (p3 - p1) ∨ p1 = p2 + t • (p3 - p2)

/-- 
If the points (2,x,y), (x,3,y), and (x,y,4) are collinear, then x + y = 6.
-/
theorem collinear_points_sum (x y : ℝ) :
  collinear (2, x, y) (x, 3, y) (x, y, 4) → x + y = 6 := by
  sorry

end collinear_points_sum_l3779_377995


namespace gcd_2197_2208_l3779_377911

theorem gcd_2197_2208 : Nat.gcd 2197 2208 = 1 := by
  sorry

end gcd_2197_2208_l3779_377911


namespace AX_length_l3779_377907

-- Define the circle and points
def Circle : Type := Unit
def Point : Type := Unit

-- Define the diameter of the circle
def diameter : ℝ := 1

-- Define the points on the circle
def A : Point := sorry
def B : Point := sorry
def C : Point := sorry
def D : Point := sorry

-- Define point X on diameter AD
def X : Point := sorry

-- Define the distance function
def distance (p q : Point) : ℝ := sorry

-- Define the angle function
def angle (p q r : Point) : ℝ := sorry

-- State the theorem
theorem AX_length (h1 : distance A D = diameter)
                  (h2 : distance B X = distance C X)
                  (h3 : 3 * angle B A C = angle B X C)
                  (h4 : angle B X C = 30 * π / 180) :
  distance A X = Real.cos (10 * π / 180) * Real.sin (20 * π / 180) * (1 / Real.sin (15 * π / 180)) :=
sorry

end AX_length_l3779_377907


namespace distribute_9_4_l3779_377959

/-- The number of ways to distribute n identical items into k boxes -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- Theorem: There are 220 ways to distribute 9 identical items into 4 boxes -/
theorem distribute_9_4 : distribute 9 4 = 220 := by
  sorry

end distribute_9_4_l3779_377959


namespace min_sum_areas_two_triangles_l3779_377933

/-- The minimum sum of areas of two equilateral triangles formed from a 12cm wire -/
theorem min_sum_areas_two_triangles : 
  ∃ (f : ℝ → ℝ), 
    (∀ x, 0 ≤ x ∧ x ≤ 12 → 
      f x = (Real.sqrt 3 / 36) * (x^2 + (12 - x)^2)) ∧
    (∀ x, 0 ≤ x ∧ x ≤ 12 → f x ≥ 2 * Real.sqrt 3) ∧
    (∃ x, 0 ≤ x ∧ x ≤ 12 ∧ f x = 2 * Real.sqrt 3) :=
by sorry

end min_sum_areas_two_triangles_l3779_377933


namespace sum_of_powers_of_i_l3779_377962

theorem sum_of_powers_of_i : ∃ (i : ℂ), i^2 = -1 ∧ 
  (i + 2*i^2 + 3*i^3 + 4*i^4 + 5*i^5 + 6*i^6 + 7*i^7 + 8*i^8 + 9*i^9 = 4 + 5*i) := by
  sorry

end sum_of_powers_of_i_l3779_377962


namespace total_cost_theorem_l3779_377917

/-- Calculates the total cost of purchasing two laptops with accessories --/
def total_cost (first_laptop_price : ℝ) (second_laptop_multiplier : ℝ) 
  (second_laptop_discount : ℝ) (hard_drive_price : ℝ) (mouse_price : ℝ) 
  (software_subscription_price : ℝ) (insurance_rate : ℝ) : ℝ :=
  let first_laptop_total := first_laptop_price + hard_drive_price + mouse_price + 
    software_subscription_price + (insurance_rate * first_laptop_price)
  let second_laptop_price := first_laptop_price * second_laptop_multiplier
  let second_laptop_discounted := second_laptop_price * (1 - second_laptop_discount)
  let second_laptop_total := second_laptop_discounted + hard_drive_price + mouse_price + 
    (2 * software_subscription_price) + (insurance_rate * second_laptop_discounted)
  first_laptop_total + second_laptop_total

/-- Theorem stating the total cost of purchasing both laptops with accessories --/
theorem total_cost_theorem : 
  total_cost 500 3 0.15 80 20 120 0.1 = 2512.5 := by
  sorry

end total_cost_theorem_l3779_377917


namespace correlation_identification_l3779_377915

/-- Represents a relationship between two variables -/
inductive Relationship
| AgeWealth
| CurvePoint
| AppleProduction
| TreeDiameterHeight

/-- Determines if a relationship exhibits correlation -/
def has_correlation (r : Relationship) : Prop :=
  match r with
  | Relationship.AgeWealth => true
  | Relationship.CurvePoint => false
  | Relationship.AppleProduction => true
  | Relationship.TreeDiameterHeight => true

/-- The main theorem stating which relationships have correlation -/
theorem correlation_identification :
  (has_correlation Relationship.AgeWealth) ∧
  (¬has_correlation Relationship.CurvePoint) ∧
  (has_correlation Relationship.AppleProduction) ∧
  (has_correlation Relationship.TreeDiameterHeight) :=
sorry


end correlation_identification_l3779_377915


namespace john_water_savings_l3779_377974

def water_savings (old_flush_volume : ℝ) (flushes_per_day : ℕ) (water_reduction_percentage : ℝ) (days_in_month : ℕ) : ℝ :=
  let old_daily_usage := old_flush_volume * flushes_per_day
  let old_monthly_usage := old_daily_usage * days_in_month
  let new_flush_volume := old_flush_volume * (1 - water_reduction_percentage)
  let new_daily_usage := new_flush_volume * flushes_per_day
  let new_monthly_usage := new_daily_usage * days_in_month
  old_monthly_usage - new_monthly_usage

theorem john_water_savings :
  water_savings 5 15 0.8 30 = 1800 := by
  sorry

end john_water_savings_l3779_377974


namespace sum_a_d_l3779_377946

theorem sum_a_d (a b c d : ℝ) 
  (h1 : a * b + a * c + b * d + c * d = 48) 
  (h2 : b + c = 6) : 
  a + d = 8 := by
sorry

end sum_a_d_l3779_377946


namespace sheila_attend_probability_l3779_377976

/-- The probability of rain tomorrow -/
def prob_rain : ℝ := 0.4

/-- The probability Sheila will go if it rains -/
def prob_go_if_rain : ℝ := 0.2

/-- The probability Sheila will go if it's sunny -/
def prob_go_if_sunny : ℝ := 0.8

/-- The probability that Sheila will attend the picnic -/
def prob_sheila_attend : ℝ := prob_rain * prob_go_if_rain + (1 - prob_rain) * prob_go_if_sunny

theorem sheila_attend_probability :
  prob_sheila_attend = 0.56 := by sorry

end sheila_attend_probability_l3779_377976


namespace worker_a_time_l3779_377987

theorem worker_a_time (worker_b_time worker_ab_time : ℝ) 
  (hb : worker_b_time = 12)
  (hab : worker_ab_time = 4.8) : ℝ :=
  let worker_a_time := (worker_b_time * worker_ab_time) / (worker_b_time - worker_ab_time)
  8

#check worker_a_time

end worker_a_time_l3779_377987


namespace no_function_satisfies_conditions_l3779_377952

/-- A function satisfying the given conditions in the problem -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∃ M : ℝ, ∀ x, |f x| ≤ M) ∧ 
  f 1 = 1 ∧
  ∀ x ≠ 0, f (x + 1/x^2) = f x + (f (1/x))^2

/-- Theorem stating that no function satisfies all the given conditions -/
theorem no_function_satisfies_conditions : ¬∃ f : ℝ → ℝ, SatisfiesConditions f := by
  sorry


end no_function_satisfies_conditions_l3779_377952


namespace theater_attendance_l3779_377903

/-- Proves that the total number of attendees is 24 given the ticket prices, revenue, and number of children --/
theorem theater_attendance
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_revenue : ℕ)
  (num_children : ℕ)
  (h1 : adult_price = 16)
  (h2 : child_price = 9)
  (h3 : total_revenue = 258)
  (h4 : num_children = 18)
  (h5 : ∃ num_adults : ℕ, adult_price * num_adults + child_price * num_children = total_revenue) :
  num_children + (total_revenue - child_price * num_children) / adult_price = 24 :=
by sorry

end theater_attendance_l3779_377903


namespace determinant_problem_l3779_377966

theorem determinant_problem (a b c d : ℝ) : 
  let M₁ : Matrix (Fin 2) (Fin 2) ℝ := !![a, b; c, d]
  let M₂ : Matrix (Fin 2) (Fin 2) ℝ := !![a+2*c, b+2*d; 3*c, 3*d]
  Matrix.det M₁ = -7 → Matrix.det M₂ = -21 := by
  sorry

end determinant_problem_l3779_377966


namespace negation_of_all_politicians_are_loyal_l3779_377931

universe u

def Politician (α : Type u) := α → Prop
def Loyal (α : Type u) := α → Prop

theorem negation_of_all_politicians_are_loyal 
  {α : Type u} (politician : Politician α) (loyal : Loyal α) :
  (¬ ∀ (x : α), politician x → loyal x) ↔ (∃ (x : α), politician x ∧ ¬ loyal x) :=
sorry

end negation_of_all_politicians_are_loyal_l3779_377931


namespace difference_is_integer_l3779_377983

/-- A linear function from ℝ to ℝ -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  map : ℝ → ℝ := fun x ↦ a * x + b
  increasing : 0 < a

/-- Two linear functions with the integer property -/
structure IntegerPropertyFunctions where
  f : LinearFunction
  g : LinearFunction
  integer_property : ∀ x : ℝ, Int.floor (f.map x) = f.map x ↔ Int.floor (g.map x) = g.map x

/-- The main theorem -/
theorem difference_is_integer (funcs : IntegerPropertyFunctions) :
  ∀ x : ℝ, ∃ n : ℤ, funcs.f.map x - funcs.g.map x = n :=
sorry

end difference_is_integer_l3779_377983


namespace multiple_of_1998_l3779_377984

theorem multiple_of_1998 (a : Fin 93 → ℕ+) (h : Function.Injective a) :
  ∃ m n p q : Fin 93, m ≠ n ∧ p ≠ q ∧ 1998 ∣ (a m - a n) * (a p - a q) := by
  sorry

end multiple_of_1998_l3779_377984


namespace linear_systems_solutions_l3779_377950

theorem linear_systems_solutions :
  -- System (1)
  let system1 : ℝ × ℝ → Prop := λ (x, y) ↦ 3 * x - 2 * y = 9 ∧ 2 * x + 3 * y = 19
  -- System (2)
  let system2 : ℝ × ℝ → Prop := λ (x, y) ↦ (2 * x + 1) / 5 - 1 = (y - 1) / 3 ∧ 2 * (y - x) - 3 * (1 - y) = 6
  -- Solutions
  let solution1 : ℝ × ℝ := (5, 3)
  let solution2 : ℝ × ℝ := (4, 17/5)
  -- Proof statements
  system1 solution1 ∧ system2 solution2 := by sorry

end linear_systems_solutions_l3779_377950


namespace books_per_shelf_l3779_377900

theorem books_per_shelf (total_books : Nat) (total_shelves : Nat) 
  (h1 : total_books = 14240) (h2 : total_shelves = 1780) : 
  total_books / total_shelves = 8 := by
  sorry

end books_per_shelf_l3779_377900


namespace inequality_solution_l3779_377935

theorem inequality_solution (x : ℝ) : 
  (x^2 - 4*x + 3) / ((x - 2)^2) < 0 ↔ 1 < x ∧ x < 3 :=
by sorry

end inequality_solution_l3779_377935


namespace M_remainder_mod_45_l3779_377968

/-- The number of digits in M -/
def num_digits : ℕ := 95

/-- The last integer in the sequence forming M -/
def last_int : ℕ := 50

/-- M is the number formed by concatenating integers from 1 to last_int -/
def M : ℕ := sorry

theorem M_remainder_mod_45 : M % 45 = 15 := by sorry

end M_remainder_mod_45_l3779_377968


namespace point_in_third_quadrant_l3779_377972

/-- Definition of a point in the third quadrant -/
def is_in_third_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 < 0

/-- The point (-2, -3) is in the third quadrant -/
theorem point_in_third_quadrant : is_in_third_quadrant (-2, -3) := by
  sorry

end point_in_third_quadrant_l3779_377972


namespace g_derivative_at_5_l3779_377919

-- Define the function g
def g (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

-- State the theorem
theorem g_derivative_at_5 : 
  (deriv g) 5 = 26 := by sorry

end g_derivative_at_5_l3779_377919


namespace hexagon_largest_angle_l3779_377965

theorem hexagon_largest_angle :
  ∀ (a b c d e f : ℝ),
    -- The angles are consecutive integers
    (∃ (x : ℝ), a = x - 2 ∧ b = x - 1 ∧ c = x ∧ d = x + 1 ∧ e = x + 2 ∧ f = x + 3) →
    -- Sum of angles in a hexagon is 720°
    a + b + c + d + e + f = 720 →
    -- The largest angle is 122.5°
    max a (max b (max c (max d (max e f)))) = 122.5 := by
  sorry

end hexagon_largest_angle_l3779_377965


namespace voldemort_remaining_calories_voldemort_specific_remaining_calories_l3779_377977

/-- Calculates the remaining calories Voldemort can consume given his intake and limit -/
theorem voldemort_remaining_calories (cake_calories : ℕ) (chips_calories : ℕ) 
  (coke_calories : ℕ) (breakfast_calories : ℕ) (lunch_calories : ℕ) 
  (daily_limit : ℕ) : ℕ :=
  by
  have dinner_calories : ℕ := cake_calories + chips_calories + coke_calories
  have breakfast_lunch_calories : ℕ := breakfast_calories + lunch_calories
  have total_consumed : ℕ := dinner_calories + breakfast_lunch_calories
  exact daily_limit - total_consumed

/-- Proves that Voldemort's remaining calories is 525 given specific intake values -/
theorem voldemort_specific_remaining_calories : 
  voldemort_remaining_calories 110 310 215 560 780 2500 = 525 :=
by
  sorry

end voldemort_remaining_calories_voldemort_specific_remaining_calories_l3779_377977


namespace window_height_is_four_l3779_377956

-- Define the room dimensions
def room_length : ℝ := 25
def room_width : ℝ := 15
def room_height : ℝ := 12

-- Define the cost of whitewashing per square foot
def cost_per_sqft : ℝ := 3

-- Define the door dimensions
def door_height : ℝ := 6
def door_width : ℝ := 3

-- Define the number of windows and their width
def num_windows : ℕ := 3
def window_width : ℝ := 3

-- Define the total cost of whitewashing
def total_cost : ℝ := 2718

-- Theorem to prove
theorem window_height_is_four :
  ∃ (h : ℝ),
    h = 4 ∧
    (2 * (room_length * room_height + room_width * room_height) -
     (door_height * door_width + num_windows * h * window_width)) * cost_per_sqft = total_cost :=
by
  sorry

end window_height_is_four_l3779_377956


namespace container_volume_increase_l3779_377913

theorem container_volume_increase (original_volume : ℝ) (scale_factor : ℝ) : 
  original_volume = 5 → 
  scale_factor = 4 → 
  (scale_factor ^ 3) * original_volume = 320 := by
sorry

end container_volume_increase_l3779_377913


namespace remainder_369975_div_6_l3779_377951

theorem remainder_369975_div_6 : 369975 % 6 = 3 := by
  sorry

end remainder_369975_div_6_l3779_377951


namespace arithmetic_sequence_problem_l3779_377981

theorem arithmetic_sequence_problem (a : ℕ → ℤ) (n : ℕ) :
  (a 1 = 1) →
  (∀ k : ℕ, a (k + 1) - a k = 3) →
  (a n = 298) →
  n = 100 := by
sorry

end arithmetic_sequence_problem_l3779_377981


namespace polynomial_unique_value_l3779_377967

theorem polynomial_unique_value (P : ℤ → ℤ) :
  (∃ x₁ x₂ x₃ : ℤ, P x₁ = 1 ∧ P x₂ = 2 ∧ P x₃ = 3 ∧ (x₁ = x₂ - 1 ∨ x₁ = x₂ + 1) ∧ (x₂ = x₃ - 1 ∨ x₂ = x₃ + 1)) →
  (∃! x : ℤ, P x = 5) :=
by sorry

end polynomial_unique_value_l3779_377967


namespace star_computation_l3779_377970

def star (a b : ℚ) : ℚ := (a - b) / (1 - a * b)

theorem star_computation :
  star 2 (star 3 (star 4 5)) = 1/4 := by sorry

end star_computation_l3779_377970


namespace sum_a_d_l3779_377909

theorem sum_a_d (a b c d : ℝ) 
  (h1 : a * b + b * c + c * a + d * b = 42) 
  (h2 : b + c = 6) : 
  a + d = 7 := by
sorry

end sum_a_d_l3779_377909


namespace quadratic_symmetry_l3779_377998

/-- A quadratic function -/
def QuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_symmetry 
  (f : ℝ → ℝ) 
  (hf : QuadraticFunction f) 
  (h0 : f 0 = 3)
  (h1 : f 1 = 2)
  (h2 : f 2 = 3)
  (h3 : f 3 = 6)
  (h4 : f 4 = 11)
  (hm2 : f (-2) = 11) :
  f (-1) = 6 := by
sorry

end quadratic_symmetry_l3779_377998


namespace total_travel_methods_eq_thirteen_l3779_377944

/-- The number of bus services from A to B -/
def bus_services : ℕ := 8

/-- The number of train services from A to B -/
def train_services : ℕ := 3

/-- The number of ship services from A to B -/
def ship_services : ℕ := 2

/-- The total number of different methods to travel from A to B -/
def total_travel_methods : ℕ := bus_services + train_services + ship_services

theorem total_travel_methods_eq_thirteen :
  total_travel_methods = 13 := by sorry

end total_travel_methods_eq_thirteen_l3779_377944


namespace sculpture_and_base_height_l3779_377920

/-- Converts feet to inches -/
def feetToInches (feet : ℕ) : ℕ := feet * 12

/-- Represents the height of an object in feet and inches -/
structure Height where
  feet : ℕ
  inches : ℕ

/-- Converts a Height to total inches -/
def heightToInches (h : Height) : ℕ := feetToInches h.feet + h.inches

/-- Theorem: The combined height of the sculpture and base is 42 inches -/
theorem sculpture_and_base_height :
  let sculpture : Height := { feet := 2, inches := 10 }
  let base_height : ℕ := 8
  heightToInches sculpture + base_height = 42 := by
  sorry

end sculpture_and_base_height_l3779_377920


namespace congruence_solution_l3779_377991

theorem congruence_solution (n : ℕ) : n ∈ Finset.range 29 → (8 * n ≡ 5 [ZMOD 29]) ↔ n = 26 := by
  sorry

end congruence_solution_l3779_377991
