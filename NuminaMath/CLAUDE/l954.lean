import Mathlib

namespace red_to_blue_ratio_l954_95445

/-- Represents the number of marbles of each color in Cara's bag. -/
structure MarbleCounts where
  total : ℕ
  yellow : ℕ
  green : ℕ
  red : ℕ
  blue : ℕ

/-- Theorem stating the ratio of red to blue marbles given the conditions -/
theorem red_to_blue_ratio (m : MarbleCounts) : 
  m.total = 60 ∧ 
  m.yellow = 20 ∧ 
  m.green = m.yellow / 2 ∧ 
  m.total = m.yellow + m.green + m.red + m.blue ∧ 
  m.blue = m.total / 4 →
  m.red / m.blue = 11 / 4 := by
  sorry

end red_to_blue_ratio_l954_95445


namespace unique_integer_divisible_by_20_with_cube_root_between_9_and_91_tenths_l954_95469

theorem unique_integer_divisible_by_20_with_cube_root_between_9_and_91_tenths : 
  ∃! n : ℕ+, 
    (∃ k : ℕ, n = 20 * k) ∧ 
    (9 : ℝ) < n.val ^ (1/3 : ℝ) ∧ 
    n.val ^ (1/3 : ℝ) < (91/10 : ℝ) ∧
    n = 740 := by
  sorry

end unique_integer_divisible_by_20_with_cube_root_between_9_and_91_tenths_l954_95469


namespace triangle_intersecting_circle_theorem_l954_95464

/-- Given a triangle ABC with sides a, b, c, and points A₁, B₁, C₁ on its sides satisfying certain ratios,
    if the circumcircle of A₁B₁C₁ intersects the sides of ABC at segments of lengths x, y, z,
    then x/a^(n-1) + y/b^(n-1) + z/c^(n-1) = 0 -/
theorem triangle_intersecting_circle_theorem
  (a b c : ℝ) (n : ℕ) (x y z : ℝ)
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_AB₁ : ∃ B₁ : ℝ, B₁ > 0 ∧ B₁ < c ∧ B₁ / (c - B₁) = (c^n) / (a^n))
  (h_BC₁ : ∃ C₁ : ℝ, C₁ > 0 ∧ C₁ < a ∧ C₁ / (a - C₁) = (a^n) / (b^n))
  (h_CA₁ : ∃ A₁ : ℝ, A₁ > 0 ∧ A₁ < b ∧ A₁ / (b - A₁) = (b^n) / (c^n))
  (h_intersect : ∃ (A₁ B₁ C₁ : ℝ), 
    (B₁ > 0 ∧ B₁ < c ∧ B₁ / (c - B₁) = (c^n) / (a^n)) ∧
    (C₁ > 0 ∧ C₁ < a ∧ C₁ / (a - C₁) = (a^n) / (b^n)) ∧
    (A₁ > 0 ∧ A₁ < b ∧ A₁ / (b - A₁) = (b^n) / (c^n)) ∧
    (∃ (x' y' z' : ℝ), x' * x = (B₁ * (c - B₁)) ∧ 
                       y' * y = (C₁ * (a - C₁)) ∧ 
                       z' * z = (A₁ * (b - A₁))))
  : x / (a^(n-1)) + y / (b^(n-1)) + z / (c^(n-1)) = 0 := by
  sorry

end triangle_intersecting_circle_theorem_l954_95464


namespace least_reducible_n_l954_95408

def is_reducible (a b : Int) : Prop :=
  Int.gcd a b > 1

def fraction_numerator (n : Int) : Int :=
  2*n - 26

def fraction_denominator (n : Int) : Int :=
  10*n + 12

theorem least_reducible_n :
  (∀ k : Nat, k > 0 ∧ k < 49 → ¬(is_reducible (fraction_numerator k) (fraction_denominator k))) ∧
  (is_reducible (fraction_numerator 49) (fraction_denominator 49)) :=
sorry

end least_reducible_n_l954_95408


namespace tangent_circle_radius_l954_95451

/-- Right isosceles triangle with legs of length 2 -/
structure RightIsoscelesTriangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ
  is_right_angle : Q.1 = 0 ∧ Q.2 = 0 ∧ P.2 = 0 ∧ R.1 = 0
  is_isosceles : P.1 = 2 ∧ R.2 = 2

/-- Circle tangent to triangle hypotenuse and coordinate axes -/
structure TangentCircle where
  S : ℝ × ℝ
  radius : ℝ
  tangent_to_hypotenuse : (S.1 - 2)^2 + (S.2 - 2)^2 = 8
  tangent_to_axes : S.1 = radius ∧ S.2 = radius

/-- The radius of the tangent circle is 4 -/
theorem tangent_circle_radius (t : RightIsoscelesTriangle) (c : TangentCircle) :
  c.radius = 4 := by sorry

end tangent_circle_radius_l954_95451


namespace vector_perpendicular_m_l954_95404

theorem vector_perpendicular_m (a b : ℝ × ℝ) (m : ℝ) : 
  a = (3, 4) → 
  b = (2, -1) → 
  (a.1 + m * b.1, a.2 + m * b.2) • (a.1 - b.1, a.2 - b.2) = 0 → 
  m = 23 / 3 := by
sorry

end vector_perpendicular_m_l954_95404


namespace x_0_interval_l954_95406

theorem x_0_interval (x_0 : ℝ) (h1 : x_0 ∈ Set.Ioo 0 π) 
  (h2 : Real.sin x_0 + Real.cos x_0 = 2/3) : 
  x_0 ∈ Set.Ioo (7*π/12) (3*π/4) := by
  sorry

end x_0_interval_l954_95406


namespace cone_base_radius_l954_95487

/-- Given a cone whose lateral surface is a sector of a circle with a central angle of 216° 
    and a radius of 15 cm, the radius of the base of the cone is 9 cm. -/
theorem cone_base_radius (central_angle : ℝ) (sector_radius : ℝ) (base_radius : ℝ) : 
  central_angle = 216 * (π / 180) →  -- Convert 216° to radians
  sector_radius = 15 →
  base_radius = sector_radius * (central_angle / (2 * π)) →
  base_radius = 9 := by
sorry


end cone_base_radius_l954_95487


namespace total_days_2004_to_2008_l954_95419

def isLeapYear (year : Nat) : Bool :=
  (year % 4 == 0 && year % 100 ≠ 0) || (year % 400 == 0)

def daysInYear (year : Nat) : Nat :=
  if isLeapYear year then 366 else 365

def totalDaysInRange (startYear endYear : Nat) : Nat :=
  (List.range (endYear - startYear + 1)).map (fun i => daysInYear (startYear + i))
    |> List.sum

theorem total_days_2004_to_2008 :
  totalDaysInRange 2004 2008 = 1827 := by
  sorry

end total_days_2004_to_2008_l954_95419


namespace trigonometric_identity_l954_95435

theorem trigonometric_identity (a b : ℝ) : 
  (∀ x : ℝ, 2 * (Real.cos (x + b / 2))^2 - 2 * Real.sin (a * x - π / 2) * Real.cos (a * x - π / 2) = 1) ↔ 
  ((a = 1 ∧ ∃ k : ℤ, b = -3 * π / 2 + 2 * ↑k * π) ∨ 
   (a = -1 ∧ ∃ k : ℤ, b = 3 * π / 2 + 2 * ↑k * π)) :=
by sorry

end trigonometric_identity_l954_95435


namespace tape_overlap_division_l954_95430

/-- Given 5 pieces of tape, each 2.7 meters long, with an overlap of 0.3 meters between pieces,
    when divided into 6 equal parts, each part is 2.05 meters long. -/
theorem tape_overlap_division (n : ℕ) (piece_length overlap_length : ℝ) (h1 : n = 5) 
    (h2 : piece_length = 2.7) (h3 : overlap_length = 0.3) : 
  (n * piece_length - (n - 1) * overlap_length) / 6 = 2.05 := by
  sorry

#check tape_overlap_division

end tape_overlap_division_l954_95430


namespace possible_student_totals_l954_95409

/-- Represents the distribution of students in groups -/
structure StudentDistribution where
  total_groups : Nat
  groups_with_13 : Nat
  total_students : Nat

/-- Checks if the distribution is valid according to the problem conditions -/
def is_valid_distribution (d : StudentDistribution) : Prop :=
  d.total_groups = 6 ∧
  d.groups_with_13 = 4 ∧
  (d.total_students = 76 ∨ d.total_students = 80)

/-- Theorem stating the possible total numbers of students -/
theorem possible_student_totals :
  ∀ d : StudentDistribution,
    is_valid_distribution d →
    (d.total_students = 76 ∨ d.total_students = 80) :=
by
  sorry

#check possible_student_totals

end possible_student_totals_l954_95409


namespace equation_roots_problem_l954_95456

/-- Given two equations with specific root conditions, prove the value of 100c + d -/
theorem equation_roots_problem (c d : ℝ) : 
  (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧
    ∀ (x : ℝ), (x + c) * (x + d) * (x + 15) = 0 ↔ x = x₁ ∨ x = x₂ ∨ x = x₃) →
  (∀ (x : ℝ), x ≠ -4 → (x + c) * (x + d) * (x + 15) ≠ 0) →
  (∃! (x : ℝ), (x + 3*c) * (x + 4) * (x + 9) = 0 ∧ (x + d) * (x + 15) ≠ 0) →
  100 * c + d = -291 := by
sorry

end equation_roots_problem_l954_95456


namespace polynomial_expansion_l954_95415

theorem polynomial_expansion (x : ℝ) :
  (3 * x^2 - 4 * x + 3) * (-2 * x^2 + 3 * x - 4) =
  -6 * x^4 + 17 * x^3 - 30 * x^2 + 25 * x - 12 := by
  sorry

end polynomial_expansion_l954_95415


namespace jason_blue_marbles_count_l954_95429

/-- The number of blue marbles Jason and Tom have in total -/
def total_blue_marbles : ℕ := 68

/-- The number of blue marbles Tom has -/
def tom_blue_marbles : ℕ := 24

/-- The number of blue marbles Jason has -/
def jason_blue_marbles : ℕ := total_blue_marbles - tom_blue_marbles

theorem jason_blue_marbles_count : jason_blue_marbles = 44 := by
  sorry

end jason_blue_marbles_count_l954_95429


namespace three_number_sum_l954_95485

theorem three_number_sum (a b c : ℝ) (h1 : a ≤ b) (h2 : b ≤ c) (h3 : b = 10) 
  (h4 : (a + b + c) / 3 = a + 20) (h5 : (a + b + c) / 3 = c - 30) : 
  a + b + c = 60 := by
sorry

end three_number_sum_l954_95485


namespace fence_cost_calculation_l954_95424

/-- Calculates the total cost of installing two types of fences around a rectangular field -/
theorem fence_cost_calculation (length width : ℝ) (barbed_wire_cost picket_fence_cost : ℝ) 
  (num_gates gate_width : ℝ) : 
  length = 500 ∧ 
  width = 150 ∧ 
  barbed_wire_cost = 1.2 ∧ 
  picket_fence_cost = 2.5 ∧ 
  num_gates = 4 ∧ 
  gate_width = 1.25 → 
  (2 * (length + width) - num_gates * gate_width) * barbed_wire_cost + 
  2 * (length + width) * picket_fence_cost = 4804 := by
  sorry


end fence_cost_calculation_l954_95424


namespace cow_herd_division_l954_95462

theorem cow_herd_division (total : ℕ) : 
  (2 : ℚ) / 3 * total + (1 : ℚ) / 6 * total + (1 : ℚ) / 9 * total + 6 = total → 
  total = 108 := by
sorry

end cow_herd_division_l954_95462


namespace circle_diameter_l954_95436

theorem circle_diameter (A : Real) (r : Real) (d : Real) : 
  A = Real.pi * r^2 → A = 64 * Real.pi → d = 2 * r → d = 16 := by
  sorry

end circle_diameter_l954_95436


namespace angle_ADB_is_right_angle_l954_95455

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Represents a circle -/
structure Circle :=
  (center : Point) (radius : ℝ)

/-- Checks if a triangle is isosceles with two sides equal -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  (t.A.x - t.C.x)^2 + (t.A.y - t.C.y)^2 = (t.B.x - t.C.x)^2 + (t.B.y - t.C.y)^2

/-- Checks if a point lies on a circle -/
def Circle.contains (c : Circle) (p : Point) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 = c.radius^2

/-- Calculates the angle between three points -/
def angle (A B C : Point) : ℝ := sorry

/-- Main theorem -/
theorem angle_ADB_is_right_angle 
  (t : Triangle) 
  (c : Circle) 
  (D : Point) 
  (h1 : t.isIsosceles)
  (h2 : c.center = t.C)
  (h3 : c.radius = 15)
  (h4 : c.contains t.B)
  (h5 : ∃ k : ℝ, D.x = t.C.x + k * (t.C.x - t.A.x) ∧ D.y = t.C.y + k * (t.C.y - t.A.y))
  (h6 : c.contains D) :
  angle t.A D t.B = 90 := by sorry

end angle_ADB_is_right_angle_l954_95455


namespace y_value_at_16_l954_95405

/-- Given a function y = k * x^(1/4) where y = 3 * √3 when x = 9, 
    prove that y = 6 when x = 16 -/
theorem y_value_at_16 (k : ℝ) (y : ℝ → ℝ) :
  (∀ x, y x = k * x^(1/4)) →
  y 9 = 3 * Real.sqrt 3 →
  y 16 = 6 := by
  sorry

end y_value_at_16_l954_95405


namespace dividend_calculation_l954_95442

theorem dividend_calculation (divisor quotient remainder dividend : ℕ) : 
  divisor = 3 →
  quotient = 7 →
  remainder = 2 →
  dividend = divisor * quotient + remainder →
  dividend = 23 := by
sorry

end dividend_calculation_l954_95442


namespace line_through_points_l954_95400

/-- Given a line y = ax + b passing through points (3, 7) and (9/2, 13), prove that a - b = 9 -/
theorem line_through_points (a b : ℝ) : 
  (7 = a * 3 + b) → (13 = a * (9/2) + b) → a - b = 9 := by
  sorry

end line_through_points_l954_95400


namespace modulus_of_3_minus_2i_l954_95496

theorem modulus_of_3_minus_2i : Complex.abs (3 - 2*Complex.I) = Real.sqrt 13 := by
  sorry

end modulus_of_3_minus_2i_l954_95496


namespace car_braking_distance_l954_95488

def braking_sequence (n : ℕ) : ℕ :=
  max (50 - 10 * n) 0

def total_distance (n : ℕ) : ℕ :=
  (List.range n).map braking_sequence |>.sum

theorem car_braking_distance :
  ∃ n : ℕ, total_distance n = 150 ∧ braking_sequence n = 0 :=
sorry

end car_braking_distance_l954_95488


namespace letters_ratio_l954_95411

/-- Proves that the ratio of letters Greta's mother received to the total letters Greta and her brother received is 2:1 -/
theorem letters_ratio (greta_letters brother_letters mother_letters : ℕ) : 
  greta_letters = brother_letters + 10 →
  brother_letters = 40 →
  greta_letters + brother_letters + mother_letters = 270 →
  ∃ k : ℕ, mother_letters = k * (greta_letters + brother_letters) →
  mother_letters = 2 * (greta_letters + brother_letters) := by
sorry

end letters_ratio_l954_95411


namespace quadratic_factorization_sum_l954_95471

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x : ℝ, x^2 + 15*x + 36 = (x + a)*(x + b)) →
  (∀ x : ℝ, x^2 + 7*x - 60 = (x + b)*(x - c)) →
  a + b + c = 20 := by
sorry

end quadratic_factorization_sum_l954_95471


namespace triplet_sum_position_l954_95431

theorem triplet_sum_position 
  (x : Fin 6 → ℝ) 
  (s : Fin 20 → ℝ) 
  (h_order : ∀ i j, i < j → x i < x j) 
  (h_sums : ∀ i j, i < j → s i < s j) 
  (h_distinct : ∀ i j k l m n, i < j → j < k → l < m → m < n → 
    x i + x j + x k ≠ x l + x m + x n) 
  (h_s11 : x 1 + x 2 + x 3 = s 10) 
  (h_s15 : x 1 + x 2 + x 5 = s 14) : 
  x 0 + x 1 + x 5 = s 6 := by
sorry

end triplet_sum_position_l954_95431


namespace dress_shoes_count_l954_95460

theorem dress_shoes_count (polished_percent : ℚ) (remaining : ℕ) : 
  polished_percent = 45/100 → remaining = 11 → (1 - polished_percent) * (2 * remaining / (1 - polished_percent)) / 2 = 10 := by
  sorry

end dress_shoes_count_l954_95460


namespace one_minus_repeating_eight_eq_one_ninth_l954_95446

/-- The value of 0.888... (repeating decimal) -/
def repeating_eight : ℚ := 8/9

/-- Proof that 1 - 0.888... = 1/9 -/
theorem one_minus_repeating_eight_eq_one_ninth :
  1 - repeating_eight = 1/9 := by
  sorry

end one_minus_repeating_eight_eq_one_ninth_l954_95446


namespace hash_2_3_2_1_l954_95449

def hash (a b c d : ℝ) : ℝ := b^2 - 4*a*c*d

theorem hash_2_3_2_1 : hash 2 3 2 1 = -7 := by
  sorry

end hash_2_3_2_1_l954_95449


namespace opposite_sides_line_range_l954_95459

theorem opposite_sides_line_range (a : ℝ) : 
  (((3 : ℝ) * 3 - 2 * 1 + a > 0 ∧ (3 : ℝ) * (-4) - 2 * 6 + a < 0) ∨
   ((3 : ℝ) * 3 - 2 * 1 + a < 0 ∧ (3 : ℝ) * (-4) - 2 * 6 + a > 0)) →
  -7 < a ∧ a < 24 := by
sorry

end opposite_sides_line_range_l954_95459


namespace distribution_equivalence_l954_95403

/-- The number of ways to distribute n indistinguishable objects among k recipients,
    with each recipient receiving at least one object. -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items. -/
def choose (n r : ℕ) : ℕ := sorry

theorem distribution_equivalence :
  distribute 10 7 = choose 9 6 := by sorry

end distribution_equivalence_l954_95403


namespace ratio_calculation_l954_95454

theorem ratio_calculation (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end ratio_calculation_l954_95454


namespace no_fixed_point_function_l954_95484

-- Define the types for our polynomials
variable (p q h : ℝ → ℝ)

-- Define m and n as natural numbers
variable (m n : ℕ)

-- Define the descending property for p
def IsDescending (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- State the theorem
theorem no_fixed_point_function
  (hp : IsDescending p)
  (hpqh : ∀ x, p (q (n * x + m) + h x) = n * (q (p x) + h x) + m) :
  ¬ ∃ f : ℝ → ℝ, ∀ x, f (q (p x) + h x) = f x ^ 2 + 1 :=
sorry

end no_fixed_point_function_l954_95484


namespace suresh_completion_time_l954_95438

/-- Proves that Suresh can complete the job alone in 15 hours given the problem conditions -/
theorem suresh_completion_time :
  ∀ (S : ℝ),
  (S > 0) →  -- Suresh's completion time is positive
  (9 / S + 10 / 25 = 1) →  -- Combined work of Suresh and Ashutosh equals the whole job
  (S = 15) :=
by
  sorry

end suresh_completion_time_l954_95438


namespace phi_values_l954_95413

theorem phi_values (φ : Real) : 
  Real.sqrt 3 * Real.sin (20 * π / 180) = 2 * Real.cos φ - Real.sin φ → 
  φ = 140 * π / 180 ∨ φ = 40 * π / 180 := by
  sorry

end phi_values_l954_95413


namespace simple_interest_problem_l954_95468

/-- Represents the simple interest calculation problem --/
theorem simple_interest_problem (P T : ℝ) (h1 : P = 2500) (h2 : T = 5) : 
  let SI := P - 2000
  let R := (SI * 100) / (P * T)
  R = 4 := by sorry

end simple_interest_problem_l954_95468


namespace red_balls_count_l954_95457

theorem red_balls_count (total : Nat) (white green yellow purple : Nat) (prob : Real) :
  total = 100 ∧ 
  white = 50 ∧ 
  green = 20 ∧ 
  yellow = 10 ∧ 
  purple = 3 ∧ 
  prob = 0.8 ∧ 
  prob = (white + green + yellow : Real) / total →
  total - (white + green + yellow + purple) = 17 := by
  sorry

end red_balls_count_l954_95457


namespace equation_solutions_l954_95494

theorem equation_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 - 8*x + 6 = 0
  let eq2 : ℝ → Prop := λ x ↦ (x - 1)^2 = 3*x - 3
  let sol1 : Set ℝ := {4 + Real.sqrt 10, 4 - Real.sqrt 10}
  let sol2 : Set ℝ := {1, 4}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ y ∉ sol1, ¬eq1 y) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ y ∉ sol2, ¬eq2 y) :=
by
  sorry

end equation_solutions_l954_95494


namespace product_of_exponents_l954_95407

theorem product_of_exponents (p r s : ℕ) : 
  3^p + 3^4 = 90 → 2^r + 44 = 76 → 5^3 + 6^s = 1421 → p * r * s = 40 := by
sorry

end product_of_exponents_l954_95407


namespace jackson_souvenirs_count_l954_95489

theorem jackson_souvenirs_count :
  let hermit_crabs : ℕ := 120
  let shells_per_crab : ℕ := 8
  let starfish_per_shell : ℕ := 5
  let sand_dollars_per_starfish : ℕ := 3
  let sand_dollars_per_coral : ℕ := 4

  let total_shells : ℕ := hermit_crabs * shells_per_crab
  let total_starfish : ℕ := total_shells * starfish_per_shell
  let total_sand_dollars : ℕ := total_starfish * sand_dollars_per_starfish
  let total_coral : ℕ := total_sand_dollars / sand_dollars_per_coral

  let total_souvenirs : ℕ := hermit_crabs + total_shells + total_starfish + total_sand_dollars + total_coral

  total_souvenirs = 22880 :=
by
  sorry

end jackson_souvenirs_count_l954_95489


namespace total_potatoes_brought_home_l954_95422

/-- The number of people who received potatoes -/
def num_people : ℕ := 3

/-- The number of potatoes each person received -/
def potatoes_per_person : ℕ := 8

/-- Theorem: The total number of potatoes brought home is 24 -/
theorem total_potatoes_brought_home : 
  num_people * potatoes_per_person = 24 := by
  sorry

end total_potatoes_brought_home_l954_95422


namespace quadratic_function_unique_l954_95499

/-- A quadratic function is a function of the form f(x) = ax^2 + bx + c where a ≠ 0 -/
def IsQuadratic (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

theorem quadratic_function_unique 
  (f : ℝ → ℝ) 
  (h_quad : IsQuadratic f) 
  (h_f0 : f 0 = 1) 
  (h_fx1 : ∀ x, f (x + 1) = f x + x + 1) : 
  ∀ x, f x = (1/2) * x^2 + (1/2) * x + 1 :=
sorry

end quadratic_function_unique_l954_95499


namespace jane_albert_same_committee_l954_95439

/-- The number of MBAs --/
def n : ℕ := 6

/-- The number of members in each committee --/
def k : ℕ := 3

/-- The number of committees to be formed --/
def num_committees : ℕ := 2

/-- The total number of ways to form the committees --/
def total_ways : ℕ := Nat.choose n k

/-- The number of ways Jane and Albert can be on the same committee --/
def favorable_ways : ℕ := Nat.choose (n - 2) (k - 2)

/-- The probability that Jane and Albert are on the same committee --/
def prob_same_committee : ℚ := favorable_ways / total_ways

theorem jane_albert_same_committee :
  prob_same_committee = 1 / 5 :=
sorry

end jane_albert_same_committee_l954_95439


namespace boat_speed_proof_l954_95481

/-- The speed of the boat in standing water -/
def boat_speed : ℝ := 9

/-- The speed of the stream -/
def stream_speed : ℝ := 6

/-- The distance traveled in one direction -/
def distance : ℝ := 170

/-- The total time taken for the round trip -/
def total_time : ℝ := 68

theorem boat_speed_proof :
  (distance / (boat_speed + stream_speed) + distance / (boat_speed - stream_speed)) = total_time :=
by sorry

end boat_speed_proof_l954_95481


namespace total_marks_is_530_l954_95461

/-- Calculates the total marks scored by Amaya in all subjects given the following conditions:
  * Amaya scored 20 marks fewer in Maths than in Arts
  * She got 10 marks more in Social Studies than in Music
  * She scored 70 in Music
  * She scored 1/10 less in Maths than in Arts
-/
def totalMarks (musicScore : ℕ) : ℕ :=
  let socialStudiesScore := musicScore + 10
  let artsScore := 200
  let mathsScore := artsScore - 20
  musicScore + socialStudiesScore + artsScore + mathsScore

theorem total_marks_is_530 : totalMarks 70 = 530 := by
  sorry

end total_marks_is_530_l954_95461


namespace special_function_value_l954_95483

/-- A function satisfying the given property -/
def SpecialFunction (f : ℝ → ℝ) : Prop :=
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x * y) = f x / y

theorem special_function_value (f : ℝ → ℝ) 
  (h : SpecialFunction f) (h250 : f 250 = 4) : f 300 = 10/3 := by
  sorry

end special_function_value_l954_95483


namespace adam_has_more_apples_l954_95440

/-- The number of apples Jackie has -/
def jackies_apples : ℕ := 6

/-- The number of apples Adam has -/
def adams_apples : ℕ := jackies_apples + 3

theorem adam_has_more_apples : adams_apples = 9 := by
  sorry

end adam_has_more_apples_l954_95440


namespace max_crayfish_revenue_l954_95478

/-- The revenue function for selling crayfish -/
def revenue (x : ℝ) : ℝ := (32 - x) * (x - 4.5)

/-- The theorem stating the maximum revenue and number of crayfish sold -/
theorem max_crayfish_revenue :
  ∃ (x : ℕ), x ≤ 32 ∧ 
  revenue (32 - x : ℝ) = 189 ∧
  ∀ (y : ℕ), y ≤ 32 → revenue (32 - y : ℝ) ≤ 189 ∧
  x = 14 :=
sorry

end max_crayfish_revenue_l954_95478


namespace shadow_boundary_equation_l954_95418

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a sphere in 3D space -/
structure Sphere where
  center : Point3D
  radius : ℝ

/-- The equation of the shadow boundary on the xy-plane -/
def shadowBoundary (s : Sphere) (lightSource : Point3D) : ℝ → ℝ := fun x ↦ -14

/-- Theorem: The shadow boundary of the given sphere with the given light source is y = -14 -/
theorem shadow_boundary_equation (s : Sphere) (lightSource : Point3D) :
  s.center = Point3D.mk 2 0 2 →
  s.radius = 2 →
  lightSource = Point3D.mk 2 (-2) 6 →
  ∀ x : ℝ, shadowBoundary s lightSource x = -14 := by
  sorry

#check shadow_boundary_equation

end shadow_boundary_equation_l954_95418


namespace complement_A_intersect_B_when_a_is_2_complement_A_union_B_equals_reals_iff_l954_95474

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1) + Real.sqrt (2 - x)}
def B (a : ℝ) : Set ℝ := {y | ∃ x ≥ a, y = 2^x}

-- Define the complement of A
def complementA : Set ℝ := {x | x ∉ A}

-- Statement I
theorem complement_A_intersect_B_when_a_is_2 :
  (complementA ∩ B 2) = {x | x ≥ 4} := by sorry

-- Statement II
theorem complement_A_union_B_equals_reals_iff (a : ℝ) :
  (complementA ∪ B a) = Set.univ ↔ a ≤ 0 := by sorry

end complement_A_intersect_B_when_a_is_2_complement_A_union_B_equals_reals_iff_l954_95474


namespace fraction_equality_l954_95450

theorem fraction_equality (p q r s : ℝ) 
  (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 4) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -1 := by
sorry

end fraction_equality_l954_95450


namespace octagon_area_in_square_l954_95466

-- Define the square's perimeter
def square_perimeter : ℝ := 72

-- Define the number of parts each side is divided into
def parts_per_side : ℕ := 3

-- Theorem statement
theorem octagon_area_in_square (square_perimeter : ℝ) (parts_per_side : ℕ) :
  square_perimeter = 72 ∧ parts_per_side = 3 →
  let side_length := square_perimeter / 4
  let segment_length := side_length / parts_per_side
  let triangle_area := 1/2 * segment_length * segment_length
  let total_removed_area := 4 * triangle_area
  let square_area := side_length * side_length
  square_area - total_removed_area = 252 :=
by sorry

end octagon_area_in_square_l954_95466


namespace poly5_with_negative_integer_roots_l954_95420

/-- A polynomial of degree 5 with integer coefficients -/
structure Poly5 where
  p : ℤ
  q : ℤ
  r : ℤ
  s : ℤ
  t : ℤ

/-- The polynomial function corresponding to a Poly5 -/
def poly5_func (g : Poly5) : ℝ → ℝ :=
  λ x => x^5 + g.p * x^4 + g.q * x^3 + g.r * x^2 + g.s * x + g.t

/-- Predicate stating that all roots of a polynomial are negative integers -/
def all_roots_negative_integers (g : Poly5) : Prop :=
  ∀ x : ℝ, poly5_func g x = 0 → (∃ n : ℤ, x = -n ∧ n > 0)

theorem poly5_with_negative_integer_roots
  (g : Poly5)
  (h1 : all_roots_negative_integers g)
  (h2 : g.p + g.q + g.r + g.s + g.t = 3024) :
  g.t = 1600 := by
  sorry

end poly5_with_negative_integer_roots_l954_95420


namespace parabola_line_intersection_l954_95473

/-- Parabola C: y² = 4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Line: y = 2x - 4 -/
def line (x y : ℝ) : Prop := y = 2*x - 4

/-- Point A is on both the parabola and the line -/
def point_A (x y : ℝ) : Prop := parabola_C x y ∧ line x y

/-- Point B is on both the parabola and the line, and is distinct from A -/
def point_B (x y : ℝ) : Prop := parabola_C x y ∧ line x y ∧ (x, y) ≠ (x_A, y_A)
  where
  x_A : ℝ := sorry
  y_A : ℝ := sorry

/-- Point P is on the parabola C -/
def point_P (x y : ℝ) : Prop := parabola_C x y

/-- The area of triangle ABP is 12 -/
def triangle_area (x_A y_A x_B y_B x_P y_P : ℝ) : Prop :=
  abs ((x_A - x_P) * (y_B - y_P) - (x_B - x_P) * (y_A - y_P)) / 2 = 12

/-- The main theorem -/
theorem parabola_line_intersection
  (x_A y_A x_B y_B x_P y_P : ℝ)
  (hA : point_A x_A y_A)
  (hB : point_B x_B y_B)
  (hP : point_P x_P y_P)
  (hArea : triangle_area x_A y_A x_B y_B x_P y_P) :
  (((x_B - x_A)^2 + (y_B - y_A)^2)^(1/2 : ℝ) = 3 * 5^(1/2 : ℝ)) ∧
  ((x_P = 9 ∧ y_P = 6) ∨ (x_P = 4 ∧ y_P = -4)) :=
sorry

end parabola_line_intersection_l954_95473


namespace symmetric_line_equation_l954_95452

/-- A line in 2D space represented by its slope and y-intercept -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- The y-axis in 2D space -/
def yAxis : Set (ℝ × ℝ) := {p | p.1 = 0}

/-- Symmetry of a line with respect to the y-axis -/
def symmetricLine (l : Line) : Line :=
  { slope := -l.slope, intercept := l.intercept }

/-- The original line y = 2x + 1 -/
def originalLine : Line :=
  { slope := 2, intercept := 1 }

theorem symmetric_line_equation :
  symmetricLine originalLine = { slope := -2, intercept := 1 } := by
  sorry

end symmetric_line_equation_l954_95452


namespace arccos_one_over_sqrt_two_l954_95447

theorem arccos_one_over_sqrt_two (π : ℝ) : Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end arccos_one_over_sqrt_two_l954_95447


namespace age_ratio_in_two_years_l954_95423

def lennon_current_age : ℕ := 8
def ophelia_current_age : ℕ := 38
def years_passed : ℕ := 2

def lennon_future_age : ℕ := lennon_current_age + years_passed
def ophelia_future_age : ℕ := ophelia_current_age + years_passed

theorem age_ratio_in_two_years :
  ophelia_future_age / lennon_future_age = 4 ∧ ophelia_future_age % lennon_future_age = 0 :=
by sorry

end age_ratio_in_two_years_l954_95423


namespace box_surface_area_l954_95444

/-- Calculates the surface area of the interior of a box formed by cutting out square corners from a rectangular sheet and folding the sides. -/
def interior_surface_area (sheet_length sheet_width corner_size : ℕ) : ℕ :=
  let remaining_area := sheet_length * sheet_width - 4 * (corner_size * corner_size)
  remaining_area

/-- The surface area of the interior of a box formed by cutting out 8-unit squares from the corners of a 40x50 unit sheet and folding the sides is 1744 square units. -/
theorem box_surface_area : interior_surface_area 40 50 8 = 1744 := by
  sorry

end box_surface_area_l954_95444


namespace tournament_result_l954_95433

/-- The number of athletes with k points after m rounds in a tournament with 2^n participants -/
def f (n m k : ℕ) : ℕ := 2^(n - m) * (m.choose k)

/-- The number of athletes with 4 points after 7 rounds in a tournament with 2^n + 6 participants -/
def athletes_with_four_points (n : ℕ) : ℕ := 35 * 2^(n - 7) + 2

theorem tournament_result (n : ℕ) (h : n > 7) :
  athletes_with_four_points n = f n 7 4 + 2 := by sorry

end tournament_result_l954_95433


namespace f_is_even_and_increasing_l954_95463

-- Define the function f(x) = |x| - 1
def f (x : ℝ) : ℝ := |x| - 1

-- State the theorem
theorem f_is_even_and_increasing :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
by sorry

end f_is_even_and_increasing_l954_95463


namespace car_transfer_equation_l954_95493

theorem car_transfer_equation (x : ℕ) : 
  (100 - x = 68 + x) ↔ 
  (∃ (team_a team_b : ℕ), 
    team_a = 100 ∧ 
    team_b = 68 ∧ 
    team_a - x = team_b + x) :=
sorry

end car_transfer_equation_l954_95493


namespace discounted_price_per_shirt_l954_95412

-- Define the given conditions
def number_of_shirts : ℕ := 3
def original_total_price : ℚ := 60
def discount_percentage : ℚ := 40

-- Define the theorem
theorem discounted_price_per_shirt :
  let discount_amount : ℚ := (discount_percentage / 100) * original_total_price
  let sale_price : ℚ := original_total_price - discount_amount
  let price_per_shirt : ℚ := sale_price / number_of_shirts
  price_per_shirt = 12 := by sorry

end discounted_price_per_shirt_l954_95412


namespace right_triangle_arctan_sum_l954_95486

theorem right_triangle_arctan_sum (a b c : ℝ) (h : 0 < a ∧ 0 < b ∧ 0 < c) 
  (right_angle : a^2 + b^2 = c^2) : 
  Real.arctan (a / (b + c)) + Real.arctan (b / (a + c)) = π / 4 := by
  sorry

end right_triangle_arctan_sum_l954_95486


namespace neighbor_rolls_l954_95428

def total_rolls : ℕ := 12
def grandmother_rolls : ℕ := 3
def uncle_rolls : ℕ := 4
def rolls_left : ℕ := 2

theorem neighbor_rolls : 
  total_rolls - grandmother_rolls - uncle_rolls - rolls_left = 3 := by
  sorry

end neighbor_rolls_l954_95428


namespace condition_sufficient_not_necessary_l954_95434

theorem condition_sufficient_not_necessary :
  (∃ x y : ℝ, x = 1 ∧ y = -1 → x * y = -1) ∧
  ¬(∀ x y : ℝ, x * y = -1 → x = 1 ∧ y = -1) :=
sorry

end condition_sufficient_not_necessary_l954_95434


namespace g_composition_equals_49_l954_95472

def g (n : ℕ) : ℕ :=
  if n < 5 then n^2 + 1 else 2*n + 3

theorem g_composition_equals_49 : g (g (g 3)) = 49 := by
  sorry

end g_composition_equals_49_l954_95472


namespace x_one_minus_f_equals_one_l954_95421

theorem x_one_minus_f_equals_one :
  let α : ℝ := 3 + 2 * Real.sqrt 2
  let x : ℝ := α ^ 50
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by sorry

end x_one_minus_f_equals_one_l954_95421


namespace gcf_lcm_sum_l954_95470

def numbers : List Nat := [18, 24, 36]

theorem gcf_lcm_sum (A B : Nat) 
  (h1 : A = Nat.gcd 18 (Nat.gcd 24 36))
  (h2 : B = Nat.lcm 18 (Nat.lcm 24 36)) : 
  A + B = 78 := by
  sorry

end gcf_lcm_sum_l954_95470


namespace xy_value_l954_95491

theorem xy_value (x y : ℝ) 
  (h : (x / (1 - Complex.I)) - (y / (1 - 2 * Complex.I)) = (5 : ℝ) / (1 - 3 * Complex.I)) : 
  x * y = 5 := by
  sorry

end xy_value_l954_95491


namespace product_of_values_l954_95427

theorem product_of_values (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x * y = 24 * Real.rpow 3 (1/3))
  (h2 : x * z = 40 * Real.rpow 3 (1/3))
  (h3 : y * z = 15 * Real.rpow 3 (1/3)) :
  x * y * z = 72 * Real.sqrt 2 := by
sorry

end product_of_values_l954_95427


namespace cafe_chairs_distribution_l954_95490

theorem cafe_chairs_distribution (indoor_tables outdoor_tables : ℕ) 
  (chairs_per_indoor_table : ℕ) (total_chairs : ℕ) : 
  indoor_tables = 9 → 
  outdoor_tables = 11 → 
  chairs_per_indoor_table = 10 → 
  total_chairs = 123 → 
  (total_chairs - indoor_tables * chairs_per_indoor_table) / outdoor_tables = 3 := by
sorry

end cafe_chairs_distribution_l954_95490


namespace base_conversion_1729_l954_95467

/-- Converts a natural number from base 10 to base 5 -/
def toBase5 (n : ℕ) : List ℕ :=
  sorry

/-- Converts a list of digits in base 5 to a natural number in base 10 -/
def fromBase5 (digits : List ℕ) : ℕ :=
  sorry

theorem base_conversion_1729 :
  toBase5 1729 = [2, 3, 4, 0, 4] ∧ fromBase5 [2, 3, 4, 0, 4] = 1729 :=
sorry

end base_conversion_1729_l954_95467


namespace sum_of_complex_magnitudes_l954_95477

theorem sum_of_complex_magnitudes : 
  Complex.abs (3 - 5*I) + Complex.abs (3 + 5*I) + Complex.abs (6 - 8*I) = 2 * Real.sqrt 34 + 10 := by
  sorry

end sum_of_complex_magnitudes_l954_95477


namespace max_sequence_length_l954_95425

def is_valid_sequence (a : ℕ → ℝ) (n : ℕ) : Prop :=
  (∀ i : ℕ, i + 4 < n → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4)) < 0) ∧
  (∀ i : ℕ, i + 8 < n → (a i + a (i+1) + a (i+2) + a (i+3) + a (i+4) + a (i+5) + a (i+6) + a (i+7) + a (i+8)) > 0)

theorem max_sequence_length :
  (∃ (a : ℕ → ℝ), is_valid_sequence a 12) ∧
  (∀ (a : ℕ → ℝ) (n : ℕ), n > 12 → ¬ is_valid_sequence a n) :=
sorry

end max_sequence_length_l954_95425


namespace decagon_triangle_probability_l954_95426

/-- The number of vertices in a regular decagon -/
def n : ℕ := 10

/-- The total number of possible triangles formed by choosing 3 vertices from n vertices -/
def total_triangles : ℕ := n.choose 3

/-- The number of triangles with exactly one side being a side of the decagon -/
def one_side_triangles : ℕ := n * (n - 4)

/-- The number of triangles with two sides being sides of the decagon (i.e., formed by three consecutive vertices) -/
def two_side_triangles : ℕ := n

/-- The total number of favorable outcomes (triangles with at least one side being a side of the decagon) -/
def favorable_outcomes : ℕ := one_side_triangles + two_side_triangles

/-- The probability of forming a triangle with at least one side being a side of the decagon -/
def probability : ℚ := favorable_outcomes / total_triangles

theorem decagon_triangle_probability : probability = 7 / 12 := by sorry

end decagon_triangle_probability_l954_95426


namespace quadratic_inequality_solution_set_l954_95465

def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h1 : a < 0)
  (h2 : quadratic_function a b c 2 = 0)
  (h3 : quadratic_function a b c (-1) = 0) :
  {x : ℝ | quadratic_function a b c x ≥ 0} = Set.Icc (-1) 2 := by
sorry

end quadratic_inequality_solution_set_l954_95465


namespace parabola_intersection_l954_95432

/-- Two parabolas intersect at exactly two points -/
theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 12 * x - 5
  let g (x : ℝ) := x^2 - 2 * x + 3
  ∃! (s : Set (ℝ × ℝ)), s = {(1, -14), (4, -5)} ∧ 
    ∀ (x y : ℝ), (x, y) ∈ s ↔ f x = g x ∧ y = f x := by
  sorry

end parabola_intersection_l954_95432


namespace equation_solution_l954_95497

theorem equation_solution (x : ℝ) (h1 : 0 < x) (h2 : x < 12) (h3 : x ≠ 1) :
  (1 + 2 * Real.log 2 / Real.log 9) / (Real.log x / Real.log 9) - 1 = 
  2 * (Real.log 3 / Real.log x) * (Real.log (12 - x) / Real.log 9) → x = 6 :=
by sorry

end equation_solution_l954_95497


namespace units_digit_of_expression_l954_95441

theorem units_digit_of_expression : ∃ n : ℕ, (9 * 19 * 1989 - 9^3) % 10 = 0 := by
  sorry

end units_digit_of_expression_l954_95441


namespace theater_seats_l954_95482

/-- Represents a theater with an arithmetic progression of seats per row -/
structure Theater where
  first_row_seats : ℕ
  seat_increment : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater -/
def total_seats (t : Theater) : ℕ :=
  let n := (t.last_row_seats - t.first_row_seats) / t.seat_increment + 1
  n * (t.first_row_seats + t.last_row_seats) / 2

/-- Theorem stating that a theater with given conditions has 570 seats -/
theorem theater_seats :
  ∃ (t : Theater), t.first_row_seats = 12 ∧ t.seat_increment = 2 ∧ t.last_row_seats = 48 ∧ total_seats t = 570 :=
by
  sorry


end theater_seats_l954_95482


namespace polynomial_factor_theorem_l954_95458

theorem polynomial_factor_theorem (a : ℝ) : 
  (∃ b : ℝ, ∀ y : ℝ, y^2 + 3*y - a = (y - 3) * (y + b)) → a = 18 := by
  sorry

end polynomial_factor_theorem_l954_95458


namespace factorial_fraction_l954_95414

theorem factorial_fraction (N : ℕ) : 
  (Nat.factorial (N + 1) * (N + 2)) / Nat.factorial (N + 3) = 1 / (N + 3) := by
  sorry

end factorial_fraction_l954_95414


namespace cube_difference_l954_95480

theorem cube_difference (a b : ℝ) (h1 : a - b = 6) (h2 : a^2 + b^2 = 66) :
  a^3 - b^3 = 486 := by sorry

end cube_difference_l954_95480


namespace diorama_time_proof_l954_95416

/-- Proves that the total time spent on a diorama is 67 minutes, given the specified conditions. -/
theorem diorama_time_proof (planning_time building_time : ℕ) : 
  building_time = 3 * planning_time - 5 →
  building_time = 49 →
  planning_time + building_time = 67 := by
  sorry

#check diorama_time_proof

end diorama_time_proof_l954_95416


namespace table_football_points_l954_95437

/-- The total points scored by four friends in table football games -/
def total_points (darius matt marius sofia : ℕ) : ℕ :=
  darius + matt + marius + sofia

/-- Theorem stating the total points scored by the four friends -/
theorem table_football_points : ∃ (darius matt marius sofia : ℕ),
  darius = 10 ∧
  marius = darius + 3 ∧
  matt = darius + 5 ∧
  sofia = 2 * matt ∧
  total_points darius matt marius sofia = 68 := by
  sorry


end table_football_points_l954_95437


namespace f_min_value_l954_95448

/-- The function f(x) = x^2 + 26x + 7 -/
def f (x : ℝ) : ℝ := x^2 + 26*x + 7

/-- The minimum value of f(x) is -162 -/
theorem f_min_value : ∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (x₀ : ℝ), f x₀ = m ∧ m = -162 := by
  sorry

end f_min_value_l954_95448


namespace calculate_expression_l954_95492

theorem calculate_expression : 2 * (-3)^2 - 4 * (-3) - 15 = 15 := by
  sorry

end calculate_expression_l954_95492


namespace parallelogram_point_D_l954_95476

/-- A parallelogram in the complex plane -/
structure ComplexParallelogram where
  A : ℂ
  B : ℂ
  C : ℂ
  D : ℂ
  is_parallelogram : (B - A) = (C - D)

/-- Theorem: Given a parallelogram ABCD in the complex plane with A = 1+3i, B = 2-i, and C = -3+i, then D = -4+5i -/
theorem parallelogram_point_D (ABCD : ComplexParallelogram) 
  (hA : ABCD.A = 1 + 3*I)
  (hB : ABCD.B = 2 - I)
  (hC : ABCD.C = -3 + I) :
  ABCD.D = -4 + 5*I :=
sorry

end parallelogram_point_D_l954_95476


namespace junk_mail_for_block_l954_95495

/-- Given a block with houses and junk mail distribution, calculate the total junk mail for the block. -/
def total_junk_mail (num_houses : ℕ) (pieces_per_house : ℕ) : ℕ :=
  num_houses * pieces_per_house

/-- Theorem: The total junk mail for a block with 6 houses, each receiving 4 pieces, is 24. -/
theorem junk_mail_for_block :
  total_junk_mail 6 4 = 24 := by
  sorry

end junk_mail_for_block_l954_95495


namespace plane_division_by_lines_l954_95402

/-- The number of regions created by n non-parallel lines in a plane --/
def num_regions (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- The number of infinite regions created by n non-parallel lines in a plane --/
def num_infinite_regions (n : ℕ) : ℕ := 2 * n

theorem plane_division_by_lines (n : ℕ) (h : n = 20) :
  num_regions n = 211 ∧ num_regions n - num_infinite_regions n = 171 :=
sorry

end plane_division_by_lines_l954_95402


namespace solution_set_for_a_equals_one_f_b_geq_f_a_l954_95479

-- Define the function f
def f (a x : ℝ) : ℝ := |x - 2*a| + |x - a|

-- Theorem for part I
theorem solution_set_for_a_equals_one :
  {x : ℝ | f 1 x > 3} = {x : ℝ | x < 0 ∨ x > 3} := by sorry

-- Theorem for part II
theorem f_b_geq_f_a (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  f a b ≥ f a a ∧
  (f a b = f a a ↔ (2*a - b) * (b - a) ≥ 0) := by sorry

end solution_set_for_a_equals_one_f_b_geq_f_a_l954_95479


namespace smallest_dual_base_representation_l954_95410

/-- 
Given two bases a and b, both greater than 2, 
this function returns the base-10 representation of 21 in base a
-/
def base_a_to_10 (a : ℕ) : ℕ := 2 * a + 1

/-- 
Given two bases a and b, both greater than 2, 
this function returns the base-10 representation of 12 in base b
-/
def base_b_to_10 (b : ℕ) : ℕ := b + 2

/--
This theorem states that 7 is the smallest base-10 integer that can be represented
as 21_a in one base and 12_b in a different base, where a and b are any bases larger than 2.
-/
theorem smallest_dual_base_representation :
  ∀ a b : ℕ, a > 2 → b > 2 →
  (∃ n : ℕ, n < 7 ∧ base_a_to_10 a = n ∧ base_b_to_10 b = n) → False :=
by sorry

end smallest_dual_base_representation_l954_95410


namespace solve_equation_l954_95443

theorem solve_equation : ∀ x : ℝ, 3 * x + 4 = x + 2 → x = -1 := by
  sorry

end solve_equation_l954_95443


namespace circle_area_ratio_l954_95453

/-- If an arc of 60° on circle X has the same length as an arc of 40° on circle Y,
    then the ratio of the area of circle X to the area of circle Y is 4/9. -/
theorem circle_area_ratio (X Y : Real) (hX : X > 0) (hY : Y > 0) :
  (60 / 360) * (2 * Real.pi * X) = (40 / 360) * (2 * Real.pi * Y) →
  (Real.pi * X^2) / (Real.pi * Y^2) = 4 / 9 := by
sorry

end circle_area_ratio_l954_95453


namespace no_solution_exponential_equation_l954_95417

theorem no_solution_exponential_equation :
  ¬∃ y : ℝ, (16 : ℝ)^(3*y - 6) = (64 : ℝ)^(2*y + 1) := by
  sorry

end no_solution_exponential_equation_l954_95417


namespace sin_plus_cos_shift_l954_95498

theorem sin_plus_cos_shift (x : ℝ) :
  Real.sin (3 * x) + Real.cos (3 * x) = Real.sqrt 2 * Real.sin (3 * x + π / 4) := by
  sorry

end sin_plus_cos_shift_l954_95498


namespace people_counting_l954_95475

theorem people_counting (first_day second_day : ℕ) : 
  first_day = 2 * second_day →
  first_day + second_day = 1500 →
  second_day = 500 := by
sorry

end people_counting_l954_95475


namespace max_volume_cone_l954_95401

/-- Given a right-angled triangle with hypotenuse c, the triangle that forms a cone
    with maximum volume when rotated around one of its legs has the following properties: -/
theorem max_volume_cone (c : ℝ) (h : c > 0) :
  ∃ (x y : ℝ),
    -- The triangle is right-angled
    x^2 + y^2 = c^2 ∧
    -- x and y are positive
    x > 0 ∧ y > 0 ∧
    -- y is the optimal radius of the cone's base
    y = c * Real.sqrt (2/3) ∧
    -- x is the optimal height of the cone
    x = c / Real.sqrt 3 ∧
    -- The volume formed by this triangle is maximum
    ∀ (x' y' : ℝ), x'^2 + y'^2 = c^2 → x' > 0 → y' > 0 →
      (1/3) * π * y'^2 * x' ≤ (1/3) * π * y^2 * x ∧
    -- The maximum volume is (2 * π * √3 * c^3) / 27
    (1/3) * π * y^2 * x = (2 * π * Real.sqrt 3 * c^3) / 27 :=
by sorry

end max_volume_cone_l954_95401
