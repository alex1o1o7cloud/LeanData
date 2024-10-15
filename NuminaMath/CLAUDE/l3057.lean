import Mathlib

namespace NUMINAMATH_CALUDE_product_of_solutions_eq_neg_nine_l3057_305750

theorem product_of_solutions_eq_neg_nine :
  ∃ (z₁ z₂ : ℂ), z₁ ≠ z₂ ∧ 
  (Complex.abs z₁ = 3 * (Complex.abs z₁ - 2)) ∧
  (Complex.abs z₂ = 3 * (Complex.abs z₂ - 2)) ∧
  (z₁ * z₂ = -9) := by
sorry

end NUMINAMATH_CALUDE_product_of_solutions_eq_neg_nine_l3057_305750


namespace NUMINAMATH_CALUDE_white_marbles_count_l3057_305747

-- Define the parameters
def total_marbles : ℕ := 60
def blue_marbles : ℕ := 5
def red_marbles : ℕ := 9
def prob_red_or_white : ℚ := 55 / 60

-- Theorem statement
theorem white_marbles_count :
  ∃ (white_marbles : ℕ),
    white_marbles = total_marbles - blue_marbles - red_marbles ∧
    (red_marbles + white_marbles : ℚ) / total_marbles = prob_red_or_white ∧
    white_marbles = 46 := by
  sorry

end NUMINAMATH_CALUDE_white_marbles_count_l3057_305747


namespace NUMINAMATH_CALUDE_plane_equation_proof_l3057_305718

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane equation in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point lies on a plane -/
def pointLiesOnPlane (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- Check if two planes are perpendicular -/
def planesArePerpendicular (eq1 eq2 : PlaneEquation) : Prop :=
  eq1.A * eq2.A + eq1.B * eq2.B + eq1.C * eq2.C = 0

/-- The greatest common divisor of the absolute values of four integers is 1 -/
def gcdOfFourIntsIsOne (a b c d : ℤ) : Prop :=
  Nat.gcd (Nat.gcd (Int.natAbs a) (Int.natAbs b)) (Nat.gcd (Int.natAbs c) (Int.natAbs d)) = 1

theorem plane_equation_proof (p1 p2 : Point3D) (givenPlane : PlaneEquation) 
    (h1 : p1 = ⟨2, -3, 4⟩) 
    (h2 : p2 = ⟨-1, 3, -2⟩)
    (h3 : givenPlane = ⟨3, -2, 1, -7⟩) :
  ∃ (resultPlane : PlaneEquation), 
    resultPlane.A > 0 ∧ 
    gcdOfFourIntsIsOne resultPlane.A resultPlane.B resultPlane.C resultPlane.D ∧
    pointLiesOnPlane p1 resultPlane ∧
    pointLiesOnPlane p2 resultPlane ∧
    planesArePerpendicular resultPlane givenPlane ∧
    resultPlane = ⟨2, 5, -4, 27⟩ := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l3057_305718


namespace NUMINAMATH_CALUDE_parallel_vectors_trig_expression_l3057_305775

/-- Given two vectors a and b in R², prove that if they are parallel,
    then a specific trigonometric expression involving their components equals 1/3. -/
theorem parallel_vectors_trig_expression (α : ℝ) :
  let a : ℝ × ℝ := (1, Real.sin α)
  let b : ℝ × ℝ := (2, Real.cos α)
  (∃ (k : ℝ), a = k • b) →
  (Real.cos α - Real.sin α) / (2 * Real.cos (-α) - Real.sin α) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_trig_expression_l3057_305775


namespace NUMINAMATH_CALUDE_classroom_handshakes_l3057_305749

theorem classroom_handshakes (m n : ℕ) (h1 : m ≥ 3) (h2 : n ≥ 3) 
  (h3 : 2 * m * n - m - n = 252) : m * n = 72 := by
  sorry

end NUMINAMATH_CALUDE_classroom_handshakes_l3057_305749


namespace NUMINAMATH_CALUDE_unique_integer_solution_l3057_305765

theorem unique_integer_solution : ∃! x : ℤ, 
  (((2 * x > 70) ∧ (x < 100)) ∨ 
   ((2 * x > 70) ∧ (4 * x > 25)) ∨ 
   ((2 * x > 70) ∧ (x > 5)) ∨ 
   ((x < 100) ∧ (4 * x > 25)) ∨ 
   ((x < 100) ∧ (x > 5)) ∨ 
   ((4 * x > 25) ∧ (x > 5))) ∧
  (((2 * x ≤ 70) ∧ (x ≥ 100)) ∨ 
   ((2 * x ≤ 70) ∧ (4 * x ≤ 25)) ∨ 
   ((2 * x ≤ 70) ∧ (x ≤ 5)) ∨ 
   ((x ≥ 100) ∧ (4 * x ≤ 25)) ∨ 
   ((x ≥ 100) ∧ (x ≤ 5)) ∨ 
   ((4 * x ≤ 25) ∧ (x ≤ 5))) ∧
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l3057_305765


namespace NUMINAMATH_CALUDE_train_length_l3057_305729

/-- The length of a train given its speed and time to cross an overbridge -/
theorem train_length (speed : ℝ) (time : ℝ) (bridge_length : ℝ) :
  speed = 36 * 1000 / 3600 →
  time = 70 →
  bridge_length = 100 →
  speed * time - bridge_length = 600 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l3057_305729


namespace NUMINAMATH_CALUDE_total_weight_of_balls_l3057_305702

theorem total_weight_of_balls (blue_weight brown_weight : ℝ) :
  blue_weight = 6 → brown_weight = 3.12 →
  blue_weight + brown_weight = 9.12 := by
  sorry

end NUMINAMATH_CALUDE_total_weight_of_balls_l3057_305702


namespace NUMINAMATH_CALUDE_triangle_not_acute_l3057_305709

theorem triangle_not_acute (A B C : ℝ) (h1 : A + B + C = 180) (h2 : A = 30) (h3 : B = 50) :
  ¬ (A < 90 ∧ B < 90 ∧ C < 90) :=
sorry

end NUMINAMATH_CALUDE_triangle_not_acute_l3057_305709


namespace NUMINAMATH_CALUDE_rectangle_perimeter_l3057_305735

theorem rectangle_perimeter (L B : ℝ) (h1 : L - B = 23) (h2 : L * B = 2030) :
  2 * (L + B) = 186 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_perimeter_l3057_305735


namespace NUMINAMATH_CALUDE_female_workers_count_l3057_305777

/-- Represents the number of workers of each type and their wages --/
structure WorkforceData where
  male_workers : ℕ
  child_workers : ℕ
  male_wage : ℕ
  female_wage : ℕ
  child_wage : ℕ
  average_wage : ℕ

/-- Calculates the number of female workers based on the given workforce data --/
def calculate_female_workers (data : WorkforceData) : ℕ :=
  sorry

/-- Theorem stating that the number of female workers is 15 --/
theorem female_workers_count (data : WorkforceData) 
  (h1 : data.male_workers = 20)
  (h2 : data.child_workers = 5)
  (h3 : data.male_wage = 35)
  (h4 : data.female_wage = 20)
  (h5 : data.child_wage = 8)
  (h6 : data.average_wage = 26) :
  calculate_female_workers data = 15 := by
  sorry

end NUMINAMATH_CALUDE_female_workers_count_l3057_305777


namespace NUMINAMATH_CALUDE_complement_of_P_relative_to_U_l3057_305706

def U : Set ℤ := {-1, 0, 1, 2, 3}
def P : Set ℤ := {-1, 2, 3}

theorem complement_of_P_relative_to_U :
  {x ∈ U | x ∉ P} = {0, 1} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_relative_to_U_l3057_305706


namespace NUMINAMATH_CALUDE_worker_y_fraction_l3057_305727

theorem worker_y_fraction (P : ℝ) (Px Py : ℝ) (h1 : P > 0) (h2 : Px ≥ 0) (h3 : Py ≥ 0) :
  Px + Py = P →
  0.005 * Px + 0.008 * Py = 0.007 * P →
  Py / P = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_worker_y_fraction_l3057_305727


namespace NUMINAMATH_CALUDE_new_person_weight_is_77_l3057_305786

/-- The weight of the new person given the conditions of the problem -/
def weight_of_new_person (total_persons : ℕ) (average_weight_increase : ℝ) (replaced_person_weight : ℝ) : ℝ :=
  replaced_person_weight + total_persons * average_weight_increase

/-- Theorem stating that the weight of the new person is 77 kg given the problem conditions -/
theorem new_person_weight_is_77 :
  weight_of_new_person 8 1.5 65 = 77 := by
  sorry

#eval weight_of_new_person 8 1.5 65

end NUMINAMATH_CALUDE_new_person_weight_is_77_l3057_305786


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3057_305764

theorem quadratic_inequality (x : ℝ) (h : x ∈ Set.Icc 0 1) :
  |x^2 - x + 1/8| ≤ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3057_305764


namespace NUMINAMATH_CALUDE_legoland_kangaroos_l3057_305713

theorem legoland_kangaroos (koalas kangaroos : ℕ) : 
  kangaroos = 5 * koalas →
  koalas + kangaroos = 216 →
  kangaroos = 180 := by
sorry

end NUMINAMATH_CALUDE_legoland_kangaroos_l3057_305713


namespace NUMINAMATH_CALUDE_no_integer_b_with_four_integer_solutions_l3057_305788

theorem no_integer_b_with_four_integer_solutions : 
  ¬ ∃ b : ℤ, ∃ x₁ x₂ x₃ x₄ : ℤ, 
    (∀ x : ℤ, x^2 + b*x + 1 ≤ 0 ↔ (x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)) ∧
    (x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₁ ≠ x₄ ∧ x₂ ≠ x₃ ∧ x₂ ≠ x₄ ∧ x₃ ≠ x₄) :=
by sorry

end NUMINAMATH_CALUDE_no_integer_b_with_four_integer_solutions_l3057_305788


namespace NUMINAMATH_CALUDE_arithmetic_mean_difference_l3057_305772

/-- Given that the arithmetic mean of p and q is 10 and the arithmetic mean of q and r is 25,
    prove that r - p = 30 -/
theorem arithmetic_mean_difference (p q r : ℝ) 
  (h1 : (p + q) / 2 = 10) 
  (h2 : (q + r) / 2 = 25) : 
  r - p = 30 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_difference_l3057_305772


namespace NUMINAMATH_CALUDE_base_4_9_digit_difference_l3057_305796

theorem base_4_9_digit_difference (n : ℕ) (h : n = 1024) : 
  (Nat.log 4 n + 1) - (Nat.log 9 n + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_base_4_9_digit_difference_l3057_305796


namespace NUMINAMATH_CALUDE_inequalities_not_necessarily_true_l3057_305714

theorem inequalities_not_necessarily_true
  (x y a b : ℝ)
  (hx : x ≠ 0)
  (hy : y ≠ 0)
  (ha : a ≠ 0)
  (hb : b ≠ 0)
  (hxa : x < a)
  (hyb : y ≠ b) :
  ∃ (x' y' a' b' : ℝ),
    x' ≠ 0 ∧ y' ≠ 0 ∧ a' ≠ 0 ∧ b' ≠ 0 ∧
    x' < a' ∧ y' ≠ b' ∧
    ¬(x' + y' < a' + b') ∧
    ¬(x' - y' < a' - b') ∧
    ¬(x' * y' < a' * b') ∧
    ¬(x' / y' < a' / b') :=
by sorry

end NUMINAMATH_CALUDE_inequalities_not_necessarily_true_l3057_305714


namespace NUMINAMATH_CALUDE_continued_fraction_theorem_l3057_305700

-- Define the continued fraction for part 1
def continued_fraction_1 : ℚ :=
  1 + 1 / (2 + 1 / (3 + 1 / 4))

-- Define the continued fraction for part 2
def continued_fraction_2 (a b c : ℕ) : ℚ :=
  a + 1 / (b + 1 / c)

-- Define the equation for part 3
def continued_fraction_equation (y : ℝ) : Prop :=
  y = 8 + 1 / y

theorem continued_fraction_theorem :
  (continued_fraction_1 = 43 / 30) ∧
  (355 / 113 = continued_fraction_2 3 7 16) ∧
  (∃ y : ℝ, continued_fraction_equation y ∧ y = 4 + Real.sqrt 17) :=
by sorry

end NUMINAMATH_CALUDE_continued_fraction_theorem_l3057_305700


namespace NUMINAMATH_CALUDE_weight_replacement_l3057_305711

theorem weight_replacement (n : ℕ) (new_weight avg_increase : ℝ) 
  (h1 : n = 8)
  (h2 : new_weight = 81)
  (h3 : avg_increase = 2) :
  let total_increase := n * avg_increase
  let replaced_weight := new_weight - total_increase
  replaced_weight = 65 := by sorry

end NUMINAMATH_CALUDE_weight_replacement_l3057_305711


namespace NUMINAMATH_CALUDE_least_positive_integer_congruence_l3057_305721

theorem least_positive_integer_congruence :
  ∃ (x : ℕ), x > 0 ∧ (x + 7351 : ℤ) ≡ 3071 [ZMOD 17] ∧
  ∀ (y : ℕ), y > 0 ∧ (y + 7351 : ℤ) ≡ 3071 [ZMOD 17] → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_least_positive_integer_congruence_l3057_305721


namespace NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3057_305738

/-- The line y = mx + 2 is tangent to the ellipse x^2 + 9y^2 = 9 if and only if m^2 = 1/3 -/
theorem line_tangent_to_ellipse (m : ℝ) :
  (∀ x y : ℝ, y = m * x + 2 ∧ x^2 + 9 * y^2 = 9 → (∃! p : ℝ × ℝ, p.1^2 + 9 * p.2^2 = 9 ∧ p.2 = m * p.1 + 2)) ↔
  m^2 = 1/3 := by
sorry


end NUMINAMATH_CALUDE_line_tangent_to_ellipse_l3057_305738


namespace NUMINAMATH_CALUDE_loss_percentage_l3057_305717

/-- Calculate the percentage of loss given the cost price and selling price -/
theorem loss_percentage (cost_price selling_price : ℝ) : 
  cost_price = 750 → selling_price = 600 → 
  (cost_price - selling_price) / cost_price * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_loss_percentage_l3057_305717


namespace NUMINAMATH_CALUDE_exists_long_period_in_range_l3057_305754

/-- The length of the period of the decimal expansion of 1/n -/
def period_length (n : ℕ) : ℕ := sorry

theorem exists_long_period_in_range :
  ∀ (start : ℕ), 
  (10^99 ≤ start) →
  ∃ (n : ℕ), 
    (start ≤ n) ∧ 
    (n < start + 100000) ∧ 
    (period_length n > 2011) := by
  sorry

end NUMINAMATH_CALUDE_exists_long_period_in_range_l3057_305754


namespace NUMINAMATH_CALUDE_units_digit_of_product_l3057_305732

def product : ℕ := 1 * 3 * 5 * 79 * 97 * 113

theorem units_digit_of_product :
  (product % 10) = 5 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_product_l3057_305732


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3057_305707

/-- Given an ellipse with equation x²/a² + y²/4 = 1 and one focus at (2,0),
    prove that its eccentricity is √2/2 -/
theorem ellipse_eccentricity (a : ℝ) (h : a > 0) :
  let c := 2  -- distance from center to focus
  let b := 2  -- √4, as y²/4 = 1 in the equation
  let e := c / a  -- definition of eccentricity
  (∀ x y, x^2 / a^2 + y^2 / 4 = 1 → (x - c)^2 + y^2 = a^2) →  -- ellipse definition
  e = Real.sqrt 2 / 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3057_305707


namespace NUMINAMATH_CALUDE_quadratic_sum_l3057_305766

/-- Given a quadratic equation 100x^2 + 80x - 144 = 0, rewritten as (dx + e)^2 = f,
    where d, e, and f are integers and d > 0, prove that d + e + f = 174 -/
theorem quadratic_sum (d e f : ℤ) : 
  d > 0 → 
  (∀ x, 100 * x^2 + 80 * x - 144 = 0 ↔ (d * x + e)^2 = f) →
  d + e + f = 174 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3057_305766


namespace NUMINAMATH_CALUDE_domino_set_size_l3057_305705

theorem domino_set_size (num_players : ℕ) (dominoes_per_player : ℕ) 
  (h1 : num_players = 4) 
  (h2 : dominoes_per_player = 7) : 
  num_players * dominoes_per_player = 28 := by
  sorry

end NUMINAMATH_CALUDE_domino_set_size_l3057_305705


namespace NUMINAMATH_CALUDE_complex_pure_imaginary_a_equals_one_l3057_305758

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem complex_pure_imaginary_a_equals_one (a : ℝ) :
  is_pure_imaginary ((1 + a * Complex.I) / (1 - Complex.I)) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_pure_imaginary_a_equals_one_l3057_305758


namespace NUMINAMATH_CALUDE_a_4_equals_18_l3057_305769

def sequence_sum (a : ℕ → ℕ) (n : ℕ) : ℕ :=
  (List.range n).map a |>.sum

theorem a_4_equals_18 (a : ℕ → ℕ) (S : ℕ → ℕ) :
  (∀ n : ℕ, S n = sequence_sum a n) →
  a 1 = 1 →
  (∀ n : ℕ+, a (n + 1) = 2 * S n) →
  a 4 = 18 := by
sorry

end NUMINAMATH_CALUDE_a_4_equals_18_l3057_305769


namespace NUMINAMATH_CALUDE_fraction_simplification_l3057_305789

theorem fraction_simplification (b c d x y : ℝ) :
  (c * x * (b^2 * x^3 + 3 * b^2 * y^3 + c^3 * y^3) + d * y * (b^2 * x^3 + 3 * c^3 * x^3 + c^3 * y^3)) / (c * x + d * y) =
  b^2 * x^3 + 3 * c^2 * x * y^3 + c^3 * y^3 :=
by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3057_305789


namespace NUMINAMATH_CALUDE_trucks_sold_l3057_305774

theorem trucks_sold (total : ℕ) (car_truck_diff : ℕ) (h1 : total = 69) (h2 : car_truck_diff = 27) :
  ∃ trucks : ℕ, trucks * 2 + car_truck_diff = total ∧ trucks = 21 :=
by sorry

end NUMINAMATH_CALUDE_trucks_sold_l3057_305774


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3057_305783

theorem partial_fraction_decomposition :
  ∀ x : ℝ, x ≠ 2 → x ≠ 4 →
  (8 * x + 1) / ((x - 4) * (x - 2)^2) =
  (33 / 4) / (x - 4) + (-19 / 4) / (x - 2) + (-17 / 2) / (x - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3057_305783


namespace NUMINAMATH_CALUDE_p_and_s_not_third_l3057_305704

-- Define the set of runners
inductive Runner : Type
  | P | Q | R | S | T | U

-- Define the finishing order relation
def finishes_before (x y : Runner) : Prop := sorry

-- Define the race conditions
axiom p_beats_q : finishes_before Runner.P Runner.Q
axiom p_beats_r : finishes_before Runner.P Runner.R
axiom q_beats_s : finishes_before Runner.Q Runner.S
axiom t_between_p_and_q : finishes_before Runner.P Runner.T ∧ finishes_before Runner.T Runner.Q
axiom u_after_r_before_t : finishes_before Runner.R Runner.U ∧ finishes_before Runner.U Runner.T

-- Define what it means to finish third
def finishes_third (x : Runner) : Prop :=
  ∃ (a b : Runner), (a ≠ x ∧ b ≠ x ∧ a ≠ b) ∧
    finishes_before a x ∧ finishes_before b x ∧
    ∀ y : Runner, y ≠ x → y ≠ a → y ≠ b → finishes_before x y

-- Theorem to prove
theorem p_and_s_not_third :
  ¬(finishes_third Runner.P) ∧ ¬(finishes_third Runner.S) :=
sorry

end NUMINAMATH_CALUDE_p_and_s_not_third_l3057_305704


namespace NUMINAMATH_CALUDE_dartboard_central_angle_l3057_305716

/-- The central angle of a region on a circular dartboard, given its probability -/
theorem dartboard_central_angle (probability : ℝ) (h : probability = 1 / 8) :
  probability * 360 = 45 := by
  sorry

end NUMINAMATH_CALUDE_dartboard_central_angle_l3057_305716


namespace NUMINAMATH_CALUDE_common_root_of_three_equations_l3057_305787

theorem common_root_of_three_equations (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) 
  (h_ab : ∃ x : ℝ, a * x^11 + b * x^4 + c = 0 ∧ b * x^11 + c * x^4 + a = 0)
  (h_bc : ∃ x : ℝ, b * x^11 + c * x^4 + a = 0 ∧ c * x^11 + a * x^4 + b = 0)
  (h_ca : ∃ x : ℝ, c * x^11 + a * x^4 + b = 0 ∧ a * x^11 + b * x^4 + c = 0) :
  a * 1^11 + b * 1^4 + c = 0 ∧ b * 1^11 + c * 1^4 + a = 0 ∧ c * 1^11 + a * 1^4 + b = 0 :=
by sorry

end NUMINAMATH_CALUDE_common_root_of_three_equations_l3057_305787


namespace NUMINAMATH_CALUDE_sector_area_l3057_305723

/-- The area of a circular sector with central angle π/3 and radius 3 is 3π/2 -/
theorem sector_area (θ : Real) (r : Real) (h1 : θ = π / 3) (h2 : r = 3) :
  (1 / 2) * θ * r^2 = 3 * π / 2 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l3057_305723


namespace NUMINAMATH_CALUDE_correlation_coefficient_relationship_l3057_305745

-- Define the data points
def x_data : List ℝ := [1, 2, 3, 4, 5]
def y_data : List ℝ := [3, 5.3, 6.9, 9.1, 10.8]
def U_data : List ℝ := [1, 2, 3, 4, 5]
def V_data : List ℝ := [12.7, 10.2, 7, 3.6, 1]

-- Define linear correlation coefficient
def linear_correlation_coefficient (x y : List ℝ) : ℝ := sorry

-- Define r₁ and r₂
def r₁ : ℝ := linear_correlation_coefficient x_data y_data
def r₂ : ℝ := linear_correlation_coefficient U_data V_data

-- Theorem to prove
theorem correlation_coefficient_relationship : r₂ < 0 ∧ 0 < r₁ := by
  sorry

end NUMINAMATH_CALUDE_correlation_coefficient_relationship_l3057_305745


namespace NUMINAMATH_CALUDE_incorrect_calculation_ratio_l3057_305793

theorem incorrect_calculation_ratio (N : ℝ) (h : N ≠ 0) : 
  (N * 16) / ((N / 16) / 8) = 2048 := by
sorry

end NUMINAMATH_CALUDE_incorrect_calculation_ratio_l3057_305793


namespace NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l3057_305782

/-- Given a hyperbola and a parabola with specific properties, prove the value of p -/
theorem hyperbola_parabola_intersection (p : ℝ) : 
  (∀ x y : ℝ, x^2 / 4 - y^2 / 12 = 1) →  -- Hyperbola equation
  (∀ x y : ℝ, x = 2 * p * y^2) →         -- Parabola equation
  (∃ e : ℝ, e = (4 : ℝ) / 2 ∧            -- Eccentricity of hyperbola
    (∀ y : ℝ, e = 2 * p * y^2)) →        -- Focus of parabola at (e, 0)
  p = 1 / 16 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_intersection_l3057_305782


namespace NUMINAMATH_CALUDE_magnitude_BD_l3057_305720

def A : ℂ := Complex.I
def B : ℂ := 1
def C : ℂ := 4 + 2 * Complex.I

def parallelogram_ABCD (A B C : ℂ) : Prop :=
  ∃ D : ℂ, (C - B) = (D - A) ∧ (D - C) = (B - A)

theorem magnitude_BD (D : ℂ) (h : parallelogram_ABCD A B C) : 
  Complex.abs (D - B) = Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_BD_l3057_305720


namespace NUMINAMATH_CALUDE_decimal_sum_l3057_305778

theorem decimal_sum : 0.5 + 0.035 + 0.0041 = 0.5391 := by
  sorry

end NUMINAMATH_CALUDE_decimal_sum_l3057_305778


namespace NUMINAMATH_CALUDE_min_seedlings_to_plant_l3057_305791

theorem min_seedlings_to_plant (min_survival : ℝ) (max_survival : ℝ) (target : ℕ) : 
  min_survival = 0.75 →
  max_survival = 0.8 →
  target = 1200 →
  ∃ n : ℕ, n ≥ 1500 ∧ ∀ m : ℕ, m < n → (m : ℝ) * max_survival < target := by
  sorry

end NUMINAMATH_CALUDE_min_seedlings_to_plant_l3057_305791


namespace NUMINAMATH_CALUDE_max_correct_answers_is_30_l3057_305719

/-- Represents the scoring system and results of a math contest. -/
structure ContestResult where
  total_questions : ℕ
  correct_points : ℤ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correct answers possible given a contest result. -/
def max_correct_answers (result : ContestResult) : ℕ :=
  sorry

/-- Theorem stating that for the given contest parameters, the maximum number of correct answers is 30. -/
theorem max_correct_answers_is_30 :
  let result : ContestResult := {
    total_questions := 50,
    correct_points := 5,
    incorrect_points := -2,
    total_score := 115
  }
  max_correct_answers result = 30 := by sorry

end NUMINAMATH_CALUDE_max_correct_answers_is_30_l3057_305719


namespace NUMINAMATH_CALUDE_specific_marathon_distance_l3057_305728

/-- A circular marathon with four checkpoints -/
structure CircularMarathon where
  /-- Number of checkpoints -/
  num_checkpoints : Nat
  /-- Distance from start to first checkpoint -/
  start_to_first : ℝ
  /-- Distance from last checkpoint to finish -/
  last_to_finish : ℝ
  /-- Distance between consecutive checkpoints -/
  checkpoint_distance : ℝ

/-- The total distance of the marathon -/
def marathon_distance (m : CircularMarathon) : ℝ :=
  m.start_to_first + 
  m.last_to_finish + 
  (m.num_checkpoints - 1 : ℝ) * m.checkpoint_distance

/-- Theorem stating the total distance of the specific marathon -/
theorem specific_marathon_distance : 
  ∀ (m : CircularMarathon), 
    m.num_checkpoints = 4 ∧ 
    m.start_to_first = 1 ∧ 
    m.last_to_finish = 1 ∧ 
    m.checkpoint_distance = 6 → 
    marathon_distance m = 20 := by
  sorry

end NUMINAMATH_CALUDE_specific_marathon_distance_l3057_305728


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3057_305759

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∀ n m : ℕ, a (n + 1) - a n = a (m + 1) - a m

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  arithmetic_sequence a → a 3 + a 7 = 37 → a 2 + a 4 + a 6 + a 8 = 74 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l3057_305759


namespace NUMINAMATH_CALUDE_number_comparison_l3057_305755

theorem number_comparison : 
  ((-3 : ℝ) < -2) ∧ 
  (|(-4 : ℝ)| ≥ -2) ∧ 
  ((0 : ℝ) ≥ -2) ∧ 
  (-(-2 : ℝ) ≥ -2) := by
  sorry

end NUMINAMATH_CALUDE_number_comparison_l3057_305755


namespace NUMINAMATH_CALUDE_perimeter_bisector_min_value_l3057_305792

/-- A line that always bisects the perimeter of a circle -/
structure PerimeterBisector where
  a : ℝ
  b : ℝ
  h1 : a > 0
  h2 : b > 0
  h3 : ∀ (x y : ℝ), a * x + b * y + 1 = 0 → x^2 + y^2 + 8*x + 2*y + 1 = 0 →
       ∃ (c : ℝ), c > 0 ∧ (x + 4)^2 + (y + 1)^2 = c^2 ∧ a * (-4) + b * (-1) + 1 = 0

/-- The minimum value of 1/a + 4/b for a perimeter bisector is 16 -/
theorem perimeter_bisector_min_value (pb : PerimeterBisector) :
  (1 / pb.a + 4 / pb.b) ≥ 16 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_bisector_min_value_l3057_305792


namespace NUMINAMATH_CALUDE_cube_of_complex_number_l3057_305744

/-- Given that z = sin(π/3) + i*cos(π/3), prove that z^3 = i -/
theorem cube_of_complex_number (z : ℂ) (h : z = Complex.exp (Complex.I * (π / 3))) :
  z^3 = Complex.I := by sorry

end NUMINAMATH_CALUDE_cube_of_complex_number_l3057_305744


namespace NUMINAMATH_CALUDE_orange_juice_price_l3057_305753

/-- The cost of a glass of orange juice -/
def orange_juice_cost : ℚ := 85/100

/-- The cost of a bagel -/
def bagel_cost : ℚ := 95/100

/-- The cost of a sandwich -/
def sandwich_cost : ℚ := 465/100

/-- The cost of milk -/
def milk_cost : ℚ := 115/100

/-- The additional amount spent on lunch compared to breakfast -/
def lunch_breakfast_difference : ℚ := 4

theorem orange_juice_price : 
  bagel_cost + orange_juice_cost + lunch_breakfast_difference = sandwich_cost + milk_cost := by
  sorry

end NUMINAMATH_CALUDE_orange_juice_price_l3057_305753


namespace NUMINAMATH_CALUDE_tan_value_from_trig_equation_l3057_305770

theorem tan_value_from_trig_equation (x : Real) 
  (h1 : 0 < x) (h2 : x < π/2) 
  (h3 : (Real.sin x)^4 / 9 + (Real.cos x)^4 / 4 = 1/13) : 
  Real.tan x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_tan_value_from_trig_equation_l3057_305770


namespace NUMINAMATH_CALUDE_alternatingArithmeticSequenceSum_l3057_305730

def alternatingArithmeticSequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : List ℤ :=
  List.range n |> List.map (λ i => a₁ + i * d * (if i % 2 = 0 then 1 else -1))

theorem alternatingArithmeticSequenceSum :
  let seq := alternatingArithmeticSequence 2 4 26
  seq.sum = -52 := by
  sorry

end NUMINAMATH_CALUDE_alternatingArithmeticSequenceSum_l3057_305730


namespace NUMINAMATH_CALUDE_basketball_free_throws_l3057_305742

theorem basketball_free_throws :
  ∀ (a b x : ℚ),
  3 * b = 2 * a →
  x = 2 * a - 2 →
  2 * a + 3 * b + x = 78 →
  x = 74 / 3 := by
sorry

end NUMINAMATH_CALUDE_basketball_free_throws_l3057_305742


namespace NUMINAMATH_CALUDE_parabola_intersection_theorem_l3057_305768

/-- Parabola E with parameter p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Line intersecting the parabola -/
structure IntersectingLine where
  k : ℝ

/-- Point on the parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Theorem about parabola intersection and slope relations -/
theorem parabola_intersection_theorem 
  (E : Parabola) 
  (L : IntersectingLine) 
  (A B : ParabolaPoint) 
  (h_on_parabola : A.x^2 = 2*E.p*A.y ∧ B.x^2 = 2*E.p*B.y)
  (h_on_line : A.y = L.k*A.x + 2 ∧ B.y = L.k*B.x + 2)
  (h_dot_product : A.x*B.x + A.y*B.y = 2) :
  (∃ (k₁ k₂ : ℝ), 
    k₁ = (A.y + 2) / A.x ∧ 
    k₂ = (B.y + 2) / B.x ∧ 
    k₁^2 + k₂^2 - 2*L.k^2 = 16) ∧
  E.p = 1/2 := by sorry

end NUMINAMATH_CALUDE_parabola_intersection_theorem_l3057_305768


namespace NUMINAMATH_CALUDE_roll_distribution_probability_l3057_305752

def total_rolls : ℕ := 9
def rolls_per_type : ℕ := 3
def num_guests : ℕ := 3

def total_arrangements : ℕ := (total_rolls.factorial) / ((rolls_per_type.factorial) ^ 3)

def favorable_outcomes : ℕ := (rolls_per_type.factorial) ^ num_guests

def probability : ℚ := favorable_outcomes / total_arrangements

theorem roll_distribution_probability :
  probability = 9 / 70 := by sorry

end NUMINAMATH_CALUDE_roll_distribution_probability_l3057_305752


namespace NUMINAMATH_CALUDE_x_greater_than_half_l3057_305762

theorem x_greater_than_half (x : ℝ) (h : (1/2) * x = 1) : 
  (x - 1/2) / (1/2) * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_x_greater_than_half_l3057_305762


namespace NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3057_305763

theorem consecutive_integers_product_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 272 → x + (x + 1) = 33 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_sum_l3057_305763


namespace NUMINAMATH_CALUDE_cloth_cost_per_metre_l3057_305722

theorem cloth_cost_per_metre (total_length : Real) (total_cost : Real) 
  (h1 : total_length = 9.25)
  (h2 : total_cost = 425.50) :
  total_cost / total_length = 46 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_per_metre_l3057_305722


namespace NUMINAMATH_CALUDE_total_statues_l3057_305757

/-- The length of the street in meters -/
def street_length : ℕ := 1650

/-- The interval between statues in meters -/
def statue_interval : ℕ := 50

/-- The number of sides of the street with statues -/
def sides : ℕ := 2

theorem total_statues : 
  (street_length / statue_interval + 1) * sides = 68 := by
  sorry

end NUMINAMATH_CALUDE_total_statues_l3057_305757


namespace NUMINAMATH_CALUDE_biology_homework_pages_l3057_305746

/-- The number of pages of math homework -/
def math_pages : ℕ := 8

/-- The total number of pages of math and biology homework -/
def total_math_biology_pages : ℕ := 11

/-- The number of pages of biology homework -/
def biology_pages : ℕ := total_math_biology_pages - math_pages

theorem biology_homework_pages : biology_pages = 3 := by
  sorry

end NUMINAMATH_CALUDE_biology_homework_pages_l3057_305746


namespace NUMINAMATH_CALUDE_bobbit_worm_aquarium_l3057_305797

def fish_count (initial_fish : ℕ) (daily_consumption : ℕ) (added_fish : ℕ) (days_before_adding : ℕ) (total_days : ℕ) : ℕ :=
  initial_fish - (daily_consumption * total_days) + added_fish

theorem bobbit_worm_aquarium (initial_fish : ℕ) (daily_consumption : ℕ) (added_fish : ℕ) (days_before_adding : ℕ) (total_days : ℕ)
  (h1 : initial_fish = 60)
  (h2 : daily_consumption = 2)
  (h3 : added_fish = 8)
  (h4 : days_before_adding = 14)
  (h5 : total_days = 21) :
  fish_count initial_fish daily_consumption added_fish days_before_adding total_days = 26 := by
  sorry

end NUMINAMATH_CALUDE_bobbit_worm_aquarium_l3057_305797


namespace NUMINAMATH_CALUDE_remainder_problem_l3057_305739

theorem remainder_problem (n m p : ℕ) 
  (hn : n % 4 = 3)
  (hm : m % 7 = 5)
  (hp : p % 5 = 2) :
  (7 * n + 3 * m - p) % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l3057_305739


namespace NUMINAMATH_CALUDE_angle_B_measure_l3057_305737

theorem angle_B_measure (a b c : ℝ) (A B C : ℝ) :
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  A = 60 * π / 180 →
  -- Sine Rule
  a / Real.sin A = b / Real.sin B →
  -- Triangle inequality (ensuring it's a valid triangle)
  a + b > c ∧ b + c > a ∧ c + a > b →
  -- Sum of angles in a triangle is π
  A + B + C = π →
  B = 45 * π / 180 := by sorry

end NUMINAMATH_CALUDE_angle_B_measure_l3057_305737


namespace NUMINAMATH_CALUDE_min_value_a_plus_b_l3057_305776

theorem min_value_a_plus_b (a b : ℝ) (ha : a > 1) (hb : b > 2) (hab : a * b = 2 * a + b) :
  (∀ x y : ℝ, x > 1 ∧ y > 2 ∧ x * y = 2 * x + y → a + b ≤ x + y) ∧ a + b = 2 * Real.sqrt 2 + 3 :=
sorry

end NUMINAMATH_CALUDE_min_value_a_plus_b_l3057_305776


namespace NUMINAMATH_CALUDE_reciprocal_opposite_square_minus_product_l3057_305701

theorem reciprocal_opposite_square_minus_product (a b c d : ℝ) 
  (h1 : a * b = 1) 
  (h2 : c + d = 0) : 
  (c + d)^2 - a * b = -1 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_opposite_square_minus_product_l3057_305701


namespace NUMINAMATH_CALUDE_smallest_positive_angle_same_terminal_side_l3057_305756

/-- Given an angle α = -3000°, this theorem states that the smallest positive angle
    with the same terminal side as α is 240°. -/
theorem smallest_positive_angle_same_terminal_side :
  let α : ℝ := -3000
  ∃ (k : ℤ), α + k * 360 = 240 ∧
    ∀ (m : ℤ), α + m * 360 > 0 → α + m * 360 ≥ 240 :=
by sorry

end NUMINAMATH_CALUDE_smallest_positive_angle_same_terminal_side_l3057_305756


namespace NUMINAMATH_CALUDE_zoo_recovery_time_l3057_305794

/-- The time spent recovering escaped animals from a zoo -/
theorem zoo_recovery_time 
  (lions : ℕ) 
  (rhinos : ℕ) 
  (recovery_time_per_animal : ℕ) 
  (h1 : lions = 3) 
  (h2 : rhinos = 2) 
  (h3 : recovery_time_per_animal = 2) : 
  (lions + rhinos) * recovery_time_per_animal = 10 := by
sorry

end NUMINAMATH_CALUDE_zoo_recovery_time_l3057_305794


namespace NUMINAMATH_CALUDE_trapezoid_division_theorem_l3057_305715

/-- A trapezoid with sides a, b, c, d where b is parallel to d -/
structure Trapezoid (α : Type*) [LinearOrderedField α] :=
  (a b c d : α)
  (parallel : b ≠ d)

/-- The ratio in which a line parallel to the bases divides a trapezoid -/
def divisionRatio {α : Type*} [LinearOrderedField α] (t : Trapezoid α) (z : α) : α :=
  (t.d + t.b) / 2 + (t.d - t.b)^2 / (2 * (t.a + t.c))

/-- The condition that two trapezoids formed by a parallel line have equal perimeters -/
def equalPerimeters {α : Type*} [LinearOrderedField α] (t : Trapezoid α) (z : α) : Prop :=
  t.a + z + t.c + (t.d - z) = t.b + z + t.a + (t.d - z)

theorem trapezoid_division_theorem {α : Type*} [LinearOrderedField α] (t : Trapezoid α) (z : α) :
  equalPerimeters t z → z = divisionRatio t z :=
sorry

end NUMINAMATH_CALUDE_trapezoid_division_theorem_l3057_305715


namespace NUMINAMATH_CALUDE_simplify_power_expression_l3057_305771

theorem simplify_power_expression (y : ℝ) : (3 * y^4)^5 = 243 * y^20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_expression_l3057_305771


namespace NUMINAMATH_CALUDE_base6_403_greater_than_base8_217_l3057_305767

/-- Converts a number from base 6 to decimal --/
def base6ToDecimal (n : ℕ) : ℕ := sorry

/-- Converts a number from base 8 to decimal --/
def base8ToDecimal (n : ℕ) : ℕ := sorry

theorem base6_403_greater_than_base8_217 :
  base6ToDecimal 403 > base8ToDecimal 217 := by sorry

end NUMINAMATH_CALUDE_base6_403_greater_than_base8_217_l3057_305767


namespace NUMINAMATH_CALUDE_initial_dimes_equation_l3057_305733

/-- The number of dimes Sam initially had -/
def initial_dimes : ℕ := sorry

/-- The number of dimes Sam gave away -/
def dimes_given_away : ℕ := 7

/-- The number of dimes Sam has left -/
def dimes_left : ℕ := 2

/-- Theorem: The initial number of dimes is equal to the sum of dimes given away and dimes left -/
theorem initial_dimes_equation : initial_dimes = dimes_given_away + dimes_left := by
  sorry

end NUMINAMATH_CALUDE_initial_dimes_equation_l3057_305733


namespace NUMINAMATH_CALUDE_inequality_proof_l3057_305724

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / c + c / b ≥ 4 * a / (a + b) ∧
  (a / c + c / b = 4 * a / (a + b) ↔ a = b ∧ b = c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l3057_305724


namespace NUMINAMATH_CALUDE_event_probability_l3057_305790

theorem event_probability (p : ℝ) :
  (0 ≤ p) ∧ (p ≤ 1) →
  (1 - (1 - p)^4 = 65/81) →
  p = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_event_probability_l3057_305790


namespace NUMINAMATH_CALUDE_polynomial_real_root_l3057_305703

theorem polynomial_real_root 
  (P : ℝ → ℝ) 
  (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h_nonzero : a₁ * a₂ * a₃ ≠ 0)
  (h_poly : ∀ x : ℝ, P (a₁ * x + b₁) + P (a₂ * x + b₂) = P (a₃ * x + b₃)) :
  ∃ r : ℝ, P r = 0 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_real_root_l3057_305703


namespace NUMINAMATH_CALUDE_shaded_area_semicircles_l3057_305731

/-- The area of the shaded region in the given semicircle configuration -/
theorem shaded_area_semicircles (r_ADB r_BEC : ℝ) (h_ADB : r_ADB = 2) (h_BEC : r_BEC = 3) : 
  let r_DFE := (r_ADB + r_BEC) / 2
  (π * r_ADB^2 / 2 + π * r_BEC^2 / 2) - (π * r_DFE^2 / 2) = 3.375 * π := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_semicircles_l3057_305731


namespace NUMINAMATH_CALUDE_inner_triangle_perimeter_is_180_l3057_305743

/-- Triangle DEF with given side lengths -/
structure Triangle :=
  (DE : ℝ)
  (EF : ℝ)
  (FD : ℝ)

/-- Parallel lines intersecting the triangle -/
structure ParallelLines :=
  (m_D : ℝ)
  (m_E : ℝ)
  (m_F : ℝ)

/-- The perimeter of the inner triangle formed by parallel lines -/
def inner_triangle_perimeter (t : Triangle) (p : ParallelLines) : ℝ :=
  p.m_D + p.m_E + p.m_F

/-- Theorem stating the perimeter of the inner triangle -/
theorem inner_triangle_perimeter_is_180 
  (t : Triangle) 
  (p : ParallelLines) 
  (h1 : t.DE = 140) 
  (h2 : t.EF = 260) 
  (h3 : t.FD = 200) 
  (h4 : p.m_D = 65) 
  (h5 : p.m_E = 85) 
  (h6 : p.m_F = 30) : 
  inner_triangle_perimeter t p = 180 := by
  sorry

#check inner_triangle_perimeter_is_180

end NUMINAMATH_CALUDE_inner_triangle_perimeter_is_180_l3057_305743


namespace NUMINAMATH_CALUDE_book_sale_revenue_l3057_305726

theorem book_sale_revenue (total_books : ℕ) (sold_fraction : ℚ) (price_per_book : ℚ) (unsold_books : ℕ) : 
  sold_fraction = 2 / 3 →
  price_per_book = 2 →
  unsold_books = 36 →
  unsold_books = (1 - sold_fraction) * total_books →
  sold_fraction * total_books * price_per_book = 144 := by
  sorry

#check book_sale_revenue

end NUMINAMATH_CALUDE_book_sale_revenue_l3057_305726


namespace NUMINAMATH_CALUDE_complementary_angles_l3057_305741

theorem complementary_angles (A B : ℝ) : 
  A + B = 90 →  -- angles are complementary
  A = 5 * B →   -- measure of A is 5 times B
  A = 75 :=     -- measure of A is 75 degrees
by
  sorry

end NUMINAMATH_CALUDE_complementary_angles_l3057_305741


namespace NUMINAMATH_CALUDE_fedya_deposit_l3057_305798

theorem fedya_deposit (n : ℕ) (X : ℕ) : 
  n < 30 →
  X * (100 - n) = 847 * 100 →
  X = 1100 := by
sorry

end NUMINAMATH_CALUDE_fedya_deposit_l3057_305798


namespace NUMINAMATH_CALUDE_swimmer_laps_theorem_l3057_305725

/-- Represents the number of laps swum by a person in a given number of weeks -/
def laps_swum (laps_per_day : ℕ) (days_per_week : ℕ) (weeks : ℕ) : ℕ :=
  laps_per_day * days_per_week * weeks

theorem swimmer_laps_theorem (x : ℕ) :
  laps_swum 12 5 x = 60 * x :=
by
  sorry

#check swimmer_laps_theorem

end NUMINAMATH_CALUDE_swimmer_laps_theorem_l3057_305725


namespace NUMINAMATH_CALUDE_triangle_problem_l3057_305710

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ A < π) (h3 : 0 < B ∧ B < π) (h4 : 0 < C ∧ C < π)
  (h5 : A + B + C = π) (h6 : b * (Real.cos C + Real.sin C) = a)
  (h7 : a * Real.sin B = b * Real.sin A) (h8 : b * Real.sin C = c * Real.sin B)
  (h9 : a * Real.sin C = c * Real.sin A) (h10 : a * (1/4) = a * Real.sin A * Real.sin C) :
  B = π/4 ∧ Real.cos A = -Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3057_305710


namespace NUMINAMATH_CALUDE_characterization_of_f_l3057_305734

-- Define the property for the function
def satisfies_property (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, (f x * f y) ∣ ((1 + 2 * x) * f y + (1 + 2 * y) * f x)

-- Define strictly increasing function
def strictly_increasing (f : ℕ → ℕ) : Prop :=
  ∀ x y : ℕ, x < y → f x < f y

-- Main theorem
theorem characterization_of_f :
  ∀ f : ℕ → ℕ, strictly_increasing f → satisfies_property f →
  (∀ x : ℕ, f x = 2 * x + 1) ∨ (∀ x : ℕ, f x = 4 * x + 2) :=
sorry

end NUMINAMATH_CALUDE_characterization_of_f_l3057_305734


namespace NUMINAMATH_CALUDE_equation_solutions_l3057_305780

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 6*x + 1 = 0 ↔ x = 3 + 2*Real.sqrt 2 ∨ x = 3 - 2*Real.sqrt 2) ∧
  (∀ x : ℝ, (2*x - 3)^2 = 5*(2*x - 3) ↔ x = 3/2 ∨ x = 4) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3057_305780


namespace NUMINAMATH_CALUDE_sarah_remaining_pages_l3057_305760

/-- Given the initial number of problems, the number of completed problems,
    and the number of problems per page, calculates the number of remaining pages. -/
def remaining_pages (initial_problems : ℕ) (completed_problems : ℕ) (problems_per_page : ℕ) : ℕ :=
  (initial_problems - completed_problems) / problems_per_page

/-- Proves that Sarah has 5 pages of problems left to do. -/
theorem sarah_remaining_pages :
  remaining_pages 60 20 8 = 5 := by
  sorry

#eval remaining_pages 60 20 8

end NUMINAMATH_CALUDE_sarah_remaining_pages_l3057_305760


namespace NUMINAMATH_CALUDE_probability_x_plus_y_less_than_4_l3057_305773

/-- A square in the 2D plane -/
structure Square where
  bottomLeft : ℝ × ℝ
  sideLength : ℝ

/-- The probability that a randomly chosen point in the square satisfies a condition -/
def probabilityInSquare (s : Square) (condition : ℝ × ℝ → Prop) : ℝ :=
  sorry

theorem probability_x_plus_y_less_than_4 :
  let s : Square := { bottomLeft := (0, 0), sideLength := 3 }
  probabilityInSquare s (fun (x, y) ↦ x + y < 4) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_x_plus_y_less_than_4_l3057_305773


namespace NUMINAMATH_CALUDE_initially_calculated_average_weight_l3057_305784

/-- Given a class of boys, prove that the initially calculated average weight
    is correct based on the given conditions. -/
theorem initially_calculated_average_weight
  (num_boys : ℕ)
  (correct_avg_weight : ℝ)
  (misread_weight : ℝ)
  (correct_weight : ℝ)
  (h1 : num_boys = 20)
  (h2 : correct_avg_weight = 58.6)
  (h3 : misread_weight = 56)
  (h4 : correct_weight = 60) :
  let correct_total_weight := correct_avg_weight * num_boys
  let initial_total_weight := correct_total_weight - (correct_weight - misread_weight)
  let initial_avg_weight := initial_total_weight / num_boys
  initial_avg_weight = 58.4 := by
sorry

end NUMINAMATH_CALUDE_initially_calculated_average_weight_l3057_305784


namespace NUMINAMATH_CALUDE_least_clock_equivalent_hour_l3057_305781

theorem least_clock_equivalent_hour : 
  ∃ (h : ℕ), h > 6 ∧ 
             h % 12 = (h^2) % 12 ∧ 
             h % 12 = (h^3) % 12 ∧ 
             (∀ (k : ℕ), k > 6 ∧ k < h → 
               (k % 12 ≠ (k^2) % 12 ∨ k % 12 ≠ (k^3) % 12)) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_least_clock_equivalent_hour_l3057_305781


namespace NUMINAMATH_CALUDE_stating_rhombus_solutions_count_l3057_305795

/-- Represents the number of solutions for inscribing a rhombus in a square and circumscribing it around a circle -/
inductive NumSolutions
  | two
  | one
  | zero

/-- 
  Given a square and a circle with the same center, determines the number of possible rhombuses 
  that can be inscribed in the square and circumscribed around the circle.
-/
def numRhombusSolutions (squareSide : ℝ) (circleRadius : ℝ) : NumSolutions :=
  sorry

/-- 
  Theorem stating that the number of rhombus solutions is either 2, 1, or 0
-/
theorem rhombus_solutions_count (squareSide : ℝ) (circleRadius : ℝ) :
  ∃ (n : NumSolutions), numRhombusSolutions squareSide circleRadius = n :=
  sorry

end NUMINAMATH_CALUDE_stating_rhombus_solutions_count_l3057_305795


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3057_305740

/-- Given an infinite geometric series with a specific pattern, prove the value of k that makes the series sum to 10 -/
theorem geometric_series_sum (k : ℝ) : 
  (∑' n : ℕ, (4 + n * k) / 5^n) = 10 → k = 19.2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3057_305740


namespace NUMINAMATH_CALUDE_power_function_through_point_l3057_305761

/-- If the point (√3/3, √3/9) lies on the graph of a power function f(x), then f(x) = x³ -/
theorem power_function_through_point (f : ℝ → ℝ) :
  (∃ α : ℝ, ∀ x : ℝ, f x = x^α) →
  f (Real.sqrt 3 / 3) = Real.sqrt 3 / 9 →
  ∀ x : ℝ, f x = x^3 := by sorry

end NUMINAMATH_CALUDE_power_function_through_point_l3057_305761


namespace NUMINAMATH_CALUDE_days_worked_together_l3057_305799

-- Define the total work as a positive real number
variable (W : ℝ) (hW : W > 0)

-- Define the time taken by a and b together to finish the work
def time_together : ℝ := 40

-- Define the time taken by a alone to finish the work
def time_a_alone : ℝ := 12

-- Define the additional time a worked after b left
def additional_time_a : ℝ := 9

-- Define the function to calculate the work done in a given time at a given rate
def work_done (time : ℝ) (rate : ℝ) : ℝ := time * rate

-- Define the theorem to prove
theorem days_worked_together (W : ℝ) (hW : W > 0) : 
  ∃ x : ℝ, x > 0 ∧ 
    work_done x (W / time_together) + 
    work_done additional_time_a (W / time_a_alone) = W ∧
    x = 10 := by
  sorry

end NUMINAMATH_CALUDE_days_worked_together_l3057_305799


namespace NUMINAMATH_CALUDE_seventh_term_value_l3057_305736

def sequence_with_sum_rule (a : ℕ → ℕ) : Prop :=
  a 1 = 5 ∧ a 4 = 13 ∧ a 6 = 40 ∧
  ∀ n ≥ 4, a n = a (n-3) + a (n-2) + a (n-1)

theorem seventh_term_value (a : ℕ → ℕ) (h : sequence_with_sum_rule a) : a 7 = 74 := by
  sorry

end NUMINAMATH_CALUDE_seventh_term_value_l3057_305736


namespace NUMINAMATH_CALUDE_cubic_sum_minus_product_l3057_305785

theorem cubic_sum_minus_product (a b c : ℝ) 
  (sum_condition : a + b + c = 13) 
  (product_sum_condition : a * b + a * c + b * c = 40) : 
  a^3 + b^3 + c^3 - 3*a*b*c = 637 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_minus_product_l3057_305785


namespace NUMINAMATH_CALUDE_equation_real_roots_l3057_305751

theorem equation_real_roots (a : ℝ) : 
  (∃ x : ℝ, 9^(-|x - 2|) - 4 * 3^(-|x - 2|) - a = 0) ↔ -3 ≤ a ∧ a < 0 := by
  sorry

end NUMINAMATH_CALUDE_equation_real_roots_l3057_305751


namespace NUMINAMATH_CALUDE_range_of_a_l3057_305779

theorem range_of_a (p : ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0) 
                   (q : ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) :
  a ≤ -2 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3057_305779


namespace NUMINAMATH_CALUDE_circle_equation_implies_expression_value_l3057_305708

theorem circle_equation_implies_expression_value (x y : ℝ) : 
  x^2 + y^2 = 1 → 
  Real.sqrt (x^2 - 4*x + 4) + Real.sqrt (x*y - 3*x + y - 3) = 3 := by
  sorry

end NUMINAMATH_CALUDE_circle_equation_implies_expression_value_l3057_305708


namespace NUMINAMATH_CALUDE_fathers_age_is_38_l3057_305712

/-- The age of the son 5 years ago -/
def sons_age_5_years_ago : ℕ := 14

/-- The current age of the son -/
def sons_current_age : ℕ := sons_age_5_years_ago + 5

/-- The age of the father when the son was born -/
def fathers_age_at_sons_birth : ℕ := sons_current_age

/-- The current age of the father -/
def fathers_current_age : ℕ := fathers_age_at_sons_birth + sons_current_age

theorem fathers_age_is_38 : fathers_current_age = 38 := by
  sorry

end NUMINAMATH_CALUDE_fathers_age_is_38_l3057_305712


namespace NUMINAMATH_CALUDE_problem_solution_l3057_305748

theorem problem_solution (x z : ℚ) : 
  x = 103 → x^3*z - 3*x^2*z + 2*x*z = 208170 → z = 5/265 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3057_305748
