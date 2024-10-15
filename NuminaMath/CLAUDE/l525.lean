import Mathlib

namespace NUMINAMATH_CALUDE_range_of_x_range_of_m_l525_52520

-- Define propositions p and q
def p (x m : ℝ) : Prop := x^2 - 3*m*x + 2*m^2 ≤ 0
def q (x : ℝ) : Prop := (x + 2)^2 < 1

-- Part 1
theorem range_of_x (x : ℝ) :
  p x (-2) ∧ q x → x ∈ Set.Ioc (-3) (-2) :=
sorry

-- Part 2
theorem range_of_m (m : ℝ) :
  m < 0 ∧ (∀ x, q x ↔ ¬p x m) →
  m ∈ Set.Iic (-3) ∪ Set.Icc (-1/2) 0 :=
sorry

end NUMINAMATH_CALUDE_range_of_x_range_of_m_l525_52520


namespace NUMINAMATH_CALUDE_train_length_train_length_proof_l525_52533

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (train_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  let speed_ms : ℝ := train_speed * 1000 / 3600
  speed_ms * crossing_time

/-- Proof that a train with speed 48 km/hr crossing a pole in 9 seconds has a length of approximately 119.97 meters -/
theorem train_length_proof :
  ∃ ε > 0, |train_length 48 9 - 119.97| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_train_length_train_length_proof_l525_52533


namespace NUMINAMATH_CALUDE_inverse_of_complex_expression_l525_52515

theorem inverse_of_complex_expression (i : ℂ) (h : i^2 = -1) :
  (3*i - 2*i⁻¹)⁻¹ = -i/5 := by sorry

end NUMINAMATH_CALUDE_inverse_of_complex_expression_l525_52515


namespace NUMINAMATH_CALUDE_older_child_age_l525_52559

def mother_charge : ℚ := 6.5
def child_charge_per_year : ℚ := 0.5
def total_bill : ℚ := 14.5
def num_children : ℕ := 4

def is_valid_age (triplet_age : ℕ) (older_age : ℕ) : Prop :=
  triplet_age > 0 ∧ 
  older_age > triplet_age ∧
  mother_charge + child_charge_per_year * (3 * triplet_age + older_age) = total_bill

theorem older_child_age :
  ∃ (triplet_age : ℕ) (older_age : ℕ), 
    is_valid_age triplet_age older_age ∧
    (older_age = 4 ∨ older_age = 7) ∧
    ¬∃ (other_age : ℕ), other_age ≠ 4 ∧ other_age ≠ 7 ∧ is_valid_age triplet_age other_age :=
by sorry

end NUMINAMATH_CALUDE_older_child_age_l525_52559


namespace NUMINAMATH_CALUDE_function_equation_solution_l525_52567

/-- A function satisfying the given functional equation -/
def satisfies_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, (f x * f y - f (x * y)) / 2 = x + y + 1

/-- The main theorem stating that any function satisfying the equation must be f(x) = x + 2 -/
theorem function_equation_solution (f : ℝ → ℝ) (h : satisfies_equation f) : 
  ∀ x : ℝ, f x = x + 2 := by
  sorry

end NUMINAMATH_CALUDE_function_equation_solution_l525_52567


namespace NUMINAMATH_CALUDE_like_terms_imply_value_l525_52518

theorem like_terms_imply_value (a b : ℤ) : 
  (1 = a - 1) → (b + 1 = 4) → (a - b)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_like_terms_imply_value_l525_52518


namespace NUMINAMATH_CALUDE_alan_shell_collection_l525_52597

/-- Proves that Alan collected 48 shells given the conditions of the problem -/
theorem alan_shell_collection (laurie_shells : ℕ) (ben_ratio : ℚ) (alan_ratio : ℕ) : 
  laurie_shells = 36 → 
  ben_ratio = 1/3 → 
  alan_ratio = 4 → 
  (alan_ratio : ℚ) * ben_ratio * laurie_shells = 48 :=
by
  sorry

#check alan_shell_collection

end NUMINAMATH_CALUDE_alan_shell_collection_l525_52597


namespace NUMINAMATH_CALUDE_profit_sharing_ratio_equal_l525_52504

/-- Represents the investment and profit calculation for two partners over a year -/
structure Investment where
  a_initial : ℕ  -- A's initial investment
  b_initial : ℕ  -- B's initial investment
  months : ℕ     -- Total number of months
  mid_months : ℕ -- Months after which A doubles investment

/-- Calculates the total capital-months for partner A -/
def capital_months_a (i : Investment) : ℕ :=
  i.a_initial * i.mid_months + (2 * i.a_initial) * (i.months - i.mid_months)

/-- Calculates the total capital-months for partner B -/
def capital_months_b (i : Investment) : ℕ :=
  i.b_initial * i.months

/-- Theorem stating that the profit-sharing ratio is 1:1 given the specific investment conditions -/
theorem profit_sharing_ratio_equal (i : Investment) 
  (h1 : i.a_initial = 3000)
  (h2 : i.b_initial = 4500)
  (h3 : i.months = 12)
  (h4 : i.mid_months = 6) :
  capital_months_a i = capital_months_b i := by
  sorry

end NUMINAMATH_CALUDE_profit_sharing_ratio_equal_l525_52504


namespace NUMINAMATH_CALUDE_lunch_break_duration_l525_52519

-- Define the painting rates and lunch break duration
variable (p : ℝ) -- Paula's painting rate (building/hour)
variable (h : ℝ) -- Combined rate of two helpers (building/hour)
variable (L : ℝ) -- Lunch break duration (hours)

-- Define the equations based on the given conditions
def monday_equation : Prop := (9 - L) * (p + h) = 0.4
def tuesday_equation : Prop := (7 - L) * h = 0.3
def wednesday_equation : Prop := (12 - L) * p = 0.3

-- Theorem statement
theorem lunch_break_duration 
  (eq1 : monday_equation p h L)
  (eq2 : tuesday_equation h L)
  (eq3 : wednesday_equation p L) :
  L = 0.5 := by sorry

end NUMINAMATH_CALUDE_lunch_break_duration_l525_52519


namespace NUMINAMATH_CALUDE_cherry_tomato_jars_l525_52596

theorem cherry_tomato_jars (total_tomatoes : ℕ) (tomatoes_per_jar : ℕ) (h1 : total_tomatoes = 56) (h2 : tomatoes_per_jar = 8) :
  (total_tomatoes / tomatoes_per_jar : ℕ) = 7 := by
  sorry

end NUMINAMATH_CALUDE_cherry_tomato_jars_l525_52596


namespace NUMINAMATH_CALUDE_average_of_multiples_of_10_l525_52584

theorem average_of_multiples_of_10 : 
  let multiples := List.filter (fun n => n % 10 = 0) (List.range 201)
  (List.sum multiples) / multiples.length = 105 := by
sorry

end NUMINAMATH_CALUDE_average_of_multiples_of_10_l525_52584


namespace NUMINAMATH_CALUDE_hospital_staff_ratio_l525_52560

theorem hospital_staff_ratio (total : ℕ) (nurses : ℕ) (doctors : ℕ) :
  total = 250 →
  nurses = 150 →
  doctors = total - nurses →
  (doctors : ℚ) / nurses = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_hospital_staff_ratio_l525_52560


namespace NUMINAMATH_CALUDE_equation_solution_l525_52509

theorem equation_solution (x : ℝ) : (x + 5)^2 = 16 ↔ x = -1 ∨ x = -9 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l525_52509


namespace NUMINAMATH_CALUDE_high_five_count_l525_52536

def number_of_people : ℕ := 12

/-- The number of unique pairs (high-fives) in a group of n people -/
def number_of_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

theorem high_five_count :
  number_of_pairs number_of_people = 66 :=
by sorry

end NUMINAMATH_CALUDE_high_five_count_l525_52536


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l525_52528

/-- The area of the shaded region in a square with circles at its vertices -/
theorem shaded_area_square_with_circles (square_side : ℝ) (circle_radius : ℝ) 
  (h1 : square_side = 8) (h2 : circle_radius = 3) : 
  ∃ (shaded_area : ℝ), shaded_area = square_side^2 - 12 * Real.sqrt 7 - 3 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l525_52528


namespace NUMINAMATH_CALUDE_find_a_l525_52508

-- Define the universal set U
def U (a : ℤ) : Set ℤ := {2, 4, a^2 - a + 1}

-- Define set A
def A (a : ℤ) : Set ℤ := {a + 4, 4}

-- Define the complement of A relative to U
def complement_A (a : ℤ) : Set ℤ := U a \ A a

-- Theorem statement
theorem find_a : ∃ (a : ℤ), 
  (U a = {2, 4, a^2 - a + 1}) ∧ 
  (A a = {a + 4, 4}) ∧ 
  (complement_A a = {7}) ∧ 
  (a = -2) :=
sorry

end NUMINAMATH_CALUDE_find_a_l525_52508


namespace NUMINAMATH_CALUDE_inequality_proof_l525_52587

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (1 + 2 * a) + 1 / (1 + 2 * b) + 1 / (1 + 2 * c) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l525_52587


namespace NUMINAMATH_CALUDE_space_line_relations_l525_52598

-- Define a type for lines in space
variable (Line : Type)

-- Define the parallel relation
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation
variable (perpendicular : Line → Line → Prop)

-- Define the intersects relation
variable (intersects : Line → Line → Prop)

-- Define a type for planes
variable (Plane : Type)

-- Define a relation for a line being in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define three non-intersecting lines
variable (a b c : Line)

-- Define two planes
variable (α β : Plane)

-- State that the lines are non-intersecting
variable (h_non_intersect : ¬(intersects a b ∨ intersects b c ∨ intersects a c))

theorem space_line_relations :
  (∀ x y z, parallel x y → parallel y z → parallel x z) ∧
  ¬(∀ x y z, perpendicular x y → perpendicular y z → parallel x z) ∧
  ¬(∀ x y z, intersects x y → intersects y z → intersects x z) ∧
  ¬(∀ x y p q, line_in_plane x p → line_in_plane y q → x ≠ y → ¬(parallel x y ∨ intersects x y)) :=
by sorry

end NUMINAMATH_CALUDE_space_line_relations_l525_52598


namespace NUMINAMATH_CALUDE_inequality_solution_l525_52593

open Real

theorem inequality_solution (x y : ℝ) : 
  (Real.sqrt 3 * Real.tan x - (Real.sin y) ^ (1/4) - 
   Real.sqrt ((3 / (Real.cos x)^2) + (Real.sin y)^(1/2) - 6) ≥ Real.sqrt 3) ↔ 
  (∃ (n k : ℤ), x = π/4 + n*π ∧ y = k*π) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l525_52593


namespace NUMINAMATH_CALUDE_arithmetic_equality_l525_52564

theorem arithmetic_equality : (18 / (5 + 2 - 3)) * 4 = 18 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l525_52564


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l525_52572

theorem max_sum_of_factors (A B C : ℕ+) : 
  A ≠ B → B ≠ C → A ≠ C → A * B * C = 1638 → 
  ∀ (X Y Z : ℕ+), X ≠ Y → Y ≠ Z → X ≠ Z → X * Y * Z = 1638 → 
  A + B + C ≤ X + Y + Z → A + B + C ≤ 126 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l525_52572


namespace NUMINAMATH_CALUDE_die_roll_probabilities_l525_52599

theorem die_roll_probabilities :
  let n : ℕ := 7  -- number of rolls
  let p : ℝ := 1/6  -- probability of rolling a 4
  let q : ℝ := 1 - p  -- probability of not rolling a 4

  -- (a) Probability of rolling at least one 4 in 7 rolls
  let prob_at_least_one : ℝ := 1 - q^n

  -- (b) Probability of rolling exactly one 4 in 7 rolls
  let prob_exactly_one : ℝ := n * p * q^(n-1)

  -- (c) Probability of rolling at most one 4 in 7 rolls
  let prob_at_most_one : ℝ := q^n + n * p * q^(n-1)

  -- Prove that the calculated probabilities are correct
  (prob_at_least_one = 1 - (5/6)^7) ∧
  (prob_exactly_one = 7 * (1/6) * (5/6)^6) ∧
  (prob_at_most_one = (5/6)^7 + 7 * (1/6) * (5/6)^6) :=
by
  sorry

end NUMINAMATH_CALUDE_die_roll_probabilities_l525_52599


namespace NUMINAMATH_CALUDE_quadratic_equations_solutions_l525_52541

theorem quadratic_equations_solutions :
  let eq1 : ℝ → Prop := λ x ↦ x^2 - 4*x + 1 = 0
  let eq2 : ℝ → Prop := λ x ↦ x^2 - 5*x + 6 = 0
  let sol1 : Set ℝ := {2 + Real.sqrt 3, 2 - Real.sqrt 3}
  let sol2 : Set ℝ := {2, 3}
  (∀ x ∈ sol1, eq1 x) ∧ (∀ x, eq1 x → x ∈ sol1) ∧
  (∀ x ∈ sol2, eq2 x) ∧ (∀ x, eq2 x → x ∈ sol2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equations_solutions_l525_52541


namespace NUMINAMATH_CALUDE_cubic_factorization_l525_52507

theorem cubic_factorization (x : ℝ) :
  (189 * x^3 + 129 * x^2 + 183 * x + 19 = (4*x - 2)^3 + (5*x + 3)^3) ∧
  (x^3 + 69 * x^2 + 87 * x + 167 = 5*(x + 3)^3 - 4*(x - 2)^3) := by
  sorry

end NUMINAMATH_CALUDE_cubic_factorization_l525_52507


namespace NUMINAMATH_CALUDE_martins_walk_l525_52575

/-- The distance between Martin's house and Lawrence's house -/
def distance : ℝ := 12

/-- The time Martin spent walking -/
def time : ℝ := 6

/-- Martin's walking speed -/
def speed : ℝ := 2

/-- Theorem: The distance between Martin's house and Lawrence's house is 12 miles -/
theorem martins_walk : distance = speed * time := by
  sorry

end NUMINAMATH_CALUDE_martins_walk_l525_52575


namespace NUMINAMATH_CALUDE_triangle_angle_measurement_l525_52516

theorem triangle_angle_measurement (A B C : ℝ) : 
  -- Triangle ABC exists
  A + B + C = 180 →
  -- Measure of angle C is three times the measure of angle B
  C = 3 * B →
  -- Angle B measures 30°
  B = 30 →
  -- Then the measure of angle A is 60°
  A = 60 := by
sorry

end NUMINAMATH_CALUDE_triangle_angle_measurement_l525_52516


namespace NUMINAMATH_CALUDE_prism_volume_given_tangent_sphere_l525_52557

-- Define the sphere
structure Sphere where
  volume : ℝ

-- Define the right triangular prism
structure RightTriangularPrism where
  baseEdgeLength : ℝ
  height : ℝ

-- Define the property of the sphere being tangent to the prism
def isTangentTo (s : Sphere) (p : RightTriangularPrism) : Prop :=
  ∃ (r : ℝ), s.volume = (4/3) * Real.pi * r^3 ∧
             p.baseEdgeLength = 2 * Real.sqrt 3 * r ∧
             p.height = 2 * r

-- Theorem statement
theorem prism_volume_given_tangent_sphere (s : Sphere) (p : RightTriangularPrism) :
  s.volume = 9 * Real.pi / 2 →
  isTangentTo s p →
  (Real.sqrt 3 / 4) * p.baseEdgeLength^2 * p.height = 81 * Real.sqrt 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_prism_volume_given_tangent_sphere_l525_52557


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l525_52517

theorem right_triangle_hypotenuse (base height hypotenuse : ℝ) :
  base = 3 →
  height * base / 2 = 6 →
  base^2 + height^2 = hypotenuse^2 →
  hypotenuse = 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l525_52517


namespace NUMINAMATH_CALUDE_fraction_simplification_l525_52532

theorem fraction_simplification (x y : ℚ) (hx : x = 4/6) (hy : y = 5/8) :
  (6*x + 8*y) / (48*x*y) = 9/20 := by sorry

end NUMINAMATH_CALUDE_fraction_simplification_l525_52532


namespace NUMINAMATH_CALUDE_max_triangles_two_lines_l525_52521

def points_on_line_a : ℕ := 5
def points_on_line_b : ℕ := 8

def triangles_type1 : ℕ := Nat.choose points_on_line_a 2 * Nat.choose points_on_line_b 1
def triangles_type2 : ℕ := Nat.choose points_on_line_a 1 * Nat.choose points_on_line_b 2

def total_triangles : ℕ := triangles_type1 + triangles_type2

theorem max_triangles_two_lines : total_triangles = 220 := by
  sorry

end NUMINAMATH_CALUDE_max_triangles_two_lines_l525_52521


namespace NUMINAMATH_CALUDE_jerrys_pool_depth_l525_52529

/-- Calculates the depth of Jerry's pool given water usage constraints -/
theorem jerrys_pool_depth :
  ∀ (total_water drinking_cooking shower_water showers pool_length pool_width : ℕ),
  total_water = 1000 →
  drinking_cooking = 100 →
  shower_water = 20 →
  showers = 15 →
  pool_length = 10 →
  pool_width = 10 →
  (total_water - (drinking_cooking + shower_water * showers)) / (pool_length * pool_width) = 6 := by
  sorry

#check jerrys_pool_depth

end NUMINAMATH_CALUDE_jerrys_pool_depth_l525_52529


namespace NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l525_52525

theorem cube_sum_and_reciprocal (x : ℝ) (h : x + 1/x = -6) :
  x^3 + 1/x^3 = -198 := by sorry

end NUMINAMATH_CALUDE_cube_sum_and_reciprocal_l525_52525


namespace NUMINAMATH_CALUDE_unfactorizable_quartic_l525_52535

theorem unfactorizable_quartic : ¬ ∃ (a b c d : ℤ), ∀ (x : ℝ),
  x^4 + 2*x^2 + 2*x + 2 = (x^2 + a*x + b) * (x^2 + c*x + d) := by
  sorry

end NUMINAMATH_CALUDE_unfactorizable_quartic_l525_52535


namespace NUMINAMATH_CALUDE_tan_theta_range_l525_52512

theorem tan_theta_range (θ : Real) (a : Real) 
  (h1 : -π/2 < θ ∧ θ < π/2) 
  (h2 : Real.sin θ + Real.cos θ = a) 
  (h3 : 0 < a ∧ a < 1) : 
  -1 < Real.tan θ ∧ Real.tan θ < 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_theta_range_l525_52512


namespace NUMINAMATH_CALUDE_skating_speed_ratio_l525_52580

theorem skating_speed_ratio (v_f v_s : ℝ) (h1 : v_f > 0) (h2 : v_s > 0) :
  (v_f + v_s) / (v_f - v_s) = 5 → v_f / v_s = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_skating_speed_ratio_l525_52580


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l525_52542

/-- An arithmetic sequence is a sequence where the difference between
    consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The theorem states that for an arithmetic sequence with given conditions,
    the general term formula is a_n = 4 - 2n. -/
theorem arithmetic_sequence_formula 
  (a : ℕ → ℤ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_3 : a 3 = -2) 
  (h_7 : a 7 = -10) :
  ∃ c : ℤ, ∀ n : ℕ, a n = 4 - 2 * n :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l525_52542


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l525_52574

theorem absolute_value_equation_solution : 
  {x : ℝ | |x - 5| = 3*x + 6} = {-11/2, -1/4} := by sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l525_52574


namespace NUMINAMATH_CALUDE_circle_center_is_two_neg_three_l525_52523

/-- The equation of a circle in the xy-plane -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 - 4*x + y^2 + 6*y - 11 = 0

/-- The center of a circle -/
def circle_center (h k : ℝ) : ℝ × ℝ := (h, k)

/-- Theorem: The center of the circle defined by x^2 - 4x + y^2 + 6y - 11 = 0 is (2, -3) -/
theorem circle_center_is_two_neg_three :
  ∃ (r : ℝ), ∀ (x y : ℝ), circle_equation x y ↔ (x - 2)^2 + (y - (-3))^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_circle_center_is_two_neg_three_l525_52523


namespace NUMINAMATH_CALUDE_interview_room_occupancy_l525_52568

/-- Given a waiting room and an interview room, prove that the number of people in the interview room is 5. -/
theorem interview_room_occupancy (waiting_room interview_room : ℕ) : interview_room = 5 :=
  by
  -- Define the initial number of people in the waiting room
  have initial_waiting : waiting_room = 22 := by sorry
  
  -- Define the number of new arrivals
  have new_arrivals : ℕ := 3
  
  -- Define the relationship between waiting room and interview room after new arrivals
  have after_arrivals : waiting_room + new_arrivals = 5 * interview_room := by sorry
  
  sorry -- Proof goes here

end NUMINAMATH_CALUDE_interview_room_occupancy_l525_52568


namespace NUMINAMATH_CALUDE_ball_drawing_properties_l525_52569

/-- Represents the number of balls drawn -/
def n : ℕ := 3

/-- Represents the initial number of red balls -/
def r : ℕ := 5

/-- Represents the initial number of black balls -/
def b : ℕ := 2

/-- Represents the total number of balls -/
def total : ℕ := r + b

/-- Represents the random variable for the number of red balls drawn without replacement -/
def X : Fin (n + 1) → ℝ := sorry

/-- Represents the random variable for the number of black balls drawn without replacement -/
def Y : Fin (n + 1) → ℝ := sorry

/-- Represents the random variable for the number of red balls drawn with replacement -/
def ξ : Fin (n + 1) → ℝ := sorry

/-- The expected value of X -/
noncomputable def E_X : ℝ := sorry

/-- The expected value of Y -/
noncomputable def E_Y : ℝ := sorry

/-- The expected value of ξ -/
noncomputable def E_ξ : ℝ := sorry

/-- The variance of X -/
noncomputable def D_X : ℝ := sorry

/-- The variance of ξ -/
noncomputable def D_ξ : ℝ := sorry

theorem ball_drawing_properties :
  (E_X / E_Y = r / b) ∧ (E_X = E_ξ) ∧ (D_X < D_ξ) := by sorry

end NUMINAMATH_CALUDE_ball_drawing_properties_l525_52569


namespace NUMINAMATH_CALUDE_max_missable_problems_l525_52502

theorem max_missable_problems (total_problems : ℕ) (passing_percentage : ℚ) 
  (hp : passing_percentage = 85 / 100) (ht : total_problems = 40) :
  ⌊total_problems * (1 - passing_percentage)⌋ = 6 :=
sorry

end NUMINAMATH_CALUDE_max_missable_problems_l525_52502


namespace NUMINAMATH_CALUDE_special_triangle_sides_l525_52514

/-- An isosceles triangle with perimeter 60 and centroid on the inscribed circle. -/
structure SpecialTriangle where
  -- Two equal sides
  a : ℝ
  -- Third side
  b : ℝ
  -- Perimeter is 60
  perimeter_eq : 2 * a + b = 60
  -- a > 0 and b > 0
  a_pos : a > 0
  b_pos : b > 0
  -- Centroid on inscribed circle condition
  centroid_on_inscribed : 3 * (a * b) = 60 * (a + b - (2 * a + b) / 2)

/-- The sides of a special triangle are 25, 25, and 10. -/
theorem special_triangle_sides (t : SpecialTriangle) : t.a = 25 ∧ t.b = 10 := by
  sorry

end NUMINAMATH_CALUDE_special_triangle_sides_l525_52514


namespace NUMINAMATH_CALUDE_jordans_initial_weight_jordans_weight_proof_l525_52503

/-- Calculates Jordan's initial weight based on his weight loss program and current weight -/
theorem jordans_initial_weight (initial_loss_rate : ℕ) (initial_weeks : ℕ) 
  (subsequent_loss_rate : ℕ) (subsequent_weeks : ℕ) (current_weight : ℕ) : ℕ :=
  let total_loss := initial_loss_rate * initial_weeks + subsequent_loss_rate * subsequent_weeks
  current_weight + total_loss

/-- Proves that Jordan's initial weight was 250 pounds -/
theorem jordans_weight_proof : 
  jordans_initial_weight 3 4 2 8 222 = 250 := by
  sorry

end NUMINAMATH_CALUDE_jordans_initial_weight_jordans_weight_proof_l525_52503


namespace NUMINAMATH_CALUDE_stratified_sampling_correct_l525_52590

/-- Represents the number of students to be chosen from a class in stratified sampling -/
def stratified_sample (total_students : ℕ) (class_size : ℕ) (sample_size : ℕ) : ℕ :=
  (class_size * sample_size) / total_students

theorem stratified_sampling_correct (class1_size class2_size sample_size : ℕ) 
  (h1 : class1_size = 54)
  (h2 : class2_size = 42)
  (h3 : sample_size = 16) :
  (stratified_sample (class1_size + class2_size) class1_size sample_size = 9) ∧
  (stratified_sample (class1_size + class2_size) class2_size sample_size = 7) := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_correct_l525_52590


namespace NUMINAMATH_CALUDE_expression_evaluation_l525_52558

theorem expression_evaluation :
  let x : ℤ := -2
  2 * x * (x - 3) - (x - 2) * (x + 1) = 16 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l525_52558


namespace NUMINAMATH_CALUDE_popsicle_consumption_l525_52582

/-- The number of Popsicles eaten in a given time period -/
def popsicles_eaten (rate : ℚ) (time : ℚ) : ℚ :=
  time / rate

/-- Proves that eating 1 Popsicle every 20 minutes for 6 hours results in 18 Popsicles -/
theorem popsicle_consumption : popsicles_eaten (20 / 60) 6 = 18 := by
  sorry

#eval popsicles_eaten (20 / 60) 6

end NUMINAMATH_CALUDE_popsicle_consumption_l525_52582


namespace NUMINAMATH_CALUDE_find_a_value_l525_52545

def A : Set ℝ := {2, 4}
def B (a : ℝ) : Set ℝ := {a, a^2 + 3}

theorem find_a_value (h : A ∩ B a = {2}) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l525_52545


namespace NUMINAMATH_CALUDE_frog_jump_distance_l525_52531

/-- The jumping contest between a grasshopper and a frog -/
theorem frog_jump_distance (grasshopper_jump : ℕ) (additional_distance : ℕ) 
  (h1 : grasshopper_jump = 25)
  (h2 : additional_distance = 15) : 
  grasshopper_jump + additional_distance = 40 := by
  sorry

#check frog_jump_distance

end NUMINAMATH_CALUDE_frog_jump_distance_l525_52531


namespace NUMINAMATH_CALUDE_train_platform_length_l525_52513

/-- Given a train passing a pole in t seconds and a platform in 6t seconds at constant velocity,
    prove that the length of the platform is 5 times the length of the train. -/
theorem train_platform_length
  (t : ℝ)
  (train_length : ℝ)
  (platform_length : ℝ)
  (velocity : ℝ)
  (h1 : velocity = train_length / t)
  (h2 : velocity = (train_length + platform_length) / (6 * t))
  : platform_length = 5 * train_length := by
  sorry

end NUMINAMATH_CALUDE_train_platform_length_l525_52513


namespace NUMINAMATH_CALUDE_no_such_sequence_l525_52501

theorem no_such_sequence : ¬∃ (a : ℕ → ℕ),
  (∀ n > 1, a n > a (n - 1)) ∧
  (∀ m n : ℕ, a (m * n) = a m + a n) :=
sorry

end NUMINAMATH_CALUDE_no_such_sequence_l525_52501


namespace NUMINAMATH_CALUDE_torturie_problem_l525_52556

/-- The number of the last remaining prisoner in the Torturie problem -/
def lastPrisoner (n : ℕ) : ℕ :=
  2 * n - 2^(Nat.log2 n + 1) + 1

/-- The Torturie problem statement -/
theorem torturie_problem (n : ℕ) (h : n > 0) :
  lastPrisoner n = 
    let k := Nat.log2 n
    2 * n - 2^(k + 1) + 1 :=
by sorry

end NUMINAMATH_CALUDE_torturie_problem_l525_52556


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l525_52566

-- Define an arithmetic sequence with common difference 2
def arithmetic_seq (a : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, a (n + 1) = a n + 2

-- Define a geometric sequence for three terms
def geometric_seq (x y z : ℤ) : Prop :=
  y * y = x * z

-- Theorem statement
theorem arithmetic_geometric_sequence (a : ℤ → ℤ) :
  arithmetic_seq a →
  geometric_seq (a 1) (a 3) (a 4) →
  a 1 = -8 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l525_52566


namespace NUMINAMATH_CALUDE_constant_d_value_l525_52585

theorem constant_d_value (e c f : ℝ) : 
  (∃ d : ℝ, ∀ x : ℝ, 
    (3 * x^3 - 2 * x^2 + x - 5/4) * (e * x^3 + d * x^2 + c * x + f) = 
    9 * x^6 - 5 * x^5 - x^4 + 20 * x^3 - 25/4 * x^2 + 15/4 * x - 5/2) →
  (∃ d : ℝ, d = 1/3) := by
sorry

end NUMINAMATH_CALUDE_constant_d_value_l525_52585


namespace NUMINAMATH_CALUDE_zane_picked_up_62_pounds_l525_52500

/-- The amount of garbage picked up by Daliah in pounds -/
def daliah_garbage : ℝ := 17.5

/-- The amount of garbage picked up by Dewei in pounds -/
def dewei_garbage : ℝ := daliah_garbage - 2

/-- The amount of garbage picked up by Zane in pounds -/
def zane_garbage : ℝ := 4 * dewei_garbage

/-- Theorem stating that Zane picked up 62 pounds of garbage -/
theorem zane_picked_up_62_pounds : zane_garbage = 62 := by
  sorry

end NUMINAMATH_CALUDE_zane_picked_up_62_pounds_l525_52500


namespace NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l525_52589

/-- A complex number z is in the fourth quadrant if its real part is positive and its imaginary part is negative. -/
def in_fourth_quadrant (z : ℂ) : Prop := 0 < z.re ∧ z.im < 0

/-- Given -1 < m < 1, the complex number (1-i) + m(1+i) is in the fourth quadrant. -/
theorem complex_in_fourth_quadrant (m : ℝ) (h : -1 < m ∧ m < 1) :
  in_fourth_quadrant ((1 - Complex.I) + m * (1 + Complex.I)) := by
  sorry

end NUMINAMATH_CALUDE_complex_in_fourth_quadrant_l525_52589


namespace NUMINAMATH_CALUDE_abs_value_of_complex_l525_52554

theorem abs_value_of_complex (z : ℂ) : z = (1 + 2 * Complex.I) / Complex.I → Complex.abs z = Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_of_complex_l525_52554


namespace NUMINAMATH_CALUDE_M_eq_open_interval_compare_values_l525_52543

/-- The function f(x) = |x| - |2x - 1| -/
def f (x : ℝ) : ℝ := |x| - |2*x - 1|

/-- The set M is defined as the solution set of f(x) > -1 -/
def M : Set ℝ := {x | f x > -1}

/-- Theorem stating that M is the open interval (0, 2) -/
theorem M_eq_open_interval : M = Set.Ioo 0 2 := by sorry

/-- Theorem comparing a^2 - a + 1 and 1/a for a ∈ M -/
theorem compare_values (a : ℝ) (h : a ∈ M) :
  (0 < a ∧ a < 1 → a^2 - a + 1 < 1/a) ∧
  (a = 1 → a^2 - a + 1 = 1/a) ∧
  (1 < a ∧ a < 2 → a^2 - a + 1 > 1/a) := by sorry

end NUMINAMATH_CALUDE_M_eq_open_interval_compare_values_l525_52543


namespace NUMINAMATH_CALUDE_binary_multiplication_addition_l525_52581

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

theorem binary_multiplication_addition :
  let a := [true, false, false, true, true]  -- 11001₂
  let b := [false, true, true]               -- 110₂
  let c := [false, true, false, true]        -- 1010₂
  let result := [false, true, true, true, true, true, false, true]  -- 10111110₂
  (binary_to_nat a * binary_to_nat b + binary_to_nat c) = binary_to_nat result := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_addition_l525_52581


namespace NUMINAMATH_CALUDE_inequality_proof_l525_52524

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hsum : x + y + z = 1) :
  (x^4)/(y*(1-y^2)) + (y^4)/(z*(1-z^2)) + (z^4)/(x*(1-x^2)) ≥ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l525_52524


namespace NUMINAMATH_CALUDE_complex_arithmetic_l525_52511

theorem complex_arithmetic (A M S : ℂ) (P : ℝ) 
  (hA : A = 5 - 2*I) 
  (hM : M = -3 + 2*I) 
  (hS : S = 2*I) 
  (hP : P = 3) : 
  A - M + S - P = 5 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_l525_52511


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l525_52576

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I * (-(2 + a) / 2) = (2 - a * Complex.I) / (1 + Complex.I)) → a = 2 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l525_52576


namespace NUMINAMATH_CALUDE_interest_calculation_l525_52561

/-- Given a principal amount and number of years, proves that if simple interest
    at 5% per annum is 50 and compound interest at the same rate is 51.25,
    then the number of years is 2. -/
theorem interest_calculation (P n : ℝ) : 
  P * n / 20 = 50 →
  P * ((1 + 5/100)^n - 1) = 51.25 →
  n = 2 := by
sorry

end NUMINAMATH_CALUDE_interest_calculation_l525_52561


namespace NUMINAMATH_CALUDE_first_problem_answer_l525_52592

/-- Given three math problems with the following properties:
    1. The second problem's answer is twice the first problem's answer.
    2. The third problem's answer is 400 less than the sum of the first two problems' answers.
    3. The total of all three answers is 3200.
    Prove that the answer to the first math problem is 600. -/
theorem first_problem_answer (a : ℕ) : 
  (∃ b c : ℕ, 
    b = 2 * a ∧ 
    c = a + b - 400 ∧ 
    a + b + c = 3200) → 
  a = 600 :=
by sorry

end NUMINAMATH_CALUDE_first_problem_answer_l525_52592


namespace NUMINAMATH_CALUDE_f_properties_l525_52540

def f (x : ℝ) : ℝ := |x| + 1

theorem f_properties :
  (∀ x : ℝ, f (-x) = f x) ∧
  (∀ x y : ℝ, x < y → x < 0 → f y < f x) ∧
  (∀ x y : ℝ, x < y → 0 < x → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l525_52540


namespace NUMINAMATH_CALUDE_quadratic_roots_nature_l525_52550

theorem quadratic_roots_nature (x : ℝ) :
  let f : ℝ → ℝ := λ x => x^2 - 4*x*Real.sqrt 2 + 8
  let discriminant := (-4*Real.sqrt 2)^2 - 4*1*8
  discriminant = 0 ∧ ∃ r : ℝ, f r = 0 ∧ (∀ s : ℝ, f s = 0 → s = r) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_nature_l525_52550


namespace NUMINAMATH_CALUDE_isosceles_triangle_l525_52591

theorem isosceles_triangle (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) 
  (h_equation : a^2 - b^2 + a*c - b*c = 0) : 
  a = b ∨ b = c ∨ c = a := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_l525_52591


namespace NUMINAMATH_CALUDE_polar_to_cartesian_l525_52527

theorem polar_to_cartesian :
  let r : ℝ := 4
  let θ : ℝ := 5 * Real.pi / 6
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = -2 * Real.sqrt 3 ∧ y = 2) := by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_l525_52527


namespace NUMINAMATH_CALUDE_prism_volume_l525_52522

/-- Given a right rectangular prism with dimensions a, b, and c satisfying certain conditions,
    prove that its volume is 200 cubic centimeters. -/
theorem prism_volume (a b c : ℝ) (h1 : a * b = 50) (h2 : b * c = 80) (h3 : a * c = 100)
    (h4 : ∃ n : ℕ, (a * c : ℝ) = n ^ 2) : a * b * c = 200 := by
  sorry

end NUMINAMATH_CALUDE_prism_volume_l525_52522


namespace NUMINAMATH_CALUDE_unknown_blanket_rate_l525_52588

/-- Proves that given the specified blanket purchases and average price, the unknown rate is 285 --/
theorem unknown_blanket_rate (num_blankets_1 num_blankets_2 num_unknown : ℕ)
                              (price_1 price_2 avg_price : ℚ) :
  num_blankets_1 = 3 →
  num_blankets_2 = 5 →
  num_unknown = 2 →
  price_1 = 100 →
  price_2 = 150 →
  avg_price = 162 →
  let total_blankets := num_blankets_1 + num_blankets_2 + num_unknown
  let total_cost := avg_price * total_blankets
  let known_cost := num_blankets_1 * price_1 + num_blankets_2 * price_2
  let unknown_cost := total_cost - known_cost
  unknown_cost / num_unknown = 285 := by sorry

end NUMINAMATH_CALUDE_unknown_blanket_rate_l525_52588


namespace NUMINAMATH_CALUDE_can_divide_into_12_l525_52594

-- Define a circular cake
structure CircularCake where
  radius : ℝ
  center : ℝ × ℝ

-- Define a function to represent dividing a cake into equal pieces
def divide_cake (cake : CircularCake) (n : ℕ) : Set (ℝ × ℝ) :=
  sorry

-- Define our given cakes
def cake1 : CircularCake := sorry
def cake2 : CircularCake := sorry
def cake3 : CircularCake := sorry

-- State that cake1 is divided into 3 pieces
axiom cake1_division : divide_cake cake1 3

-- State that cake2 is divided into 4 pieces
axiom cake2_division : divide_cake cake2 4

-- State that all cakes have the same radius
axiom same_radius : cake1.radius = cake2.radius ∧ cake2.radius = cake3.radius

-- State that we know the center of cake3
axiom known_center3 : cake3.center = (0, 0)

-- Theorem to prove
theorem can_divide_into_12 : 
  ∃ (division : Set (ℝ × ℝ)), division = divide_cake cake3 12 :=
sorry

end NUMINAMATH_CALUDE_can_divide_into_12_l525_52594


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l525_52570

/-- Given that i² = -1, prove that (2 - 3i) / (4 - 5i) = 23/41 - (2/41)i -/
theorem complex_fraction_simplification (i : ℂ) (h : i^2 = -1) :
  (2 - 3*i) / (4 - 5*i) = 23/41 - (2/41)*i := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l525_52570


namespace NUMINAMATH_CALUDE_cups_in_box_l525_52549

/-- Given an initial quantity of cups and a number of cups added, 
    calculate the total number of cups -/
def total_cups (initial : ℕ) (added : ℕ) : ℕ := initial + added

/-- Theorem stating that given 17 initial cups and 16 cups added, 
    the total number of cups is 33 -/
theorem cups_in_box : total_cups 17 16 = 33 := by
  sorry

end NUMINAMATH_CALUDE_cups_in_box_l525_52549


namespace NUMINAMATH_CALUDE_max_m_value_l525_52505

theorem max_m_value (b a m : ℝ) (hb : b > 0) :
  (∀ a, (b - (a - 2))^2 + (Real.log b - (a - 1))^2 ≥ m^2 - m) →
  m ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_m_value_l525_52505


namespace NUMINAMATH_CALUDE_correct_article_usage_l525_52546

-- Define the possible article types
inductive Article
  | Definite
  | Indefinite
  | NoArticle

-- Define the context of a noun
structure NounContext where
  isSpecific : Bool

-- Define the function to determine the correct article
def correctArticle (context : NounContext) : Article :=
  if context.isSpecific then Article.Definite else Article.Indefinite

-- Theorem statement
theorem correct_article_usage 
  (keyboard_context : NounContext)
  (computer_context : NounContext)
  (h1 : keyboard_context.isSpecific = true)
  (h2 : computer_context.isSpecific = false) :
  (correctArticle keyboard_context = Article.Definite) ∧
  (correctArticle computer_context = Article.Indefinite) := by
  sorry


end NUMINAMATH_CALUDE_correct_article_usage_l525_52546


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l525_52547

theorem line_parabola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 - 1 ∧ p.2^2 = 4 * p.1) → k = 0 ∨ k = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l525_52547


namespace NUMINAMATH_CALUDE_cos_180_deg_l525_52555

/-- The cosine of an angle in degrees -/
noncomputable def cos_deg (θ : ℝ) : ℝ := 
  (Complex.exp (θ * Complex.I * Real.pi / 180)).re

/-- Theorem: The cosine of 180 degrees is -1 -/
theorem cos_180_deg : cos_deg 180 = -1 := by sorry

end NUMINAMATH_CALUDE_cos_180_deg_l525_52555


namespace NUMINAMATH_CALUDE_division_problem_l525_52551

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 1565 → 
  quotient = 65 → 
  remainder = 5 → 
  dividend = divisor * quotient + remainder → 
  divisor = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_division_problem_l525_52551


namespace NUMINAMATH_CALUDE_triangle_problem_l525_52552

-- Define the triangle ABC
variable (A B C : Real) -- Angles
variable (a b c : Real) -- Sides
variable (S : Real) -- Area

-- State the theorem
theorem triangle_problem 
  (h1 : 2 * Real.cos (2 * B) = 4 * Real.cos B - 3)
  (h2 : S = Real.sqrt 3)
  (h3 : a * Real.sin A + c * Real.sin C = 5 * Real.sin B) :
  B = π / 3 ∧ b = (5 + Real.sqrt 21) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l525_52552


namespace NUMINAMATH_CALUDE_binomial_150_150_l525_52577

theorem binomial_150_150 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_l525_52577


namespace NUMINAMATH_CALUDE_power_equality_l525_52510

theorem power_equality (n : ℕ) : 4^6 = 8^n → n = 4 := by sorry

end NUMINAMATH_CALUDE_power_equality_l525_52510


namespace NUMINAMATH_CALUDE_school_bought_fifty_marker_cartons_l525_52530

/-- Represents the purchase of school supplies -/
structure SchoolSupplies where
  pencil_cartons : ℕ
  pencil_boxes_per_carton : ℕ
  pencil_box_cost : ℕ
  marker_carton_cost : ℕ
  total_spent : ℕ

/-- Calculates the number of marker cartons bought -/
def marker_cartons_bought (supplies : SchoolSupplies) : ℕ :=
  (supplies.total_spent - supplies.pencil_cartons * supplies.pencil_boxes_per_carton * supplies.pencil_box_cost) / supplies.marker_carton_cost

/-- Theorem stating that the school bought 50 cartons of markers -/
theorem school_bought_fifty_marker_cartons :
  let supplies : SchoolSupplies := {
    pencil_cartons := 20,
    pencil_boxes_per_carton := 10,
    pencil_box_cost := 2,
    marker_carton_cost := 4,
    total_spent := 600
  }
  marker_cartons_bought supplies = 50 := by
  sorry


end NUMINAMATH_CALUDE_school_bought_fifty_marker_cartons_l525_52530


namespace NUMINAMATH_CALUDE_fraction_subtraction_l525_52539

theorem fraction_subtraction : 
  (((2 + 4 + 6 + 8) : ℚ) / (1 + 3 + 5 + 7)) - ((1 + 3 + 5 + 7) / (2 + 4 + 6 + 8)) = 9 / 20 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l525_52539


namespace NUMINAMATH_CALUDE_apple_solution_l525_52553

/-- The number of apples each person has. -/
structure Apples where
  rebecca : ℕ
  jackie : ℕ
  adam : ℕ

/-- The conditions of the apple distribution problem. -/
def AppleConditions (a : Apples) : Prop :=
  a.rebecca = 2 * a.jackie ∧
  a.adam = a.jackie + 3 ∧
  a.adam = 9

/-- The solution to the apple distribution problem. -/
theorem apple_solution (a : Apples) (h : AppleConditions a) : a.jackie = 6 ∧ a.rebecca = 12 := by
  sorry


end NUMINAMATH_CALUDE_apple_solution_l525_52553


namespace NUMINAMATH_CALUDE_smallest_positive_a_l525_52534

/-- A function with period 20 -/
def IsPeriodic20 (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x - 20) = g x

/-- The property we want to prove for the smallest positive a -/
def HasProperty (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g ((x - a) / 10) = g (x / 10)

theorem smallest_positive_a (g : ℝ → ℝ) (h : IsPeriodic20 g) :
  (∃ a > 0, HasProperty g a) →
  (∃ a > 0, HasProperty g a ∧ ∀ b, 0 < b → b < a → ¬HasProperty g b) →
  (∃! a, a = 200 ∧ a > 0 ∧ HasProperty g a ∧ ∀ b, 0 < b → b < a → ¬HasProperty g b) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_a_l525_52534


namespace NUMINAMATH_CALUDE_diagonal_passes_through_720_cubes_l525_52565

/-- The number of unit cubes an internal diagonal passes through in a rectangular solid -/
def cubes_passed_through (l w h : ℕ) : ℕ :=
  l + w + h - (Nat.gcd l w + Nat.gcd w h + Nat.gcd h l) + Nat.gcd l (Nat.gcd w h)

/-- Theorem stating that for a 120 × 360 × 400 rectangular solid, 
    an internal diagonal passes through 720 unit cubes -/
theorem diagonal_passes_through_720_cubes :
  cubes_passed_through 120 360 400 = 720 := by
  sorry

end NUMINAMATH_CALUDE_diagonal_passes_through_720_cubes_l525_52565


namespace NUMINAMATH_CALUDE_codemaster_combinations_l525_52538

/-- The number of colors available for the pegs -/
def num_colors : ℕ := 8

/-- The number of slots in the code -/
def num_slots : ℕ := 5

/-- Theorem: The number of possible secret codes in Codemaster -/
theorem codemaster_combinations : num_colors ^ num_slots = 32768 := by
  sorry

end NUMINAMATH_CALUDE_codemaster_combinations_l525_52538


namespace NUMINAMATH_CALUDE_sets_equality_l525_52586

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem sets_equality : M = N := by sorry

end NUMINAMATH_CALUDE_sets_equality_l525_52586


namespace NUMINAMATH_CALUDE_negation_equivalence_l525_52506

theorem negation_equivalence :
  (¬ ∀ m : ℝ, m > 0 → m^2 > 0) ↔ (∃ m : ℝ, m ≤ 0 ∧ m^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l525_52506


namespace NUMINAMATH_CALUDE_factor_implies_h_value_l525_52544

theorem factor_implies_h_value (h : ℝ) (m : ℝ) : 
  (∃ k : ℝ, m^2 - h*m - 24 = (m - 8) * k) → h = 5 := by
sorry

end NUMINAMATH_CALUDE_factor_implies_h_value_l525_52544


namespace NUMINAMATH_CALUDE_distinct_triangles_in_tetrahedron_l525_52526

/-- The number of vertices in a regular tetrahedron -/
def tetrahedron_vertices : ℕ := 4

/-- The number of vertices needed to form a triangle -/
def triangle_vertices : ℕ := 3

/-- Calculates the number of combinations of k items chosen from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- Theorem: The number of distinct triangles in a regular tetrahedron is 4 -/
theorem distinct_triangles_in_tetrahedron :
  choose tetrahedron_vertices triangle_vertices = 4 := by sorry

end NUMINAMATH_CALUDE_distinct_triangles_in_tetrahedron_l525_52526


namespace NUMINAMATH_CALUDE_triangle_cosine_law_l525_52573

theorem triangle_cosine_law (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  let S := (1/2) * Real.sqrt (a^2 * b^2 - ((a^2 + b^2 - c^2) / 2)^2)
  (∃ (C : ℝ), S = (1/2) * a * b * Real.sin C) →
  ∃ (C : ℝ), Real.cos C = (a^2 + b^2 - c^2) / (2 * a * b) := by
sorry

end NUMINAMATH_CALUDE_triangle_cosine_law_l525_52573


namespace NUMINAMATH_CALUDE_two_by_one_cuboid_net_l525_52583

/-- Represents a cuboid with integer dimensions -/
structure Cuboid where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the number of unit squares in the net of a cuboid -/
def net_squares (c : Cuboid) : ℕ :=
  2 * (c.length * c.width + c.length * c.height + c.width * c.height)

/-- Theorem: A 2x1x1 cuboid's net has 10 unit squares, and removing any one leaves 9 -/
theorem two_by_one_cuboid_net :
  let c : Cuboid := ⟨2, 1, 1⟩
  net_squares c = 10 ∧ net_squares c - 1 = 9 := by
  sorry

#eval net_squares ⟨2, 1, 1⟩

end NUMINAMATH_CALUDE_two_by_one_cuboid_net_l525_52583


namespace NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l525_52563

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, a < x^2 + 1
def q (a : ℝ) : Prop := ∃ x₀ : ℝ, a < 3 - x₀^2

-- Theorem stating that p is sufficient but not necessary for q
theorem p_sufficient_not_necessary_for_q :
  (∀ a : ℝ, p a → q a) ∧ (∃ a : ℝ, q a ∧ ¬(p a)) :=
sorry

end NUMINAMATH_CALUDE_p_sufficient_not_necessary_for_q_l525_52563


namespace NUMINAMATH_CALUDE_sqrt_inequality_l525_52562

theorem sqrt_inequality : Real.sqrt 3 - Real.sqrt 2 > Real.sqrt 6 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l525_52562


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l525_52579

theorem arithmetic_mean_of_special_set (n : ℕ) (h : n > 2) :
  let set := [1 - 2 / n, 1 - 2 / n] ++ List.replicate (n - 2) 1
  (set.sum / n : ℚ) = 1 - 4 / n^2 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l525_52579


namespace NUMINAMATH_CALUDE_cyclist_travel_time_l525_52578

/-- Proves that a cyclist who travels 2.5 miles in 10 minutes will take 20 minutes
    to travel 4 miles when their speed is reduced by 20% due to a headwind -/
theorem cyclist_travel_time (initial_distance : ℝ) (initial_time : ℝ) 
  (new_distance : ℝ) (speed_reduction : ℝ) :
  initial_distance = 2.5 →
  initial_time = 10 →
  new_distance = 4 →
  speed_reduction = 0.2 →
  (new_distance / ((initial_distance / initial_time) * (1 - speed_reduction))) = 20 := by
  sorry

#check cyclist_travel_time

end NUMINAMATH_CALUDE_cyclist_travel_time_l525_52578


namespace NUMINAMATH_CALUDE_tan_plus_reciprocal_l525_52595

theorem tan_plus_reciprocal (θ : Real) (h : Real.sin (2 * θ) = 2/3) :
  Real.tan θ + (Real.tan θ)⁻¹ = 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_plus_reciprocal_l525_52595


namespace NUMINAMATH_CALUDE_hoseok_number_division_l525_52537

theorem hoseok_number_division (x : ℤ) (h : x + 8 = 88) : x / 10 = 8 := by
  sorry

end NUMINAMATH_CALUDE_hoseok_number_division_l525_52537


namespace NUMINAMATH_CALUDE_complement_B_union_A_equals_open_interval_l525_52548

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | x^2 - 3*x < 4}

-- Define set B
def B : Set ℝ := {x : ℝ | |x| ≥ 2}

-- Theorem statement
theorem complement_B_union_A_equals_open_interval :
  (Set.compl B) ∪ A = Set.Ioo (-2 : ℝ) 4 :=
sorry

end NUMINAMATH_CALUDE_complement_B_union_A_equals_open_interval_l525_52548


namespace NUMINAMATH_CALUDE_coin_sum_theorem_l525_52571

def coin_values : List Nat := [5, 10, 25, 50, 100]

def sum_three_coins (a b c : Nat) : Nat := a + b + c

def is_valid_sum (sum : Nat) : Prop :=
  ∃ (a b c : Nat), a ∈ coin_values ∧ b ∈ coin_values ∧ c ∈ coin_values ∧ sum_three_coins a b c = sum

theorem coin_sum_theorem :
  ¬(is_valid_sum 52) ∧
  (is_valid_sum 60) ∧
  (is_valid_sum 115) ∧
  (is_valid_sum 165) ∧
  (is_valid_sum 180) :=
sorry

end NUMINAMATH_CALUDE_coin_sum_theorem_l525_52571
