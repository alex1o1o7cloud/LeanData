import Mathlib

namespace NUMINAMATH_CALUDE_bus_empty_seats_after_second_stop_l146_14614

/-- Represents the state of the bus at different stages --/
structure BusState where
  total_seats : ℕ
  occupied_seats : ℕ

/-- Calculates the number of empty seats in the bus --/
def empty_seats (state : BusState) : ℕ :=
  state.total_seats - state.occupied_seats

/-- Updates the bus state after passenger movement --/
def update_state (state : BusState) (board : ℕ) (leave : ℕ) : BusState :=
  { total_seats := state.total_seats,
    occupied_seats := state.occupied_seats + board - leave }

theorem bus_empty_seats_after_second_stop :
  let initial_state : BusState := { total_seats := 23 * 4, occupied_seats := 16 }
  let first_stop := update_state initial_state 15 3
  let second_stop := update_state first_stop 17 10
  empty_seats second_stop = 57 := by sorry


end NUMINAMATH_CALUDE_bus_empty_seats_after_second_stop_l146_14614


namespace NUMINAMATH_CALUDE_team_size_is_five_l146_14652

/-- The length of the relay race in meters -/
def relay_length : ℕ := 150

/-- The distance each team member runs in meters -/
def member_distance : ℕ := 30

/-- The number of people on the team -/
def team_size : ℕ := relay_length / member_distance

theorem team_size_is_five : team_size = 5 := by
  sorry

end NUMINAMATH_CALUDE_team_size_is_five_l146_14652


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l146_14649

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a) 
  (h_a2 : a 2 = 3) 
  (h_a5 : a 5 = 12) : 
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l146_14649


namespace NUMINAMATH_CALUDE_cos_alpha_value_l146_14611

theorem cos_alpha_value (α : Real) 
  (h1 : Real.sin (α - Real.pi/4) = -Real.sqrt 2 / 10)
  (h2 : 0 < α) (h3 : α < Real.pi/2) : 
  Real.cos α = 4/5 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_value_l146_14611


namespace NUMINAMATH_CALUDE_triangle_construction_feasibility_l146_14694

/-- Given a triangle with sides a and c, and angle condition α = 2β, 
    the triangle construction is feasible if and only if a > (2/3)c -/
theorem triangle_construction_feasibility (a c : ℝ) (α β : ℝ) 
  (h_positive_a : a > 0) (h_positive_c : c > 0) (h_angle : α = 2 * β) :
  (∃ b : ℝ, b > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b) ↔ a > (2/3) * c := by
sorry

end NUMINAMATH_CALUDE_triangle_construction_feasibility_l146_14694


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l146_14690

theorem coefficient_x_cubed_expansion : 
  let expansion := (1 + X : Polynomial ℤ)^5 - (1 + X : Polynomial ℤ)^6
  expansion.coeff 3 = -10 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x_cubed_expansion_l146_14690


namespace NUMINAMATH_CALUDE_min_keystrokes_to_243_l146_14684

-- Define the allowed operations
def add_one (n : ℕ) : ℕ := n + 1
def multiply_two (n : ℕ) : ℕ := n * 2
def multiply_three (n : ℕ) : ℕ := if n % 3 = 0 then n * 3 else n

-- Define a function to represent a sequence of operations
def apply_operations (ops : List (ℕ → ℕ)) (start : ℕ) : ℕ :=
  ops.foldl (λ acc op => op acc) start

-- Define the theorem
theorem min_keystrokes_to_243 :
  ∃ (ops : List (ℕ → ℕ)), 
    (∀ op ∈ ops, op ∈ [add_one, multiply_two, multiply_three]) ∧
    apply_operations ops 1 = 243 ∧
    ops.length = 5 ∧
    (∀ (other_ops : List (ℕ → ℕ)), 
      (∀ op ∈ other_ops, op ∈ [add_one, multiply_two, multiply_three]) →
      apply_operations other_ops 1 = 243 →
      other_ops.length ≥ 5) :=
sorry

end NUMINAMATH_CALUDE_min_keystrokes_to_243_l146_14684


namespace NUMINAMATH_CALUDE_least_number_remainder_l146_14632

theorem least_number_remainder (n : ℕ) : 
  (n % 20 = 1929 % 20 ∧ n % 2535 = 1929 ∧ n % 40 = 34) →
  (∀ m : ℕ, m < n → ¬(m % 20 = 1929 % 20 ∧ m % 2535 = 1929 ∧ m % 40 = 34)) →
  n = 1394 →
  n % 20 = 14 := by
sorry

end NUMINAMATH_CALUDE_least_number_remainder_l146_14632


namespace NUMINAMATH_CALUDE_sally_lost_balloons_l146_14617

/-- The number of orange balloons Sally lost -/
def balloons_lost (initial_count current_count : ℕ) : ℕ :=
  initial_count - current_count

theorem sally_lost_balloons (initial_count current_count : ℕ) 
  (h1 : initial_count = 9)
  (h2 : current_count = 7) :
  balloons_lost initial_count current_count = 2 := by
  sorry

end NUMINAMATH_CALUDE_sally_lost_balloons_l146_14617


namespace NUMINAMATH_CALUDE_smallest_stair_count_l146_14613

theorem smallest_stair_count (n : ℕ) : 
  (n > 20 ∧ n % 3 = 1 ∧ n % 5 = 4) → 
  (∀ m : ℕ, m > 20 ∧ m % 3 = 1 ∧ m % 5 = 4 → m ≥ n) → 
  n = 34 := by
sorry

end NUMINAMATH_CALUDE_smallest_stair_count_l146_14613


namespace NUMINAMATH_CALUDE_proportion_with_one_half_one_third_l146_14625

def forms_proportion (a b c d : ℚ) : Prop := a / b = c / d

theorem proportion_with_one_half_one_third :
  forms_proportion (1/2) (1/3) 3 2 ∧
  ¬forms_proportion (1/2) (1/3) 5 4 ∧
  ¬forms_proportion (1/2) (1/3) (1/3) (1/4) ∧
  ¬forms_proportion (1/2) (1/3) (1/3) (1/2) :=
by sorry

end NUMINAMATH_CALUDE_proportion_with_one_half_one_third_l146_14625


namespace NUMINAMATH_CALUDE_cyclic_inequality_sqrt_l146_14689

theorem cyclic_inequality_sqrt (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (Real.sqrt (3 * x * (x + y) * (y + z)) +
   Real.sqrt (3 * y * (y + z) * (z + x)) +
   Real.sqrt (3 * z * (z + x) * (x + y))) ≤
  Real.sqrt (4 * (x + y + z)^3) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_sqrt_l146_14689


namespace NUMINAMATH_CALUDE_max_rect_box_length_l146_14681

-- Define the dimensions of the wooden box in centimeters
def wooden_box_length : ℝ := 800
def wooden_box_width : ℝ := 700
def wooden_box_height : ℝ := 600

-- Define the dimensions of the rectangular box in centimeters
def rect_box_width : ℝ := 7
def rect_box_height : ℝ := 6

-- Define the maximum number of rectangular boxes
def max_boxes : ℕ := 2000000

-- Theorem statement
theorem max_rect_box_length :
  ∀ x : ℝ,
  x > 0 →
  (x * rect_box_width * rect_box_height * max_boxes : ℝ) ≤ wooden_box_length * wooden_box_width * wooden_box_height →
  x ≤ 4 := by
sorry


end NUMINAMATH_CALUDE_max_rect_box_length_l146_14681


namespace NUMINAMATH_CALUDE_circle_covering_theorem_l146_14695

/-- A point in the plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A set of n points in the plane -/
def PointSet (n : ℕ) := Fin n → Point

/-- A circle in the plane -/
structure Circle where
  center : Point
  radius : ℝ

/-- Predicate to check if a point is covered by a circle -/
def covered (p : Point) (c : Circle) : Prop :=
  (p.x - c.center.x)^2 + (p.y - c.center.y)^2 ≤ c.radius^2

/-- Predicate to check if a set of points can be covered by a circle -/
def canBeCovered (S : Set Point) (c : Circle) : Prop :=
  ∀ p ∈ S, covered p c

theorem circle_covering_theorem (n : ℕ) (points : PointSet n) :
  (∀ (i j k : Fin n), ∃ (c : Circle), c.radius = 1 ∧ 
    canBeCovered {points i, points j, points k} c) →
  ∃ (c : Circle), c.radius = 1 ∧ canBeCovered (Set.range points) c :=
sorry

end NUMINAMATH_CALUDE_circle_covering_theorem_l146_14695


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l146_14643

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = 3 * x * y) :
  1 / x + 1 / y = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l146_14643


namespace NUMINAMATH_CALUDE_largest_after_erasing_100_l146_14615

/-- Concatenates numbers from 1 to n as a string -/
def concatenateNumbers (n : ℕ) : String :=
  (List.range n).map (fun i => toString (i + 1)) |> String.join

/-- Checks if a number is the largest possible after erasing digits -/
def isLargestAfterErasing (original : String) (erased : ℕ) (result : String) : Prop :=
  result.length = original.length - erased ∧
  ∀ (other : String), other.length = original.length - erased →
    other.toNat! ≤ result.toNat!

theorem largest_after_erasing_100 :
  isLargestAfterErasing (concatenateNumbers 60) 100 "99999785960" := by
  sorry

end NUMINAMATH_CALUDE_largest_after_erasing_100_l146_14615


namespace NUMINAMATH_CALUDE_circle_origin_outside_l146_14664

theorem circle_origin_outside (a : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 + 2*y + a - 2 = 0 → x^2 + y^2 > 0) ↔ (2 < a ∧ a < 3) :=
sorry

end NUMINAMATH_CALUDE_circle_origin_outside_l146_14664


namespace NUMINAMATH_CALUDE_companion_pair_expression_zero_l146_14623

/-- Definition of companion number pairs -/
def is_companion_pair (a b : ℚ) : Prop :=
  a / 2 + b / 3 = (a + b) / 5

/-- Theorem: For any companion number pair (m,n), 
    the expression 14m-5n-[5m-3(3n-1)]+3 always evaluates to 0 -/
theorem companion_pair_expression_zero (m n : ℚ) 
  (h : is_companion_pair m n) : 
  14*m - 5*n - (5*m - 3*(3*n - 1)) + 3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_companion_pair_expression_zero_l146_14623


namespace NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l146_14621

/-- Given a, b, s, t, u, v are real numbers satisfying the following conditions:
    - 0 < a < b
    - a, s, t, b form an arithmetic sequence
    - a, u, v, b form a geometric sequence
    Prove that s * t * (s + t) > u * v * (u + v)
-/
theorem arithmetic_geometric_inequality (a b s t u v : ℝ) 
  (h1 : 0 < a) (h2 : a < b)
  (h3 : s = (2*a + b)/3) (h4 : t = (a + 2*b)/3)  -- arithmetic sequence condition
  (h5 : u = (a^2 * b)^(1/3)) (h6 : v = (a * b^2)^(1/3))  -- geometric sequence condition
  : s * t * (s + t) > u * v * (u + v) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_inequality_l146_14621


namespace NUMINAMATH_CALUDE_work_completion_time_l146_14629

/-- Given that A can do a work in 15 days and when A and B work together for 4 days
    they complete 0.4666666666666667 of the work, prove that B can do the work alone in 20 days. -/
theorem work_completion_time (a b : ℝ) (ha : a = 1 / 15) 
    (h_together : 4 * (a + 1 / b) = 0.4666666666666667) : b = 20 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l146_14629


namespace NUMINAMATH_CALUDE_area_is_twenty_l146_14612

/-- The equation of the graph --/
def graph_equation (x y : ℝ) : Prop := abs (5 * x) + abs (2 * y) = 10

/-- The set of points satisfying the graph equation --/
def graph_set : Set (ℝ × ℝ) := {p | graph_equation p.1 p.2}

/-- The area enclosed by the graph --/
noncomputable def enclosed_area : ℝ := sorry

/-- Theorem stating that the area enclosed by the graph is 20 --/
theorem area_is_twenty : enclosed_area = 20 := by sorry

end NUMINAMATH_CALUDE_area_is_twenty_l146_14612


namespace NUMINAMATH_CALUDE_base5_multiplication_addition_l146_14600

/-- Converts a base 5 number to decimal --/
def base5ToDecimal (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a decimal number to base 5 --/
def decimalToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
  let rec aux (m : Nat) (acc : List Nat) :=
    if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
  aux n []

/-- The main theorem --/
theorem base5_multiplication_addition :
  decimalToBase5 (base5ToDecimal [1, 3, 2] * base5ToDecimal [1, 3] + base5ToDecimal [4, 1]) =
  [0, 3, 1, 0, 1] := by sorry

end NUMINAMATH_CALUDE_base5_multiplication_addition_l146_14600


namespace NUMINAMATH_CALUDE_sin_product_identity_l146_14650

theorem sin_product_identity (α β : ℝ) :
  Real.sin α * Real.sin β = (Real.sin ((α + β) / 2))^2 - (Real.sin ((α - β) / 2))^2 := by
  sorry

end NUMINAMATH_CALUDE_sin_product_identity_l146_14650


namespace NUMINAMATH_CALUDE_algebra_problem_percentage_l146_14669

theorem algebra_problem_percentage (total_problems : ℕ) 
  (linear_equations : ℕ) (h1 : total_problems = 140) 
  (h2 : linear_equations = 28) : 
  (linear_equations * 2 : ℚ) / total_problems * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_algebra_problem_percentage_l146_14669


namespace NUMINAMATH_CALUDE_marble_probability_l146_14659

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) 
  (h_total : total = 84)
  (h_white : p_white = 1/4)
  (h_green : p_green = 2/7) :
  1 - p_white - p_green = 13/28 := by
  sorry

end NUMINAMATH_CALUDE_marble_probability_l146_14659


namespace NUMINAMATH_CALUDE_married_men_fraction_l146_14602

theorem married_men_fraction (total_women : ℕ) (total_people : ℕ) 
  (h1 : total_women > 0)
  (h2 : total_people > total_women)
  (h3 : (3 : ℚ) / 7 = (total_women - (total_people - total_women)) / total_women) :
  (total_people - total_women : ℚ) / total_people = 4 / 11 := by
sorry

end NUMINAMATH_CALUDE_married_men_fraction_l146_14602


namespace NUMINAMATH_CALUDE_ellipse_t_squared_range_l146_14651

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define point H
def H : ℝ × ℝ := (3, 0)

-- Define the condition for points A and B
def intersects_ellipse (A B : ℝ × ℝ) : Prop :=
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 ∧ ∃ k : ℝ, A.2 - B.2 = k * (A.1 - B.1) ∧ A.2 = k * (A.1 - H.1) + H.2

-- Define the condition for point P
def P_condition (P : ℝ × ℝ) : Prop := ellipse P.1 P.2

-- Define the vector relation
def vector_relation (O A B P : ℝ × ℝ) (t : ℝ) : Prop :=
  (A.1 - O.1, A.2 - O.2) + (B.1 - O.1, B.2 - O.2) = t • (P.1 - O.1, P.2 - O.2)

-- Define the distance condition
def distance_condition (P A B : ℝ × ℝ) : Prop :=
  ((P.1 - A.1)^2 + (P.2 - A.2)^2)^(1/2) - ((P.1 - B.1)^2 + (P.2 - B.2)^2)^(1/2) < Real.sqrt 3

theorem ellipse_t_squared_range :
  ∀ (O A B P : ℝ × ℝ) (t : ℝ),
    intersects_ellipse A B →
    P_condition P →
    vector_relation O A B P t →
    distance_condition P A B →
    20 - Real.sqrt 283 < t^2 ∧ t^2 < 4 :=
sorry

end NUMINAMATH_CALUDE_ellipse_t_squared_range_l146_14651


namespace NUMINAMATH_CALUDE_pqr_positive_iff_p_q_r_positive_l146_14699

theorem pqr_positive_iff_p_q_r_positive
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (P : ℝ) (hP : P = a + b - c)
  (Q : ℝ) (hQ : Q = b + c - a)
  (R : ℝ) (hR : R = c + a - b) :
  P * Q * R > 0 ↔ P > 0 ∧ Q > 0 ∧ R > 0 := by
sorry

end NUMINAMATH_CALUDE_pqr_positive_iff_p_q_r_positive_l146_14699


namespace NUMINAMATH_CALUDE_algebraic_simplification_l146_14686

theorem algebraic_simplification (x y : ℝ) :
  (18 * x^3 * y) * (8 * x * y^2) * (1 / (6 * x * y)^2) = 4 * x * y := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l146_14686


namespace NUMINAMATH_CALUDE_angle_c_measure_l146_14698

/-- Given a triangle ABC where the sum of angles A and B is 110°, prove that the measure of angle C is 70°. -/
theorem angle_c_measure (A B C : ℝ) (h1 : A + B = 110) (h2 : A + B + C = 180) : C = 70 := by
  sorry

end NUMINAMATH_CALUDE_angle_c_measure_l146_14698


namespace NUMINAMATH_CALUDE_parabola_equation_l146_14610

/-- A parabola in the Cartesian coordinate system with directrix y = 4 -/
structure Parabola where
  /-- The equation of the parabola -/
  equation : ℝ → ℝ → Prop

/-- The standard form of a parabola equation -/
def StandardForm (p : ℝ) : ℝ → ℝ → Prop :=
  fun x y => x^2 = -4*p*y

/-- Theorem: The standard equation of a parabola with directrix y = 4 is x^2 = -16y -/
theorem parabola_equation (P : Parabola) : 
  P.equation = StandardForm 4 := by sorry

end NUMINAMATH_CALUDE_parabola_equation_l146_14610


namespace NUMINAMATH_CALUDE_tangent_roots_expression_value_l146_14692

theorem tangent_roots_expression_value (α β : Real) : 
  (∃ x y : Real, x^2 - 4*x - 2 = 0 ∧ y^2 - 4*y - 2 = 0 ∧ x ≠ y ∧ Real.tan α = x ∧ Real.tan β = y) →
  (Real.cos (α + β))^2 + 2*(Real.sin (α + β))*(Real.cos (α + β)) - 2*(Real.sin (α + β))^2 = 1/25 := by
  sorry

end NUMINAMATH_CALUDE_tangent_roots_expression_value_l146_14692


namespace NUMINAMATH_CALUDE_lena_tennis_win_probability_l146_14682

theorem lena_tennis_win_probability :
  ∀ (p_lose : ℚ),
  p_lose = 3/7 →
  (∀ (p_win : ℚ), p_win + p_lose = 1 → p_win = 4/7) :=
by sorry

end NUMINAMATH_CALUDE_lena_tennis_win_probability_l146_14682


namespace NUMINAMATH_CALUDE_number_of_men_is_correct_l146_14634

/-- The number of men in a group where:
  1. The average age of the group increases by 2 years when two men are replaced.
  2. The ages of the two men being replaced are 21 and 23 years.
  3. The average age of the two new men is 37 years. -/
def number_of_men : ℕ :=
  let age_increase : ℕ := 2
  let replaced_men_ages : Fin 2 → ℕ := ![21, 23]
  let new_men_average_age : ℕ := 37
  15

theorem number_of_men_is_correct :
  let age_increase : ℕ := 2
  let replaced_men_ages : Fin 2 → ℕ := ![21, 23]
  let new_men_average_age : ℕ := 37
  number_of_men = 15 := by
  sorry

end NUMINAMATH_CALUDE_number_of_men_is_correct_l146_14634


namespace NUMINAMATH_CALUDE_g_difference_l146_14647

/-- Given a function g(n) = 1/2 * n^2 * (n+3), prove that g(s) - g(s-1) = 1/2 * (3s - 2) for any real number s. -/
theorem g_difference (s : ℝ) : 
  let g : ℝ → ℝ := λ n => (1/2) * n^2 * (n + 3)
  g s - g (s - 1) = (1/2) * (3*s - 2) := by
sorry

end NUMINAMATH_CALUDE_g_difference_l146_14647


namespace NUMINAMATH_CALUDE_watch_cost_price_l146_14620

theorem watch_cost_price (loss_price gain_price : ℝ) : 
  loss_price = 0.9 * 1500 →
  gain_price = 1.04 * 1500 →
  gain_price - loss_price = 210 →
  1500 = (210 : ℝ) / 0.14 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_price_l146_14620


namespace NUMINAMATH_CALUDE_monotonic_quadratic_function_l146_14654

/-- A function f(x) = ax² + x + 1 is monotonically increasing in the interval [-2, +∞) if and only if 0 ≤ a ≤ 1/4 -/
theorem monotonic_quadratic_function (a : ℝ) :
  (∀ x : ℝ, x ≥ -2 → Monotone (fun x => a * x^2 + x + 1)) ↔ 0 ≤ a ∧ a ≤ 1/4 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_quadratic_function_l146_14654


namespace NUMINAMATH_CALUDE_complement_intersection_AB_l146_14679

open Set

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_intersection_AB : (U \ (A ∩ B)) = {1, 4, 5} := by
  sorry

end NUMINAMATH_CALUDE_complement_intersection_AB_l146_14679


namespace NUMINAMATH_CALUDE_brigade_task_completion_time_l146_14670

theorem brigade_task_completion_time :
  ∀ (x : ℝ),
  (x > 0) →
  (x - 15 > 0) →
  (18 / x + 6 / (x - 15) = 0.6) →
  x = 62.25 :=
by
  sorry

end NUMINAMATH_CALUDE_brigade_task_completion_time_l146_14670


namespace NUMINAMATH_CALUDE_c_value_l146_14658

theorem c_value (x y : ℝ) (h : 2 * x + 5 * y = 3) :
  let c := Real.sqrt ((4 ^ (x + 1/2)) * (32 ^ y))
  c = 4 := by
sorry

end NUMINAMATH_CALUDE_c_value_l146_14658


namespace NUMINAMATH_CALUDE_square_side_length_average_l146_14662

theorem square_side_length_average (a b c : ℝ) (ha : a = 25) (hb : b = 64) (hc : c = 121) :
  (a.sqrt + b.sqrt + c.sqrt) / 3 = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_side_length_average_l146_14662


namespace NUMINAMATH_CALUDE_find_a_l146_14672

theorem find_a : ∃ a : ℝ, (2 * 1 - a * (-1) = 3) ∧ a = 1 := by sorry

end NUMINAMATH_CALUDE_find_a_l146_14672


namespace NUMINAMATH_CALUDE_sqrt_fraction_simplification_l146_14618

theorem sqrt_fraction_simplification :
  Real.sqrt ((25 : ℝ) / 49 - 16 / 81) = Real.sqrt 1241 / 63 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_simplification_l146_14618


namespace NUMINAMATH_CALUDE_silk_diameter_scientific_notation_l146_14674

/-- The diameter of a certain silk in meters -/
def silk_diameter : ℝ := 0.000014

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def to_scientific_notation (x : ℝ) : ScientificNotation :=
  sorry

theorem silk_diameter_scientific_notation :
  to_scientific_notation silk_diameter = ScientificNotation.mk 1.4 (-5) sorry :=
sorry

end NUMINAMATH_CALUDE_silk_diameter_scientific_notation_l146_14674


namespace NUMINAMATH_CALUDE_fourth_power_sum_l146_14660

theorem fourth_power_sum (a b c : ℝ) 
  (sum_1 : a + b + c = 1)
  (sum_2 : a^2 + b^2 + c^2 = 2)
  (sum_3 : a^3 + b^3 + c^3 = 3) :
  a^4 + b^4 + c^4 = 25/6 := by sorry

end NUMINAMATH_CALUDE_fourth_power_sum_l146_14660


namespace NUMINAMATH_CALUDE_eventually_periodic_sequence_l146_14635

def RecursiveSequence (p : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, n ≥ 4 → p n * (p (n-1) * p (n-2) + p (n-3) + p (n-4)) = 
    p (n-1) + p (n-2) + p (n-3) * p (n-4)

theorem eventually_periodic_sequence 
  (p : ℕ → ℤ) 
  (h_bounded : ∃ M : ℤ, ∀ n : ℕ, |p n| ≤ M) 
  (h_recursive : RecursiveSequence p) : 
  ∃ (T : ℕ) (N : ℕ), T > 0 ∧ ∀ n : ℕ, n ≥ N → p n = p (n + T) := by
sorry

end NUMINAMATH_CALUDE_eventually_periodic_sequence_l146_14635


namespace NUMINAMATH_CALUDE_no_integer_solution_for_cornelia_age_l146_14606

theorem no_integer_solution_for_cornelia_age :
  ∀ (C : ℕ) (K : ℕ),
    K = 30 →
    C + 20 = 2 * (K + 20) →
    (K - 5)^2 = 3 * (C - 5) →
    False :=
by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_for_cornelia_age_l146_14606


namespace NUMINAMATH_CALUDE_simplify_expression_l146_14627

theorem simplify_expression (x : ℝ) : 5*x + 9*x^2 + 8 - (6 - 5*x - 3*x^2) = 12*x^2 + 10*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l146_14627


namespace NUMINAMATH_CALUDE_employee_wage_calculation_l146_14691

theorem employee_wage_calculation (revenue : ℝ) (num_employees : ℕ) 
  (tax_rate : ℝ) (marketing_rate : ℝ) (operational_rate : ℝ) (wage_rate : ℝ) :
  revenue = 400000 →
  num_employees = 10 →
  tax_rate = 0.1 →
  marketing_rate = 0.05 →
  operational_rate = 0.2 →
  wage_rate = 0.15 →
  let after_tax := revenue * (1 - tax_rate)
  let after_marketing := after_tax * (1 - marketing_rate)
  let after_operational := after_marketing * (1 - operational_rate)
  let total_wages := after_operational * wage_rate
  let wage_per_employee := total_wages / num_employees
  wage_per_employee = 4104 :=
by sorry

end NUMINAMATH_CALUDE_employee_wage_calculation_l146_14691


namespace NUMINAMATH_CALUDE_systems_solutions_l146_14667

-- Define the systems of equations
def system1 (x y : ℝ) : Prop :=
  y = 2 * x - 3 ∧ 3 * x + 2 * y = 8

def system2 (x y : ℝ) : Prop :=
  x + 2 * y = 3 ∧ 2 * x - 4 * y = -10

-- State the theorem
theorem systems_solutions :
  (∃ x y : ℝ, system1 x y ∧ x = 2 ∧ y = 1) ∧
  (∃ x y : ℝ, system2 x y ∧ x = -1 ∧ y = 2) :=
sorry

end NUMINAMATH_CALUDE_systems_solutions_l146_14667


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_sixty_one_satisfies_conditions_smallest_integer_is_sixty_one_l146_14676

theorem smallest_integer_with_remainder_one (n : ℕ) : n > 1 ∧ 
  n % 4 = 1 ∧ n % 5 = 1 ∧ n % 6 = 1 → n ≥ 61 :=
by
  sorry

theorem sixty_one_satisfies_conditions : 
  61 > 1 ∧ 61 % 4 = 1 ∧ 61 % 5 = 1 ∧ 61 % 6 = 1 :=
by
  sorry

theorem smallest_integer_is_sixty_one : 
  ∃ (n : ℕ), n > 1 ∧ n % 4 = 1 ∧ n % 5 = 1 ∧ n % 6 = 1 ∧ 
  ∀ (m : ℕ), m > 1 ∧ m % 4 = 1 ∧ m % 5 = 1 ∧ m % 6 = 1 → m ≥ n :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainder_one_sixty_one_satisfies_conditions_smallest_integer_is_sixty_one_l146_14676


namespace NUMINAMATH_CALUDE_prob_red_then_black_is_three_fourths_l146_14624

/-- A deck of cards with red and black cards -/
structure Deck :=
  (total_cards : ℕ)
  (red_cards : ℕ)
  (black_cards : ℕ)
  (h_total : total_cards = red_cards + black_cards)
  (h_equal : red_cards = black_cards)

/-- The probability of drawing a red card first and a black card second -/
def prob_red_then_black (d : Deck) : ℚ :=
  (d.red_cards : ℚ) * d.black_cards / (d.total_cards * (d.total_cards - 1))

/-- Theorem: For a deck with 64 cards, half red and half black,
    the probability of drawing a red card first and a black card second is 3/4 -/
theorem prob_red_then_black_is_three_fourths (d : Deck) 
    (h_total : d.total_cards = 64) : prob_red_then_black d = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_black_is_three_fourths_l146_14624


namespace NUMINAMATH_CALUDE_inequality_proof_l146_14603

theorem inequality_proof (x y : ℝ) : (1 / 2) * (x^2 + y^2) - x * y ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l146_14603


namespace NUMINAMATH_CALUDE_range_of_m_l146_14637

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, -1 < x ∧ x < 4 → x > 2 * m^2 - 3) ∧ 
  (∃ x : ℝ, x > 2 * m^2 - 3 ∧ (x ≤ -1 ∨ x ≥ 4)) → 
  -1 ≤ m ∧ m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l146_14637


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l146_14636

theorem fraction_to_decimal : (45 : ℚ) / (2^2 * 5^3) = (9 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l146_14636


namespace NUMINAMATH_CALUDE_problem_1_l146_14653

theorem problem_1 : -36 * (3/4 - 1/6 + 2/9 - 5/12) + |(-21/5) / (7/25)| = 61 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l146_14653


namespace NUMINAMATH_CALUDE_solution_set_l146_14641

theorem solution_set : 
  {x : ℝ | x * (x + 3)^2 * (5 - x) = 0 ∧ x^2 + 3*x + 2 > 0} = {-3, 0, 5} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_l146_14641


namespace NUMINAMATH_CALUDE_equilateral_triangle_tiling_l146_14640

theorem equilateral_triangle_tiling (m : ℕ) : 
  (∃ (t₁ t₂ : ℕ), 
    m = t₁ + t₂ ∧ 
    t₁ - t₂ = 5 ∧ 
    t₁ ≥ 5 ∧ 
    3 * t₁ + t₂ + 2 * (25 - t₁ - t₂) = 55) ↔ 
  (m % 2 = 1 ∧ m ≥ 5 ∧ m ≤ 25) :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_tiling_l146_14640


namespace NUMINAMATH_CALUDE_quadratic_congruence_solution_unique_solution_modulo_l146_14648

theorem quadratic_congruence_solution :
  ∃ (x : ℕ), x^2 - x + 2 ≡ 0 [ZMOD 7] ∧ x ≡ 4 [ZMOD 7] := by sorry

theorem unique_solution_modulo :
  ∀ (n : ℕ), n ≥ 2 →
    (∃! (x : ℕ), x^2 - x + 2 ≡ 0 [ZMOD n]) ↔ n = 7 := by sorry

end NUMINAMATH_CALUDE_quadratic_congruence_solution_unique_solution_modulo_l146_14648


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l146_14678

/-- A hyperbola with given asymptotes -/
structure Hyperbola where
  /-- The asymptotes of the hyperbola are x ± 2y = 0 -/
  asymptotes : ∀ (x y : ℝ), x = 2*y ∨ x = -2*y

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := 
  sorry

/-- Theorem stating that the eccentricity of the hyperbola is either √5 or √5/2 -/
theorem hyperbola_eccentricity (h : Hyperbola) : 
  eccentricity h = Real.sqrt 5 ∨ eccentricity h = (Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l146_14678


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l146_14633

theorem sum_of_coefficients : ∃ (a b c : ℕ+), 
  (a.val : ℝ) * Real.sqrt 6 + (b.val : ℝ) * Real.sqrt 8 = c.val * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) ∧ 
  (∀ (a' b' c' : ℕ+), (a'.val : ℝ) * Real.sqrt 6 + (b'.val : ℝ) * Real.sqrt 8 = c'.val * (Real.sqrt 6 + 1 / Real.sqrt 6 + Real.sqrt 8 + 1 / Real.sqrt 8) → c'.val ≥ c.val) →
  a.val + b.val + c.val = 67 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l146_14633


namespace NUMINAMATH_CALUDE_min_dot_product_on_hyperbola_l146_14644

theorem min_dot_product_on_hyperbola (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let m : ℝ × ℝ := (1, Real.sqrt (a^2 + 1/a^2))
  let B : ℝ × ℝ := (b, 1/b)
  m.1 * B.1 + m.2 * B.2 ≥ 2 * Real.sqrt (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_min_dot_product_on_hyperbola_l146_14644


namespace NUMINAMATH_CALUDE_min_valid_positions_l146_14638

/-- Represents a disk with a certain number of sectors and red sectors. -/
structure Disk :=
  (total_sectors : ℕ)
  (red_sectors : ℕ)
  (h_red_le_total : red_sectors ≤ total_sectors)

/-- Represents the configuration of two overlapping disks. -/
structure DiskOverlay :=
  (disk1 : Disk)
  (disk2 : Disk)
  (h_same_sectors : disk1.total_sectors = disk2.total_sectors)

/-- Calculates the number of positions with at most 20 overlapping red sectors. -/
def count_valid_positions (overlay : DiskOverlay) : ℕ :=
  overlay.disk1.total_sectors - (overlay.disk1.red_sectors * overlay.disk2.red_sectors) / 21 + 1

theorem min_valid_positions (overlay : DiskOverlay) 
  (h_total : overlay.disk1.total_sectors = 1965)
  (h_red1 : overlay.disk1.red_sectors = 200)
  (h_red2 : overlay.disk2.red_sectors = 200) :
  count_valid_positions overlay = 61 :=
sorry

end NUMINAMATH_CALUDE_min_valid_positions_l146_14638


namespace NUMINAMATH_CALUDE_rectangle_area_with_tangent_circle_l146_14605

/-- Given a rectangle ABCD with a circle of radius r tangent to sides AB, AD, and CD,
    and passing through a point one-third the distance from A to C along diagonal AC,
    the area of the rectangle is (2√2)/3 * r^2. -/
theorem rectangle_area_with_tangent_circle (r : ℝ) (h : r > 0) :
  ∃ (w h : ℝ),
    w > 0 ∧ h > 0 ∧
    h = r ∧
    (w^2 + h^2) = 9 * r^2 ∧
    w * h = (2 * Real.sqrt 2 / 3) * r^2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_with_tangent_circle_l146_14605


namespace NUMINAMATH_CALUDE_inverse_proportion_relationship_l146_14687

theorem inverse_proportion_relationship (x₁ x₂ y₁ y₂ : ℝ) :
  x₁ < x₂ → x₂ < 0 → y₁ = 2 / x₁ → y₂ = 2 / x₂ → y₂ < y₁ ∧ y₁ < 0 := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_relationship_l146_14687


namespace NUMINAMATH_CALUDE_base_7_to_10_conversion_l146_14616

/-- Converts a list of digits in a given base to its decimal (base 10) representation -/
def to_decimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldl (fun acc d => acc * base + d) 0

/-- The base 7 number represented by the digits [3, 2, 0, 6] -/
def base_7_number : List Nat := [3, 2, 0, 6]

/-- Theorem stating that the base 10 representation of 3206 in base 7 is 1133 -/
theorem base_7_to_10_conversion :
  to_decimal base_7_number 7 = 1133 := by
  sorry

end NUMINAMATH_CALUDE_base_7_to_10_conversion_l146_14616


namespace NUMINAMATH_CALUDE_negation_equivalence_l146_14668

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀^2 + x₀ + 1 < 0) ↔ (∀ x : ℝ, x^2 + x + 1 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l146_14668


namespace NUMINAMATH_CALUDE_parabola_p_value_l146_14608

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  h : p > 0

/-- Point on a parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  h : y^2 = 2 * C.p * x

/-- Theorem: Value of p for a parabola given specific point conditions -/
theorem parabola_p_value (C : Parabola) (A : PointOnParabola C)
  (h1 : Real.sqrt ((A.x - C.p/2)^2 + A.y^2) = 12)  -- Distance from A to focus is 12
  (h2 : A.x = 9)  -- Distance from A to y-axis is 9
  : C.p = 6 := by
  sorry

end NUMINAMATH_CALUDE_parabola_p_value_l146_14608


namespace NUMINAMATH_CALUDE_ellipse_foci_distance_l146_14604

/-- The distance between the foci of the ellipse 9x^2 + y^2 = 900 is 40√2 -/
theorem ellipse_foci_distance :
  let ellipse := {(x, y) : ℝ × ℝ | 9 * x^2 + y^2 = 900}
  ∃ f₁ f₂ : ℝ × ℝ, f₁ ∈ ellipse ∧ f₂ ∈ ellipse ∧ ‖f₁ - f₂‖ = 40 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_foci_distance_l146_14604


namespace NUMINAMATH_CALUDE_marco_score_percentage_l146_14671

/-- Proves that Marco scored 10% less than the average test score -/
theorem marco_score_percentage (average_score : ℝ) (margaret_score : ℝ) (marco_score : ℝ) :
  average_score = 90 →
  margaret_score = 86 →
  margaret_score = marco_score + 5 →
  (average_score - marco_score) / average_score = 0.1 := by
  sorry

end NUMINAMATH_CALUDE_marco_score_percentage_l146_14671


namespace NUMINAMATH_CALUDE_probability_perfect_square_sum_l146_14685

def roll_outcomes : ℕ := 64

def favorable_outcomes : ℕ := 12

theorem probability_perfect_square_sum (roll_outcomes : ℕ) (favorable_outcomes : ℕ) :
  (favorable_outcomes : ℚ) / (roll_outcomes : ℚ) = 3 / 16 :=
by sorry

end NUMINAMATH_CALUDE_probability_perfect_square_sum_l146_14685


namespace NUMINAMATH_CALUDE_factorization_of_cubic_minus_linear_l146_14628

theorem factorization_of_cubic_minus_linear (x : ℝ) :
  3 * x^3 - 12 * x = 3 * x * (x - 2) * (x + 2) := by sorry

end NUMINAMATH_CALUDE_factorization_of_cubic_minus_linear_l146_14628


namespace NUMINAMATH_CALUDE_no_solution_implies_m_equals_one_l146_14626

theorem no_solution_implies_m_equals_one (m : ℝ) : 
  (∀ x : ℝ, x ≠ 2 → (x - 3) / (x - 2) ≠ m / (2 - x)) →
  m = 1 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_implies_m_equals_one_l146_14626


namespace NUMINAMATH_CALUDE_largest_m_for_inequality_l146_14607

theorem largest_m_for_inequality : ∃ m : ℕ+, 
  (m = 27) ∧ 
  (∀ n : ℕ+, n ≤ m → (2*n + 1)/(3*n + 8) < (Real.sqrt 5 - 1)/2 ∧ (Real.sqrt 5 - 1)/2 < (n + 7)/(2*n + 1)) ∧
  (∀ m' : ℕ+, m' > m → ∃ n : ℕ+, n ≤ m' ∧ ((2*n + 1)/(3*n + 8) ≥ (Real.sqrt 5 - 1)/2 ∨ (Real.sqrt 5 - 1)/2 ≥ (n + 7)/(2*n + 1))) :=
by sorry

end NUMINAMATH_CALUDE_largest_m_for_inequality_l146_14607


namespace NUMINAMATH_CALUDE_milton_books_l146_14663

theorem milton_books (z b : ℕ) : z + b = 80 → b = 4 * z → z = 16 := by
  sorry

end NUMINAMATH_CALUDE_milton_books_l146_14663


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l146_14642

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∀ x y : ℝ, x + m * y + 1 = 0 ∧ m^2 * x - 2 * y - 1 = 0 → 
    (1 : ℝ) * (-m^2 : ℝ) = -1) → 
  m = 0 ∨ m = 2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l146_14642


namespace NUMINAMATH_CALUDE_circle_center_sum_l146_14688

/-- Given a circle with equation x^2 + y^2 = 6x + 18y - 63, 
    prove that the sum of the coordinates of its center is 12. -/
theorem circle_center_sum (h k : ℝ) : 
  (∀ x y : ℝ, x^2 + y^2 = 6*x + 18*y - 63 ↔ (x - h)^2 + (y - k)^2 = (h^2 + k^2 - 6*h - 18*k + 63)) →
  h + k = 12 := by
sorry

end NUMINAMATH_CALUDE_circle_center_sum_l146_14688


namespace NUMINAMATH_CALUDE_min_value_fraction_l146_14656

theorem min_value_fraction (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (a + b) * (a + 2) * (b + 2) / (16 * a * b) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l146_14656


namespace NUMINAMATH_CALUDE_candy_bar_earnings_difference_l146_14683

/-- The problem of calculating the difference in earnings between Tina and Marvin from selling candy bars. -/
theorem candy_bar_earnings_difference :
  let candy_bar_price : ℕ := 2
  let marvins_sales : ℕ := 35
  let tinas_sales : ℕ := 3 * marvins_sales
  let marvins_earnings : ℕ := candy_bar_price * marvins_sales
  let tinas_earnings : ℕ := candy_bar_price * tinas_sales
  tinas_earnings - marvins_earnings = 140 := by
sorry

end NUMINAMATH_CALUDE_candy_bar_earnings_difference_l146_14683


namespace NUMINAMATH_CALUDE_ternary_221_greater_than_binary_10111_l146_14677

/-- Converts a ternary number (represented as a list of digits) to decimal --/
def ternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3^i)) 0

/-- Converts a binary number (represented as a list of digits) to decimal --/
def binary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (2^i)) 0

/-- The ternary number 221 --/
def a : List Nat := [1, 2, 2]

/-- The binary number 10111 --/
def b : List Nat := [1, 1, 1, 0, 1]

theorem ternary_221_greater_than_binary_10111 :
  ternary_to_decimal a > binary_to_decimal b := by
  sorry

end NUMINAMATH_CALUDE_ternary_221_greater_than_binary_10111_l146_14677


namespace NUMINAMATH_CALUDE_leading_coefficient_of_p_l146_14630

/-- The polynomial in question -/
def p (x : ℝ) : ℝ := -5 * (x^4 - 2*x^3 + 3*x) + 9 * (x^4 - x + 4) - 6 * (3*x^4 - x^3 + 2)

/-- The leading coefficient of a polynomial is the coefficient of its highest degree term -/
def leading_coefficient (p : ℝ → ℝ) : ℝ :=
  sorry -- Definition of leading coefficient

theorem leading_coefficient_of_p :
  leading_coefficient p = -14 := by
  sorry

end NUMINAMATH_CALUDE_leading_coefficient_of_p_l146_14630


namespace NUMINAMATH_CALUDE_joan_apples_l146_14661

/-- The number of apples Joan gave to Melanie -/
def apples_given : ℕ := 27

/-- The number of apples Joan has now -/
def apples_left : ℕ := 16

/-- The total number of apples Joan picked from the orchard -/
def total_apples : ℕ := apples_given + apples_left

theorem joan_apples : total_apples = 43 := by
  sorry

end NUMINAMATH_CALUDE_joan_apples_l146_14661


namespace NUMINAMATH_CALUDE_triangle_area_theorem_l146_14697

noncomputable def f (x : ℝ) : ℝ := Real.cos x ^ 2 - Real.sin x ^ 2 + 1/2

def is_monotone_increasing (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x ≤ f y

structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)
  (acute : A < π/2 ∧ B < π/2 ∧ C < π/2)
  (side_a : a = Real.sqrt 19)
  (side_b : b = 5)
  (angle_condition : f A = 0)

theorem triangle_area_theorem (t : Triangle) :
  is_monotone_increasing f (π/2) π ∧ 
  (1/2 * t.b * Real.sqrt (19 - t.b^2 + 2*t.b*Real.sqrt 19 * Real.cos t.A)) = 15 * Real.sqrt 3 / 4 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_theorem_l146_14697


namespace NUMINAMATH_CALUDE_tower_remainder_l146_14619

/-- Represents the number of different towers that can be built with cubes up to size n -/
def T : ℕ → ℕ
| 0 => 1
| 1 => 1
| 2 => 2
| (n+1) => if n ≤ 9 then T n * (min n 4) else T n

/-- The main theorem stating the remainder when T(10) is divided by 1000 -/
theorem tower_remainder : T 10 % 1000 = 216 := by sorry

end NUMINAMATH_CALUDE_tower_remainder_l146_14619


namespace NUMINAMATH_CALUDE_students_surveyed_l146_14631

theorem students_surveyed : ℕ :=
  let total_students : ℕ := sorry
  let french_speakers : ℕ := sorry
  let french_english_speakers : ℕ := 10
  let french_only_speakers : ℕ := 40

  have h1 : french_speakers = french_english_speakers + french_only_speakers := by sorry
  have h2 : french_speakers = 50 := by sorry
  have h3 : french_speakers = total_students / 4 := by sorry

  200

/- Proof omitted -/

end NUMINAMATH_CALUDE_students_surveyed_l146_14631


namespace NUMINAMATH_CALUDE_max_sum_of_four_numbers_l146_14655

theorem max_sum_of_four_numbers (a b c d : ℕ) : 
  a < b → b < c → c < d → (c + d) + (a + b + c) = 2017 → 
  a + b + c + d ≤ 806 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_four_numbers_l146_14655


namespace NUMINAMATH_CALUDE_min_distance_to_line_l146_14657

/-- Given a line x + y + 1 = 0 in a 2D plane, the minimum distance from the point (-2, -3) to this line is 2√2 -/
theorem min_distance_to_line :
  ∀ x y : ℝ, x + y + 1 = 0 →
  (2 * Real.sqrt 2 : ℝ) ≤ Real.sqrt ((x + 2)^2 + (y + 3)^2) :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l146_14657


namespace NUMINAMATH_CALUDE_quadratic_roots_condition_l146_14646

theorem quadratic_roots_condition (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 - 4 * x + 2 = 0 ∧ a * y^2 - 4 * y + 2 = 0) ↔ 
  (a ≤ 2 ∧ a ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_condition_l146_14646


namespace NUMINAMATH_CALUDE_sqrt_neg_two_squared_l146_14601

theorem sqrt_neg_two_squared : Real.sqrt ((-2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_neg_two_squared_l146_14601


namespace NUMINAMATH_CALUDE_sylvia_earnings_l146_14693

-- Define the work durations
def monday_hours : ℚ := 5/2
def tuesday_minutes : ℕ := 40
def wednesday_start : ℕ := 9 * 60 + 15  -- 9:15 AM in minutes
def wednesday_end : ℕ := 11 * 60 + 50   -- 11:50 AM in minutes
def thursday_minutes : ℕ := 45

-- Define the hourly rate
def hourly_rate : ℚ := 4

-- Define the function to calculate total earnings
def total_earnings : ℚ :=
  let total_minutes : ℚ := 
    monday_hours * 60 + 
    tuesday_minutes + 
    (wednesday_end - wednesday_start) + 
    thursday_minutes
  let total_hours : ℚ := total_minutes / 60
  total_hours * hourly_rate

-- Theorem statement
theorem sylvia_earnings : total_earnings = 26 := by
  sorry

end NUMINAMATH_CALUDE_sylvia_earnings_l146_14693


namespace NUMINAMATH_CALUDE_circle_in_square_area_ratio_l146_14645

/-- The ratio of the area of a circle inscribed in a square 
    (where the circle's diameter is equal to the square's side length) 
    to the area of the square is π/4. -/
theorem circle_in_square_area_ratio : 
  ∀ s : ℝ, s > 0 → (π * (s/2)^2) / (s^2) = π/4 := by
sorry

end NUMINAMATH_CALUDE_circle_in_square_area_ratio_l146_14645


namespace NUMINAMATH_CALUDE_gas_station_candy_boxes_l146_14696

/-- The number of boxes of chocolate candy sold by a gas station -/
def chocolate_boxes : ℕ := 9 - (5 + 2)

/-- The total number of boxes sold -/
def total_boxes : ℕ := 9

/-- The number of boxes of sugar candy sold -/
def sugar_boxes : ℕ := 5

/-- The number of boxes of gum sold -/
def gum_boxes : ℕ := 2

theorem gas_station_candy_boxes :
  chocolate_boxes = 2 ∧
  chocolate_boxes + sugar_boxes + gum_boxes = total_boxes :=
sorry

end NUMINAMATH_CALUDE_gas_station_candy_boxes_l146_14696


namespace NUMINAMATH_CALUDE_polynomial_remainder_l146_14666

theorem polynomial_remainder (x : ℝ) : 
  (5*x^8 - x^7 + 3*x^6 - 5*x^4 + 6*x^3 - 7) % (3*x - 6) = 1305 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l146_14666


namespace NUMINAMATH_CALUDE_total_packs_eq_51_l146_14639

/-- The number of cookie packs sold in the first village -/
def village1_packs : ℕ := 23

/-- The number of cookie packs sold in the second village -/
def village2_packs : ℕ := 28

/-- The total number of cookie packs sold in both villages -/
def total_packs : ℕ := village1_packs + village2_packs

theorem total_packs_eq_51 : total_packs = 51 := by
  sorry

end NUMINAMATH_CALUDE_total_packs_eq_51_l146_14639


namespace NUMINAMATH_CALUDE_expression_value_l146_14680

theorem expression_value (x y z w : ℝ) 
  (h1 : 4 * x * z + y * w = 3) 
  (h2 : x * w + y * z = 6) : 
  (2 * x + y) * (2 * z + w) = 15 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l146_14680


namespace NUMINAMATH_CALUDE_y_over_x_equals_two_l146_14673

theorem y_over_x_equals_two (x y : ℝ) (h : y / 2 = (2 * y - x) / 3) : y / x = 2 := by
  sorry

end NUMINAMATH_CALUDE_y_over_x_equals_two_l146_14673


namespace NUMINAMATH_CALUDE_square_of_98_l146_14609

theorem square_of_98 : (98 : ℕ) ^ 2 = 9604 := by sorry

end NUMINAMATH_CALUDE_square_of_98_l146_14609


namespace NUMINAMATH_CALUDE_mango_purchase_proof_l146_14622

def grape_quantity : ℕ := 11
def grape_price : ℕ := 98
def mango_price : ℕ := 50
def total_payment : ℕ := 1428

def mango_quantity : ℕ := (total_payment - grape_quantity * grape_price) / mango_price

theorem mango_purchase_proof : mango_quantity = 7 := by
  sorry

end NUMINAMATH_CALUDE_mango_purchase_proof_l146_14622


namespace NUMINAMATH_CALUDE_albert_needs_twelve_dollars_l146_14675

/-- The amount of additional money Albert needs to buy art supplies -/
def additional_money_needed (paintbrush_cost set_of_paints_cost wooden_easel_cost current_money : ℚ) : ℚ :=
  paintbrush_cost + set_of_paints_cost + wooden_easel_cost - current_money

theorem albert_needs_twelve_dollars :
  additional_money_needed 1.50 4.35 12.65 6.50 = 12.00 := by
  sorry

end NUMINAMATH_CALUDE_albert_needs_twelve_dollars_l146_14675


namespace NUMINAMATH_CALUDE_smallest_tangent_circle_l146_14665

-- Define the given line
def given_line (x y : ℝ) : Prop := x - y - 4 = 0

-- Define the given circle
def given_circle (x y : ℝ) : Prop := (x + 1)^2 + (y - 1)^2 = 2

-- Define the solution circle
def solution_circle (x y : ℝ) : Prop := (x - 1)^2 + (y + 1)^2 = 2

-- Define a general circle with center (a, b) and radius r
def general_circle (x y a b r : ℝ) : Prop := (x - a)^2 + (y - b)^2 = r^2

-- Theorem statement
theorem smallest_tangent_circle :
  ∃! (a b r : ℝ),
    (∀ x y, general_circle x y a b r → 
      (∃ x₀ y₀, given_line x₀ y₀ ∧ general_circle x₀ y₀ a b r) ∧
      (∃ x₁ y₁, given_circle x₁ y₁ ∧ general_circle x₁ y₁ a b r)) ∧
    (∀ a' b' r', 
      (∀ x y, general_circle x y a' b' r' → 
        (∃ x₀ y₀, given_line x₀ y₀ ∧ general_circle x₀ y₀ a' b' r') ∧
        (∃ x₁ y₁, given_circle x₁ y₁ ∧ general_circle x₁ y₁ a' b' r')) →
      r ≤ r') ∧
    (∀ x y, general_circle x y a b r ↔ solution_circle x y) :=
sorry

end NUMINAMATH_CALUDE_smallest_tangent_circle_l146_14665
