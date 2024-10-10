import Mathlib

namespace bookstore_max_revenue_l2880_288073

/-- The revenue function for the bookstore -/
def R (p : ℝ) : ℝ := p * (200 - 8 * p)

/-- The theorem stating the maximum revenue and optimal price -/
theorem bookstore_max_revenue :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 25 ∧
  R p = 1250 ∧
  ∀ (q : ℝ), 0 ≤ q ∧ q ≤ 25 → R q ≤ R p ∧
  p = 12.5 := by
  sorry

end bookstore_max_revenue_l2880_288073


namespace car_value_reduction_l2880_288024

theorem car_value_reduction (original_price current_value : ℝ) : 
  current_value = 0.7 * original_price → 
  current_value = 2800 → 
  original_price = 4000 := by
sorry

end car_value_reduction_l2880_288024


namespace reciprocal_of_negative_three_l2880_288002

theorem reciprocal_of_negative_three :
  ∃ x : ℚ, x * (-3) = 1 ∧ x = -1/3 := by
  sorry

end reciprocal_of_negative_three_l2880_288002


namespace valid_sequences_of_length_21_l2880_288098

/-- Counts valid binary sequences of given length -/
def count_valid_sequences (n : ℕ) : ℕ :=
  if n < 3 then 0
  else if n = 3 then 1
  else if n = 4 then 1
  else if n = 5 then 1
  else if n = 6 then 2
  else count_valid_sequences (n - 4) + 2 * count_valid_sequences (n - 5) + 2 * count_valid_sequences (n - 6)

/-- The main theorem stating the number of valid sequences of length 21 -/
theorem valid_sequences_of_length_21 :
  count_valid_sequences 21 = 135 := by sorry

end valid_sequences_of_length_21_l2880_288098


namespace motion_of_q_l2880_288075

/-- Point on a circle -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Motion of a point on a circle -/
structure CircularMotion where
  center : Point2D
  radius : ℝ
  angular_velocity : ℝ
  clockwise : Bool

/-- Given a point P moving counterclockwise on the unit circle with angular velocity ω,
    prove that the point Q(-2xy, y^2 - x^2) moves clockwise on the unit circle
    with angular velocity 2ω -/
theorem motion_of_q (ω : ℝ) (h_ω : ω > 0) :
  let p_motion : CircularMotion :=
    { center := ⟨0, 0⟩
    , radius := 1
    , angular_velocity := ω
    , clockwise := false }
  let q (p : Point2D) : Point2D :=
    ⟨-2 * p.x * p.y, p.y^2 - p.x^2⟩
  ∃ (q_motion : CircularMotion),
    q_motion.center = ⟨0, 0⟩ ∧
    q_motion.radius = 1 ∧
    q_motion.angular_velocity = 2 * ω ∧
    q_motion.clockwise = true :=
by sorry

end motion_of_q_l2880_288075


namespace sandwich_combinations_l2880_288086

def num_meats : ℕ := 10
def num_cheeses : ℕ := 12
def num_condiments : ℕ := 5

theorem sandwich_combinations :
  (num_meats) * (num_cheeses.choose 2) * (num_condiments) = 3300 := by
  sorry

end sandwich_combinations_l2880_288086


namespace birch_trees_not_adjacent_probability_l2880_288012

def total_trees : ℕ := 17
def birch_trees : ℕ := 6
def non_birch_trees : ℕ := total_trees - birch_trees

theorem birch_trees_not_adjacent_probability : 
  (Nat.choose (non_birch_trees + 1) birch_trees) / (Nat.choose total_trees birch_trees) = 77 / 1033 := by
  sorry

end birch_trees_not_adjacent_probability_l2880_288012


namespace smallest_n_congruence_l2880_288015

theorem smallest_n_congruence : ∃ (n : ℕ), n > 0 ∧ (623 * n) % 32 = (1319 * n) % 32 ∧ ∀ (m : ℕ), m > 0 → m < n → (623 * m) % 32 ≠ (1319 * m) % 32 :=
by sorry

end smallest_n_congruence_l2880_288015


namespace solution_set_quadratic_inequality_l2880_288094

theorem solution_set_quadratic_inequality (x : ℝ) :
  x^2 - |x| - 2 < 0 ↔ -2 < x ∧ x < 2 := by
  sorry

end solution_set_quadratic_inequality_l2880_288094


namespace quadratic_sum_l2880_288032

/-- A quadratic function f(x) = px^2 + qx + r with vertex (-3, 4) passing through (0, 1) -/
def QuadraticFunction (p q r : ℝ) : ℝ → ℝ := fun x ↦ p * x^2 + q * x + r

/-- The vertex of the quadratic function -/
def vertex (p q r : ℝ) : ℝ × ℝ := (-3, 4)

/-- The function passes through the point (0, 1) -/
def passes_through_origin (p q r : ℝ) : Prop :=
  QuadraticFunction p q r 0 = 1

theorem quadratic_sum (p q r : ℝ) :
  vertex p q r = (-3, 4) →
  passes_through_origin p q r →
  p + q + r = -4/3 := by
  sorry

end quadratic_sum_l2880_288032


namespace percentage_of_juniors_l2880_288057

theorem percentage_of_juniors (total : ℕ) (seniors : ℕ) :
  total = 800 →
  seniors = 160 →
  let sophomores := (total : ℚ) * (1 / 4)
  let freshmen := sophomores + 16
  let juniors := total - (freshmen + sophomores + seniors)
  (juniors / total) * 100 = 28 := by
  sorry

end percentage_of_juniors_l2880_288057


namespace cubic_equation_equivalence_l2880_288065

theorem cubic_equation_equivalence (y : ℝ) :
  6 * y^(1/3) - 3 * (y^2 / y^(2/3)) = 12 + y^(1/3) + y →
  ∃ z : ℝ, z = y^(1/3) ∧ 3 * z^4 + z^3 - 5 * z + 12 = 0 :=
by sorry

end cubic_equation_equivalence_l2880_288065


namespace fraction_value_l2880_288031

theorem fraction_value (a b c d : ℚ) 
  (h1 : a = 4 * b) 
  (h2 : b = 3 * c) 
  (h3 : c = 5 * d) : 
  (a * c) / (b * d) = 20 := by
  sorry

end fraction_value_l2880_288031


namespace surface_area_difference_l2880_288055

/-- Calculates the difference between the sum of surface areas of smaller cubes
    and the surface area of a larger cube containing them. -/
theorem surface_area_difference (larger_volume : ℝ) (num_smaller_cubes : ℕ) (smaller_volume : ℝ) :
  larger_volume = 64 →
  num_smaller_cubes = 64 →
  smaller_volume = 1 →
  (num_smaller_cubes : ℝ) * (6 * smaller_volume ^ (2/3)) - 6 * larger_volume ^ (2/3) = 288 := by
  sorry

end surface_area_difference_l2880_288055


namespace regular_quad_pyramid_angle_relation_l2880_288051

/-- A regular quadrilateral pyramid -/
structure RegularQuadPyramid where
  /-- The dihedral angle between a lateral face and the base -/
  α : ℝ
  /-- The dihedral angle between two adjacent lateral faces -/
  β : ℝ

/-- Theorem: For a regular quadrilateral pyramid, 2 cos β + cos 2α = -1 -/
theorem regular_quad_pyramid_angle_relation (P : RegularQuadPyramid) : 
  2 * Real.cos P.β + Real.cos (2 * P.α) = -1 := by
  sorry

end regular_quad_pyramid_angle_relation_l2880_288051


namespace line_y_axis_intersection_l2880_288067

/-- A line passing through two points intersects the y-axis at a specific point -/
theorem line_y_axis_intersection (x₁ y₁ x₂ y₂ : ℝ) :
  x₁ = 3 →
  y₁ = 20 →
  x₂ = -7 →
  y₂ = 2 →
  ∃ y : ℝ, y = 14.6 ∧ (y - y₁) / (0 - x₁) = (y₂ - y₁) / (x₂ - x₁) :=
by sorry

end line_y_axis_intersection_l2880_288067


namespace a_minus_b_equals_seven_l2880_288030

theorem a_minus_b_equals_seven (a b : ℝ) 
  (ha : a^2 = 9)
  (hb : |b| = 4)
  (hgt : a > b) : 
  a - b = 7 := by
sorry

end a_minus_b_equals_seven_l2880_288030


namespace total_spent_is_88_70_l2880_288003

-- Define the constants
def pizza_price : ℝ := 10
def pizza_quantity : ℕ := 5
def pizza_discount_threshold : ℕ := 3
def pizza_discount_rate : ℝ := 0.15

def soft_drink_price : ℝ := 1.5
def soft_drink_quantity : ℕ := 10

def hamburger_price : ℝ := 3
def hamburger_quantity : ℕ := 6
def hamburger_discount_threshold : ℕ := 5
def hamburger_discount_rate : ℝ := 0.1

-- Define the function to calculate the total spent
def total_spent : ℝ :=
  let robert_pizza_cost := 
    if pizza_quantity > pizza_discount_threshold
    then pizza_price * pizza_quantity * (1 - pizza_discount_rate)
    else pizza_price * pizza_quantity
  let robert_drinks_cost := soft_drink_price * soft_drink_quantity
  let teddy_hamburger_cost := 
    if hamburger_quantity > hamburger_discount_threshold
    then hamburger_price * hamburger_quantity * (1 - hamburger_discount_rate)
    else hamburger_price * hamburger_quantity
  let teddy_drinks_cost := soft_drink_price * soft_drink_quantity
  robert_pizza_cost + robert_drinks_cost + teddy_hamburger_cost + teddy_drinks_cost

-- Theorem statement
theorem total_spent_is_88_70 : total_spent = 88.70 := by sorry

end total_spent_is_88_70_l2880_288003


namespace domain_of_g_l2880_288020

/-- The domain of f(x) -/
def DomainF : Set ℝ := {x : ℝ | 0 ≤ x ∧ x ≤ 6}

/-- The function g(x) -/
noncomputable def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2 * x) / (x - 2)

/-- The domain of g(x) -/
def DomainG : Set ℝ := {x : ℝ | (0 ≤ x ∧ x < 2) ∨ (2 < x ∧ x ≤ 3)}

/-- Theorem: The domain of g(x) is correct given the domain of f(x) -/
theorem domain_of_g (f : ℝ → ℝ) (hf : ∀ x, x ∈ DomainF → f x ≠ 0) :
  ∀ x, x ∈ DomainG ↔ (2 * x ∈ DomainF ∧ x ≠ 2) :=
sorry

end domain_of_g_l2880_288020


namespace rias_initial_savings_l2880_288000

theorem rias_initial_savings (r f : ℚ) : 
  r / f = 5 / 3 →  -- Initial ratio
  (r - 160) / f = 3 / 5 →  -- New ratio after withdrawal
  r = 250 := by
sorry

end rias_initial_savings_l2880_288000


namespace riverdale_rangers_loss_percentage_l2880_288040

/-- Represents the statistics of a sports team --/
structure TeamStats where
  totalGames : ℕ
  winLossRatio : ℚ

/-- Calculates the percentage of games lost --/
def percentLost (stats : TeamStats) : ℚ :=
  let lostGames := stats.totalGames / (1 + stats.winLossRatio)
  (lostGames / stats.totalGames) * 100

/-- Theorem stating that for a team with given statistics, the percentage of games lost is 38% --/
theorem riverdale_rangers_loss_percentage :
  let stats : TeamStats := { totalGames := 65, winLossRatio := 8/5 }
  percentLost stats = 38 := by sorry


end riverdale_rangers_loss_percentage_l2880_288040


namespace ratio_of_divisors_sums_l2880_288004

def P : ℕ := 45 * 45 * 98 * 480

def sum_of_odd_divisors (n : ℕ) : ℕ := sorry

def sum_of_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisors_sums :
  (sum_of_odd_divisors P) * 126 = sum_of_even_divisors P := by sorry

end ratio_of_divisors_sums_l2880_288004


namespace xy_sum_l2880_288005

theorem xy_sum (x y : ℕ) (hx : x < 15) (hy : y < 25) (hxy : x + y + x * y = 119) :
  x + y = 20 ∨ x + y = 21 := by
  sorry

end xy_sum_l2880_288005


namespace fathers_savings_l2880_288035

theorem fathers_savings (total : ℝ) : 
  (total / 2 - (total / 2) * 0.6) = 2000 → total = 10000 := by
  sorry

end fathers_savings_l2880_288035


namespace cards_added_l2880_288038

theorem cards_added (initial_cards final_cards : ℕ) 
  (h1 : initial_cards = 9) 
  (h2 : final_cards = 13) : 
  final_cards - initial_cards = 4 := by
  sorry

end cards_added_l2880_288038


namespace parallel_planes_transitivity_l2880_288058

structure Plane

/-- Two planes are parallel -/
def parallel (p q : Plane) : Prop := sorry

theorem parallel_planes_transitivity 
  (α β γ : Plane) 
  (h1 : α ≠ β) 
  (h2 : α ≠ γ) 
  (h3 : β ≠ γ) 
  (h4 : parallel α β) 
  (h5 : parallel α γ) : 
  parallel β γ := by sorry

end parallel_planes_transitivity_l2880_288058


namespace boys_neither_happy_nor_sad_l2880_288027

/-- Given a group of children with various emotional states and genders, 
    prove the number of boys who are neither happy nor sad. -/
theorem boys_neither_happy_nor_sad 
  (total_children : ℕ) 
  (happy_children sad_children confused_children excited_children neither_happy_nor_sad : ℕ)
  (total_boys total_girls : ℕ)
  (happy_boys sad_girls confused_boys excited_girls : ℕ)
  (h1 : total_children = 80)
  (h2 : happy_children = 35)
  (h3 : sad_children = 15)
  (h4 : confused_children = 10)
  (h5 : excited_children = 5)
  (h6 : neither_happy_nor_sad = 15)
  (h7 : total_boys = 45)
  (h8 : total_girls = 35)
  (h9 : happy_boys = 8)
  (h10 : sad_girls = 7)
  (h11 : confused_boys = 4)
  (h12 : excited_girls = 3)
  (h13 : total_children = happy_children + sad_children + confused_children + excited_children + neither_happy_nor_sad)
  (h14 : total_children = total_boys + total_girls) :
  total_boys - (happy_boys + (sad_children - sad_girls) + confused_boys + (excited_children - excited_girls)) = 23 :=
by sorry

end boys_neither_happy_nor_sad_l2880_288027


namespace geometric_sequence_a3_l2880_288039

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

-- State the theorem
theorem geometric_sequence_a3 (a : ℕ → ℝ) :
  geometric_sequence a → a 1 = 1 → a 5 = 5 → a 3 = Real.sqrt 5 := by
  sorry

end geometric_sequence_a3_l2880_288039


namespace division_problem_l2880_288061

theorem division_problem : (5 + 1/2) / (2/11) = 121/4 := by
  sorry

end division_problem_l2880_288061


namespace rectangle_perimeter_l2880_288014

theorem rectangle_perimeter (area : ℝ) (side_ratio : ℝ) (perimeter : ℝ) : 
  area = 500 →
  side_ratio = 2 →
  let shorter_side := Real.sqrt (area / side_ratio)
  let longer_side := side_ratio * shorter_side
  perimeter = 2 * (shorter_side + longer_side) →
  perimeter = 30 * Real.sqrt 10 := by
  sorry

end rectangle_perimeter_l2880_288014


namespace marias_age_l2880_288097

/-- 
Given that Jose is 12 years older than Maria and the sum of their ages is 40,
prove that Maria is 14 years old.
-/
theorem marias_age (maria jose : ℕ) 
  (h1 : jose = maria + 12) 
  (h2 : maria + jose = 40) : 
  maria = 14 := by
  sorry

end marias_age_l2880_288097


namespace distance_travelled_l2880_288066

theorem distance_travelled (initial_speed : ℝ) (faster_speed : ℝ) (additional_distance : ℝ) :
  initial_speed = 12 →
  faster_speed = 18 →
  additional_distance = 30 →
  ∃ (actual_distance : ℝ),
    actual_distance / initial_speed = (actual_distance + additional_distance) / faster_speed ∧
    actual_distance = 60 := by
  sorry

end distance_travelled_l2880_288066


namespace arithmetic_sequence_first_term_l2880_288056

/-- Sum of first n terms of an arithmetic sequence -/
def T (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a₁ + (n - 1) * d) / 2

/-- The problem statement -/
theorem arithmetic_sequence_first_term
  (h : ∃ (k : ℚ), ∀ (n : ℕ), n > 0 → T a₁ 5 (2*n) / T a₁ 5 n = k) :
  a₁ = 5/2 := by
  sorry

end arithmetic_sequence_first_term_l2880_288056


namespace least_sum_of_equal_multiples_l2880_288044

theorem least_sum_of_equal_multiples (x y z : ℕ+) (h : (2 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (6 : ℕ) * z.val) :
  ∃ (a b c : ℕ+), (2 : ℕ) * a.val = (5 : ℕ) * b.val ∧ (5 : ℕ) * b.val = (6 : ℕ) * c.val ∧
  (∀ (p q r : ℕ+), (2 : ℕ) * p.val = (5 : ℕ) * q.val ∧ (5 : ℕ) * q.val = (6 : ℕ) * r.val →
    a.val + b.val + c.val ≤ p.val + q.val + r.val) ∧
  a.val + b.val + c.val = 26 :=
sorry

end least_sum_of_equal_multiples_l2880_288044


namespace tangent_line_determines_b_l2880_288062

/-- A curve of the form y = x³ + ax + b -/
def curve (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x + b

/-- The derivative of the curve -/
def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + a

theorem tangent_line_determines_b (a b : ℝ) :
  curve a b 1 = 3 →
  curve_derivative a 1 = 2 →
  b = 3 := by
  sorry

#check tangent_line_determines_b

end tangent_line_determines_b_l2880_288062


namespace twenty_percent_greater_than_52_l2880_288064

theorem twenty_percent_greater_than_52 (x : ℝ) : x = 52 * (1 + 0.2) → x = 62.4 := by
  sorry

end twenty_percent_greater_than_52_l2880_288064


namespace square_sum_theorem_l2880_288017

theorem square_sum_theorem (x y z a b c : ℝ) 
  (h1 : x * y = a) 
  (h2 : x * z = b) 
  (h3 : y * z = c) 
  (hx : x ≠ 0) 
  (hy : y ≠ 0) 
  (hz : z ≠ 0) 
  (ha : a ≠ 0) 
  (hb : b ≠ 0) 
  (hc : c ≠ 0) : 
  x^2 + y^2 + z^2 = ((a*b)^2 + (a*c)^2 + (b*c)^2) / (a*b*c) := by
sorry

end square_sum_theorem_l2880_288017


namespace function_inequality_iff_a_geq_half_l2880_288090

/-- Given a function f(x) = ln x - a(x - 1), where a is a real number and x ≥ 1,
    prove that f(x) ≤ (ln x) / (x + 1) if and only if a ≥ 1/2 -/
theorem function_inequality_iff_a_geq_half (a : ℝ) :
  (∀ x : ℝ, x ≥ 1 → (Real.log x - a * (x - 1)) ≤ (Real.log x) / (x + 1)) ↔ a ≥ 1/2 := by
  sorry

end function_inequality_iff_a_geq_half_l2880_288090


namespace function_property_l2880_288080

theorem function_property (f : ℤ → ℤ) :
  (∀ a b c : ℤ, a + b + c = 0 → f a + f b + f c = a^2 + b^2 + c^2) →
  ∃ c : ℤ, ∀ x : ℤ, f x = x^2 + c * x :=
by sorry

end function_property_l2880_288080


namespace x_over_y_is_negative_two_l2880_288008

theorem x_over_y_is_negative_two (x y : ℝ) 
  (h1 : 1 < (x - y) / (x + y) ∧ (x - y) / (x + y) < 4)
  (h2 : (x + y) / (x - y) ≠ 1)
  (h3 : ∃ (n : ℤ), x / y = n) : 
  x / y = -2 := by
sorry

end x_over_y_is_negative_two_l2880_288008


namespace pigeons_eating_breadcrumbs_l2880_288078

theorem pigeons_eating_breadcrumbs (initial_pigeons : ℕ) (new_pigeons : ℕ) : 
  initial_pigeons = 1 → new_pigeons = 1 → initial_pigeons + new_pigeons = 2 := by
  sorry

end pigeons_eating_breadcrumbs_l2880_288078


namespace negation_of_universal_quantifier_negation_of_quadratic_inequality_l2880_288041

theorem negation_of_universal_quantifier (p : ℝ → Prop) :
  (¬ ∀ x : ℝ, p x) ↔ (∃ x : ℝ, ¬ p x) := by sorry

theorem negation_of_quadratic_inequality :
  (¬ ∀ x : ℝ, x^2 - 3*x + 3 > 0) ↔ (∃ x : ℝ, x^2 - 3*x + 3 ≤ 0) := by sorry

end negation_of_universal_quantifier_negation_of_quadratic_inequality_l2880_288041


namespace train_length_train_length_proof_l2880_288026

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (cross_time_s : ℝ) : ℝ :=
  let speed_ms := speed_kmh * 1000 / 3600
  speed_ms * cross_time_s

/-- Proof that a train's length is approximately 100.02 meters -/
theorem train_length_proof (speed_kmh : ℝ) (cross_time_s : ℝ) 
  (h1 : speed_kmh = 60) 
  (h2 : cross_time_s = 6) : 
  ∃ ε > 0, |train_length speed_kmh cross_time_s - 100.02| < ε :=
sorry

end train_length_train_length_proof_l2880_288026


namespace rhombus_area_l2880_288053

/-- The area of a rhombus with side length 4 and an interior angle of 45 degrees is 8√2 -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π / 4) :
  s * s * Real.sin θ = 8 * Real.sqrt 2 := by
  sorry

end rhombus_area_l2880_288053


namespace surface_area_ratio_l2880_288037

/-- A regular tetrahedron with its inscribed sphere -/
structure RegularTetrahedronWithInscribedSphere where
  /-- The surface area of the regular tetrahedron -/
  S₁ : ℝ
  /-- The surface area of the inscribed sphere -/
  S₂ : ℝ
  /-- The surface area of the tetrahedron is positive -/
  h_S₁_pos : 0 < S₁
  /-- The surface area of the sphere is positive -/
  h_S₂_pos : 0 < S₂

/-- The ratio of the surface area of a regular tetrahedron to its inscribed sphere -/
theorem surface_area_ratio (t : RegularTetrahedronWithInscribedSphere) :
  t.S₁ / t.S₂ = 6 * Real.sqrt 3 / Real.pi := by sorry

end surface_area_ratio_l2880_288037


namespace min_square_value_l2880_288059

theorem min_square_value (a b : ℕ+) 
  (h1 : ∃ r : ℕ, (15 * a + 16 * b : ℕ) = r^2)
  (h2 : ∃ s : ℕ, (16 * a - 15 * b : ℕ) = s^2) :
  min (15 * a + 16 * b) (16 * a - 15 * b) ≥ 231361 := by
sorry

end min_square_value_l2880_288059


namespace success_rate_is_70_percent_l2880_288077

def games_played : ℕ := 15
def games_won : ℕ := 9
def remaining_games : ℕ := 5

def total_games : ℕ := games_played + remaining_games
def total_wins : ℕ := games_won + remaining_games

def success_rate : ℚ := (total_wins : ℚ) / (total_games : ℚ)

theorem success_rate_is_70_percent :
  success_rate = 7/10 :=
sorry

end success_rate_is_70_percent_l2880_288077


namespace teacher_selection_theorem_l2880_288001

/-- The number of ways to select k items from n items -/
def binomial (n k : ℕ) : ℕ := sorry

/-- The number of ways to select 6 teachers out of 10, where two specific teachers cannot be selected together -/
def selectTeachers (totalTeachers invitedTeachers : ℕ) : ℕ :=
  binomial totalTeachers invitedTeachers - binomial (totalTeachers - 2) (invitedTeachers - 2)

theorem teacher_selection_theorem :
  selectTeachers 10 6 = 140 := by sorry

end teacher_selection_theorem_l2880_288001


namespace sum_of_absolute_roots_l2880_288052

theorem sum_of_absolute_roots (x : ℂ) : 
  x^4 - 6*x^3 + 13*x^2 - 12*x + 4 = 0 →
  ∃ r1 r2 r3 r4 : ℂ, 
    (x = r1 ∨ x = r2 ∨ x = r3 ∨ x = r4) ∧
    (Complex.abs r1 + Complex.abs r2 + Complex.abs r3 + Complex.abs r4 = 2 * Real.sqrt 6 + 2 * Real.sqrt 2) :=
by sorry

end sum_of_absolute_roots_l2880_288052


namespace min_four_digit_quotient_l2880_288082

/-- A type representing a base-ten digit (1-9) -/
def Digit := { n : Nat // 1 ≤ n ∧ n ≤ 9 }

/-- The function to be minimized -/
def f (a b c d : Digit) : ℚ :=
  (1000 * a.val + 100 * b.val + 10 * c.val + d.val) / (a.val + b.val + c.val + d.val)

/-- The theorem stating the minimum value of the function -/
theorem min_four_digit_quotient :
  ∀ (a b c d : Digit),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
    f a b c d ≥ 80.56 :=
sorry

end min_four_digit_quotient_l2880_288082


namespace juelz_sisters_count_l2880_288095

theorem juelz_sisters_count (total_pieces : ℕ) (eaten_percentage : ℚ) (pieces_per_sister : ℕ) : 
  total_pieces = 240 →
  eaten_percentage = 60 / 100 →
  pieces_per_sister = 32 →
  (total_pieces - (eaten_percentage * total_pieces).num) / pieces_per_sister = 3 :=
by sorry

end juelz_sisters_count_l2880_288095


namespace quadratic_equation_problem_l2880_288018

theorem quadratic_equation_problem : 
  (∀ m : ℝ, ∃ x : ℝ, x^2 - m*x - 1 = 0) ∧ 
  (∃ x₀ : ℕ, x₀^2 - 2*x₀ - 1 ≤ 0) → 
  ¬((∀ m : ℝ, ∃ x : ℝ, x^2 - m*x - 1 = 0) ∧ 
    ¬(∃ x₀ : ℕ, x₀^2 - 2*x₀ - 1 ≤ 0)) :=
by sorry

end quadratic_equation_problem_l2880_288018


namespace identity_implies_a_minus_b_equals_one_l2880_288068

theorem identity_implies_a_minus_b_equals_one :
  ∀ (a b : ℚ),
  (∀ (y : ℚ), y > 0 → a / (y - 3) + b / (y + 5) = (3 * y + 7) / ((y - 3) * (y + 5))) →
  a - b = 1 := by
sorry

end identity_implies_a_minus_b_equals_one_l2880_288068


namespace equation_solution_l2880_288019

theorem equation_solution :
  ∃ x : ℚ, x ≠ 1 ∧ (1 / (x - 1) + 1 = 3 / (2 * x - 2)) ∧ x = 3 / 2 := by
  sorry

end equation_solution_l2880_288019


namespace remaining_soup_feeds_20_adults_l2880_288010

/-- Represents the number of adults a can of soup can feed -/
def adults_per_can : ℕ := 4

/-- Represents the number of children a can of soup can feed -/
def children_per_can : ℕ := 6

/-- Represents the total number of cans of soup -/
def total_cans : ℕ := 8

/-- Represents the number of children fed -/
def children_fed : ℕ := 20

/-- Represents the fraction of soup left in a can after feeding children -/
def leftover_fraction : ℚ := 1/3

/-- Calculates the number of adults that can be fed with the remaining soup -/
def adults_fed (adults_per_can : ℕ) (children_per_can : ℕ) (total_cans : ℕ) (children_fed : ℕ) (leftover_fraction : ℚ) : ℕ :=
  sorry

/-- Theorem stating that the remaining soup can feed 20 adults -/
theorem remaining_soup_feeds_20_adults : 
  adults_fed adults_per_can children_per_can total_cans children_fed leftover_fraction = 20 :=
sorry

end remaining_soup_feeds_20_adults_l2880_288010


namespace inequality_solution_l2880_288063

theorem inequality_solution (x : ℝ) : 
  (x / (x + 1) + (x - 3) / (2 * x) ≥ 4) ↔ (x ∈ Set.Icc (-3) (-1/5)) :=
sorry

end inequality_solution_l2880_288063


namespace student_average_greater_than_true_average_l2880_288029

theorem student_average_greater_than_true_average (x y z : ℝ) (h : x < z ∧ z < y) :
  (x + z) / 2 / 2 + y / 2 > (x + y + z) / 3 := by
  sorry

end student_average_greater_than_true_average_l2880_288029


namespace product_B_percentage_l2880_288011

theorem product_B_percentage (X : ℝ) : 
  X ≥ 0 → X ≤ 100 →
  ∃ (total : ℕ), total ≥ 100 ∧
  ∃ (A B both neither : ℕ),
    A + B + neither = total ∧
    both ≤ A ∧ both ≤ B ∧
    (X : ℝ) = (A : ℝ) / total * 100 ∧
    (23 : ℝ) = (both : ℝ) / total * 100 ∧
    (23 : ℝ) = (neither : ℝ) / total * 100 →
  (B : ℝ) / total * 100 = 100 - X :=
by sorry

end product_B_percentage_l2880_288011


namespace median_eq_twelve_l2880_288070

/-- An isosceles trapezoid with given properties -/
structure IsoscelesTrapezoid where
  -- The height of the trapezoid
  height : ℝ
  -- The angle AOD, where O is the intersection of diagonals
  angle_AOD : ℝ
  -- Assumption that the height is 4√3
  height_eq : height = 4 * Real.sqrt 3
  -- Assumption that ∠AOD is 120°
  angle_AOD_eq : angle_AOD = 120

/-- The median of an isosceles trapezoid -/
def median (t : IsoscelesTrapezoid) : ℝ := sorry

/-- Theorem: The median of the given isosceles trapezoid is 12 -/
theorem median_eq_twelve (t : IsoscelesTrapezoid) : median t = 12 := by sorry

end median_eq_twelve_l2880_288070


namespace brownies_per_neighbor_l2880_288096

/-- Calculates the number of brownies each neighbor receives given the following conditions:
  * Melanie baked 15 batches of brownies
  * Each batch contains 30 brownies
  * She set aside 13/15 of the brownies in each batch for a bake sale
  * She placed 7/10 of the remaining brownies in a container
  * She donated 3/5 of what was left to a local charity
  * She wants to evenly distribute the rest among x neighbors
-/
theorem brownies_per_neighbor (x : ℕ) (x_pos : x > 0) : 
  let total_brownies := 15 * 30
  let bake_sale_brownies := (13 / 15 : ℚ) * total_brownies
  let remaining_after_bake_sale := total_brownies - bake_sale_brownies.floor
  let container_brownies := (7 / 10 : ℚ) * remaining_after_bake_sale
  let remaining_after_container := remaining_after_bake_sale - container_brownies.floor
  let charity_brownies := (3 / 5 : ℚ) * remaining_after_container
  let final_remaining := remaining_after_container - charity_brownies.floor
  (final_remaining / x : ℚ) = 8 / x := by
    sorry

#check brownies_per_neighbor

end brownies_per_neighbor_l2880_288096


namespace basketball_shots_mode_and_median_l2880_288006

def data_set : List Nat := [6, 7, 6, 9, 8]

def mode (l : List Nat) : Nat := sorry

def median (l : List Nat) : Nat := sorry

theorem basketball_shots_mode_and_median :
  mode data_set = 6 ∧ median data_set = 7 := by sorry

end basketball_shots_mode_and_median_l2880_288006


namespace quadratic_inequality_l2880_288074

theorem quadratic_inequality (m : ℝ) : (∃ x : ℝ, x^2 - x - m = 0) → m ≥ -1/4 := by
  sorry

end quadratic_inequality_l2880_288074


namespace multiple_of_four_is_multiple_of_two_l2880_288042

theorem multiple_of_four_is_multiple_of_two (n : ℕ) :
  (∀ k : ℕ, 4 ∣ k → 2 ∣ k) →
  4 ∣ n →
  2 ∣ n :=
by sorry

end multiple_of_four_is_multiple_of_two_l2880_288042


namespace ratio_to_nine_l2880_288050

theorem ratio_to_nine (x : ℝ) : (x / 9 = 5 / 1) → x = 45 := by
  sorry

end ratio_to_nine_l2880_288050


namespace two_numbers_with_ratio_and_square_difference_l2880_288088

theorem two_numbers_with_ratio_and_square_difference (p q : ℝ) (hp : p > 0) (hpn : p ≠ 1) (hq : q > 0) :
  let x : ℝ := q / (p - 1)
  let y : ℝ := p * q / (p - 1)
  y / x = p ∧ (y^2 - x^2) / (y + x) = q := by
  sorry

end two_numbers_with_ratio_and_square_difference_l2880_288088


namespace shaded_area_semicircles_pattern_l2880_288046

/-- The area of the shaded region in a 1-foot length of alternating semicircles pattern --/
theorem shaded_area_semicircles_pattern (foot_to_inch : ℝ) (diameter : ℝ) (π : ℝ) : 
  foot_to_inch = 12 →
  diameter = 2 →
  (foot_to_inch / diameter) * (π * (diameter / 2)^2) = 6 * π := by
  sorry

end shaded_area_semicircles_pattern_l2880_288046


namespace log_equality_implies_ratio_l2880_288083

theorem log_equality_implies_ratio (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (Real.log a / Real.log 9 = Real.log b / Real.log 12) ∧ 
  (Real.log a / Real.log 9 = Real.log (3*a + b) / Real.log 16) →
  b / a = (1 + Real.sqrt 13) / 2 := by
sorry

end log_equality_implies_ratio_l2880_288083


namespace infinite_geometric_series_first_term_l2880_288081

theorem infinite_geometric_series_first_term
  (r : ℚ) (S : ℚ) (h1 : r = 1 / 8)
  (h2 : S = 60)
  (h3 : S = a / (1 - r)) :
  a = 105 / 2 :=
by sorry

end infinite_geometric_series_first_term_l2880_288081


namespace line_through_points_l2880_288069

/-- Given a line y = mx + c passing through points (3,2) and (7,14), prove that m - c = 10 -/
theorem line_through_points (m c : ℝ) : 
  (2 = m * 3 + c) → (14 = m * 7 + c) → m - c = 10 := by
  sorry

end line_through_points_l2880_288069


namespace negation_of_exists_is_forall_l2880_288033

theorem negation_of_exists_is_forall :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by
  sorry

end negation_of_exists_is_forall_l2880_288033


namespace triangle_tangent_product_l2880_288013

theorem triangle_tangent_product (A B C : Real) (h1 : C = 2 * Real.pi / 3) 
  (h2 : Real.tan A + Real.tan B = 2 * Real.sqrt 3 / 3) : 
  Real.tan A * Real.tan B = 1 / 3 := by
  sorry

end triangle_tangent_product_l2880_288013


namespace nine_times_s_on_half_l2880_288016

def s (θ : ℚ) : ℚ := 1 / (2 - θ)

theorem nine_times_s_on_half : s (s (s (s (s (s (s (s (s (1/2)))))))))  = 13/15 := by
  sorry

end nine_times_s_on_half_l2880_288016


namespace amanda_quizzes_l2880_288054

/-- The number of quizzes Amanda has taken so far -/
def n : ℕ := sorry

/-- Amanda's average score on quizzes taken so far (as a percentage) -/
def current_average : ℚ := 92

/-- The required score on the final quiz to get an A (as a percentage) -/
def final_quiz_score : ℚ := 97

/-- The required average score over all quizzes to get an A (as a percentage) -/
def required_average : ℚ := 93

/-- The total number of quizzes including the final quiz -/
def total_quizzes : ℕ := 5

theorem amanda_quizzes : 
  n * current_average + final_quiz_score = required_average * total_quizzes ∧ n = 4 := by sorry

end amanda_quizzes_l2880_288054


namespace park_area_l2880_288060

/-- Proves that a rectangular park with sides in ratio 3:2 and fencing cost of 125 at 50 ps per meter has an area of 3750 square meters -/
theorem park_area (length width : ℝ) (h1 : length / width = 3 / 2) 
  (h2 : 2 * (length + width) * 0.5 = 125) : length * width = 3750 := by
  sorry

end park_area_l2880_288060


namespace field_trip_adults_l2880_288091

theorem field_trip_adults (van_capacity : ℕ) (num_students : ℕ) (num_vans : ℕ) :
  van_capacity = 5 →
  num_students = 25 →
  num_vans = 6 →
  ∃ (num_adults : ℕ), num_adults = num_vans * van_capacity - num_students ∧ num_adults = 5 :=
by sorry

end field_trip_adults_l2880_288091


namespace tom_green_marbles_l2880_288028

/-- The number of green marbles Sara has -/
def sara_green : ℕ := 3

/-- The total number of green marbles Sara and Tom have together -/
def total_green : ℕ := 7

/-- The number of green marbles Tom has -/
def tom_green : ℕ := total_green - sara_green

theorem tom_green_marbles : tom_green = 4 := by
  sorry

end tom_green_marbles_l2880_288028


namespace perpendicular_vector_scalar_l2880_288007

/-- Given two vectors a and b in ℝ³, prove that k = -2 when (k * a + b) is perpendicular to a. -/
theorem perpendicular_vector_scalar (a b : ℝ × ℝ × ℝ) (k : ℝ) : 
  a = (1, 1, 1) → 
  b = (1, 2, 3) → 
  (k * a.1 + b.1, k * a.2.1 + b.2.1, k * a.2.2 + b.2.2) • a = 0 → 
  k = -2 := by sorry

end perpendicular_vector_scalar_l2880_288007


namespace paco_ate_fifteen_sweet_cookies_l2880_288084

/-- The number of sweet cookies Paco ate -/
def sweet_cookies_eaten (initial_sweet : ℕ) (sweet_left : ℕ) : ℕ :=
  initial_sweet - sweet_left

/-- Theorem stating that Paco ate 15 sweet cookies -/
theorem paco_ate_fifteen_sweet_cookies : 
  sweet_cookies_eaten 34 19 = 15 := by
  sorry

end paco_ate_fifteen_sweet_cookies_l2880_288084


namespace two_digit_number_difference_divisibility_l2880_288099

theorem two_digit_number_difference_divisibility (A B : Nat) 
  (h1 : A ≠ B) (h2 : A > B) (h3 : A < 10) (h4 : B < 10) : 
  ∃ k : Int, (10 * A + B) - ((10 * B + A) - 5) = 3 * k := by
  sorry

end two_digit_number_difference_divisibility_l2880_288099


namespace speaker_sale_profit_l2880_288071

theorem speaker_sale_profit (selling_price : ℝ) 
  (profit_percentage : ℝ) (loss_percentage : ℝ) : 
  selling_price = 1.44 →
  profit_percentage = 0.2 →
  loss_percentage = 0.1 →
  let cost_price_1 := selling_price / (1 + profit_percentage)
  let cost_price_2 := selling_price / (1 - loss_percentage)
  let total_cost := cost_price_1 + cost_price_2
  let total_revenue := 2 * selling_price
  total_revenue - total_cost = 0.08 := by
sorry

end speaker_sale_profit_l2880_288071


namespace f_sum_logs_l2880_288076

-- Define the function f
def f (x : ℝ) : ℝ := 1 + x^3

-- State the theorem
theorem f_sum_logs : f (Real.log 2) + f (Real.log (1/2)) = 2 := by
  sorry

end f_sum_logs_l2880_288076


namespace friday_earnings_calculation_l2880_288034

/-- Represents the earnings of Johannes' vegetable shop over three days -/
structure VegetableShopEarnings where
  wednesday : ℝ
  friday : ℝ
  today : ℝ

/-- Calculates the total earnings over three days -/
def total_earnings (e : VegetableShopEarnings) : ℝ :=
  e.wednesday + e.friday + e.today

theorem friday_earnings_calculation (e : VegetableShopEarnings) 
  (h1 : e.wednesday = 30)
  (h2 : e.today = 42)
  (h3 : total_earnings e = 48 * 2) : 
  e.friday = 24 := by
  sorry

end friday_earnings_calculation_l2880_288034


namespace jennifer_book_expense_l2880_288021

theorem jennifer_book_expense (total : ℚ) (sandwich_fraction : ℚ) (ticket_fraction : ℚ) (leftover : ℚ) :
  total = 180 →
  sandwich_fraction = 1 / 5 →
  ticket_fraction = 1 / 6 →
  leftover = 24 →
  ∃ (book_fraction : ℚ),
    book_fraction = 1 / 2 ∧
    total * sandwich_fraction + total * ticket_fraction + total * book_fraction + leftover = total :=
by sorry

end jennifer_book_expense_l2880_288021


namespace plate_acceleration_l2880_288079

noncomputable def α : Real := Real.arccos 0.82
noncomputable def g : Real := 10

theorem plate_acceleration (R r m : Real) (h_R : R = 1) (h_r : r = 0.5) (h_m : m = 75) :
  let a := g * Real.sqrt ((1 - Real.cos α) / 2)
  let direction := α / 2
  a = 3 ∧ direction = Real.arcsin 0.2 := by sorry

end plate_acceleration_l2880_288079


namespace y1_greater_than_y2_l2880_288043

/-- A quadratic function passing through points (0, y₁) and (4, y₂) -/
def quadratic_function (c y₁ y₂ : ℝ) : Prop :=
  y₁ = c ∧ y₂ = 16 - 24 + c

/-- Theorem stating that y₁ > y₂ for the given quadratic function -/
theorem y1_greater_than_y2 (c y₁ y₂ : ℝ) (h : quadratic_function c y₁ y₂) : y₁ > y₂ := by
  sorry

end y1_greater_than_y2_l2880_288043


namespace computer_table_price_l2880_288025

/-- The selling price of an item given its cost price and markup percentage -/
def sellingPrice (costPrice : ℚ) (markupPercentage : ℚ) : ℚ :=
  costPrice * (1 + markupPercentage / 100)

/-- Theorem: The selling price of a computer table with cost price 3840 and markup 25% is 4800 -/
theorem computer_table_price : sellingPrice 3840 25 = 4800 := by
  sorry

end computer_table_price_l2880_288025


namespace max_a_is_maximum_l2880_288036

/-- The polynomial function f(x) = ax^2 - ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 - a * x + 1

/-- The condition that |f(x)| ≤ 1 for all x in [0, 1] -/
def condition (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc 0 1 → |f a x| ≤ 1

/-- The maximum value of a that satisfies the condition -/
def max_a : ℝ := 8

/-- Theorem stating that max_a is the maximum value satisfying the condition -/
theorem max_a_is_maximum :
  (condition max_a) ∧ (∀ a : ℝ, a > max_a → ¬(condition a)) :=
sorry

end max_a_is_maximum_l2880_288036


namespace drop_is_negative_of_rise_is_positive_l2880_288022

/-- Represents the change in water level -/
structure WaterLevelChange where
  magnitude : ℝ
  isRise : Bool

/-- Records a water level change as a signed real number -/
def recordChange (change : WaterLevelChange) : ℝ :=
  if change.isRise then change.magnitude else -change.magnitude

theorem drop_is_negative_of_rise_is_positive 
  (h : ∀ (rise : WaterLevelChange), rise.isRise → recordChange rise = rise.magnitude) :
  ∀ (drop : WaterLevelChange), ¬drop.isRise → recordChange drop = -drop.magnitude :=
by sorry

end drop_is_negative_of_rise_is_positive_l2880_288022


namespace flour_weight_qualified_l2880_288089

def nominal_weight : ℝ := 25
def tolerance : ℝ := 0.25
def flour_weight : ℝ := 24.80

theorem flour_weight_qualified :
  flour_weight ≥ nominal_weight - tolerance ∧
  flour_weight ≤ nominal_weight + tolerance := by
  sorry

end flour_weight_qualified_l2880_288089


namespace tank_inflow_rate_l2880_288009

theorem tank_inflow_rate (capacity : ℝ) (time_diff : ℝ) (slow_rate : ℝ) : 
  capacity > 0 → time_diff > 0 → slow_rate > 0 →
  let slow_time := capacity / slow_rate
  let fast_time := slow_time - time_diff
  fast_time > 0 →
  capacity / fast_time = 2 * slow_rate := by
  sorry

-- Example usage with given values
example : 
  let capacity := 20
  let time_diff := 5
  let slow_rate := 2
  let slow_time := capacity / slow_rate
  let fast_time := slow_time - time_diff
  capacity / fast_time = 4 := by
  sorry

end tank_inflow_rate_l2880_288009


namespace bill_apples_left_l2880_288045

/-- The number of apples Bill has left after distributing them -/
def apples_left (total : ℕ) (children : ℕ) (apples_per_child : ℕ) (pies : ℕ) (apples_per_pie : ℕ) : ℕ :=
  total - (children * apples_per_child + pies * apples_per_pie)

/-- Theorem: Bill has 24 apples left -/
theorem bill_apples_left :
  apples_left 50 2 3 2 10 = 24 := by
  sorry

end bill_apples_left_l2880_288045


namespace infinite_solutions_imply_c_equals_three_l2880_288092

theorem infinite_solutions_imply_c_equals_three :
  (∀ y : ℝ, 3 * (3 + 2 * c * y) = 18 * y + 9) → c = 3 :=
by
  sorry

end infinite_solutions_imply_c_equals_three_l2880_288092


namespace article_cost_l2880_288049

/-- 
Given an article with two selling prices and a relationship between the gains,
prove that the cost of the article is 60.
-/
theorem article_cost (selling_price_1 selling_price_2 : ℝ) 
  (h1 : selling_price_1 = 360)
  (h2 : selling_price_2 = 340)
  (h3 : selling_price_1 - selling_price_2 = 0.05 * (selling_price_2 - cost)) :
  cost = 60 := by
  sorry

end article_cost_l2880_288049


namespace set_operations_l2880_288072

def U : Set Nat := {1,2,3,4,5,6,7,8,9,10,11,13}
def A : Set Nat := {2,4,6,8}
def B : Set Nat := {3,4,5,6,8,9,11}

theorem set_operations :
  (A ∪ B = {2,3,4,5,6,8,9,11}) ∧
  (U \ A = {1,3,5,7,9,10,11,13}) ∧
  (U \ (A ∩ B) = {1,2,3,5,7,9,10,11,13}) ∧
  (A ∪ (U \ B) = {1,2,4,6,7,8,10,13}) := by
  sorry

end set_operations_l2880_288072


namespace tangent_half_angle_sum_l2880_288023

theorem tangent_half_angle_sum (α β γ : Real) (h : α + β + γ = Real.pi) :
  Real.tan (α/2) * Real.tan (β/2) + Real.tan (β/2) * Real.tan (γ/2) + Real.tan (γ/2) * Real.tan (α/2) = 1 := by
  sorry

end tangent_half_angle_sum_l2880_288023


namespace apple_picking_ratio_l2880_288047

theorem apple_picking_ratio : 
  ∀ (frank_apples susan_apples : ℕ) (x : ℚ),
    frank_apples = 36 →
    susan_apples = frank_apples * x →
    (susan_apples / 2 + frank_apples * 2 / 3 : ℚ) = 78 →
    x = 3 := by
  sorry

end apple_picking_ratio_l2880_288047


namespace typing_area_percentage_l2880_288085

/-- Calculates the percentage of a rectangular sheet used for typing, given the sheet dimensions and margins. -/
theorem typing_area_percentage (sheet_width sheet_length side_margin top_bottom_margin : ℝ) :
  sheet_width = 20 ∧ 
  sheet_length = 30 ∧ 
  side_margin = 2 ∧ 
  top_bottom_margin = 3 →
  (sheet_width - 2 * side_margin) * (sheet_length - 2 * top_bottom_margin) / (sheet_width * sheet_length) * 100 = 64 := by
  sorry

#check typing_area_percentage

end typing_area_percentage_l2880_288085


namespace bangle_packing_optimal_solution_l2880_288093

/-- Represents the number of dozens of bangles that can be packed in each box size -/
structure BoxCapacity where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the maximum number of boxes available for each size -/
structure MaxBoxes where
  small : Nat
  medium : Nat
  large : Nat

/-- Represents the number of boxes used for packing -/
structure UsedBoxes where
  small : Nat
  medium : Nat
  large : Nat

/-- Check if the given number of used boxes is within the maximum allowed -/
def isValidBoxCount (used : UsedBoxes) (max : MaxBoxes) : Prop :=
  used.small ≤ max.small ∧ used.medium ≤ max.medium ∧ used.large ≤ max.large

/-- Calculate the total number of dozens packed given the box capacities and used boxes -/
def totalPacked (capacity : BoxCapacity) (used : UsedBoxes) : Nat :=
  used.small * capacity.small + used.medium * capacity.medium + used.large * capacity.large

/-- Check if the given solution packs all bangles and uses the minimum number of boxes -/
def isOptimalSolution (totalDozens : Nat) (capacity : BoxCapacity) (max : MaxBoxes) (solution : UsedBoxes) : Prop :=
  isValidBoxCount solution max ∧
  totalPacked capacity solution = totalDozens ∧
  ∀ (other : UsedBoxes), isValidBoxCount other max → totalPacked capacity other = totalDozens →
    solution.small + solution.medium + solution.large ≤ other.small + other.medium + other.large

theorem bangle_packing_optimal_solution :
  let totalDozens : Nat := 40
  let capacity : BoxCapacity := { small := 2, medium := 3, large := 4 }
  let max : MaxBoxes := { small := 6, medium := 5, large := 4 }
  let solution : UsedBoxes := { small := 5, medium := 5, large := 4 }
  isOptimalSolution totalDozens capacity max solution := by
  sorry

end bangle_packing_optimal_solution_l2880_288093


namespace digit_product_is_30_l2880_288048

/-- Represents a 3x3 grid of digits -/
def Grid := Fin 3 → Fin 3 → Fin 9

/-- Check if all digits from 1 to 9 are used exactly once in the grid -/
def allDigitsUsedOnce (g : Grid) : Prop := ∀ d : Fin 9, ∃! (i j : Fin 3), g i j = d

/-- Product of digits in a row -/
def rowProduct (g : Grid) (row : Fin 3) : ℕ := (g row 0).val.succ * (g row 1).val.succ * (g row 2).val.succ

/-- Product of digits in a column -/
def colProduct (g : Grid) (col : Fin 3) : ℕ := (g 0 col).val.succ * (g 1 col).val.succ * (g 2 col).val.succ

/-- Product of digits in the shaded cells (top-left, center, bottom-right) -/
def shadedProduct (g : Grid) : ℕ := (g 0 0).val.succ * (g 1 1).val.succ * (g 2 2).val.succ

theorem digit_product_is_30 (g : Grid) 
  (h1 : allDigitsUsedOnce g)
  (h2 : rowProduct g 0 = 12)
  (h3 : rowProduct g 1 = 112)
  (h4 : colProduct g 0 = 216)
  (h5 : colProduct g 1 = 12) :
  shadedProduct g = 30 := by
  sorry

end digit_product_is_30_l2880_288048


namespace room_dimension_increase_l2880_288087

/-- Given the cost of painting a room and the cost of painting an enlarged version of the same room,
    calculate the factor by which the room's dimensions were increased. -/
theorem room_dimension_increase (original_cost enlarged_cost : ℝ) 
    (h1 : original_cost = 350)
    (h2 : enlarged_cost = 3150) :
    ∃ (n : ℝ), n = 3 ∧ enlarged_cost = n^2 * original_cost := by
  sorry

end room_dimension_increase_l2880_288087
