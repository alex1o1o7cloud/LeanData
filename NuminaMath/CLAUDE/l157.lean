import Mathlib

namespace circle_tangent_and_symmetric_points_l157_15789

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 6*y - 6 = 0

-- Define point M
def point_M : ℝ × ℝ := (-5, 11)

-- Define the line equation
def line_eq (m : ℝ) (x y : ℝ) : Prop := x + m*y + 4 = 0

-- Define the dot product of OP and OQ
def dot_product_OP_OQ (P Q : ℝ × ℝ) : ℝ := P.1 * Q.1 + P.2 * Q.2

-- Theorem statement
theorem circle_tangent_and_symmetric_points :
  ∃ (P Q : ℝ × ℝ) (m : ℝ),
    (∀ x y, circle_C x y ↔ (x + 1)^2 + (y - 3)^2 = 16) ∧
    (∀ x y, (x = -5 ∨ 3*x + 4*y - 29 = 0) ↔ 
      (circle_C x y ∧ ∃ t, x = point_M.1 + t * (x - point_M.1) ∧ 
                           y = point_M.2 + t * (y - point_M.2) ∧ 
                           t ≠ 0)) ∧
    circle_C P.1 P.2 ∧ 
    circle_C Q.1 Q.2 ∧
    line_eq m P.1 P.2 ∧
    line_eq m Q.1 Q.2 ∧
    dot_product_OP_OQ P Q = -7 ∧
    m = -1 ∧
    (∀ x y, (y = -x ∨ y = -x + 2) ↔ 
      (∃ t, x = P.1 + t * (Q.1 - P.1) ∧ y = P.2 + t * (Q.2 - P.2))) :=
by
  sorry

end circle_tangent_and_symmetric_points_l157_15789


namespace correct_algebraic_equality_l157_15790

theorem correct_algebraic_equality (x y : ℝ) : -4 * x * y + 3 * x * y = -x * y := by
  sorry

end correct_algebraic_equality_l157_15790


namespace tim_vocabulary_proof_l157_15701

/-- Proves that given the conditions of Tim's word learning, his original vocabulary was 14600 words --/
theorem tim_vocabulary_proof (words_per_day : ℕ) (learning_days : ℕ) (increase_percentage : ℚ) : 
  words_per_day = 10 →
  learning_days = 730 →
  increase_percentage = 1/2 →
  (words_per_day * learning_days : ℚ) = increase_percentage * (words_per_day * learning_days + 14600) :=
by
  sorry

end tim_vocabulary_proof_l157_15701


namespace sqrt_D_always_irrational_l157_15775

-- Define the relationship between a and b as consecutive integers
def consecutive (a b : ℤ) : Prop := b = a + 1

-- Define D in terms of a and b
def D (a b : ℤ) : ℤ := a^2 + b^2 + (a + b)^2

-- Theorem statement
theorem sqrt_D_always_irrational (a b : ℤ) (h : consecutive a b) :
  Irrational (Real.sqrt (D a b)) := by
  sorry

end sqrt_D_always_irrational_l157_15775


namespace abc_product_range_l157_15758

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 9 then |Real.log x / Real.log 3 - 1|
  else if x > 9 then 4 - Real.sqrt x
  else 0

theorem abc_product_range (a b c : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  f a = f b ∧ f b = f c →
  81 < a * b * c ∧ a * b * c < 144 := by
sorry

end abc_product_range_l157_15758


namespace sqrt_sum_fractions_l157_15734

theorem sqrt_sum_fractions : Real.sqrt (25 / 36 + 16 / 9) = Real.sqrt 89 / 6 := by
  sorry

end sqrt_sum_fractions_l157_15734


namespace square_root_problem_l157_15795

theorem square_root_problem (x : ℝ) : Real.sqrt x - (Real.sqrt 625 / Real.sqrt 25) = 12 → x = 289 := by
  sorry

end square_root_problem_l157_15795


namespace unpainted_area_proof_l157_15744

def board_width_1 : ℝ := 4
def board_width_2 : ℝ := 6
def intersection_angle : ℝ := 60

theorem unpainted_area_proof :
  let parallelogram_base := board_width_2 / Real.sin (intersection_angle * Real.pi / 180)
  let parallelogram_height := board_width_1
  parallelogram_base * parallelogram_height = 16 * Real.sqrt 3 := by
  sorry

end unpainted_area_proof_l157_15744


namespace oil_added_to_mixture_l157_15757

/-- Proves that the amount of oil added to mixture A is 2 kilograms -/
theorem oil_added_to_mixture (mixture_a_weight : ℝ) (oil_percentage : ℝ) (material_b_percentage : ℝ)
  (added_mixture_a : ℝ) (final_material_b_percentage : ℝ) :
  mixture_a_weight = 8 →
  oil_percentage = 0.2 →
  material_b_percentage = 0.8 →
  added_mixture_a = 6 →
  final_material_b_percentage = 0.7 →
  ∃ (x : ℝ),
    x = 2 ∧
    (material_b_percentage * mixture_a_weight + material_b_percentage * added_mixture_a) =
      final_material_b_percentage * (mixture_a_weight + x + added_mixture_a) :=
by sorry

end oil_added_to_mixture_l157_15757


namespace box_volume_constraint_l157_15792

theorem box_volume_constraint (x : ℕ+) : 
  (∃! x, (2 * x + 6 : ℝ) * ((x : ℝ)^3 - 8) * ((x : ℝ)^2 + 4) < 1200) := by
  sorry

end box_volume_constraint_l157_15792


namespace log_relationship_l157_15706

theorem log_relationship (c d x : ℝ) (hc : c > 0) (hd : d > 0) (hx : x > 0 ∧ x ≠ 1) :
  6 * (Real.log x / Real.log c)^2 + 5 * (Real.log x / Real.log d)^2 = 12 * (Real.log x)^2 / (Real.log c * Real.log d) →
  d = c^(5 / (6 + Real.sqrt 6)) ∨ d = c^(5 / (6 - Real.sqrt 6)) :=
by sorry

end log_relationship_l157_15706


namespace particle_average_velocity_l157_15750

/-- Given a particle with motion law s = t^2 + 3, its average velocity 
    during the time interval (3, 3+Δx) is equal to 6 + Δx. -/
theorem particle_average_velocity (Δx : ℝ) : 
  let s (t : ℝ) := t^2 + 3
  ((s (3 + Δx) - s 3) / Δx) = 6 + Δx :=
sorry

end particle_average_velocity_l157_15750


namespace power_sum_equals_zero_l157_15762

theorem power_sum_equals_zero : (-1)^2021 + 1^2022 = 0 := by
  sorry

end power_sum_equals_zero_l157_15762


namespace fence_poles_count_l157_15722

-- Define the parameters
def total_path_length : ℕ := 900
def bridge_length : ℕ := 42
def pole_spacing : ℕ := 6

-- Define the function to calculate the number of fence poles
def fence_poles : ℕ :=
  let path_to_line := total_path_length - bridge_length
  let poles_one_side := path_to_line / pole_spacing
  2 * poles_one_side

-- Theorem statement
theorem fence_poles_count : fence_poles = 286 := by
  sorry

end fence_poles_count_l157_15722


namespace isosceles_at_second_iteration_l157_15771

/-- Represents a triangle with angles α, β, and γ -/
structure Triangle where
  α : Real
  β : Real
  γ : Real

/-- Generates the next triangle in the sequence -/
def nextTriangle (t : Triangle) : Triangle :=
  { α := t.β, β := t.α, γ := 90 }

/-- Checks if a triangle is isosceles -/
def isIsosceles (t : Triangle) : Prop :=
  t.α = t.β ∨ t.β = t.γ ∨ t.γ = t.α

/-- The initial triangle A₀B₀C₀ -/
def A₀B₀C₀ : Triangle :=
  { α := 30, β := 60, γ := 90 }

/-- Generates the nth triangle in the sequence -/
def nthTriangle (n : Nat) : Triangle :=
  match n with
  | 0 => A₀B₀C₀
  | n + 1 => nextTriangle (nthTriangle n)

theorem isosceles_at_second_iteration :
  ∃ n : Nat, n > 0 ∧ isIsosceles (nthTriangle n) ∧ ∀ m : Nat, 0 < m ∧ m < n → ¬isIsosceles (nthTriangle m) :=
  sorry

end isosceles_at_second_iteration_l157_15771


namespace worker_travel_time_l157_15787

/-- Proves that the usual travel time is 40 minutes given the conditions -/
theorem worker_travel_time (normal_speed : ℝ) (normal_time : ℝ) 
  (h1 : normal_speed > 0) 
  (h2 : normal_time > 0) 
  (h3 : normal_speed * normal_time = (4/5 * normal_speed) * (normal_time + 10)) : 
  normal_time = 40 := by
sorry

end worker_travel_time_l157_15787


namespace irrationality_of_32_minus_sqrt3_l157_15794

theorem irrationality_of_32_minus_sqrt3 : Irrational (32 - Real.sqrt 3) := by
  sorry

end irrationality_of_32_minus_sqrt3_l157_15794


namespace intersection_perpendicular_line_max_distance_to_origin_l157_15740

-- Define the lines
def line1 (x y : ℝ) : Prop := 2*x - 3*y - 3 = 0
def line2 (x y : ℝ) : Prop := x + y + 2 = 0
def line3 (x y : ℝ) : Prop := 3*x + y - 1 = 0

-- Define the general form of line l
def line_l (m x y : ℝ) : Prop := m*x + y - 2*(m+1) = 0

-- Part I
theorem intersection_perpendicular_line :
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, line1 x y ∧ line2 x y → a*x + b*y + c = 0) ∧
    (∀ x y : ℝ, (a*x + b*y + c = 0) → (3*a + b = 0)) ∧
    (a = 5 ∧ b = -15 ∧ c = -18) :=
sorry

-- Part II
theorem max_distance_to_origin :
  ∃ (d : ℝ), 
    (∀ m x y : ℝ, line_l m x y → (x^2 + y^2 ≤ d^2)) ∧
    (∃ m x y : ℝ, line_l m x y ∧ x^2 + y^2 = d^2) ∧
    (d = 2 * Real.sqrt 2) :=
sorry

end intersection_perpendicular_line_max_distance_to_origin_l157_15740


namespace tetrahedron_volume_zero_l157_15768

-- Define the arithmetic progression
def arithmetic_progression (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ k, k < 11 → a (k + 1) = a k + d

-- Define the vertices of the tetrahedron
def tetrahedron_vertices (a : ℕ → ℝ) : Fin 4 → ℝ × ℝ × ℝ
| 0 => (a 1 ^ 2, a 2 ^ 2, a 3 ^ 2)
| 1 => (a 4 ^ 2, a 5 ^ 2, a 6 ^ 2)
| 2 => (a 7 ^ 2, a 8 ^ 2, a 9 ^ 2)
| 3 => (a 10 ^ 2, a 11 ^ 2, a 12 ^ 2)

-- Define the volume of a tetrahedron
def tetrahedron_volume (vertices : Fin 4 → ℝ × ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem tetrahedron_volume_zero (a : ℕ → ℝ) (d : ℝ) :
  arithmetic_progression a d →
  tetrahedron_volume (tetrahedron_vertices a) = 0 := by sorry

end tetrahedron_volume_zero_l157_15768


namespace one_student_in_all_activities_l157_15781

/-- Represents the number of students participating in various combinations of activities -/
structure ActivityParticipation where
  total : ℕ
  chess : ℕ
  soccer : ℕ
  music : ℕ
  atLeastTwo : ℕ

/-- The conditions of the problem -/
def clubConditions : ActivityParticipation where
  total := 30
  chess := 15
  soccer := 18
  music := 12
  atLeastTwo := 14

/-- Theorem stating that exactly one student participates in all three activities -/
theorem one_student_in_all_activities (ap : ActivityParticipation) 
  (h1 : ap = clubConditions) : 
  ∃! x : ℕ, x = (ap.chess + ap.soccer + ap.music) - (2 * ap.atLeastTwo) + ap.total - ap.atLeastTwo :=
by sorry

end one_student_in_all_activities_l157_15781


namespace inequality_proof_l157_15731

theorem inequality_proof :
  (∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + b) ≤ (1 / 4) * (1 / a + 1 / b)) ∧
  (∀ x₁ x₂ x₃ : ℝ, x₁ > 0 → x₂ > 0 → x₃ > 0 → 1 / x₁ + 1 / x₂ + 1 / x₃ = 1 →
    (x₁ + x₂ + x₃) / (x₁ * x₃ + x₃ * x₂) + 
    (x₁ + x₂ + x₃) / (x₁ * x₂ + x₃ * x₁) + 
    (x₁ + x₂ + x₃) / (x₂ * x₁ + x₃ * x₂) ≤ 3 / 2) := by
  sorry

end inequality_proof_l157_15731


namespace cos_alpha_minus_pi_half_l157_15723

/-- If the terminal side of angle α passes through the point P(-1, √3), then cos(α - π/2) = √3/2 -/
theorem cos_alpha_minus_pi_half (α : Real) :
  (∃ (x y : Real), x = -1 ∧ y = Real.sqrt 3 ∧ x = Real.cos α * Real.cos 0 - Real.sin α * Real.sin 0 ∧ 
                    y = Real.sin α * Real.cos 0 + Real.cos α * Real.sin 0) →
  Real.cos (α - π/2) = Real.sqrt 3 / 2 := by
sorry

end cos_alpha_minus_pi_half_l157_15723


namespace sqrt_0_1681_l157_15735

theorem sqrt_0_1681 (h : Real.sqrt 16.81 = 4.1) : Real.sqrt 0.1681 = 0.41 := by
  sorry

end sqrt_0_1681_l157_15735


namespace range_of_a_in_fourth_quadrant_l157_15747

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (a + 1, a - 1)

-- Define the condition for a point to be in the fourth quadrant
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

-- Theorem statement
theorem range_of_a_in_fourth_quadrant :
  ∀ a : ℝ, in_fourth_quadrant (P a) ↔ -1 < a ∧ a < 1 := by sorry

end range_of_a_in_fourth_quadrant_l157_15747


namespace restaurant_ratio_change_l157_15714

/-- Given a restaurant with an initial ratio of cooks to waiters of 3:8,
    9 cooks, and 12 additional waiters hired, prove that the new ratio
    of cooks to waiters is 1:4. -/
theorem restaurant_ratio_change (initial_cooks : ℕ) (initial_waiters : ℕ) 
    (additional_waiters : ℕ) :
  initial_cooks = 9 →
  initial_waiters = (8 * initial_cooks) / 3 →
  additional_waiters = 12 →
  (initial_cooks : ℚ) / (initial_waiters + additional_waiters : ℚ) = 1 / 4 := by
sorry

end restaurant_ratio_change_l157_15714


namespace max_sum_squares_and_products_l157_15770

def S : Finset ℕ := {2, 4, 6, 8}

theorem max_sum_squares_and_products (f g h j : ℕ) 
  (hf : f ∈ S) (hg : g ∈ S) (hh : h ∈ S) (hj : j ∈ S)
  (hsum : f + g + h + j = 20) :
  (∃ (f' g' h' j' : ℕ), f' ∈ S ∧ g' ∈ S ∧ h' ∈ S ∧ j' ∈ S ∧ 
    f' + g' + h' + j' = 20 ∧
    f'^2 + g'^2 + h'^2 + j'^2 ≤ 120 ∧
    (f'^2 + g'^2 + h'^2 + j'^2 = 120 → 
      f' * g' + g' * h' + h' * j' + f' * j' = 100)) ∧
  f^2 + g^2 + h^2 + j^2 ≤ 120 ∧
  (f^2 + g^2 + h^2 + j^2 = 120 → 
    f * g + g * h + h * j + f * j = 100) :=
sorry

end max_sum_squares_and_products_l157_15770


namespace simplified_cow_bull_ratio_l157_15716

/-- Represents the number of cattle on the farm -/
def total_cattle : ℕ := 555

/-- Represents the number of bulls on the farm -/
def bulls : ℕ := 405

/-- Calculates the number of cows on the farm -/
def cows : ℕ := total_cattle - bulls

/-- Represents the ratio of cows to bulls as a pair of natural numbers -/
def cow_bull_ratio : ℕ × ℕ := (cows, bulls)

/-- The theorem stating that the simplified ratio of cows to bulls is 10:27 -/
theorem simplified_cow_bull_ratio : 
  ∃ (k : ℕ), k > 0 ∧ cow_bull_ratio.1 = 10 * k ∧ cow_bull_ratio.2 = 27 * k := by
  sorry

end simplified_cow_bull_ratio_l157_15716


namespace factors_of_45_proportion_l157_15709

theorem factors_of_45_proportion :
  ∃ (a b c d : ℕ), 
    (a ∣ 45) ∧ (b ∣ 45) ∧ (c ∣ 45) ∧ (d ∣ 45) ∧
    (b = 3 * a) ∧ (d = 3 * c) ∧
    (b : ℚ) / a = (d : ℚ) / c ∧ (b : ℚ) / a = 3 := by
  sorry

end factors_of_45_proportion_l157_15709


namespace polynomial_factorization_l157_15779

theorem polynomial_factorization (x : ℝ) :
  4 * (x + 3) * (x + 7) * (x + 8) * (x + 12) - 5 * x^2 =
  (2 * x^2 + (60 - Real.sqrt 5) * x + 80) * (2 * x^2 + (60 + Real.sqrt 5) * x + 80) := by
  sorry

end polynomial_factorization_l157_15779


namespace smallest_integer_satisfying_inequality_l157_15748

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x < 3 * x - 15 → x ≥ 8 :=
by sorry

end smallest_integer_satisfying_inequality_l157_15748


namespace cindys_calculation_l157_15766

theorem cindys_calculation (x : ℝ) : (x - 12) / 4 = 72 → (x - 5) / 9 = 33 := by
  sorry

end cindys_calculation_l157_15766


namespace decagon_interior_exterior_angle_sum_l157_15741

theorem decagon_interior_exterior_angle_sum (n : ℕ) : 
  (n - 2) * 180 = 4 * 360 ↔ n = 10 :=
sorry

end decagon_interior_exterior_angle_sum_l157_15741


namespace arithmetic_progression_terms_l157_15751

/-- 
Given an arithmetic progression with:
- First term: 2
- Last term: 62
- Common difference: 2

Prove that the number of terms in this arithmetic progression is 31.
-/
theorem arithmetic_progression_terms : 
  let a := 2  -- First term
  let L := 62 -- Last term
  let d := 2  -- Common difference
  let n := (L - a) / d + 1 -- Number of terms formula
  n = 31 := by sorry

end arithmetic_progression_terms_l157_15751


namespace barons_claim_impossible_l157_15721

/-- Represents the number of games played by each participant -/
def GameDistribution := List ℕ

/-- A chess tournament with the given rules -/
structure ChessTournament where
  participants : ℕ
  initialGamesPerParticipant : ℕ
  claimedDistribution : GameDistribution

/-- Checks if a game distribution is valid for the given tournament rules -/
def isValidDistribution (t : ChessTournament) (d : GameDistribution) : Prop :=
  d.length = t.participants ∧
  d.sum = t.participants * t.initialGamesPerParticipant + 2 * (d.sum / 2 - t.participants * t.initialGamesPerParticipant / 2)

/-- The specific tournament described in the problem -/
def baronsTournament : ChessTournament where
  participants := 8
  initialGamesPerParticipant := 7
  claimedDistribution := [11, 11, 10, 8, 8, 8, 7, 7]

/-- Theorem stating that the Baron's claim is impossible -/
theorem barons_claim_impossible :
  ¬ isValidDistribution baronsTournament baronsTournament.claimedDistribution :=
sorry

end barons_claim_impossible_l157_15721


namespace axis_of_symmetry_for_quadratic_with_roots_1_and_5_l157_15752

-- Define the quadratic function
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem axis_of_symmetry_for_quadratic_with_roots_1_and_5 
  (a b c : ℝ) (h_a : a ≠ 0) :
  (∀ x, quadratic a b c x = 0 ↔ x = 1 ∨ x = 5) →
  (∃ k, ∀ x, quadratic a b c (k + x) = quadratic a b c (k - x)) ∧
  (∀ k, (∀ x, quadratic a b c (k + x) = quadratic a b c (k - x)) → k = 3) :=
sorry

end axis_of_symmetry_for_quadratic_with_roots_1_and_5_l157_15752


namespace race_winning_post_distance_l157_15797

theorem race_winning_post_distance
  (speed_ratio : ℝ)
  (head_start : ℝ)
  (h_speed_ratio : speed_ratio = 1.75)
  (h_head_start : head_start = 84)
  : ∃ (distance : ℝ),
    distance = 196 ∧
    distance / speed_ratio = (distance - head_start) / 1 :=
by sorry

end race_winning_post_distance_l157_15797


namespace hexagon_diagonals_l157_15798

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexagon is a polygon with 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: The number of diagonals in a hexagon is 9 -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end hexagon_diagonals_l157_15798


namespace semicircle_radius_with_inscribed_circles_l157_15732

/-- The radius of a semicircle that inscribes two externally touching circles -/
theorem semicircle_radius_with_inscribed_circles 
  (r₁ r₂ R : ℝ) 
  (h₁ : r₁ = Real.sqrt 19)
  (h₂ : r₂ = Real.sqrt 76)
  (h_touch : r₁ + r₂ = R - r₁ + R - r₂) 
  (h_inscribed : R^2 = (R - r₁)^2 + r₁^2 ∧ R^2 = (R - r₂)^2 + r₂^2) :
  R = 4 * Real.sqrt 19 := by
sorry

end semicircle_radius_with_inscribed_circles_l157_15732


namespace soda_cost_l157_15782

theorem soda_cost (alice_burgers alice_sodas alice_total bill_burgers bill_sodas bill_total : ℕ)
  (h_alice : 4 * alice_burgers + 3 * alice_sodas = alice_total)
  (h_bill : 3 * bill_burgers + 2 * bill_sodas = bill_total)
  (h_alice_total : alice_total = 500)
  (h_bill_total : bill_total = 370)
  (h_same_prices : alice_burgers = bill_burgers ∧ alice_sodas = bill_sodas) :
  alice_sodas = 20 := by
  sorry

end soda_cost_l157_15782


namespace dividing_line_slope_absolute_value_is_one_l157_15730

/-- A circle in a 2D plane --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The line that equally divides the total area of the circles --/
structure DividingLine where
  slope : ℝ
  passes_through : ℝ × ℝ

/-- The problem setup --/
def problem_setup : (Circle × Circle × Circle) × DividingLine := 
  let c1 : Circle := ⟨(10, 90), 4⟩
  let c2 : Circle := ⟨(15, 70), 4⟩
  let c3 : Circle := ⟨(20, 80), 4⟩
  let line : DividingLine := ⟨0, (15, 70)⟩  -- slope initialized to 0
  ((c1, c2, c3), line)

/-- The theorem to be proved --/
theorem dividing_line_slope_absolute_value_is_one 
  (setup : (Circle × Circle × Circle) × DividingLine) : 
  abs setup.2.slope = 1 := by
  sorry

end dividing_line_slope_absolute_value_is_one_l157_15730


namespace x35x_divisible_by_18_l157_15793

def is_single_digit (x : ℕ) : Prop := x ≥ 0 ∧ x ≤ 9

def four_digit_number (x : ℕ) : ℕ := 1000 * x + 350 + x

theorem x35x_divisible_by_18 : 
  ∃! (x : ℕ), is_single_digit x ∧ (four_digit_number x) % 18 = 0 ∧ x = 8 :=
sorry

end x35x_divisible_by_18_l157_15793


namespace pentagon_reconstruction_l157_15728

/-- A point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- A pentagon in 2D space -/
structure Pentagon where
  A : Point2D
  B : Point2D
  C : Point2D
  D : Point2D
  E : Point2D

/-- The reflected points of a pentagon -/
structure ReflectedPoints where
  A1 : Point2D
  B1 : Point2D
  C1 : Point2D
  D1 : Point2D
  E1 : Point2D

/-- Function to reflect a point with respect to another point -/
def reflect (p : Point2D) (center : Point2D) : Point2D :=
  { x := 2 * center.x - p.x
    y := 2 * center.y - p.y }

/-- Theorem stating that a pentagon can be reconstructed from its reflected points -/
theorem pentagon_reconstruction (reflectedPoints : ReflectedPoints) :
  ∃! (original : Pentagon),
    reflectedPoints.A1 = reflect original.A original.B ∧
    reflectedPoints.B1 = reflect original.B original.C ∧
    reflectedPoints.C1 = reflect original.C original.D ∧
    reflectedPoints.D1 = reflect original.D original.E ∧
    reflectedPoints.E1 = reflect original.E original.A :=
  sorry


end pentagon_reconstruction_l157_15728


namespace f_properties_l157_15784

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := Real.log (10 - 2 * x) / Real.log (1/2)

-- Theorem statement
theorem f_properties :
  -- 1. Domain of f(x) is (-∞, 5)
  (∀ x, f x ≠ 0 → x < 5) ∧
  -- 2. f(x) is increasing on its domain
  (∀ x y, x < y → x < 5 → y < 5 → f x < f y) ∧
  -- 3. Maximum value of m for which f(x) ≥ (1/2)ˣ + m holds for all x ∈ [3, 4] is -17/8
  (∀ m, (∀ x, x ∈ Set.Icc 3 4 → f x ≥ (1/2)^x + m) → m ≤ -17/8) ∧
  (∃ x, x ∈ Set.Icc 3 4 ∧ f x = (1/2)^x + (-17/8)) := by
  sorry

end f_properties_l157_15784


namespace existence_of_n_l157_15746

theorem existence_of_n : ∃ n : ℕ+, 
  ∀ k : ℕ, 2 ≤ k ∧ k ≤ 10 → 
    ∃ p : ℕ+, 
      (↑p + 2015/10000 : ℝ)^k < n ∧ n < (↑p + 2016/10000 : ℝ)^k := by
  sorry

end existence_of_n_l157_15746


namespace no_integer_solution_l157_15738

theorem no_integer_solution :
  ∀ (a b : ℤ), 
    0 ≤ a ∧ 
    0 < b ∧ 
    a < 9 ∧ 
    b < 4 →
    ¬(1 < (a : ℝ) + (b : ℝ) * Real.sqrt 5 ∧ (a : ℝ) + (b : ℝ) * Real.sqrt 5 < 9 + 4 * Real.sqrt 5) :=
by
  sorry


end no_integer_solution_l157_15738


namespace mechanic_average_earning_l157_15720

/-- The average earning of a mechanic for a week, given specific conditions -/
theorem mechanic_average_earning
  (first_four_avg : ℝ)
  (last_four_avg : ℝ)
  (fourth_day_earning : ℝ)
  (h1 : first_four_avg = 25)
  (h2 : last_four_avg = 22)
  (h3 : fourth_day_earning = 20) :
  (4 * first_four_avg + 4 * last_four_avg - fourth_day_earning) / 7 = 24 := by
  sorry

#check mechanic_average_earning

end mechanic_average_earning_l157_15720


namespace geometric_sequence_ratio_is_two_l157_15799

/-- A geometric sequence with specified properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a (n + 2) / a (n + 1)
  a_5_eq_2 : a 5 = 2
  a_6_a_8_eq_8 : a 6 * a 8 = 8

/-- The ratio of differences in a geometric sequence with specific properties is 2 -/
theorem geometric_sequence_ratio_is_two (seq : GeometricSequence) :
  (seq.a 2018 - seq.a 2016) / (seq.a 2014 - seq.a 2012) = 2 := by
  sorry

end geometric_sequence_ratio_is_two_l157_15799


namespace vanilla_syrup_cost_vanilla_syrup_cost_is_correct_l157_15788

/-- The cost of vanilla syrup in a coffee order -/
theorem vanilla_syrup_cost : ℝ :=
  let drip_coffee_cost : ℝ := 2.25
  let drip_coffee_quantity : ℕ := 2
  let espresso_cost : ℝ := 3.50
  let espresso_quantity : ℕ := 1
  let latte_cost : ℝ := 4.00
  let latte_quantity : ℕ := 2
  let cold_brew_cost : ℝ := 2.50
  let cold_brew_quantity : ℕ := 2
  let cappuccino_cost : ℝ := 3.50
  let cappuccino_quantity : ℕ := 1
  let total_order_cost : ℝ := 25.00

  have h1 : ℝ := drip_coffee_cost * drip_coffee_quantity +
                 espresso_cost * espresso_quantity +
                 latte_cost * latte_quantity +
                 cold_brew_cost * cold_brew_quantity +
                 cappuccino_cost * cappuccino_quantity

  have h2 : ℝ := total_order_cost - h1

  h2

theorem vanilla_syrup_cost_is_correct : vanilla_syrup_cost = 0.50 := by
  sorry

end vanilla_syrup_cost_vanilla_syrup_cost_is_correct_l157_15788


namespace arithmetic_evaluation_l157_15736

theorem arithmetic_evaluation : 12 / 4 - 3 - 6 + 3 * 5 = 9 := by sorry

end arithmetic_evaluation_l157_15736


namespace comparison_theorem_l157_15764

theorem comparison_theorem :
  (2 * Real.sqrt 3 < 3 * Real.sqrt 2) ∧
  ((Real.sqrt 10 - 1) / 3 > 2 / 3) := by
  sorry

end comparison_theorem_l157_15764


namespace water_percentage_in_fresh_grapes_l157_15724

/-- The percentage of water in fresh grapes -/
def water_percentage_fresh : ℝ := 90

/-- The percentage of water in dried grapes -/
def water_percentage_dried : ℝ := 20

/-- The weight of fresh grapes in kg -/
def fresh_weight : ℝ := 5

/-- The weight of dried grapes in kg -/
def dried_weight : ℝ := 0.625

theorem water_percentage_in_fresh_grapes :
  (100 - water_percentage_fresh) / 100 * fresh_weight = 
  (100 - water_percentage_dried) / 100 * dried_weight := by sorry

end water_percentage_in_fresh_grapes_l157_15724


namespace inequality_solution_l157_15777

theorem inequality_solution (x : ℝ) : (2 * x) / 5 ≤ 3 + x ∧ 3 + x < -3 * (1 + x) ↔ x ∈ Set.Ici (-5) ∩ Set.Iio (-3/2) := by
  sorry

end inequality_solution_l157_15777


namespace probability_3_or_more_babies_speak_l157_15780

def probability_at_least_3_out_of_7 (p : ℝ) : ℝ :=
  1 - (Nat.choose 7 0 * p^0 * (1-p)^7 +
       Nat.choose 7 1 * p^1 * (1-p)^6 +
       Nat.choose 7 2 * p^2 * (1-p)^5)

theorem probability_3_or_more_babies_speak :
  probability_at_least_3_out_of_7 (1/3) = 939/2187 := by
  sorry

end probability_3_or_more_babies_speak_l157_15780


namespace essay_pages_theorem_l157_15702

/-- Calculates the number of pages needed for a given number of words -/
def pages_needed (words : ℕ) : ℕ := (words + 259) / 260

/-- Represents the essay writing scenario -/
def essay_pages : Prop :=
  let johnny_words : ℕ := 150
  let madeline_words : ℕ := 2 * johnny_words
  let timothy_words : ℕ := madeline_words + 30
  let total_pages : ℕ := pages_needed johnny_words + pages_needed madeline_words + pages_needed timothy_words
  total_pages = 5

theorem essay_pages_theorem : essay_pages := by
  sorry

end essay_pages_theorem_l157_15702


namespace angle_ABG_measure_l157_15711

/-- A regular octagon is a polygon with 8 sides of equal length and 8 interior angles of equal measure. -/
structure RegularOctagon where
  vertices : Fin 8 → ℝ × ℝ

/-- Angle ABG in a regular octagon ABCDEFGH -/
def angle_ABG (octagon : RegularOctagon) : ℝ := sorry

/-- The measure of angle ABG in a regular octagon is 22.5 degrees -/
theorem angle_ABG_measure (octagon : RegularOctagon) : angle_ABG octagon = 22.5 := by sorry

end angle_ABG_measure_l157_15711


namespace sequence_arrangements_l157_15710

-- Define a type for our sequence
def Sequence := Fin 5 → Fin 5

-- Define a predicate for valid permutations
def is_valid_permutation (s : Sequence) : Prop :=
  Function.Injective s ∧ Function.Surjective s

-- Define a predicate for non-adjacent odd and even numbers
def non_adjacent_odd_even (s : Sequence) : Prop :=
  ∀ i : Fin 4, (s i).val % 2 ≠ (s (i + 1)).val % 2

-- Define a predicate for decreasing then increasing sequence
def decreasing_then_increasing (s : Sequence) : Prop :=
  ∃ j : Fin 4, (∀ i : Fin 5, i < j → s i > s (i + 1)) ∧
               (∀ i : Fin 5, i ≥ j → s i < s (i + 1))

-- Define a predicate for the specific inequality condition
def specific_inequality (s : Sequence) : Prop :=
  s 0 < s 1 ∧ s 1 > s 2 ∧ s 2 > s 3 ∧ s 3 < s 4

-- State the theorem
theorem sequence_arrangements (s : Sequence) 
  (h : is_valid_permutation s) : 
  (∃ l : List Sequence, (∀ s' ∈ l, is_valid_permutation s' ∧ non_adjacent_odd_even s') ∧ l.length = 12) ∧
  (∃ l : List Sequence, (∀ s' ∈ l, is_valid_permutation s' ∧ decreasing_then_increasing s') ∧ l.length = 14) ∧
  (∃ l : List Sequence, (∀ s' ∈ l, is_valid_permutation s' ∧ specific_inequality s') ∧ l.length = 11) :=
sorry

end sequence_arrangements_l157_15710


namespace parallel_lines_perpendicular_lines_l157_15733

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := m * x + 2 * y + 4 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := x + (1 + m) * y - 2 = 0

-- Theorem for parallel lines
theorem parallel_lines (m : ℝ) :
  (∀ x y : ℝ, l₁ m x y ↔ l₂ m x y) → m = 1 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines (m : ℝ) :
  (∀ x y : ℝ, l₁ m x y → l₂ m x y → x * x + y * y = 0) → m = -2/3 :=
sorry

end parallel_lines_perpendicular_lines_l157_15733


namespace cookie_is_circle_with_radius_sqrt35_l157_15776

-- Define the equation of the cookie's boundary
def cookie_boundary (x y : ℝ) : Prop :=
  x^2 + y^2 + 10 = 6*x + 12*y

-- Theorem stating that the cookie's boundary is a circle with radius √35
theorem cookie_is_circle_with_radius_sqrt35 :
  ∃ (h k : ℝ), ∀ (x y : ℝ),
    cookie_boundary x y ↔ (x - h)^2 + (y - k)^2 = 35 :=
sorry

end cookie_is_circle_with_radius_sqrt35_l157_15776


namespace w_range_l157_15785

theorem w_range (x y w : ℝ) : 
  -x + y = 2 → x < 3 → y ≥ 0 → w = x + y - 2 → -4 ≤ w ∧ w < 6 :=
by sorry

end w_range_l157_15785


namespace inscribed_circle_area_l157_15739

/-- The area of a circle inscribed in a sector of a circle -/
theorem inscribed_circle_area (R a : ℝ) (h₁ : R > 0) (h₂ : a > 0) :
  let r := R * a / (R + a)
  π * r^2 = π * (R * a / (R + a))^2 := by
  sorry

end inscribed_circle_area_l157_15739


namespace complex_equation_solution_l157_15773

theorem complex_equation_solution :
  ∀ z : ℂ, (3 - z) * Complex.I = 2 * Complex.I → z = 3 + 2 * Complex.I :=
by
  sorry

end complex_equation_solution_l157_15773


namespace wattage_increase_percentage_l157_15778

theorem wattage_increase_percentage (original_wattage new_wattage : ℝ) 
  (h1 : original_wattage = 110)
  (h2 : new_wattage = 143) : 
  (new_wattage - original_wattage) / original_wattage * 100 = 30 := by
  sorry

end wattage_increase_percentage_l157_15778


namespace green_ball_probability_l157_15755

/-- Represents a container with red and green balls -/
structure Container where
  red : ℕ
  green : ℕ

/-- The probability of selecting a green ball from a given container -/
def greenProbability (c : Container) : ℚ :=
  c.green / (c.red + c.green)

/-- The probability of selecting a green ball given three containers -/
def totalGreenProbability (c1 c2 c3 : Container) : ℚ :=
  (1 / 3) * (greenProbability c1 + greenProbability c2 + greenProbability c3)

theorem green_ball_probability :
  let c1 := Container.mk 8 4
  let c2 := Container.mk 3 5
  let c3 := Container.mk 4 6
  totalGreenProbability c1 c2 c3 = 187 / 360 := by
  sorry

end green_ball_probability_l157_15755


namespace sum_of_coefficients_eq_value_at_one_l157_15725

/-- The polynomial for which we want to find the sum of coefficients -/
def p (x : ℝ) : ℝ := 3*(x^8 - x^5 + 2*x^3 - 6) - 5*(x^4 + 3*x^2) + 2*(x^6 - 5)

/-- The sum of coefficients of a polynomial is equal to its value at x = 1 -/
theorem sum_of_coefficients_eq_value_at_one :
  p 1 = -40 := by sorry

end sum_of_coefficients_eq_value_at_one_l157_15725


namespace quadratic_two_distinct_roots_l157_15786

theorem quadratic_two_distinct_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 + m*x + 4 = 0 ∧ y^2 + m*y + 4 = 0) ↔ 
  (m < -4 ∨ m > 4) :=
sorry

end quadratic_two_distinct_roots_l157_15786


namespace last_two_digits_same_l157_15753

def sequence_property (a : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, (a (n + 1) = a n + 54) ∨ (a (n + 1) = a n + 77)

theorem last_two_digits_same (a : ℕ → ℕ) (h : sequence_property a) :
  ∃ k : ℕ, a k % 100 = a (k + 1) % 100 :=
sorry

end last_two_digits_same_l157_15753


namespace trading_cards_theorem_l157_15712

/-- The number of cards in a partially filled box -/
def partially_filled_box (total_cards : ℕ) (cards_per_box : ℕ) : ℕ :=
  total_cards % cards_per_box

theorem trading_cards_theorem :
  let pokemon_cards := 65
  let magic_cards := 55
  let yugioh_cards := 40
  let pokemon_per_box := 8
  let magic_per_box := 10
  let yugioh_per_box := 12
  (partially_filled_box pokemon_cards pokemon_per_box = 1) ∧
  (partially_filled_box magic_cards magic_per_box = 5) ∧
  (partially_filled_box yugioh_cards yugioh_per_box = 4) :=
by sorry

end trading_cards_theorem_l157_15712


namespace max_cone_volume_in_sphere_l157_15754

/-- The maximum volume of a cone formed by a circular section of a sphere --/
theorem max_cone_volume_in_sphere (R : ℝ) (h : R = 9) : 
  ∃ (V : ℝ), V = 54 * Real.sqrt 3 * Real.pi ∧ 
  ∀ (r h : ℝ), r^2 + h^2 = R^2 → 
  (1/3 : ℝ) * Real.pi * r^2 * h ≤ V := by
  sorry

end max_cone_volume_in_sphere_l157_15754


namespace simplify_and_rationalize_l157_15742

theorem simplify_and_rationalize :
  (Real.sqrt 3 / Real.sqrt 7) * (Real.sqrt 5 / Real.sqrt 11) * (Real.sqrt 6 / Real.sqrt 13) = 
  (3 * Real.sqrt 10010) / 1001 := by
sorry

end simplify_and_rationalize_l157_15742


namespace fraction_value_l157_15760

theorem fraction_value : 
  let a : ℕ := 2003
  let b : ℕ := 2002
  let four : ℕ := 2^2
  let six : ℕ := 2 * 3
  (four^a * 3^b) / (six^b * 2^a) = 2 := by
sorry

end fraction_value_l157_15760


namespace melissa_shoe_repair_time_l157_15756

/-- The total time Melissa spends repairing shoes -/
theorem melissa_shoe_repair_time (buckle_time heel_time strap_time sole_time : ℕ) 
  (num_pairs : ℕ) : 
  buckle_time = 5 → 
  heel_time = 10 → 
  strap_time = 7 → 
  sole_time = 12 → 
  num_pairs = 8 → 
  (buckle_time + heel_time + strap_time + sole_time) * 2 * num_pairs = 544 :=
by sorry

end melissa_shoe_repair_time_l157_15756


namespace triangle_at_most_one_obtuse_l157_15783

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  all_positive : ∀ i, angles i > 0

-- Define an obtuse angle
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Theorem statement
theorem triangle_at_most_one_obtuse (t : Triangle) : 
  ¬(∃ i j : Fin 3, i ≠ j ∧ is_obtuse (t.angles i) ∧ is_obtuse (t.angles j)) := by
  sorry

end triangle_at_most_one_obtuse_l157_15783


namespace proposition_b_correct_l157_15703

theorem proposition_b_correct :
  (∃ x : ℕ, x^3 ≤ x^2) ∧
  ((∀ x : ℝ, x > 1 → x^2 > 1) ∧ (∃ x : ℝ, x ≤ 1 ∧ x^2 > 1)) ∧
  (∃ a b : ℝ, a > b ∧ a^2 ≤ b^2) :=
by sorry

end proposition_b_correct_l157_15703


namespace consecutive_blue_gumballs_probability_l157_15772

theorem consecutive_blue_gumballs_probability 
  (pink_prob : ℝ) 
  (h_pink_prob : pink_prob = 1/3) : 
  let blue_prob := 1 - pink_prob
  (blue_prob * blue_prob) = 4/9 := by sorry

end consecutive_blue_gumballs_probability_l157_15772


namespace green_pieces_count_l157_15707

/-- The number of green pieces of candy in a jar, given the total number of pieces and the number of red and blue pieces. -/
def green_pieces (total red blue : ℚ) : ℚ :=
  total - red - blue

/-- Theorem: The number of green pieces is 9468 given the specified conditions. -/
theorem green_pieces_count :
  let total : ℚ := 12509.72
  let red : ℚ := 568.29
  let blue : ℚ := 2473.43
  green_pieces total red blue = 9468 := by
  sorry

end green_pieces_count_l157_15707


namespace number_calculation_l157_15749

theorem number_calculation (n : ℝ) : (0.1 * 0.2 * 0.35 * 0.4 * n = 84) → n = 300000 := by
  sorry

end number_calculation_l157_15749


namespace necessary_condition_transitivity_l157_15713

theorem necessary_condition_transitivity (A B C : Prop) :
  (B → C) → (A → B) → (A → C) := by
  sorry

end necessary_condition_transitivity_l157_15713


namespace union_equals_first_set_l157_15761

theorem union_equals_first_set (I M N : Set α) : 
  M ⊂ I → N ⊂ I → M ≠ N → M.Nonempty → N.Nonempty → N ∩ (I \ M) = ∅ → M ∪ N = M := by
  sorry

end union_equals_first_set_l157_15761


namespace five_students_three_companies_l157_15727

/-- The number of ways to assign n students to k companies, where each company must receive at least one student. -/
def assignment_count (n k : ℕ) : ℕ :=
  k^n - k * (k-1)^n + (k.choose 2)

/-- Theorem stating that the number of ways to assign 5 students to 3 companies, where each company must receive at least one student, is 150. -/
theorem five_students_three_companies : assignment_count 5 3 = 150 := by
  sorry

end five_students_three_companies_l157_15727


namespace constant_jump_returns_to_start_increasing_jump_returns_to_start_l157_15704

-- Define the number of stones
def num_stones : ℕ := 10

-- Define the number of jumps
def num_jumps : ℕ := 100

-- Function to calculate the position after constant jumps
def constant_jump_position (jump_size : ℕ) : ℕ :=
  (1 + jump_size * num_jumps) % num_stones

-- Function to calculate the position after increasing jumps
def increasing_jump_position : ℕ :=
  (1 + (num_jumps * (num_jumps + 1) / 2)) % num_stones

-- Theorem for constant jump scenario
theorem constant_jump_returns_to_start :
  constant_jump_position 2 = 1 := by sorry

-- Theorem for increasing jump scenario
theorem increasing_jump_returns_to_start :
  increasing_jump_position = 1 := by sorry

end constant_jump_returns_to_start_increasing_jump_returns_to_start_l157_15704


namespace mod_seven_difference_powers_l157_15765

theorem mod_seven_difference_powers : (81^1814 - 25^1814) % 7 = 0 := by
  sorry

end mod_seven_difference_powers_l157_15765


namespace syllogism_invalid_l157_15719

-- Define the sets and properties
def Geese : Type := Unit
def Senators : Type := Unit
def eats_cabbage (α : Type) : α → Prop := fun _ => True

-- Define the syllogism
def invalid_syllogism (g : Geese) (s : Senators) : Prop :=
  eats_cabbage Geese g ∧ eats_cabbage Senators s → s = g

-- Theorem stating that the syllogism is invalid
theorem syllogism_invalid :
  ¬∀ (g : Geese) (s : Senators), invalid_syllogism g s :=
sorry

end syllogism_invalid_l157_15719


namespace parallelogram_base_l157_15705

/-- Given a parallelogram with area 308 square centimeters and height 14 cm, its base is 22 cm. -/
theorem parallelogram_base (area height base : ℝ) : 
  area = 308 ∧ height = 14 ∧ area = base * height → base = 22 := by
  sorry

end parallelogram_base_l157_15705


namespace inequalities_given_sum_positive_l157_15745

/-- Given two real numbers a and b such that a + b > 0, 
    the following statements are true:
    1. a^5 * b^2 + a^4 * b^3 ≥ 0
    2. a^21 + b^21 > 0
    3. (a+2)*(b+2) > a*b
-/
theorem inequalities_given_sum_positive (a b : ℝ) (h : a + b > 0) :
  (a^5 * b^2 + a^4 * b^3 ≥ 0) ∧ 
  (a^21 + b^21 > 0) ∧ 
  ((a+2)*(b+2) > a*b) := by
  sorry

end inequalities_given_sum_positive_l157_15745


namespace negation_of_sum_even_both_even_l157_15729

theorem negation_of_sum_even_both_even :
  (¬ ∀ (a b : ℤ), Even (a + b) → (Even a ∧ Even b)) ↔
  (∃ (a b : ℤ), Even (a + b) ∧ (¬ Even a ∨ ¬ Even b)) :=
sorry

end negation_of_sum_even_both_even_l157_15729


namespace subset_implies_a_range_l157_15767

theorem subset_implies_a_range (a : ℝ) : 
  let M := {x : ℝ | (x - 1) * (x - 2) < 0}
  let N := {x : ℝ | x < a}
  (M ⊆ N) → a ∈ Set.Ici 2 := by
  sorry

end subset_implies_a_range_l157_15767


namespace smallest_positive_integer_ending_in_6_divisible_by_11_l157_15774

def is_smallest_positive_integer_ending_in_6_divisible_by_11 (n : ℕ) : Prop :=
  n > 0 ∧ n % 10 = 6 ∧ n % 11 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 6 → m % 11 = 0 → m ≥ n

theorem smallest_positive_integer_ending_in_6_divisible_by_11 :
  is_smallest_positive_integer_ending_in_6_divisible_by_11 116 := by
  sorry

end smallest_positive_integer_ending_in_6_divisible_by_11_l157_15774


namespace square_of_1085_l157_15796

theorem square_of_1085 : (1085 : ℕ)^2 = 1177225 := by
  sorry

end square_of_1085_l157_15796


namespace geometric_sequence_sum_l157_15791

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  (a 1 + a 2 + a 3 = 4) →
  (a 2 + a 3 + a 4 = -2) →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 = 7/8) :=
by sorry

end geometric_sequence_sum_l157_15791


namespace largest_common_difference_and_terms_l157_15763

def is_decreasing_arithmetic_progression (a b c : ℤ) : Prop :=
  ∃ d : ℤ, d < 0 ∧ b = a + d ∧ c = a + 2*d

def has_two_roots (a b c : ℤ) : Prop :=
  b^2 - 4*a*c ≥ 0

theorem largest_common_difference_and_terms 
  (a b c : ℤ) 
  (h1 : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h2 : is_decreasing_arithmetic_progression a b c)
  (h3 : has_two_roots (2*a) (2*b) c)
  (h4 : has_two_roots (2*a) c (2*b))
  (h5 : has_two_roots (2*b) (2*a) c)
  (h6 : has_two_roots (2*b) c (2*a))
  (h7 : has_two_roots c (2*a) (2*b))
  (h8 : has_two_roots c (2*b) (2*a)) :
  ∃ d : ℤ, d = -5 ∧ a = 4 ∧ b = -1 ∧ c = -6 ∧ 
  ∀ d' : ℤ, (∃ a' b' c' : ℤ, 
    a' ≠ 0 ∧ b' ≠ 0 ∧ c' ≠ 0 ∧
    is_decreasing_arithmetic_progression a' b' c' ∧
    has_two_roots (2*a') (2*b') c' ∧
    has_two_roots (2*a') c' (2*b') ∧
    has_two_roots (2*b') (2*a') c' ∧
    has_two_roots (2*b') c' (2*a') ∧
    has_two_roots c' (2*a') (2*b') ∧
    has_two_roots c' (2*b') (2*a') ∧
    d' < 0) → d' ≥ d :=
by sorry

end largest_common_difference_and_terms_l157_15763


namespace arithmetic_sequence_formula_l157_15715

def f (x : ℝ) : ℝ := x^2 - 4*x + 2

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_formula (a : ℕ → ℝ) (x : ℝ) :
  arithmetic_sequence a →
  a 1 = f (x + 1) →
  a 2 = 0 →
  a 3 = f (x - 1) →
  (∀ n : ℕ, a n = 2*n - 4) ∨ (∀ n : ℕ, a n = 4 - 2*n) :=
by sorry

end arithmetic_sequence_formula_l157_15715


namespace unique_solution_for_specific_k_and_a_l157_15708

/-- The equation (x + 2) / (kx - ax - 1) = x has exactly one solution when k = 0 and a = 1/2 -/
theorem unique_solution_for_specific_k_and_a :
  ∃! x : ℝ, (x + 2) / (0 * x - (1/2) * x - 1) = x :=
sorry

end unique_solution_for_specific_k_and_a_l157_15708


namespace fair_coin_prob_TTHH_l157_15717

/-- The probability of getting tails on a single flip of a fair coin -/
def prob_tails : ℚ := 1 / 2

/-- The number of times the coin is flipped -/
def num_flips : ℕ := 4

/-- The probability of getting tails on the first two flips and heads on the last two flips -/
def prob_TTHH : ℚ := prob_tails * prob_tails * (1 - prob_tails) * (1 - prob_tails)

theorem fair_coin_prob_TTHH :
  prob_TTHH = 1 / 16 :=
sorry

end fair_coin_prob_TTHH_l157_15717


namespace terminal_side_of_half_angle_l157_15726

theorem terminal_side_of_half_angle (θ : Real) 
  (h1 : |Real.cos θ| = Real.cos θ) 
  (h2 : |Real.tan θ| = -Real.tan θ) : 
  (∃ (k : Int), 
    (k * Real.pi + Real.pi / 2 < θ / 2 ∧ θ / 2 ≤ k * Real.pi + Real.pi) ∨
    (k * Real.pi + 3 * Real.pi / 2 < θ / 2 ∧ θ / 2 ≤ k * Real.pi + 2 * Real.pi) ∨
    (∃ (n : Int), θ / 2 = n * Real.pi)) :=
sorry

end terminal_side_of_half_angle_l157_15726


namespace linear_system_solution_l157_15769

theorem linear_system_solution (k : ℚ) (x y z : ℚ) : 
  x ≠ 0 → y ≠ 0 → z ≠ 0 →
  x + 2*k*y + 4*z = 0 →
  4*x + k*y + z = 0 →
  3*x + 6*y + 5*z = 0 →
  (k = 90/41 ∧ y*z/x^2 = 41/30) :=
by sorry

end linear_system_solution_l157_15769


namespace min_value_problem_l157_15743

theorem min_value_problem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let line := {(x, y) : ℝ × ℝ | a * x - b * y + 2 = 0}
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 + 4*x - 4*y - 1 = 0}
  (∃ (p q : ℝ × ℝ), p ∈ line ∧ q ∈ line ∧ p ∈ circle ∧ q ∈ circle ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 6) →
  (2 / a + 3 / b ≥ 5 + 2 * Real.sqrt 6 ∧ 
   ∃ (a' b' : ℝ), a' > 0 ∧ b' > 0 ∧ 
     2 / a' + 3 / b' = 5 + 2 * Real.sqrt 6 ∧
     ∃ (p q : ℝ × ℝ), p ∈ {(x, y) : ℝ × ℝ | a' * x - b' * y + 2 = 0} ∧ 
       q ∈ {(x, y) : ℝ × ℝ | a' * x - b' * y + 2 = 0} ∧
       p ∈ circle ∧ q ∈ circle ∧
       Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = 6) :=
by sorry

end min_value_problem_l157_15743


namespace sandys_comic_books_l157_15759

theorem sandys_comic_books (initial : ℕ) : 
  (initial / 2 + 6 = 13) → initial = 14 := by
  sorry

end sandys_comic_books_l157_15759


namespace max_of_two_numbers_l157_15718

theorem max_of_two_numbers (a b : ℕ) (ha : a = 2) (hb : b = 3) :
  max a b = 3 := by
  sorry

end max_of_two_numbers_l157_15718


namespace min_distance_to_A_l157_15737

-- Define the space
variable (X : Type) [NormedAddCommGroup X] [InnerProductSpace ℝ X] [CompleteSpace X]

-- Define points A, B, and P
variable (A B P : X)

-- Define the conditions
variable (h1 : ‖A - B‖ = 4)
variable (h2 : ‖P - A‖ - ‖P - B‖ = 3)

-- State the theorem
theorem min_distance_to_A :
  ∃ (min_dist : ℝ), min_dist = 7/2 ∧ ∀ P, ‖P - A‖ - ‖P - B‖ = 3 → ‖P - A‖ ≥ min_dist :=
sorry

end min_distance_to_A_l157_15737


namespace ben_remaining_amount_l157_15700

/-- Calculates the remaining amount after a series of transactions -/
def remaining_amount (initial: Int) (supplier_payment: Int) (debtor_payment: Int) (maintenance_cost: Int) : Int :=
  initial - supplier_payment + debtor_payment - maintenance_cost

/-- Proves that given the specified transactions, the remaining amount is $1000 -/
theorem ben_remaining_amount :
  remaining_amount 2000 600 800 1200 = 1000 := by
  sorry

end ben_remaining_amount_l157_15700
