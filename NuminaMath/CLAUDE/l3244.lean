import Mathlib

namespace fifth_term_is_1280_l3244_324432

/-- A geometric sequence of positive integers with first term 5 and fourth term 320 -/
def geometric_sequence (n : ℕ) : ℕ :=
  5 * (320 / 5) ^ ((n - 1) / 3)

/-- The fifth term of the geometric sequence is 1280 -/
theorem fifth_term_is_1280 : geometric_sequence 5 = 1280 := by
  sorry

end fifth_term_is_1280_l3244_324432


namespace trajectory_is_ellipse_l3244_324463

/-- The trajectory of point M satisfying the given conditions is an ellipse -/
theorem trajectory_is_ellipse (x y : ℝ) : 
  let F : ℝ × ℝ := (0, 2)
  let line_y : ℝ := 8
  let distance_to_F := Real.sqrt ((x - F.1)^2 + (y - F.2)^2)
  let distance_to_line := |y - line_y|
  distance_to_F / distance_to_line = 1 / 2 → x^2 / 12 + y^2 / 16 = 1 :=
by sorry

end trajectory_is_ellipse_l3244_324463


namespace sector_arc_length_l3244_324410

theorem sector_arc_length (θ : Real) (A : Real) (l : Real) : 
  θ = 120 → A = π → l = (2 * Real.sqrt 3 * π) / 3 → 
  l = (θ * Real.sqrt (3 * A / θ) * π) / 180 := by
  sorry

end sector_arc_length_l3244_324410


namespace union_of_A_and_B_l3244_324413

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x^2 - x - 6 ≥ 0}
def B : Set ℝ := {x | (1 - x) / (x - 3) ≥ 0}

-- State the theorem
theorem union_of_A_and_B : A ∪ B = {x : ℝ | x ≥ 1 ∨ x ≤ -3/2} := by sorry

end union_of_A_and_B_l3244_324413


namespace expression_simplification_l3244_324453

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 2 + 1) :
  (x^2 - 1) / (x^2 + x) / (x - (2*x - 1) / x) = Real.sqrt 2 / 2 := by
  sorry

end expression_simplification_l3244_324453


namespace no_valid_base_for_450_l3244_324450

def is_four_digit (n : ℕ) (b : ℕ) : Prop :=
  b ^ 3 ≤ n ∧ n < b ^ 4

def last_digit (n : ℕ) (b : ℕ) : ℕ :=
  n % b

theorem no_valid_base_for_450 :
  ¬ ∃ (b : ℕ), b > 1 ∧ is_four_digit 450 b ∧ Odd (last_digit 450 b) :=
sorry

end no_valid_base_for_450_l3244_324450


namespace quadratic_no_real_roots_l3244_324418

theorem quadratic_no_real_roots (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x + k - 1 ≠ 0) → k > 2 := by
  sorry

end quadratic_no_real_roots_l3244_324418


namespace burning_time_3x5_grid_l3244_324428

/-- Represents a rectangular grid of toothpicks -/
structure ToothpickGrid where
  rows : ℕ
  cols : ℕ
  toothpicks : ℕ

/-- Represents the burning properties of toothpicks -/
structure BurningProperties where
  burn_time : ℕ  -- Time for one toothpick to burn completely
  spread_speed : ℝ  -- Speed at which fire spreads (assumed constant)

/-- Calculates the maximum burning time for a toothpick grid -/
def max_burning_time (grid : ToothpickGrid) (props : BurningProperties) : ℕ :=
  sorry  -- The actual calculation would go here

/-- Theorem stating the maximum burning time for the specific grid -/
theorem burning_time_3x5_grid :
  let grid := ToothpickGrid.mk 3 5 38
  let props := BurningProperties.mk 10 1
  max_burning_time grid props = 65 :=
sorry

#check burning_time_3x5_grid

end burning_time_3x5_grid_l3244_324428


namespace sum_of_reciprocals_l3244_324430

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 22*x^2 + 80*x - 67

-- Define the roots
variables (p q r : ℝ)

-- Define A, B, C
variables (A B C : ℝ)

-- Axioms
axiom distinct_roots : p ≠ q ∧ q ≠ r ∧ p ≠ r
axiom roots : f p = 0 ∧ f q = 0 ∧ f r = 0

axiom partial_fraction_decomposition :
  ∀ s : ℝ, s ≠ p ∧ s ≠ q ∧ s ≠ r →
    1 / (s^3 - 22*s^2 + 80*s - 67) = A / (s - p) + B / (s - q) + C / (s - r)

-- Theorem to prove
theorem sum_of_reciprocals : 1/A + 1/B + 1/C = 244 := by sorry

end sum_of_reciprocals_l3244_324430


namespace divisibility_problem_l3244_324403

theorem divisibility_problem (A B C : Nat) : 
  A < 10 → B < 10 → C < 10 →
  (7 * 1000000 + A * 100000 + 5 * 10000 + 1 * 1000 + B * 10 + 2) % 15 = 0 →
  (3 * 1000000 + 2 * 100000 + 6 * 10000 + A * 1000 + B * 100 + 4 * 10 + C) % 15 = 0 →
  C = 4 := by
sorry

end divisibility_problem_l3244_324403


namespace teachers_arrangement_count_l3244_324482

def number_of_seats : ℕ := 25
def number_of_teachers : ℕ := 5
def min_gap : ℕ := 2

def arrange_teachers (seats : ℕ) (teachers : ℕ) (gap : ℕ) : ℕ :=
  Nat.choose (seats + teachers - (teachers - 1) * (gap + 1)) teachers

theorem teachers_arrangement_count :
  arrange_teachers number_of_seats number_of_teachers min_gap = 26334 := by
  sorry

end teachers_arrangement_count_l3244_324482


namespace optimal_furniture_purchase_l3244_324431

def maximize_furniture (budget chair_price table_price : ℕ) : ℕ × ℕ :=
  let (tables, chairs) := (25, 37)
  have budget_constraint : tables * table_price + chairs * chair_price ≤ budget := by sorry
  have chair_lower_bound : chairs ≥ tables := by sorry
  have chair_upper_bound : chairs ≤ (3 * tables) / 2 := by sorry
  have is_optimal : ∀ (t c : ℕ), t * table_price + c * chair_price ≤ budget → 
                    c ≥ t → c ≤ (3 * t) / 2 → t + c ≤ tables + chairs := by sorry
  (tables, chairs)

theorem optimal_furniture_purchase :
  let (tables, chairs) := maximize_furniture 2000 20 50
  tables = 25 ∧ chairs = 37 := by sorry

end optimal_furniture_purchase_l3244_324431


namespace set_operations_l3244_324439

def U : Set ℝ := Set.univ

def A : Set ℝ := {x | 0 ≤ x ∧ x < 5}

def B : Set ℝ := {x | -2 ≤ x ∧ x < 4}

theorem set_operations :
  (A ∩ B = {x | 0 ≤ x ∧ x < 4}) ∧
  (A ∪ B = {x | -2 ≤ x ∧ x < 5}) ∧
  (A ∩ (U \ B) = {x | 4 ≤ x ∧ x < 5}) := by
  sorry

end set_operations_l3244_324439


namespace sum_of_numbers_l3244_324406

theorem sum_of_numbers : 1357 + 3571 + 5713 + 7135 + 1357 = 19133 := by
  sorry

end sum_of_numbers_l3244_324406


namespace R_duration_approx_l3244_324456

/-- Represents the investment and profit information for three partners -/
structure PartnershipData where
  inv_ratio_P : ℚ
  inv_ratio_Q : ℚ
  inv_ratio_R : ℚ
  profit_ratio_P : ℚ
  profit_ratio_Q : ℚ
  profit_ratio_R : ℚ
  duration_P : ℚ
  duration_Q : ℚ

/-- Calculates the investment duration for partner R given the partnership data -/
def calculate_R_duration (data : PartnershipData) : ℚ :=
  (data.profit_ratio_R * data.inv_ratio_Q * data.duration_Q) /
  (data.profit_ratio_Q * data.inv_ratio_R)

/-- Theorem stating that R's investment duration is approximately 5.185 months -/
theorem R_duration_approx (data : PartnershipData)
  (h1 : data.inv_ratio_P = 7)
  (h2 : data.inv_ratio_Q = 5)
  (h3 : data.inv_ratio_R = 3)
  (h4 : data.profit_ratio_P = 7)
  (h5 : data.profit_ratio_Q = 9)
  (h6 : data.profit_ratio_R = 4)
  (h7 : data.duration_P = 5)
  (h8 : data.duration_Q = 7) :
  abs (calculate_R_duration data - 5.185) < 0.001 := by
  sorry

end R_duration_approx_l3244_324456


namespace no_prime_in_first_15_cumulative_sums_l3244_324414

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def nthPrime (n : ℕ) : ℕ := sorry

def cumulativePrimeSum (n : ℕ) : ℕ := 
  if n = 0 then 0 else cumulativePrimeSum (n-1) + nthPrime (n+1)

theorem no_prime_in_first_15_cumulative_sums : 
  ∀ n : ℕ, n > 0 → n ≤ 15 → ¬(isPrime (cumulativePrimeSum n)) :=
sorry

end no_prime_in_first_15_cumulative_sums_l3244_324414


namespace isosceles_triangle_base_length_l3244_324476

/-- Given an equilateral triangle with perimeter 60 and an isosceles triangle with perimeter 55,
    where one side of the equilateral triangle is a side of the isosceles triangle,
    the base of the isosceles triangle is 15 units long. -/
theorem isosceles_triangle_base_length
  (equilateral_perimeter : ℝ)
  (isosceles_perimeter : ℝ)
  (h_equilateral_perimeter : equilateral_perimeter = 60)
  (h_isosceles_perimeter : isosceles_perimeter = 55)
  (h_shared_side : equilateral_perimeter / 3 = isosceles_perimeter / 2 - isosceles_base / 2) :
  isosceles_base = 15 :=
by
  sorry

#check isosceles_triangle_base_length

end isosceles_triangle_base_length_l3244_324476


namespace reflected_ray_equation_l3244_324498

-- Define the points
def M : ℝ × ℝ := (-2, 3)
def P : ℝ × ℝ := (1, 0)

-- Define the x-axis
def x_axis : Set (ℝ × ℝ) := {p | p.2 = 0}

-- Define the Law of Reflection
def law_of_reflection (incident : ℝ × ℝ → ℝ × ℝ → Prop) (reflected : ℝ × ℝ → ℝ × ℝ → Prop) : Prop :=
  ∀ p q r, incident p q → reflected q r → (q.2 = 0) → 
    (p.2 - q.2) * (r.1 - q.1) = (r.2 - q.2) * (p.1 - q.1)

-- State the theorem
theorem reflected_ray_equation :
  ∃ (incident reflected : ℝ × ℝ → ℝ × ℝ → Prop),
    incident M P ∧ P ∈ x_axis ∧ law_of_reflection incident reflected →
    ∃ a b c : ℝ, a * P.1 + b * P.2 + c = 0 ∧
               ∀ x y : ℝ, reflected P (x, y) ↔ a * x + b * y + c = 0 ∧
               a = 1 ∧ b = 1 ∧ c = -1 := by
  sorry

end reflected_ray_equation_l3244_324498


namespace r_plus_s_value_l3244_324457

/-- The line equation y = -5/3 * x + 15 -/
def line_equation (x y : ℝ) : Prop := y = -5/3 * x + 15

/-- Point P is where the line crosses the x-axis -/
def P : ℝ × ℝ := (9, 0)

/-- Point Q is where the line crosses the y-axis -/
def Q : ℝ × ℝ := (0, 15)

/-- T is a point on line segment PQ -/
def T_on_PQ (r s : ℝ) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ r = t * P.1 + (1 - t) * Q.1 ∧ s = t * P.2 + (1 - t) * Q.2

/-- Area of triangle POQ is 4 times the area of triangle TOP -/
def area_condition (r s : ℝ) : Prop :=
  abs ((P.1 * Q.2 - Q.1 * P.2) / 2) = 4 * abs ((P.1 * s - r * P.2) / 2)

/-- Main theorem: Given the conditions, r + s = 10.5 -/
theorem r_plus_s_value (r s : ℝ) 
  (h1 : line_equation r s) 
  (h2 : T_on_PQ r s) 
  (h3 : area_condition r s) : 
  r + s = 10.5 := by
  sorry

end r_plus_s_value_l3244_324457


namespace arctan_sum_identity_l3244_324452

theorem arctan_sum_identity : 
  Real.arctan (3/4) + 2 * Real.arctan (4/3) = π - Real.arctan (3/4) := by
  sorry

end arctan_sum_identity_l3244_324452


namespace movies_on_shelves_l3244_324460

theorem movies_on_shelves (total_movies : ℕ) (num_shelves : ℕ) (h1 : total_movies = 999) (h2 : num_shelves = 5) :
  ∃ (additional_movies : ℕ), 
    additional_movies = 1 ∧ 
    (total_movies + additional_movies) % num_shelves = 0 :=
by sorry

end movies_on_shelves_l3244_324460


namespace count_negative_numbers_l3244_324468

def given_set : Finset Int := {-3, -2, 0, 5}

theorem count_negative_numbers : 
  (given_set.filter (λ x => x < 0)).card = 2 := by sorry

end count_negative_numbers_l3244_324468


namespace fifth_term_is_648_l3244_324490

/-- A geometric sequence with 7 terms, first term 8, and last term 5832 -/
def GeometricSequence : Type := 
  { a : Fin 7 → ℝ // a 0 = 8 ∧ a 6 = 5832 ∧ ∀ i j, i < j → (a j) / (a i) = (a 1) / (a 0) }

/-- The fifth term of the geometric sequence is 648 -/
theorem fifth_term_is_648 (seq : GeometricSequence) : seq.val 4 = 648 := by
  sorry

end fifth_term_is_648_l3244_324490


namespace system_solution_unique_l3244_324479

theorem system_solution_unique (x y z : ℚ) : 
  x + 2*y - z = 100 ∧
  y - z = 25 ∧
  3*x - 5*y + 4*z = 230 →
  x = 101.25 ∧ y = -26.25 ∧ z = -51.25 := by
sorry

end system_solution_unique_l3244_324479


namespace solve_a_and_b_l3244_324472

def A : Set ℝ := {x | -2 < x ∧ x < -1 ∨ x > 1}

def B (a b : ℝ) : Set ℝ := {x | x^2 + a*x + b ≤ 0}

theorem solve_a_and_b :
  ∃ (a b : ℝ),
    (A ∪ B a b = {x | x > -2}) ∧
    (A ∩ B a b = {x | 1 < x ∧ x ≤ 3}) ∧
    a = -4 ∧ b = 3 := by
  sorry

end solve_a_and_b_l3244_324472


namespace problem_solution_l3244_324487

-- Define the quadratic equation
def quadratic_equation (a : ℝ) (t : ℝ) : Prop :=
  2 * a * t^2 + 12 * t + 9 = 0

-- Define parallel lines
def parallel_lines (a b : ℝ) : Prop :=
  ∃ (x y : ℝ), a * x + b * y = 1 ∧ 4 * x + 18 * y = 3

-- Define the b-th prime number
def nth_prime (b : ℕ) (p : ℕ) : Prop :=
  p.Prime ∧ (Finset.filter Nat.Prime (Finset.range p)).card = b

-- Define the trigonometric equation
def trig_equation (k θ : ℝ) : Prop :=
  k = (4 * Real.sin θ + 3 * Real.cos θ) / (2 * Real.sin θ - Real.cos θ) ∧
  Real.tan θ = 3

theorem problem_solution :
  (∃ a : ℝ, quadratic_equation a has_equal_roots) →
  (∃ b : ℝ, parallel_lines 2 b) →
  (∃ p : ℕ, nth_prime 9 p) →
  (∃ k θ : ℝ, trig_equation k θ) →
  ∃ (a b : ℝ) (p : ℕ) (k : ℝ),
    a = 2 ∧ b = 9 ∧ p = 23 ∧ k = 3 :=
by sorry

end problem_solution_l3244_324487


namespace angle_sum_around_point_l3244_324435

/-- Given a point W with four angles around it, where one angle is 90°, 
    another is y°, a third is 3y°, and the sum of all angles is 360°, 
    prove that y = 67.5° -/
theorem angle_sum_around_point (y : ℝ) : 
  90 + y + 3*y = 360 → y = 67.5 := by sorry

end angle_sum_around_point_l3244_324435


namespace juan_oranges_picked_l3244_324474

def total_oranges : ℕ := 107
def del_daily_pick : ℕ := 23
def del_days : ℕ := 2

theorem juan_oranges_picked : 
  total_oranges - (del_daily_pick * del_days) = 61 := by
  sorry

end juan_oranges_picked_l3244_324474


namespace find_y_l3244_324475

theorem find_y : ∃ y : ℝ, y > 0 ∧ 16 * y = 256 ∧ ∃ n : ℕ, y^2 = n ∧ y = 16 := by
  sorry

end find_y_l3244_324475


namespace min_prime_no_solution_l3244_324455

theorem min_prime_no_solution : 
  ∀ p : ℕ, Prime p → p > 3 →
    (∀ n : ℕ, n > 0 → ¬(2^n + 3^n) % p = 0) →
    p ≥ 19 :=
by sorry

end min_prime_no_solution_l3244_324455


namespace range_of_expression_l3244_324465

theorem range_of_expression (x y : ℝ) (h1 : x * y = 1) (h2 : 3 ≥ x) (h3 : x ≥ 4 * y) (h4 : y > 0) :
  ∃ (a b : ℝ), a = 4 ∧ b = 5 ∧
  (∀ z, (z = (x^2 + 4*y^2) / (x - 2*y)) → a ≤ z ∧ z ≤ b) ∧
  (∃ z1 z2, z1 = (x^2 + 4*y^2) / (x - 2*y) ∧ z2 = (x^2 + 4*y^2) / (x - 2*y) ∧ z1 = a ∧ z2 = b) :=
by sorry

end range_of_expression_l3244_324465


namespace quadratic_polynomial_proof_l3244_324409

/-- A quadratic polynomial M in terms of x -/
def M (a : ℝ) (x : ℝ) : ℝ := (a + 4) * x^3 + 6 * x^2 - 2 * x + 5

/-- The coefficient of the quadratic term -/
def b : ℝ := 6

/-- Point A on the number line -/
def A : ℝ := -4

/-- Point B on the number line -/
def B : ℝ := 6

/-- Position of P after t seconds -/
def P (t : ℝ) : ℝ := A + 2 * t

/-- Position of Q after t seconds (starting 2 seconds after P) -/
def Q (t : ℝ) : ℝ := B - 2 * (t - 2)

/-- Distance between two points -/
def distance (x y : ℝ) : ℝ := |x - y|

theorem quadratic_polynomial_proof :
  (∀ x, M A x = 6 * x^2 - 2 * x + 5) ∧
  (∃ t, t > 0 ∧ (distance (P t) B = (1/2) * distance (P t) A)) ∧
  (∃ m, m > 2 ∧ distance (P m) (Q m) = 8) :=
by sorry

end quadratic_polynomial_proof_l3244_324409


namespace circle_intersection_theorem_l3244_324411

-- Define the types for points and circles
variable (Point Circle : Type)
-- Define the predicate for a point lying on a circle
variable (lies_on : Point → Circle → Prop)
-- Define the predicate for two circles intersecting
variable (intersect : Circle → Circle → Prop)
-- Define the predicate for a circle being tangent to another circle
variable (tangent : Circle → Circle → Prop)
-- Define the predicate for a point being the intersection of a line and a circle
variable (line_circle_intersection : Point → Point → Circle → Point → Prop)
-- Define the predicate for four points being concyclic
variable (concyclic : Point → Point → Point → Point → Prop)

-- State the theorem
theorem circle_intersection_theorem 
  (Γ₁ Γ₂ Γ : Circle) 
  (A B C D E F G H I : Point) :
  intersect Γ₁ Γ₂ →
  lies_on A Γ₁ ∧ lies_on A Γ₂ →
  lies_on B Γ₁ ∧ lies_on B Γ₂ →
  tangent Γ Γ₁ ∧ tangent Γ Γ₂ →
  lies_on D Γ ∧ lies_on D Γ₁ →
  lies_on E Γ ∧ lies_on E Γ₂ →
  line_circle_intersection A B Γ C →
  line_circle_intersection E C Γ₂ F →
  line_circle_intersection D C Γ₁ G →
  line_circle_intersection E D Γ₁ H →
  line_circle_intersection E D Γ₂ I →
  concyclic F G H I := by
  sorry

end circle_intersection_theorem_l3244_324411


namespace cosine_power_identity_l3244_324492

theorem cosine_power_identity (θ : ℝ) (u : ℝ) (n : ℤ) :
  2 * Real.cos θ = u + 1 / u →
  2 * Real.cos (n * θ) = u^n + 1 / u^n :=
by sorry

end cosine_power_identity_l3244_324492


namespace even_6digit_integers_count_l3244_324440

/-- The count of even 6-digit positive integers -/
def count_even_6digit_integers : ℕ :=
  9 * 10^4 * 5

/-- Theorem: The count of even 6-digit positive integers is 450,000 -/
theorem even_6digit_integers_count : count_even_6digit_integers = 450000 := by
  sorry

end even_6digit_integers_count_l3244_324440


namespace minimum_points_tenth_game_l3244_324438

def first_five_games : List ℕ := [18, 15, 13, 17, 19]
def next_four_games : List ℕ := [14, 20, 12, 21]

def total_points_nine_games : ℕ := (first_five_games.sum + next_four_games.sum)

theorem minimum_points_tenth_game :
  ∀ x : ℕ, 
    (((total_points_nine_games + x) : ℚ) / 10 > 17) ∧ 
    (∀ y : ℕ, y < x → ((total_points_nine_games + y : ℚ) / 10 ≤ 17)) → 
    x = 22 :=
by sorry

end minimum_points_tenth_game_l3244_324438


namespace total_shared_amount_l3244_324429

theorem total_shared_amount 
  (T a b c d : ℚ) 
  (h1 : a = (1/3) * (b + c + d))
  (h2 : b = (2/7) * (a + c + d))
  (h3 : c = (4/9) * (a + b + d))
  (h4 : d = (5/11) * (a + b + c))
  (h5 : a = b + 20)
  (h6 : c = d - 15)
  (h7 : T = a + b + c + d)
  (h8 : ∃ k : ℤ, T = 10 * k) :
  T = 1330 := by
sorry

end total_shared_amount_l3244_324429


namespace perpendicular_line_exists_l3244_324459

-- Define the circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line
structure Line where
  point1 : ℝ × ℝ
  point2 : ℝ × ℝ

-- Define perpendicularity
def perpendicular (l1 l2 : Line) : Prop := sorry

-- Define a point being on a line
def point_on_line (p : ℝ × ℝ) (l : Line) : Prop := sorry

-- Define a point being on a circle
def point_on_circle (p : ℝ × ℝ) (c : Circle) : Prop := sorry

-- Define a line passing through a point
def line_through_point (l : Line) (p : ℝ × ℝ) : Prop := sorry

-- Theorem statement
theorem perpendicular_line_exists 
  (C : Circle) (A B M : ℝ × ℝ) (diameter : Line) :
  point_on_circle A C →
  point_on_circle B C →
  point_on_line A diameter →
  point_on_line B diameter →
  ∃ (L : Line), line_through_point L M ∧ perpendicular L diameter :=
sorry

end perpendicular_line_exists_l3244_324459


namespace money_difference_equals_5p_minus_20_l3244_324495

/-- The number of pennies in a nickel -/
def nickel_value : ℕ := 5

/-- The number of nickels Jessica has -/
def jessica_nickels (p : ℕ) : ℕ := 3 * p + 2

/-- The number of nickels Samantha has -/
def samantha_nickels (p : ℕ) : ℕ := 2 * p + 6

/-- The difference in money (in pennies) between Jessica and Samantha -/
def money_difference (p : ℕ) : ℤ :=
  nickel_value * (jessica_nickels p - samantha_nickels p)

theorem money_difference_equals_5p_minus_20 (p : ℕ) :
  money_difference p = 5 * p - 20 := by sorry

end money_difference_equals_5p_minus_20_l3244_324495


namespace nested_square_root_twenty_l3244_324486

theorem nested_square_root_twenty : 
  ∃ x : ℝ, x = Real.sqrt (20 + x) ∧ x = 5 := by sorry

end nested_square_root_twenty_l3244_324486


namespace inscribed_cube_volume_l3244_324421

/-- The volume of a cube inscribed in a cylinder, which is inscribed in a larger cube --/
theorem inscribed_cube_volume (outer_cube_edge : ℝ) (h : outer_cube_edge = 12) :
  let cylinder_radius : ℝ := outer_cube_edge / 2
  let cylinder_diameter : ℝ := outer_cube_edge
  let inscribed_cube_face_diagonal : ℝ := cylinder_diameter
  let inscribed_cube_edge : ℝ := inscribed_cube_face_diagonal / Real.sqrt 2
  let inscribed_cube_volume : ℝ := inscribed_cube_edge ^ 3
  inscribed_cube_volume = 432 * Real.sqrt 2 := by
  sorry

end inscribed_cube_volume_l3244_324421


namespace digit_placement_combinations_l3244_324467

def grid_size : ℕ := 6
def num_digits : ℕ := 4

theorem digit_placement_combinations : 
  (grid_size * (grid_size - 1) * (grid_size - 2) * (grid_size - 3) * (grid_size - 4)) = 720 :=
by sorry

end digit_placement_combinations_l3244_324467


namespace arman_second_week_hours_l3244_324461

/-- Calculates the number of hours worked in the second week given the conditions of Arman's work schedule and earnings. -/
def hours_worked_second_week (
  first_week_hours : ℕ)
  (first_week_rate : ℚ)
  (rate_increase : ℚ)
  (total_earnings : ℚ) : ℚ :=
  let first_week_earnings := first_week_hours * first_week_rate
  let second_week_earnings := total_earnings - first_week_earnings
  let new_rate := first_week_rate + rate_increase
  second_week_earnings / new_rate

/-- Theorem stating that given the conditions of Arman's work schedule and earnings, 
    the number of hours worked in the second week is 40. -/
theorem arman_second_week_hours :
  hours_worked_second_week 35 10 0.5 770 = 40 := by
  sorry

end arman_second_week_hours_l3244_324461


namespace constant_value_l3244_324407

theorem constant_value (t : ℝ) (x y : ℝ → ℝ) (c : ℝ) :
  (∀ t, x t = c - 4 * t) →
  (∀ t, y t = 2 * t - 2) →
  x 0.5 = y 0.5 →
  c = 1 := by
  sorry

end constant_value_l3244_324407


namespace eight_members_prefer_b_first_l3244_324401

/-- Represents the number of ballots for each permutation of candidates A, B, C -/
structure BallotCounts where
  abc : ℕ
  acb : ℕ
  cab : ℕ
  cba : ℕ
  bca : ℕ
  bac : ℕ

/-- The committee voting system with given conditions -/
def CommitteeVoting (counts : BallotCounts) : Prop :=
  -- Total number of ballots is 20
  counts.abc + counts.acb + counts.cab + counts.cba + counts.bca + counts.bac = 20 ∧
  -- Each permutation appears at least once
  counts.abc ≥ 1 ∧ counts.acb ≥ 1 ∧ counts.cab ≥ 1 ∧
  counts.cba ≥ 1 ∧ counts.bca ≥ 1 ∧ counts.bac ≥ 1 ∧
  -- 11 members prefer A to B
  counts.abc + counts.acb + counts.cab = 11 ∧
  -- 12 members prefer C to A
  counts.cab + counts.cba + counts.bca = 12 ∧
  -- 14 members prefer B to C
  counts.abc + counts.bca + counts.bac = 14

/-- The theorem stating that 8 members have B as their first choice -/
theorem eight_members_prefer_b_first (counts : BallotCounts) :
  CommitteeVoting counts → counts.bca + counts.bac = 8 := by
  sorry

end eight_members_prefer_b_first_l3244_324401


namespace soda_barrel_leak_time_l3244_324491

/-- The time it takes to fill one barrel with the leak -/
def leak_fill_time : ℝ := 5

/-- The normal filling time for one barrel -/
def normal_fill_time : ℝ := 3

/-- The number of barrels -/
def num_barrels : ℝ := 12

/-- The additional time it takes to fill all barrels with the leak -/
def additional_time : ℝ := 24

theorem soda_barrel_leak_time :
  leak_fill_time * num_barrels = normal_fill_time * num_barrels + additional_time :=
by sorry

end soda_barrel_leak_time_l3244_324491


namespace regular_18gon_symmetry_sum_l3244_324473

/-- Represents a regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle of rotational symmetry in degrees for a regular polygon -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ := 360 / n

theorem regular_18gon_symmetry_sum :
  ∀ (p : RegularPolygon 18),
    (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by sorry

end regular_18gon_symmetry_sum_l3244_324473


namespace limit_special_function_l3244_324477

/-- The limit of ((x+1)/(2x))^((ln(x+2))/(ln(2-x))) as x approaches 1 is √3 -/
theorem limit_special_function :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |((x + 1) / (2 * x))^((Real.log (x + 2)) / (Real.log (2 - x))) - Real.sqrt 3| < ε :=
sorry

end limit_special_function_l3244_324477


namespace binomial_floor_divisibility_l3244_324446

theorem binomial_floor_divisibility (p n : ℕ) (hp : Prime p) (hn : n ≥ p) :
  p ∣ (Nat.choose n p - n / p) := by
  sorry

end binomial_floor_divisibility_l3244_324446


namespace tony_initial_money_l3244_324444

/-- Given Tony's expenses and remaining money, prove his initial amount --/
theorem tony_initial_money :
  ∀ (initial spent_ticket spent_hotdog remaining : ℕ),
    spent_ticket = 8 →
    spent_hotdog = 3 →
    remaining = 9 →
    initial = spent_ticket + spent_hotdog + remaining →
    initial = 20 := by
  sorry

end tony_initial_money_l3244_324444


namespace circle_tangent_to_semicircles_radius_bounds_l3244_324471

/-- Given a triangle ABC with semiperimeter s and inradius r, and semicircles drawn on its sides,
    the radius t of the circle tangent to all three semicircles satisfies:
    s/2 < t ≤ s/2 + (1 - √3/2)r -/
theorem circle_tangent_to_semicircles_radius_bounds
  (s r t : ℝ) -- semiperimeter, inradius, and radius of tangent circle
  (h_s_pos : 0 < s)
  (h_r_pos : 0 < r)
  (h_t_pos : 0 < t)
  (h_triangle : ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ s = (a + b + c) / 2)
  (h_inradius : ∃ (area : ℝ), area > 0 ∧ r = area / s)
  (h_tangent : ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
               t + x/2 = t + y/2 ∧ t + y/2 = t + z/2 ∧ x + y + z = 2 * s) :
  s / 2 < t ∧ t ≤ s / 2 + (1 - Real.sqrt 3 / 2) * r := by
  sorry

end circle_tangent_to_semicircles_radius_bounds_l3244_324471


namespace pascal_triangle_row_34_l3244_324437

theorem pascal_triangle_row_34 : 
  let row_34 := List.range 35
  let nth_elem (n : ℕ) := Nat.choose 34 n
  (nth_elem 29 = 278256) ∧ (nth_elem 30 = 46376) := by
sorry

end pascal_triangle_row_34_l3244_324437


namespace min_sum_with_condition_min_sum_equality_l3244_324499

theorem min_sum_with_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  a + b ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

theorem min_sum_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 1/a + 2/b = 1) :
  a + b = 3 + 2 * Real.sqrt 2 ↔ a = 1 + Real.sqrt 2 ∧ b = 2 + Real.sqrt 2 := by
  sorry

end min_sum_with_condition_min_sum_equality_l3244_324499


namespace chlorine_and_hcl_moles_l3244_324433

/-- Represents the stoichiometric coefficients of the chemical reaction:
    C2H6 + 6Cl2 → C2Cl6 + 6HCl -/
structure ReactionCoefficients where
  ethane : ℕ
  chlorine : ℕ
  hexachloroethane : ℕ
  hydrochloric_acid : ℕ

/-- The given chemical reaction -/
def reaction : ReactionCoefficients :=
  { ethane := 1
  , chlorine := 6
  , hexachloroethane := 1
  , hydrochloric_acid := 6 }

/-- The number of moles of ethane given -/
def ethane_moles : ℕ := 3

/-- Theorem stating the number of moles of chlorine required and hydrochloric acid formed -/
theorem chlorine_and_hcl_moles :
  (ethane_moles * reaction.chlorine = 18) ∧
  (ethane_moles * reaction.hydrochloric_acid = 18) := by
  sorry

end chlorine_and_hcl_moles_l3244_324433


namespace tooth_arrangements_count_l3244_324436

/-- The number of unique arrangements of letters in TOOTH -/
def toothArrangements : ℕ :=
  Nat.factorial 5 / (Nat.factorial 3 * Nat.factorial 2)

/-- Theorem stating that the number of unique arrangements of letters in TOOTH is 10 -/
theorem tooth_arrangements_count : toothArrangements = 10 := by
  sorry

end tooth_arrangements_count_l3244_324436


namespace speed_ratio_proof_l3244_324483

/-- Proves that the speed ratio of return to outbound trip is 2:1 given specific conditions -/
theorem speed_ratio_proof (total_distance : ℝ) (total_time : ℝ) (return_speed : ℝ) : 
  total_distance = 40 ∧ 
  total_time = 6 ∧ 
  return_speed = 10 → 
  (return_speed / (total_distance / 2 / (total_time - total_distance / 2 / return_speed))) = 2 := by
  sorry

end speed_ratio_proof_l3244_324483


namespace parabola_c_value_l3244_324426

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_c_value (p : Parabola) :
  (p.x_coord 4 = 5) →  -- vertex at (5,4)
  (p.x_coord 6 = 1) →  -- passes through (1,6)
  (p.x_coord 0 = -27) →  -- passes through (-27,0)
  p.c = -27 := by
  sorry

end parabola_c_value_l3244_324426


namespace possible_m_values_l3244_324496

-- Define the sets A and B
def A (m : ℝ) : Set ℝ := {1, 3, Real.sqrt m}
def B (m : ℝ) : Set ℝ := {1, m}

-- Define the theorem
theorem possible_m_values :
  ∀ m : ℝ, (B m ⊆ A m) → (m = 0 ∨ m = 3) :=
by
  sorry

-- Note: The proof is omitted as per the instructions

end possible_m_values_l3244_324496


namespace yarn_balls_per_sweater_l3244_324416

/-- The number of balls of yarn needed for each sweater -/
def balls_per_sweater : ℕ := sorry

/-- The cost of each ball of yarn in dollars -/
def yarn_cost : ℕ := 6

/-- The selling price of each sweater in dollars -/
def sweater_price : ℕ := 35

/-- The number of sweaters sold -/
def sweaters_sold : ℕ := 28

/-- The total profit from selling all sweaters in dollars -/
def total_profit : ℕ := 308

theorem yarn_balls_per_sweater :
  (sweaters_sold * (sweater_price - yarn_cost * balls_per_sweater) = total_profit) →
  balls_per_sweater = 4 := by sorry

end yarn_balls_per_sweater_l3244_324416


namespace basketball_games_played_l3244_324422

theorem basketball_games_played (team_a_win_ratio : Rat) (team_b_win_ratio : Rat)
  (team_b_more_wins : ℕ) (team_b_more_losses : ℕ) :
  team_a_win_ratio = 3/4 →
  team_b_win_ratio = 2/3 →
  team_b_more_wins = 9 →
  team_b_more_losses = 9 →
  ∃ (team_a_games : ℕ),
    team_a_games = 36 ∧
    (team_a_games : Rat) * team_a_win_ratio + (team_a_games : Rat) * (1 - team_a_win_ratio) = team_a_games ∧
    ((team_a_games : Rat) + (team_b_more_wins + team_b_more_losses : Rat)) * team_b_win_ratio = 
      team_a_games * team_a_win_ratio + team_b_more_wins :=
by sorry

end basketball_games_played_l3244_324422


namespace arithmetic_sequence_min_sum_l3244_324419

/-- Arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1 : ℤ) * d

/-- Sum of the first n terms of an arithmetic sequence -/
def arithmetic_sum (a₁ d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1 : ℤ) * d) / 2

/-- The value of n that minimizes the sum of the first n terms -/
def minimizing_n (a₁ d : ℤ) : Set ℕ :=
  {n : ℕ | ∀ m : ℕ, arithmetic_sum a₁ d n ≤ arithmetic_sum a₁ d m}

theorem arithmetic_sequence_min_sum :
  minimizing_n (-28) 4 = {7, 8} := by
  sorry

end arithmetic_sequence_min_sum_l3244_324419


namespace quadratic_equation_with_given_root_properties_l3244_324462

theorem quadratic_equation_with_given_root_properties :
  ∀ (a b c p q : ℝ),
    a ≠ 0 →
    (∀ x, a * x^2 + b * x + c = 0 ↔ x = p ∨ x = q) →
    p + q = 12 →
    |p - q| = 4 →
    a * x^2 + b * x + c = x^2 - 12 * x + 32 :=
by sorry

end quadratic_equation_with_given_root_properties_l3244_324462


namespace calculation_proof_l3244_324449

theorem calculation_proof : 8 * 5.4 - 0.6 * 10 / 1.2 = 38.2 := by
  sorry

end calculation_proof_l3244_324449


namespace partial_fraction_decomposition_l3244_324405

theorem partial_fraction_decomposition :
  ∃ (A B C : ℚ),
    (∀ x : ℚ, x ≠ 0 ∧ x ≠ 1 ∧ x ≠ -1 →
      (x^2 - 5*x + 6) / (x^3 - x) = A / x + (B*x + C) / (x^2 - 1)) ∧
    A = -6 ∧ B = 7 ∧ C = -5 := by
  sorry

end partial_fraction_decomposition_l3244_324405


namespace batsman_average_after_20th_innings_l3244_324425

/-- Represents a batsman's statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored) / (stats.innings + 1)

/-- Theorem: Given the conditions, prove that the new average is 92 -/
theorem batsman_average_after_20th_innings 
  (stats : BatsmanStats)
  (h1 : stats.innings = 19)
  (h2 : newAverage stats 130 = stats.average + 2)
  : newAverage stats 130 = 92 := by
  sorry

#check batsman_average_after_20th_innings

end batsman_average_after_20th_innings_l3244_324425


namespace basketball_team_selection_l3244_324424

def total_players : ℕ := 18
def quadruplets : ℕ := 4
def team_size : ℕ := 7
def quadruplets_in_team : ℕ := 2

theorem basketball_team_selection :
  (Nat.choose quadruplets quadruplets_in_team) *
  (Nat.choose (total_players - quadruplets) (team_size - quadruplets_in_team)) = 12012 :=
by sorry

end basketball_team_selection_l3244_324424


namespace fifth_observation_value_l3244_324469

theorem fifth_observation_value (x1 x2 x3 x4 x5 x6 x7 x8 x9 : ℝ) :
  (x1 + x2 + x3 + x4 + x5) / 5 = 10 →
  (x5 + x6 + x7 + x8 + x9) / 5 = 8 →
  (x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9) / 9 = 8 →
  x5 = 18 := by
  sorry

end fifth_observation_value_l3244_324469


namespace apple_sale_loss_l3244_324493

/-- The fraction of the cost price lost by a seller when selling an item -/
def fractionLost (sellingPrice costPrice : ℚ) : ℚ :=
  (costPrice - sellingPrice) / costPrice

/-- Theorem: The fraction of the cost price lost when selling an apple for 19 Rs with a cost price of 20 Rs is 1/20 -/
theorem apple_sale_loss : fractionLost 19 20 = 1 / 20 := by
  sorry

end apple_sale_loss_l3244_324493


namespace total_toll_for_week_l3244_324451

/-- Calculate the total toll for a week for an 18-wheel truck -/
theorem total_toll_for_week (total_wheels : Nat) (front_axle_wheels : Nat) (other_axle_wheels : Nat)
  (weekday_base_toll : Real) (weekday_rate : Real) (weekend_base_toll : Real) (weekend_rate : Real) :
  total_wheels = 18 →
  front_axle_wheels = 2 →
  other_axle_wheels = 4 →
  weekday_base_toll = 2.50 →
  weekday_rate = 0.70 →
  weekend_base_toll = 3.00 →
  weekend_rate = 0.80 →
  let total_axles := (total_wheels - front_axle_wheels) / other_axle_wheels + 1
  let weekday_toll := weekday_base_toll + weekday_rate * (total_axles - 1)
  let weekend_toll := weekend_base_toll + weekend_rate * (total_axles - 1)
  let total_toll := 5 * weekday_toll + 2 * weekend_toll
  total_toll = 38.90 := by
  sorry

end total_toll_for_week_l3244_324451


namespace intersection_of_M_and_N_l3244_324458

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.2 = -p.1 + 1}
def N : Set (ℝ × ℝ) := {p | p.2 = p.1 - 1}

-- State the theorem
theorem intersection_of_M_and_N : M ∩ N = {(1, 0)} := by sorry

end intersection_of_M_and_N_l3244_324458


namespace max_correct_answers_l3244_324400

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (wrong_points : ℤ) (total_score : ℤ) :
  total_questions = 60 →
  correct_points = 5 →
  wrong_points = -2 →
  total_score = 150 →
  (∃ (correct unpicked wrong : ℕ),
    correct + unpicked + wrong = total_questions ∧
    correct * correct_points + wrong * wrong_points = total_score) →
  (∀ (x : ℕ), x > 38 →
    ¬∃ (unpicked wrong : ℕ),
      x + unpicked + wrong = total_questions ∧
      x * correct_points + wrong * wrong_points = total_score) :=
by sorry

end max_correct_answers_l3244_324400


namespace canteen_banana_requirement_l3244_324494

/-- The number of bananas required for the given period -/
def total_bananas : ℕ := 9828

/-- The number of weeks in the given period -/
def num_weeks : ℕ := 9

/-- The number of days in a week -/
def days_per_week : ℕ := 7

/-- The number of bananas in a dozen -/
def bananas_per_dozen : ℕ := 12

/-- Theorem: The canteen needs 13 dozens of bananas per day -/
theorem canteen_banana_requirement :
  (total_bananas / (num_weeks * days_per_week)) / bananas_per_dozen = 13 := by
  sorry

end canteen_banana_requirement_l3244_324494


namespace watch_gain_percentage_l3244_324434

/-- Calculates the gain percentage when a watch is sold at a different price -/
def gainPercentage (costPrice sellPrice : ℚ) : ℚ :=
  (sellPrice - costPrice) / costPrice * 100

/-- Theorem: The gain percentage is 5% under the given conditions -/
theorem watch_gain_percentage :
  let costPrice : ℚ := 933.33
  let initialLossPercentage : ℚ := 10
  let initialSellPrice : ℚ := costPrice * (1 - initialLossPercentage / 100)
  let newSellPrice : ℚ := initialSellPrice + 140
  gainPercentage costPrice newSellPrice = 5 := by
  sorry

end watch_gain_percentage_l3244_324434


namespace fourth_largest_divisor_of_n_l3244_324420

def n : ℕ := 1000800000

def fourth_largest_divisor (m : ℕ) : ℕ := sorry

theorem fourth_largest_divisor_of_n :
  fourth_largest_divisor n = 62550000 := by sorry

end fourth_largest_divisor_of_n_l3244_324420


namespace count_valid_sequences_l3244_324466

/-- The set of digits to be used -/
def Digits : Finset Nat := {0, 1, 2, 3, 4}

/-- A function to check if a number is even -/
def isEven (n : Nat) : Bool := n % 2 = 0

/-- A function to check if a digit sequence satisfies the condition -/
def validSequence (seq : List Nat) : Bool :=
  seq.length = 5 ∧ 
  seq.toFinset = Digits ∧
  ∃ i, i ∈ [1, 2, 3] ∧ 
    isEven (seq.nthLe i sorry) ∧ 
    ¬isEven (seq.nthLe (i-1) sorry) ∧ 
    ¬isEven (seq.nthLe (i+1) sorry)

/-- The main theorem -/
theorem count_valid_sequences : 
  (Digits.toList.permutations.filter validSequence).length = 28 := by
  sorry

end count_valid_sequences_l3244_324466


namespace discriminant_not_necessary_nor_sufficient_l3244_324417

/-- The function f(x) = ax^2 + bx + c --/
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The condition that the graph of f is always above the x-axis --/
def always_above (a b c : ℝ) : Prop :=
  ∀ x, f a b c x > 0

/-- The discriminant condition --/
def discriminant_condition (a b c : ℝ) : Prop :=
  b^2 - 4*a*c < 0

theorem discriminant_not_necessary_nor_sufficient :
  ¬(∀ a b c : ℝ, discriminant_condition a b c ↔ always_above a b c) :=
sorry

end discriminant_not_necessary_nor_sufficient_l3244_324417


namespace rectangular_window_width_l3244_324402

/-- Represents the width of a rectangular window with specific pane arrangements and dimensions. -/
def window_width (pane_width : ℝ) : ℝ :=
  3 * pane_width + 4  -- 3 panes across plus 4 borders

/-- Theorem stating the width of the rectangular window under given conditions. -/
theorem rectangular_window_width :
  ∃ (pane_width : ℝ),
    pane_width > 0 ∧
    (3 : ℝ) / 4 * pane_width = 3 / 4 * pane_width ∧  -- height-to-width ratio of 3:4
    window_width pane_width = 28 := by
  sorry

end rectangular_window_width_l3244_324402


namespace profit_percentage_problem_l3244_324412

/-- Calculates the profit percentage given the cost price and selling price -/
def profit_percentage (cost_price selling_price : ℚ) : ℚ :=
  (selling_price - cost_price) / cost_price * 100

/-- Theorem stating that the profit percentage is 25% for the given problem -/
theorem profit_percentage_problem : profit_percentage 96 120 = 25 := by
  sorry

end profit_percentage_problem_l3244_324412


namespace reciprocal_of_negative_2023_l3244_324489

theorem reciprocal_of_negative_2023 :
  ∃ x : ℚ, x * (-2023) = 1 ∧ x = -1/2023 := by
sorry

end reciprocal_of_negative_2023_l3244_324489


namespace quadratic_function_proof_l3244_324485

/-- The quadratic function a(x) -/
def a (x : ℝ) : ℝ := 2*x^2 - 14*x + 20

/-- The shape function y = 2x² -/
def shape (x : ℝ) : ℝ := 2*x^2

theorem quadratic_function_proof :
  a 2 = 0 ∧ a 5 = 0 ∧ ∃ k, ∀ x, a x = k * shape x + (a 0 - k * shape 0) :=
sorry

end quadratic_function_proof_l3244_324485


namespace k_range_for_two_solutions_l3244_324427

open Real

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := k * x

noncomputable def g (x : ℝ) : ℝ := (log x) / x

theorem k_range_for_two_solutions (k : ℝ) :
  (∃ x y, x ∈ Set.Icc (1/ℯ) ℯ ∧ y ∈ Set.Icc (1/ℯ) ℯ ∧ x ≠ y ∧ f k x = g x ∧ f k y = g y) →
  k ∈ Set.Ioo (1/ℯ^2) (1/(2*ℯ)) :=
sorry

end k_range_for_two_solutions_l3244_324427


namespace difference_of_expressions_l3244_324448

theorem difference_of_expressions : 
  (Real.sqrt (0.9 * 40) - (4/5 * (2/3 * 25))) = -22/3 := by
  sorry

end difference_of_expressions_l3244_324448


namespace icosahedron_edges_l3244_324484

/-- A regular icosahedron is a polyhedron with 20 faces and 12 vertices, 
    where each vertex is connected to 5 edges. -/
structure RegularIcosahedron where
  faces : ℕ
  vertices : ℕ
  edges_per_vertex : ℕ
  faces_eq : faces = 20
  vertices_eq : vertices = 12
  edges_per_vertex_eq : edges_per_vertex = 5

/-- The number of edges in a regular icosahedron is 30. -/
theorem icosahedron_edges (i : RegularIcosahedron) : 
  (i.vertices * i.edges_per_vertex) / 2 = 30 := by
  sorry

#check icosahedron_edges

end icosahedron_edges_l3244_324484


namespace jackson_decorations_given_l3244_324454

/-- The number of decorations given to the neighbor -/
def decorations_given_to_neighbor (num_boxes : ℕ) (decorations_per_box : ℕ) (decorations_used : ℕ) : ℕ :=
  num_boxes * decorations_per_box - decorations_used

/-- Theorem: Mrs. Jackson gave 92 decorations to her neighbor -/
theorem jackson_decorations_given :
  decorations_given_to_neighbor 6 25 58 = 92 := by
  sorry

end jackson_decorations_given_l3244_324454


namespace cylinder_volume_from_rectangle_l3244_324488

/-- The volume of a cylinder formed by rotating a rectangle about its vertical line of symmetry -/
theorem cylinder_volume_from_rectangle (length width : ℝ) (h_length : length = 20) (h_width : width = 10) :
  let radius := width / 2
  let height := length
  let volume := π * radius^2 * height
  volume = 500 * π := by
  sorry

end cylinder_volume_from_rectangle_l3244_324488


namespace base_2_representation_of_123_l3244_324445

/-- Converts a natural number to its binary representation as a list of bits -/
def to_binary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec aux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: aux (m / 2)
  aux n

/-- Converts a list of bits to a natural number -/
def from_binary (l : List Bool) : ℕ :=
  l.foldr (fun b n => 2 * n + if b then 1 else 0) 0

theorem base_2_representation_of_123 :
  to_binary 123 = [true, true, true, true, false, true, true] :=
by sorry

end base_2_representation_of_123_l3244_324445


namespace dilution_proof_l3244_324404

/-- Proves that adding 7.2 ounces of water to 12 ounces of 40% alcohol shaving lotion 
    results in a solution with 25% alcohol concentration -/
theorem dilution_proof (initial_volume : ℝ) (initial_concentration : ℝ) 
  (target_concentration : ℝ) (water_added : ℝ) : 
  initial_volume = 12 ∧ 
  initial_concentration = 0.4 ∧ 
  target_concentration = 0.25 ∧
  water_added = 7.2 →
  (initial_volume * initial_concentration) / (initial_volume + water_added) = target_concentration :=
by sorry

end dilution_proof_l3244_324404


namespace tangent_slope_product_l3244_324497

theorem tangent_slope_product (x₀ : ℝ) : 
  let y₁ : ℝ → ℝ := λ x => 2 - 1/x
  let y₂ : ℝ → ℝ := λ x => x^3 - x^2 + x
  let y₁' : ℝ → ℝ := λ x => 1/x^2
  let y₂' : ℝ → ℝ := λ x => 3*x^2 - 2*x + 1
  (y₁' x₀) * (y₂' x₀) = 3 → x₀ = 1/2 := by
sorry

end tangent_slope_product_l3244_324497


namespace sum_a_b_equals_one_l3244_324481

theorem sum_a_b_equals_one (a b : ℝ) (h : a = Real.sqrt (2 * b - 4) + Real.sqrt (4 - 2 * b) - 1) : a + b = 1 := by
  sorry

end sum_a_b_equals_one_l3244_324481


namespace bill_donut_purchase_l3244_324408

/-- The number of ways to distribute donuts among types with constraints -/
def donut_combinations (total_donuts : ℕ) (num_types : ℕ) (min_types : ℕ) : ℕ :=
  -- Definition to be implemented
  sorry

/-- Theorem stating the specific case for Bill's donut purchase -/
theorem bill_donut_purchase :
  donut_combinations 8 5 4 = 425 :=
sorry

end bill_donut_purchase_l3244_324408


namespace imaginary_unit_power_sum_l3244_324423

theorem imaginary_unit_power_sum : ∀ i : ℂ, i^2 = -1 → i^45 + i^205 + i^365 = 3*i := by
  sorry

end imaginary_unit_power_sum_l3244_324423


namespace derivative_f_zero_dne_l3244_324480

noncomputable def f (x : ℝ) : ℝ :=
  if x ≠ 0 then 6 * x + x * Real.sin (1 / x) else 0

theorem derivative_f_zero_dne :
  ¬ ∃ (L : ℝ), ∀ ε > 0, ∃ δ > 0, ∀ h : ℝ, 0 < |h| → |h| < δ →
    |((f (0 + h) - f 0) / h) - L| < ε :=
sorry

end derivative_f_zero_dne_l3244_324480


namespace batsman_average_after_12_innings_l3244_324443

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : Nat
  totalRuns : Nat
  lastInningsScore : Nat
  averageIncrease : Nat

/-- Calculates the average score after a given number of innings -/
def calculateAverage (stats : BatsmanStats) : Nat :=
  (stats.totalRuns) / (stats.innings)

/-- Theorem stating the batsman's average after 12 innings -/
theorem batsman_average_after_12_innings 
  (stats : BatsmanStats)
  (h1 : stats.innings = 12)
  (h2 : stats.lastInningsScore = 48)
  (h3 : stats.averageIncrease = 2)
  (h4 : calculateAverage stats = calculateAverage { stats with 
    innings := stats.innings - 1, 
    totalRuns := stats.totalRuns - stats.lastInningsScore 
  } + stats.averageIncrease) :
  calculateAverage stats = 26 := by
  sorry

#check batsman_average_after_12_innings

end batsman_average_after_12_innings_l3244_324443


namespace arithmetic_progression_squares_l3244_324441

theorem arithmetic_progression_squares (a b c : ℝ) :
  (∃ d : ℝ, b = a + d ∧ c = a + 2*d) →
  ∃ q : ℝ, (a^2 + a*c + c^2) - (a^2 + a*b + b^2) = q ∧
           (b^2 + b*c + c^2) - (a^2 + a*c + c^2) = q :=
by sorry

end arithmetic_progression_squares_l3244_324441


namespace line_vector_to_slope_intercept_l3244_324470

/-- Given a line in vector form, prove it's equivalent to slope-intercept form --/
theorem line_vector_to_slope_intercept :
  ∀ (x y : ℝ), 
  (2 : ℝ) * (x - 3) + (-1 : ℝ) * (y + 4) = 0 ↔ y = 2 * x - 10 := by
  sorry

end line_vector_to_slope_intercept_l3244_324470


namespace unique_abc_solution_l3244_324478

/-- Represents a base-7 number with two digits -/
def Base7TwoDigit (a b : Nat) : Nat := 7 * a + b

/-- Represents a base-7 number with one digit -/
def Base7OneDigit (c : Nat) : Nat := c

theorem unique_abc_solution :
  ∀ A B C : Nat,
    A < 7 → B < 7 → C < 7 →
    A ≠ B → B ≠ C → A ≠ C →
    Base7TwoDigit A B + Base7OneDigit C = Base7TwoDigit C 0 →
    Base7TwoDigit A B + Base7TwoDigit B A = Base7TwoDigit C C →
    A = 5 ∧ B = 1 ∧ C = 6 :=
by sorry

end unique_abc_solution_l3244_324478


namespace solution_set_of_inequality_l3244_324442

-- Define an even function f: ℝ → ℝ
def even_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

-- Define increasing function on [0, +∞)
def increasing_on_nonneg (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, 0 ≤ x → 0 ≤ y → x < y → f x < f y

-- Main theorem
theorem solution_set_of_inequality
  (f : ℝ → ℝ)
  (h_even : even_function f)
  (h_increasing : increasing_on_nonneg f)
  (h_f2 : f 2 = 0) :
  {x : ℝ | f (x + 1) > 0} = Set.Ioo (-1 : ℝ) 1 := by
sorry

end solution_set_of_inequality_l3244_324442


namespace systematic_sample_fourth_member_l3244_324464

def systematic_sample (total : ℕ) (sample_size : ℕ) (known_members : List ℕ) : Prop :=
  ∃ (start : ℕ) (k : ℕ),
    k = total / sample_size ∧
    ∀ (i : ℕ), i < sample_size →
      (start + i * k) % total + 1 ∈ known_members ∪ {(start + (sample_size - 1) * k) % total + 1}

theorem systematic_sample_fourth_member 
  (total : ℕ) (sample_size : ℕ) (known_members : List ℕ) 
  (h_total : total = 52)
  (h_sample_size : sample_size = 4)
  (h_known_members : known_members = [6, 32, 45]) :
  systematic_sample total sample_size known_members →
  (19 : ℕ) ∈ known_members ∪ {19} :=
by sorry

end systematic_sample_fourth_member_l3244_324464


namespace intersection_sum_l3244_324415

-- Define the two equations
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 2*x + 1
def g (x y : ℝ) : Prop := x + 3*y = 3

-- Define the intersection points
def intersection_points : Set (ℝ × ℝ) :=
  {p | f p.1 = p.2 ∧ g p.1 p.2}

-- State the theorem
theorem intersection_sum :
  ∃ (p₁ p₂ p₃ : ℝ × ℝ),
    p₁ ∈ intersection_points ∧
    p₂ ∈ intersection_points ∧
    p₃ ∈ intersection_points ∧
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₂ ≠ p₃ ∧
    p₁.1 + p₂.1 + p₃.1 = 3 ∧
    p₁.2 + p₂.2 + p₃.2 = 2 :=
sorry

end intersection_sum_l3244_324415


namespace triangle_area_comparison_l3244_324447

-- Define the triangles
def Triangle := Fin 3 → ℝ × ℝ

-- Define the area of a triangle
noncomputable def area (t : Triangle) : ℝ := sorry

-- Define the side length between two points of a triangle
def side_length (t : Triangle) (i k : Fin 3) : ℝ := sorry

-- Define if a triangle is obtuse-angled
def is_obtuse (t : Triangle) : Prop := sorry

-- Theorem statement
theorem triangle_area_comparison 
  (A B : Triangle) 
  (h_sides : ∀ (i k : Fin 3), side_length A i k ≥ side_length B i k) 
  (h_not_obtuse : ¬ is_obtuse A) : 
  area A ≥ area B := by sorry

end triangle_area_comparison_l3244_324447
