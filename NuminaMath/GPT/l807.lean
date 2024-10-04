import Mathlib

namespace regular_octagon_interior_angle_l807_807414

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807414


namespace count_six_letter_strings_l807_807983

open Nat

def vowels_count : List ℕ := [6, 6, 6, 6, 3]

noncomputable def count_strings (n : ℕ) : ℕ :=
  (Finset.range 4).sum (λ k => choose n k * 6^(n - k))

theorem count_six_letter_strings : count_strings 6 = 117072 := by
  sorry

end count_six_letter_strings_l807_807983


namespace stockings_total_cost_l807_807619

-- Define a function to compute the total cost given the conditions
def total_cost (n_grandchildren n_children : Nat) 
               (stocking_price discount monogram_cost : Nat) : Nat :=
  let total_stockings := n_grandchildren + n_children
  let discounted_price := stocking_price - (stocking_price * discount / 100)
  let total_stockings_cost := discounted_price * total_stockings
  let total_monogram_cost := monogram_cost * total_stockings
  total_stockings_cost + total_monogram_cost

-- Prove that the total cost calculation is correct given the conditions
theorem stockings_total_cost :
  total_cost 5 4 20 10 5 = 207 :=
by
  -- A placeholder for the proof
  sorry

end stockings_total_cost_l807_807619


namespace angle_measure_l807_807841

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807841


namespace interior_angle_regular_octagon_l807_807456

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807456


namespace scientific_notation_l807_807117

theorem scientific_notation :
  56.9 * 10^9 = 5.69 * 10^(10 - 1) :=
by
  sorry

end scientific_notation_l807_807117


namespace cyclic_pentagon_angle_sum_l807_807956

theorem cyclic_pentagon_angle_sum (ABCDE : Type)
  [CyclicPentagon ABCDE] -- Assuming CyclicPentagon is defined elsewhere
  (angle_CEB : ℝ)
  (h_angle_CEB : angle_CEB = 17) :
  ∃ (angle_CDE angle_EAB : ℝ), angle_CDE + angle_EAB = 163 :=
by
  sorry

end cyclic_pentagon_angle_sum_l807_807956


namespace interior_angle_regular_octagon_l807_807444

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807444


namespace angle_supplement_complement_l807_807815

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l807_807815


namespace hiker_cyclist_wait_time_l807_807962

theorem hiker_cyclist_wait_time
    (hiker_speed : ℝ)
    (cyclist_speed : ℝ)
    (wait_time_minutes : ℝ)
    (wait_time_start_minutes : ℝ)
    (hiker_speed_pos : 0 < hiker_speed)
    (cyclist_speed_pos : 0 < cyclist_speed)
    (begin_wait_time : wait_time_start_minutes = 5)
    (hiker_speed_eq : hiker_speed = 7)
    (cyclist_speed_eq : cyclist_speed = 28) :
    wait_time_minutes = 20 :=
  begin
    sorry,
  end

end hiker_cyclist_wait_time_l807_807962


namespace angle_measure_l807_807848

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807848


namespace composites_from_factorial_l807_807623

def is_composite (m : ℕ) : Prop :=
  ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = m

theorem composites_from_factorial (n : ℕ) : ∃ N : ℕ, N = nat.factorial n + 1 ∧ ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → is_composite (N + i) :=
by
  sorry

end composites_from_factorial_l807_807623


namespace angle_supplement_complement_l807_807811

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l807_807811


namespace correct_statement_is_B_l807_807062

theorem correct_statement_is_B :
  let neg1 : Prop := ∀ x : ℝ, x^2 - 1 ≥ 0
  let neg2 : Prop := ∀ x : ℝ, x ≠ 3 → x^2 - 2*x - 3 ≠ 0
  let statement3 : Prop := ∃ (q : {q : Type | quadrilateral q}) (h : sides_equal q), ¬ square q
  let statement4 : Prop := ∀ x y : ℝ, cos x ≠ cos y → x ≠ y
  neg1 = (∀ x : ℝ, x^2 - 1 > 0) ∧ neg2 = (∀ x : ℝ, x ≠ 3 → x^2 - 2*x - 3 ≠ 0) ∧ statement3 = false ∧ statement4 = true →
  neg2 :=
sorry

end correct_statement_is_B_l807_807062


namespace regular_octagon_angle_l807_807316

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807316


namespace total_number_of_students_l807_807681

variable (T : ℕ)
variable (total_students : T = (2 / 5) * T + 90 + 150)

theorem total_number_of_students (h : total_students) : T = 400 :=
by
  sorry

end total_number_of_students_l807_807681


namespace expression_evaluation_l807_807640

variable (x y : ℤ)

theorem expression_evaluation (h₁ : x = -1) (h₂ : y = 1) : 
  (x + y) * (x - y) - (4 * x^3 * y - 8 * x * y^3) / (2 * x * y) = 2 :=
by
  rw [h₁, h₂]
  have h₃ : (-1 + 1) * (-1 - 1) - (4 * (-1)^3 * 1 - 8 * (-1) * 1^3) / (2 * (-1) * 1) = (-2) - (-10 / -2) := by sorry
  have h₄ : (-2) - 5 = 2 := by sorry
  sorry

end expression_evaluation_l807_807640


namespace cos_sum_angle_identity_l807_807017

theorem cos_sum_angle_identity : 
  cos (12 * Real.pi / 180) * cos (18 * Real.pi / 180) - sin (12 * Real.pi / 180) * sin (18 * Real.pi / 180) = cos (30 * Real.pi / 180) :=
by sorry

end cos_sum_angle_identity_l807_807017


namespace maximum_m_value_l807_807072

variable {a b c : ℝ}

noncomputable def maximum_m : ℝ := 9/8

theorem maximum_m_value 
  (h1 : (a - b)^2 + (b - c)^2 + (c - a)^2 ≥ maximum_m * a^2)
  (h2 : b^2 - 4 * a * c ≥ 0) : 
  maximum_m = 9 / 8 :=
sorry

end maximum_m_value_l807_807072


namespace restaurant_pizzas_l807_807096

theorem restaurant_pizzas (lunch dinner total : ℕ) (h_lunch : lunch = 9) (h_dinner : dinner = 6) 
  (h_total : total = lunch + dinner) : total = 15 :=
by
  rw [h_lunch, h_dinner, h_total]
  norm_num

end restaurant_pizzas_l807_807096


namespace sum_max_min_f_l807_807015

noncomputable def f (x : ℝ) : ℝ :=
  1 + (Real.sin x / (2 + Real.cos x))

theorem sum_max_min_f {a b : ℝ} (ha : ∀ x, f x ≤ a) (hb : ∀ x, b ≤ f x) (h_max : ∃ x, f x = a) (h_min : ∃ x, f x = b) :
  a + b = 2 :=
sorry

end sum_max_min_f_l807_807015


namespace angle_solution_l807_807886

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l807_807886


namespace minimal_constant_inequality_l807_807598

theorem minimal_constant_inequality (n : ℕ) (h : 2 ≤ n) (x : Fin n → ℝ) (hx : ∀ i, 0 ≤ x i) :
    ∑ i in Finset.range (n - 1), ∑ j in Finset.Ico (i + 1) n, x i * x j * (x i ^ 2 + x j ^ 2) ≤
    (1 / 8) * (∑ i in Finset.range n, x i) ^ 4 := 
sorry

end minimal_constant_inequality_l807_807598


namespace regular_octagon_interior_angle_l807_807279

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807279


namespace angle_measure_is_60_l807_807781

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l807_807781


namespace regular_octagon_interior_angle_l807_807253

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807253


namespace restore_grid_values_l807_807766

def gridCondition (A B C D : Nat) : Prop :=
  A < 9 ∧ 1 < 9 ∧ 9 < 9 ∧
  3 < 9 ∧ 5 < 9 ∧ D < 9 ∧
  B < 9 ∧ C < 9 ∧ 7 < 9 ∧
  -- Sum of any two numbers in adjacent cells is less than 12
  A + 1 < 12 ∧ A + 3 < 12 ∧
  B + C < 12 ∧ B + 3 < 12 ∧
  D + 2 < 12 ∧ C + 4 < 12 ∧
  -- The grid values are from 1 to 9
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  C ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
  -- Even numbers erased are 2, 4, 6, and 8

theorem restore_grid_values :
  ∃ (A B C D : Nat), gridCondition A B C D ∧ A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  sorry

end restore_grid_values_l807_807766


namespace angle_supplement_complement_l807_807804

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l807_807804


namespace regular_octagon_interior_angle_l807_807291

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807291


namespace standard_eq_of_ellipse_range_of_m_l807_807188

-- Definition and conditions for the ellipse
noncomputable def e (c a : ℝ) : ℝ := c / a
noncomputable def ellipse_eq (a b : ℝ) : Prop := (a > b ∧ b > 0)

-- Proof problem for the standard equation of ellipse
theorem standard_eq_of_ellipse (e := 1 / 2) (distance : ℝ := 1):
  ∃ (a b : ℝ), ellipse_eq a b ∧ (a - e * a = distance) ∧ 
  ((a - e * a = distance) ∧ (a = 2 ∧ (b^2 = a^2 - e^2))) ∧
  (∀ (x y : ℝ), ((x^2 / a^2) + (y^2 / b^2) = 1) ↔ (x^2 / 4 + y^2 / 3 = 1)) := sorry

-- Proof problem for the range of m
theorem range_of_m (k m : ℝ) :
  (3 + 4 * k^2 > m^2) → (7 * m^2 = 12 + 12 * k^2) → 
  (m^2 > 3 / 4) → 
  (m^2 ≥ 12 / 7) → 
  ∃ m, 
  (- ∞ < m ∧ m ≤ - 2 / 7 * real.sqrt 21) ∨ 
  (2 / 7 * real.sqrt 21 ≤ m ∧ m < + ∞) := sorry

end standard_eq_of_ellipse_range_of_m_l807_807188


namespace regular_octagon_interior_angle_l807_807321

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807321


namespace four_digit_numbers_divisible_by_7_l807_807510

theorem four_digit_numbers_divisible_by_7 :
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_n := Nat.ceil (lower_bound / 7)
  let largest_n := upper_bound / 7
  ∃ n : ℕ, smallest_n = 143 ∧ largest_n = 1428 ∧ (largest_n - smallest_n + 1 = 1286) :=
by
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_n := Nat.ceil (lower_bound / 7)
  let largest_n := upper_bound / 7
  use smallest_n, largest_n
  have h1 : smallest_n = 143 := sorry
  have h2 : largest_n = 1428 := sorry
  have h3 : largest_n - smallest_n + 1 = 1286 := sorry
  exact ⟨h1, h2, h3⟩

end four_digit_numbers_divisible_by_7_l807_807510


namespace problem_statement_l807_807058

variables {a b y x : ℝ}

theorem problem_statement :
  (3 * a + 2 * b ≠ 5 * a * b) ∧
  (5 * y - 3 * y ≠ 2) ∧
  (7 * a + a ≠ 7 * a ^ 2) ∧
  (3 * x^2 * y - 2 * y * x^2 = x^2 * y) :=
by
  split
  · intro h
    have h₁ : 3 * a + 2 * b = 3 * a + 2 * b := rfl
    rw h₁ at h
    sorry
  split
  · intro h
    have h₁ : 5 * y - 3 * y = 2 * y := by ring
    rw h₁ at h
    sorry
  split
  · intro h
    have h₁ : 7 * a + a = 8 * a := by ring
    rw h₁ at h
    sorry
  · ring

end problem_statement_l807_807058


namespace angle_measure_l807_807863

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l807_807863


namespace angle_measure_l807_807800

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l807_807800


namespace interior_angle_regular_octagon_l807_807491

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807491


namespace verify_differential_eq_l807_807035

noncomputable def function_z (x y : ℝ) : ℝ := 2 * Real.cos (y - x / 2) ^ 2

theorem verify_differential_eq (x y : ℝ) :
  2 * (deriv (λ x, deriv (λ x, function_z x y) x) x) +
    (deriv (λ y, deriv (λ x, function_z x y) x) y) = 0 :=
by
  sorry

end verify_differential_eq_l807_807035


namespace jackson_entertainment_expense_l807_807570

noncomputable def total_spent_on_entertainment_computer_game_original_price : ℝ :=
  66 / 0.85

noncomputable def movie_ticket_price_with_tax : ℝ :=
  12 * 1.10

noncomputable def total_movie_tickets_cost : ℝ :=
  3 * movie_ticket_price_with_tax

noncomputable def total_snacks_and_transportation_cost : ℝ :=
  7 + 5

noncomputable def total_spent : ℝ :=
  66 + total_movie_tickets_cost + total_snacks_and_transportation_cost

theorem jackson_entertainment_expense :
  total_spent = 117.60 :=
by
  sorry

end jackson_entertainment_expense_l807_807570


namespace arithmetic_geometric_sequences_l807_807005

theorem arithmetic_geometric_sequences :
  ∃ (A B C D : ℤ), A < B ∧ B > 0 ∧ C > 0 ∧ -- Ensure A, B, C are positive
  (B - A) = (C - B) ∧  -- Arithmetic sequence condition
  B * (49 : ℚ) = C * (49 / 9 : ℚ) ∧ -- Geometric sequence condition written using fractional equality
  A + B + C + D = 76 := 
by {
  sorry -- Placeholder for actual proof
}

end arithmetic_geometric_sequences_l807_807005


namespace min_value_x_2y_l807_807198

theorem min_value_x_2y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 2 * y + 2 * x * y = 8) : x + 2 * y ≥ 4 :=
sorry

end min_value_x_2y_l807_807198


namespace exists_points_convex_hull_contains_origin_l807_807579

noncomputable theory

open Set
open Convex

universe u

variables {d : ℕ}
variables {C : Fin d → Set (EuclideanSpace ℝ (Fin d))}

/-- Given compact, connected sets whose convex hull contains the origin, there exists points such
that the origin is contained in the convex hull of these points. -/
theorem exists_points_convex_hull_contains_origin
  (h_compact : ∀ i, IsCompact (C i))
  (h_connected : ∀ i, IsConnected (C i))
  (h_convex_hull_contains_origin : ∀ i, (0 : EuclideanSpace ℝ (Fin d)) ∈ convexHull ℝ (C i)) :
  ∃ (c : Fin d → EuclideanSpace ℝ (Fin d)), (∀ i, c i ∈ C i) ∧ 
    (0 : EuclideanSpace ℝ (Fin d)) ∈ convexHull ℝ (range c) :=
sorry

end exists_points_convex_hull_contains_origin_l807_807579


namespace angle_measure_l807_807798

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l807_807798


namespace ratio_is_correct_l807_807563

-- Definitions of given conditions
variables {B C A K : Type} [Inhabited B] [Inhabited C] [Inhabited A] [Inhabited K]
variable (BC : ℝ) (cos_ACB : ℚ) (CK KA : ℝ)
axiom BC_eq : BC = 5
axiom CK_eq : CK = 3
axiom KA_eq : KA = 1
axiom cos_ACB_eq : cos_ACB = 4/5

-- Target: the ratio of the circumradius of the circle passing through B and C to the radius of the circle inscribed in triangle ABK
def ratio_of_radii : ℝ := (10 * real.sqrt 10 + 25) / 9

-- The proof statement
theorem ratio_is_correct :
  ∃ (R r : ℝ), ((R / r) = ratio_of_radii) ∧ 
  -- Given conditions
  (BC = 5) ∧ (cos_ACB = 4/5) ∧ (CK = 3) ∧ (KA = 1) := 
sorry

end ratio_is_correct_l807_807563


namespace interior_angle_regular_octagon_l807_807458

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807458


namespace regular_octagon_interior_angle_l807_807285

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807285


namespace original_cost_of_book_l807_807537

theorem original_cost_of_book (y d t x : ℝ) (h1 : y = 10) (h2 : d = 0.5) (h3 : t = 45) : 10 * (x - 0.5) = 45 → x = 5 :=
by
  intros h
  calc
    10 * (x - 0.5) = 45 : by assumption
    10 * x - 5 = 45     : by ring
    10 * x = 50         : by linarith
    x = 5               : by linarith

end original_cost_of_book_l807_807537


namespace each_interior_angle_of_regular_octagon_l807_807345

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807345


namespace cuberoot_condition_l807_807517

/-- If \(\sqrt[3]{x-1}=3\), then \((x-1)^2 = 729\). -/
theorem cuberoot_condition (x : ℝ) (h : (x - 1)^(1/3) = 3) : (x - 1)^2 = 729 := 
  sorry

end cuberoot_condition_l807_807517


namespace angle_measure_l807_807852

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l807_807852


namespace angle_measure_l807_807829

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l807_807829


namespace angle_supplement_complement_l807_807864

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l807_807864


namespace regular_octagon_angle_l807_807317

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807317


namespace restore_grid_l807_807759

noncomputable def grid_values (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 9 < 12 ∧
  3 + 1 < 12 ∧ 3 + 5 < 12 ∧
  B + 3 < 12 ∧ B + 4 < 12 ∧
  B + 7 < 12 ∧ 
  C + 4 < 12 ∧
  C + 7 < 12 ∧ 
  D + 9 < 12 ∧
  D + 7 < 12 ∧
  D + 5 < 12

def even_numbers_erased (A B C D : ℕ) : Prop :=
  A ∈ {8} ∧ B ∈ {6} ∧ C ∈ {4} ∧ D ∈ {2}

theorem restore_grid :
  ∃ (A B C D : ℕ), grid_values A B C D ∧ even_numbers_erased A B C D :=
by {
  use [8, 6, 4, 2],
  dsimp [grid_values, even_numbers_erased],
  exact ⟨⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩, 
         ⟨rfl, rfl, rfl, rfl⟩⟩
}

end restore_grid_l807_807759


namespace man_is_older_by_l807_807089

theorem man_is_older_by :
  ∀ (M S : ℕ), S = 22 → (M + 2) = 2 * (S + 2) → (M - S) = 24 :=
by
  intros M S h1 h2
  sorry

end man_is_older_by_l807_807089


namespace regular_octagon_angle_l807_807314

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807314


namespace regular_octagon_interior_angle_l807_807248

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807248


namespace angle_measure_l807_807856

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l807_807856


namespace exists_lattice_point_l807_807182

-- Definitions of points and their properties
structure Point :=
  (x : ℤ)
  (y : ℤ)

-- Definition for convex pentagon with integer coordinates
structure ConvexPentagon :=
  (A B C D E : Point)
  (isConvex : convex {A, B, C, D, E})

-- Helper function to find intersection of diagonals
noncomputable def intersection (p1 p2 p3 p4 : Point) : Point := sorry

-- Definition for the interior pentagon formed by diagonal intersections
def InteriorPentagon (P : ConvexPentagon) : ConvexPentagon :=
  let A1 := intersection P.A P.C P.B P.D
  let B1 := intersection P.B P.D P.C P.E
  let C1 := intersection P.C P.E P.D P.A
  let D1 := intersection P.D P.A P.E P.B
  let E1 := intersection P.E P.B P.A P.C
  { A := A1, B := B1, C := C1, D := D1, E := E1, isConvex := sorry }

-- Main theorem statement
theorem exists_lattice_point (P : ConvexPentagon) : ∃ Q : Point, Q ∈ InteriorPentagon P.vertices ∧ Q.x ∈ ℤ ∧ Q.y ∈ ℤ :=
sorry

end exists_lattice_point_l807_807182


namespace restaurant_pizzas_l807_807097

theorem restaurant_pizzas (lunch dinner total : ℕ) (h_lunch : lunch = 9) (h_dinner : dinner = 6) 
  (h_total : total = lunch + dinner) : total = 15 :=
by
  rw [h_lunch, h_dinner, h_total]
  norm_num

end restaurant_pizzas_l807_807097


namespace total_cost_of_vacation_l807_807069

variable (C : ℚ)

def cost_per_person_divided_among_3 := C / 3
def cost_per_person_divided_among_4 := C / 4
def per_person_difference := 40

theorem total_cost_of_vacation
  (h : cost_per_person_divided_among_3 C - cost_per_person_divided_among_4 C = per_person_difference) :
  C = 480 := by
  sorry

end total_cost_of_vacation_l807_807069


namespace original_number_l807_807967

theorem original_number (x : ℕ) (h : ∃ k, 14 * x = 112 * k) : x = 8 :=
sorry

end original_number_l807_807967


namespace directrix_of_parabola_l807_807219

-- Define the parabola and the line conditions
def parabola (p : ℝ) := ∀ x y : ℝ, y^2 = 2 * p * x
def focus_line (x y : ℝ) := 2 * x + 3 * y - 8 = 0

-- Theorem stating that the directrix of the parabola is x = -4
theorem directrix_of_parabola (p : ℝ) (hx : ∃ x, ∃ y, focus_line x y) (hp : parabola p) :
  ∃ k : ℝ, k = 4 → ∀ x y : ℝ, (-x) = -4 :=
by
  sorry

end directrix_of_parabola_l807_807219


namespace angle_measure_l807_807832

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l807_807832


namespace probability_of_xyz_eq_72_l807_807914

open ProbabilityTheory Finset

def dice_values : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem probability_of_xyz_eq_72 :
  (∑ x in dice_values, ∑ y in dice_values, ∑ z in dice_values, 
   if x * y * z = 72 then 1 else 0) / (dice_values.card ^ 3) = 1 / 36 :=
by
  sorry -- Proof omitted

end probability_of_xyz_eq_72_l807_807914


namespace area_TANF_half_area_ABC_l807_807552

-- Define the conditions
variable {α : Type*}
variable (a b : ℝ) -- side length a, segment length b
variables (A B C N T F : α)
variable [EuclideanGeometry α]

-- Conditions on the triangle and points
axiom h_eq_triangle : equilateral_triangle A B C
axiom h_point_on_AC : on_line_segment A C N
axiom h_point_on_AB : on_line_segment A B T
axiom h_point_on_BC : on_line_segment B C F
axiom h_AN_TB : distance A N = distance T B
axiom h_CF_FB : distance C F = distance F B

-- Goal
theorem area_TANF_half_area_ABC :
  area (quadrilateral T A N F) = (1/2) * area (triangle A B C) :=
sorry

end area_TANF_half_area_ABC_l807_807552


namespace regular_octagon_angle_l807_807315

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807315


namespace regular_octagon_interior_angle_l807_807422

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807422


namespace evaluate_power_l807_807143

theorem evaluate_power :
  (64 : ℝ) = 2^6 →
  64^(3/4 : ℝ) = 16 * Real.sqrt 2 :=
by
  intro h₁
  rw [h₁]
  sorry

end evaluate_power_l807_807143


namespace reconstruct_trapezoid_from_segments_l807_807980

-- Definitions of the geometrical elements involved based on the problem's conditions
variables {A B C D O K L N M E F: Type}
noncomputable def is_trapezoid (A B C D : Type) : Prop := 
  sorry -- a function to check if ABCD is a trapezoid where AD is parallel to BC

noncomputable def is_perpendicular (OK AD : Type) : Prop := 
  sorry -- a function to check if OK is perpendicular to AD

noncomputable def is_midline (EF : Type) (A B C D : Type) : Prop := 
  sorry -- a function to check if EF is the midline of trapezoid ABCD

theorem reconstruct_trapezoid_from_segments :
  ∀ (A B C D O K L N M E F : Type),
  is_trapezoid A B C D →
  is_perpendicular O K A D → 
  is_midline E F A B C D → 
  ∃ (A' B' C' D' : Type),
  is_trapezoid A' B' C' D' ∧ sorry :=
sorry -- proof omitted

end reconstruct_trapezoid_from_segments_l807_807980


namespace weeks_to_work_l807_807567

-- Definitions of conditions as per step a)
def isabelle_ticket_cost : ℕ := 20
def brother_ticket_cost : ℕ := 10
def brothers_total_savings : ℕ := 5
def isabelle_savings : ℕ := 5
def job_pay_per_week : ℕ := 3
def total_ticket_cost := isabelle_ticket_cost + 2 * brother_ticket_cost
def total_savings := isabelle_savings + brothers_total_savings
def remaining_amount := total_ticket_cost - total_savings

-- Theorem statement to match the question
theorem weeks_to_work : remaining_amount / job_pay_per_week = 10 := by
  -- Lean expects a proof here, replaced with sorry to skip it
  sorry

end weeks_to_work_l807_807567


namespace maximum_value_sqrt_l807_807197

theorem maximum_value_sqrt {a b c : ℝ} (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) (h4 : a + b + 9*c^2 = 1) :
  (sqrt a + sqrt b + sqrt 3 * c) ≤ sqrt (7 / 3) ∧ 
  (∃ a b c : ℝ, (a = 3 / 7) ∧ (b = 3 / 7) ∧ (c = sqrt 7 / 21) ∧ a + b + 9*c^2 = 1 ∧ sqrt a + sqrt b + sqrt 3 * c = (sqrt 21) / 3) :=
by
  sorry

end maximum_value_sqrt_l807_807197


namespace regular_octagon_interior_angle_l807_807428

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807428


namespace line_through_origin_in_quadrants_l807_807670

theorem line_through_origin_in_quadrants (A B C : ℝ) :
  (-A * x - B * y + C = 0) ∧ (0 = 0) ∧ (exists x y, 0 < x * y) →
  (C = 0) ∧ (A * B < 0) :=
sorry

end line_through_origin_in_quadrants_l807_807670


namespace tom_first_part_speed_l807_807026

theorem tom_first_part_speed (v : ℝ) 
  (h1 : ∀ total_distance, total_distance = 60) 
  (h2 : ∀ first_part_distance, first_part_distance = 12) 
  (h3 : ∀ second_part_speed, second_part_speed = 48) 
  (h4 : ∀ avg_speed, avg_speed = 40) 
  (h5 : ∀ remaining_distance, remaining_distance = 60 - 12) 
  (h6 : ∀ time_second_part, time_second_part = remaining_distance / second_part_speed) 
  (h7 : ∀ total_time, total_time = (first_part_distance / v) + time_second_part) 
  (h8 : ∀ avg_speed, avg_speed = total_distance / total_time) :
  v = 24 := 
by 
  sorry

end tom_first_part_speed_l807_807026


namespace angle_supplement_complement_l807_807874

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l807_807874


namespace oil_drop_in_tank_l807_807082

def volume_cylinder (r h : ℝ) : ℝ := π * r^2 * h

theorem oil_drop_in_tank :
  let 
    r_stationary := 100,
    h_stationary := 25,
    r1 := 5,
    h1 := 10,
    r2 := 8,
    h2 := 12,
    r3 := 10,
    h3 := 7 
  in
  (π * r1^2 * h1 + π * r2^2 * h2 + π * r3^2 * h3) / (π * r_stationary^2) = 0.1718 :=
by
  sorry

end oil_drop_in_tank_l807_807082


namespace order_of_magnitude_l807_807221

noncomputable def a : ℝ := Real.log 6 / Real.log 0.3
noncomputable def b : ℝ := 0.3 ^ 6
noncomputable def c : ℝ := 6 ^ 0.3

theorem order_of_magnitude : a < b ∧ b < c := by
  sorry

end order_of_magnitude_l807_807221


namespace max_volume_height_l807_807974

noncomputable def volume (x : ℝ) : ℝ :=
  x * (x + 0.5) * (3.2 - 2 * x)

theorem max_volume_height : ∃ h, h = 1.2 ∧
  ∀ (x : ℝ), volume' x = 0 → volume (1) = volume x :=
begin
  -- Defining the derivative
  let volume' := λ x, -6 * x ^ 2 + 4.4 * x + 1.6,
  -- The volume achieves its maximum at x = 1
  have deriv_zero : volume' 1 = 0,
  { sorry },
  -- Given the constraints, the height h should be 1.2 at maximum volume
  use 1.2,
  split,
  { refl, },
  { sorry }
end

end max_volume_height_l807_807974


namespace girl_cannot_visit_in_6_trips_l807_807084

def can_visit_all_friends (trips: ℕ) (current_floor target1 target2: ℤ) (move1 move2: ℤ) : Prop :=
    ∃ (x y: ℕ), x + y ≤ trips ∧ 
               (current_floor + x * move1 + y * move2 = target1) ∧ 
               (current_floor + (x + y) * move1 + (x + y) * move2 = target2)

theorem girl_cannot_visit_in_6_trips : ¬can_visit_all_friends 6 1 12 14 3 7 :=
begin
  sorry
end

end girl_cannot_visit_in_6_trips_l807_807084


namespace regular_octagon_interior_angle_l807_807413

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807413


namespace minimal_AH_value_l807_807110

noncomputable def triangle_inscribed (A B C : Point) : Prop :=
circle_contains (circle_of_radius A B 1) C

noncomputable def angle_BAC_60 (A B C : Point) : Prop :=
angle A B C = 60 * degree

noncomputable def intersect_at_H (A B C H D E : Point) : Prop :=
altitudes_intersect A B C H D E

theorem minimal_AH_value (A B C H D E : Point) (r : ℝ) (BAC: ℝ):
  triangle_inscribed A B C → 
  angle_BAC_60 A B C →
  intersect_at_H A B C H D E →
  radius (circumcircle A B C) = 1 →
  (BAC = 60 * degree) →
  (0 < r) →
  (segment_length A H = r) :=
begin
  sorry
end

end minimal_AH_value_l807_807110


namespace angle_measure_l807_807796

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l807_807796


namespace sum_binom_remainder_l807_807587

theorem sum_binom_remainder :
  let S := ∑ n in finset.range (334 + 1), (-1)^n * nat.choose 1002 (3 * n)
  S % 1000 = 2 :=
begin
  let S := ∑ n in finset.range (334 + 1), (-1)^n * nat.choose 1002 (3 * n),
  sorry
end

end sum_binom_remainder_l807_807587


namespace checkerboard_corners_sum_l807_807610

theorem checkerboard_corners_sum : 
  let N : ℕ := 9 
  let corners := [1, 9, 73, 81]
  (corners.sum = 164) := by
  sorry

end checkerboard_corners_sum_l807_807610


namespace chessboard_colorings_l807_807647

-- Definitions based on conditions
def valid_chessboard_colorings_count : ℕ :=
  2 ^ 33

-- Theorem statement with the question, conditions, and the correct answer
theorem chessboard_colorings : 
  valid_chessboard_colorings_count = 2 ^ 33 := by
  sorry

end chessboard_colorings_l807_807647


namespace angle_measure_is_60_l807_807786

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l807_807786


namespace set_y_cardinality_l807_807532

theorem set_y_cardinality (x y : Set ℤ) (h1 : x.card = 8) (h2 : (x ∩ y).card = 6) (h3 : (x ∪ y \ (x ∩ y)).card = 6) : y.card = 10 :=
sorry

end set_y_cardinality_l807_807532


namespace part1_part2_l807_807675

noncomputable def A : Set ℝ := {x | 0 < x ∧ x ≤ 3}

noncomputable def B (m : ℝ) : Set ℝ := {x | x < m - 2} ∪ {x | x > m}

theorem part1 (m : ℝ) (h : m = 1) : A ∩ (B m)ᶜ = {x | 0 < x ∧ x ≤ 1} :=
by
  sorry

theorem part2 (A B : Set ℝ) (h : A = {x | 0 < x ∧ x ≤ 3}) (h' : B = B m) : 
  (A ⊆ B m) ↔ (m ≤ 0 ∨ 5 < m) :=
by
  sorry

end part1_part2_l807_807675


namespace book_arrangement_count_l807_807511

theorem book_arrangement_count :
  let total_books := 6
  let identical_science_books := 3
  let unique_other_books := total_books - identical_science_books
  (total_books! / (identical_science_books! * unique_other_books!)) = 120 := by
  sorry

end book_arrangement_count_l807_807511


namespace anna_total_earnings_l807_807988

-- Define the conditions
def monday_work : ℝ := 2.5 -- 2 1/2 hours in decimal
def tuesday_work : ℕ := 80 -- 80 minutes
def thursday_start : ℕ := 915 -- Start time: 9:15 AM as 915
def thursday_end : ℕ := 1200 -- End time: 12:00 PM as 1200
def saturday_work : ℕ := 45 -- 45 minutes
def hourly_rate : ℝ := 5 -- Wage of 5 dollars per hour

-- Translate the problem to Lean statement
theorem anna_total_earnings : 
  let total_minutes := 150 + tuesday_work + 165 + saturday_work,
      total_hours := total_minutes / 60,
      earnings := total_hours * hourly_rate
  in earnings = 37 := 
by
  -- 150 minutes for Monday, 165 minutes from 9:15 AM to 12:00 PM
  let total_minutes := 150 + tuesday_work + 165 + saturday_work,
      total_hours := total_minutes / 60,
      earnings := total_hours * hourly_rate;
  sorry

end anna_total_earnings_l807_807988


namespace probability_neither_red_nor_purple_l807_807959

section Probability

def total_balls : ℕ := 60
def red_balls : ℕ := 15
def purple_balls : ℕ := 3
def total_red_or_purple_balls : ℕ := red_balls + purple_balls
def non_red_or_purple_balls : ℕ := total_balls - total_red_or_purple_balls

theorem probability_neither_red_nor_purple :
  (non_red_or_purple_balls : ℚ) / (total_balls : ℚ) = 7 / 10 :=
by
  sorry

end Probability

end probability_neither_red_nor_purple_l807_807959


namespace find_grid_values_l807_807757

-- Define the grid and the conditions in Lean
variables (A B C D : ℕ)
assume h1: A, B, C, D ∈ {2, 4, 6, 8}
assume h2: A + 1 < 12
assume h3: A + 9 < 12
assume h4: A + 3 < 12
assume h5: A + 5 < 12
assume h6: A + D < 12
assume h7: B + C < 12
assume h8: B + 5 < 12
assume h9: C + 7 < 12
assume h10: D + 9 < 12

-- Prove the specific values for A, B, C, and D
theorem find_grid_values : 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by sorry

end find_grid_values_l807_807757


namespace prob_xyz_eq_72_l807_807929

-- Define the set of possible outcomes for a standard six-sided die
def dice_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define a predicate that checks if three dice rolls multiply to 72
def is_valid_combination (x y z : ℕ) : Prop := (x * y * z = 72)

-- Define the event space for three dice rolls
def event_space : Finset (ℕ × ℕ × ℕ) := Finset.product dice_outcomes (Finset.product dice_outcomes dice_outcomes)

-- Define the probability of an event
def probability {α : Type*} [Fintype α] (s : Finset α) (event : α → Prop) : ℚ :=
  (s.filter event).card.to_rat / s.card.to_rat

-- State the theorem
theorem prob_xyz_eq_72 : probability event_space (λ t, is_valid_combination t.1 t.2.1 t.2.2) = (7 / 216) := 
by { sorry }

end prob_xyz_eq_72_l807_807929


namespace half_water_remains_after_tilt_l807_807960

/--
Given a cylindrical barrel completely filled with water, if you tilt the barrel until the water surface intersects
the center of the bottom base and the diametrically opposite top edge, then exactly half of the water will remain.
-/
theorem half_water_remains_after_tilt (r h : ℝ) (V : ℝ) (Hfill : V = π * r^2 * h) :
  ∃ θ : ℝ, ∀ θ (Htilt : _, sorry -- exact condition for tilting) -- Use a sorry to handle the tilt condition.
  , volume (remaining_water_barrel_after_tilt r h θ) = V / 2 :=
sorry -- Proof goes here

end half_water_remains_after_tilt_l807_807960


namespace interior_angle_regular_octagon_l807_807498

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807498


namespace greatest_value_a4_b4_l807_807581

theorem greatest_value_a4_b4
    (a b : Nat → ℝ)
    (h_arith_seq : ∀ n, a (n + 1) = a n + a 1)
    (h_geom_seq : ∀ n, b (n + 1) = b n * b 1)
    (h_a1b1 : a 1 * b 1 = 20)
    (h_a2b2 : a 2 * b 2 = 19)
    (h_a3b3 : a 3 * b 3 = 14) :
    ∃ m : ℝ, a 4 * b 4 = 8 ∧ ∀ x, a 4 * b 4 ≤ x -> x = 8 := by
  sorry

end greatest_value_a4_b4_l807_807581


namespace cos_expression_l807_807177

-- Define the condition
def condition (θ : ℝ) : Prop :=
  sin (π / 3 - θ) = 1 / 2

-- State the proposition we want to prove
theorem cos_expression (θ : ℝ) (h : condition θ) : cos (π / 6 + θ) = 1 / 2 :=
  sorry

end cos_expression_l807_807177


namespace fixed_circle_center_circumcircle_l807_807589

open EuclideanGeometry

theorem fixed_circle_center_circumcircle 
  (Γ1 Γ2 : Circle) (P Q : Point) 
  (hPQ : P ∈ Γ1 ∧ P ∈ Γ2 ∧ Q ∈ Γ1 ∧ Q ∈ Γ2)
  (A1 B1 : Point) (hA1B1 : A1 ∈ Γ1 ∧ B1 ∈ Γ1)
  (A2 B2 : Point)
  (hA2 : line (through A1 P) ∩ Γ2 = A2 ∧ A2 ≠ P ∧ A2 ≠ A1)
  (hB2 : line (through B1 P) ∩ Γ2 = B2 ∧ B2 ≠ P ∧ B2 ≠ B1)
  (C : Point) (hC : ∃ l1 l2 : Line, A1 ∈ l1 ∧ B1 ∈ l1 ∧ A2 ∈ l2 ∧ B2 ∈ l2 ∧ C ∈ l1 ∧ C ∈ l2) :
  ∃ (O : Point) (fixed_circle : Circle), 
  center (circumcircle C A1 A2) = O ∧ O ∈ fixed_circle :=
by
  sorry

end fixed_circle_center_circumcircle_l807_807589


namespace angle_measure_l807_807902

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807902


namespace infinitely_many_perfect_squares_sum_of_perfect_square_and_prime_infinitely_many_perfect_squares_not_sum_of_perfect_square_and_prime_l807_807636

theorem infinitely_many_perfect_squares_sum_of_perfect_square_and_prime :
  ∃ᶠ n : ℕ in at_top, ∃ p : ℕ, prime p ∧ (n + 1) ^ 2 = n ^ 2 + p :=
by 
  sorry

theorem infinitely_many_perfect_squares_not_sum_of_perfect_square_and_prime :
  ∃ᶠ n : ℕ in at_top, ∀ p : ℕ, prime p → (n + 1) ^ 2 ≠ n ^ 2 + p :=
by
  sorry

end infinitely_many_perfect_squares_sum_of_perfect_square_and_prime_infinitely_many_perfect_squares_not_sum_of_perfect_square_and_prime_l807_807636


namespace isabelle_work_weeks_l807_807565

-- Define the costs and savings
def isabelle_ticket_cost := 20
def brother_ticket_cost := 10
def brothers_count := 2
def brothers_savings := 5
def isabelle_savings := 5
def weekly_earnings := 3

-- Calculate total required work weeks
theorem isabelle_work_weeks :
  let total_ticket_cost := isabelle_ticket_cost + brother_ticket_cost * brothers_count in
  let total_savings := isabelle_savings + brothers_savings in
  let required_savings := total_ticket_cost - total_savings in
  required_savings / weekly_earnings = 10 :=
by
  sorry

end isabelle_work_weeks_l807_807565


namespace regular_octagon_interior_angle_eq_135_l807_807407

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807407


namespace vinny_final_weight_l807_807036

theorem vinny_final_weight :
  let initial_weight := 300
  let first_month_loss := 20
  let second_month_loss := first_month_loss / 2
  let third_month_loss := second_month_loss / 2
  let fourth_month_loss := third_month_loss / 2
  let fifth_month_loss := 12
  let total_loss := first_month_loss + second_month_loss + third_month_loss + fourth_month_loss + fifth_month_loss
  let final_weight := initial_weight - total_loss
  final_weight = 250.5 :=
by
  sorry

end vinny_final_weight_l807_807036


namespace determine_grid_numbers_l807_807724

theorem determine_grid_numbers (A B C D : ℕ) :
  (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧ (1 ≤ D ∧ D ≤ 9) ∧ 
  (A ≠ 1) ∧ (A ≠ 3) ∧ (A ≠ 5) ∧ (A ≠ 7) ∧ (A ≠ 9) ∧ 
  (B ≠ 1) ∧ (B ≠ 3) ∧ (B ≠ 5) ∧ (B ≠ 7) ∧ (B ≠ 9) ∧ 
  (C ≠ 1) ∧ (C ≠ 3) ∧ (C ≠ 5) ∧ (C ≠ 7) ∧ (C ≠ 9) ∧ 
  (D ≠ 1) ∧ (D ≠ 3) ∧ (D ≠ 5) ∧ (D ≠ 7) ∧ (D ≠ 9) ∧ 
  (A ≠ 2 ∧ A ≠ 4 ∧ A ≠ 6 ∧ A ≠ 8 ∨ B ≠ 2 ∧ B ≠ 4 ∧ B ≠ 6 ∧ B ≠ 8 ∨ C ≠ 2 ∧ C ≠ 4 ∧ C ≠ 6 ∧ C ≠ 8 ∨ D ≠ 2 ∧ D ≠ 4 ∧ D ≠ 6 ∧ D ≠ 8) ∧ 
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ 
  (3 + 5 < 12) ∧ (3 + D < 12) ∧ (B + 5 < 12) ∧ 
  (B + C < 12) ∧ (C + 7 < 12) ∧ (B + 7 < 12) 
  → (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) :=
by 
  intros A B C D
  exact sorry

end determine_grid_numbers_l807_807724


namespace focus_of_conic_section_l807_807954

theorem focus_of_conic_section (x y t : ℝ) (h1 : x = t^2) (h2 : y = 2 * t) : (1, 0) = (1 : ℝ, 0 : ℝ) :=
by sorry

end focus_of_conic_section_l807_807954


namespace property_4_28_reproved_l807_807630

open_locale classical

noncomputable theory

def harmonic_division (P X D C : Point) : Prop :=
  -- Definition of harmonic division should be based on projective geometry.
  sorry

variables (A B C D P X : Point)
variables (polar_AB : Line)
variables (h1 : is_polar P polar_AB)
variables (h2 : X = intersection polar_AB (line C D))

theorem property_4_28_reproved :
  harmonic_division P X D C :=
sorry

end property_4_28_reproved_l807_807630


namespace child_age_five_l807_807964

theorem child_age_five (total_bill_discounted : ℝ) (adult_charge : ℝ) (child_charge_per_year : ℝ) (family_discount : ℝ) (number_of_adults : ℕ) :
  total_bill_discounted = 13.20 →
  adult_charge = 6.00 →
  child_charge_per_year = 0.60 →
  family_discount = 2.00 →
  number_of_adults = 2 →
  (∃ age_of_child : ℕ, (total_bill_discounted + family_discount = (number_of_adults * adult_charge) + (age_of_child * child_charge_per_year)) ∧ age_of_child = 5) :=
begin
  sorry
end

end child_age_five_l807_807964


namespace min_value_frac_sq_l807_807156

theorem min_value_frac_sq (x : ℝ) (h : x > 12) : (x^2 / (x - 12)) >= 48 :=
by
  sorry

end min_value_frac_sq_l807_807156


namespace probability_point_on_smaller_segment_l807_807612

theorem probability_point_on_smaller_segment
    (L : ℝ) (ℓ : ℝ)
    (hL : L = 40)
    (hℓ : ℓ = 15)
    (h_prob : ∀ x : ℝ, x ∈ ℓ → P(x ∈ ℓ) = ℓ / L) :
  P(x ∈ ℓ) = 3 / 8 :=
by {
  rw [hL, hℓ, h_prob],
  norm_num,
  sorry
}

end probability_point_on_smaller_segment_l807_807612


namespace abs_inequality_solution_l807_807014

theorem abs_inequality_solution :
  {x : ℝ | |2 * x + 1| > 3} = {x : ℝ | x < -2} ∪ {x : ℝ | x > 1} :=
by
  sorry

end abs_inequality_solution_l807_807014


namespace radio_advertiser_savings_l807_807697

def total_store_price : ℚ := 299.99
def ad_payment : ℚ := 55.98
def payments_count : ℚ := 5
def shipping_handling : ℚ := 12.99

def total_ad_price : ℚ := payments_count * ad_payment + shipping_handling

def savings_in_dollars : ℚ := total_store_price - total_ad_price
def savings_in_cents : ℚ := savings_in_dollars * 100

theorem radio_advertiser_savings :
  savings_in_cents = 710 := by
  sorry

end radio_advertiser_savings_l807_807697


namespace angle_measure_is_60_l807_807788

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l807_807788


namespace restore_grid_l807_807714

-- Definitions for each cell in the grid.
variable (A B C D : ℕ)

-- The given grid and conditions
noncomputable def grid : list (list ℕ) :=
  [[A, 1, 9],
   [3, 5, D],
   [B, C, 7]]

-- The condition that the sum of numbers in adjacent cells is less than 12.
def valid (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 3 < 12 ∧
  1 + 9 < 12 ∧ 5 + D < 12 ∧
  3 + 5 < 12 ∧  B + C < 12 ∧
  9 + D < 12 ∧ 7 + C < 12 ∧
  9 + 1 < 12 ∧ B + 7 < 12

-- Assertion to be proved.
theorem restore_grid (A B C D : ℕ) (valid : valid A B C D) :
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 := by
  sorry

end restore_grid_l807_807714


namespace circulation_r_value_circulation_p_value_circulation_q_value_curl_q_at_A_value_l807_807071

noncomputable def circulation_r (a : ℝ) : ℝ :=
  ∮ (x = a * cos t, y = a * sin t) (x * (differential y))

theorem circulation_r_value (a : ℝ) : circulation_r a = π * a^2 := 
  sorry

noncomputable def circulation_p : ℝ :=
  let A := (1, 0, 0)
  let B := (0, 1, 0)
  let C := (0, 0, 1)
  ∮ (~π ∙ perimeter ABC) ((x-2) * dx + (x+y) * dy - 2 * dz)

theorem circulation_p_value : circulation_p = 5 / 2 := 
  sorry

noncomputable def circulation_q (a : ℝ) : ℝ :=
  ∮ (z = x^2 - y^2 + 2 * a^2, x^2 + y^2 = a^2) (xz * dx + (-yz^2) * dy + xy * dz)

theorem circulation_q_value (a : ℝ) : circulation_q a = -π * a^4 := 
  sorry

noncomputable def curl_q_at_A (a : ℝ) : Vector3 :=
  let A := (0, -a, a^2)
  ∇ × (x*z, -y*z^2, x*y) at A

theorem curl_q_at_A_value (a : ℝ) : curl_q_at_A a = -- define the expected curl value here
  sorry

end circulation_r_value_circulation_p_value_circulation_q_value_curl_q_at_A_value_l807_807071


namespace count_valid_integers_l807_807505

def four_digit_integer (n : ℕ) : bool := n >= 1000 ∧ n < 10000

def non_zero_and_non_seven_digits (n : ℕ) : bool :=
  ∀ d ∈ (n.digits 10), d ≠ 0 ∧ d ≠ 7

def product_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).product

def valid_integer (n : ℕ) : bool :=
  four_digit_integer n ∧ non_zero_and_non_seven_digits n ∧ product_of_digits n = 18

theorem count_valid_integers : 
  {n : ℕ // valid_integer n} . length = 24 :=
sorry

end count_valid_integers_l807_807505


namespace solutions_count_l807_807530

noncomputable def number_of_solutions (x y z : ℚ) : ℕ :=
if (x^2 - y * z = 1) ∧ (y^2 - x * z = 1) ∧ (z^2 - x * y = 1)
then 6
else 0

theorem solutions_count : number_of_solutions x y z = 6 :=
sorry

end solutions_count_l807_807530


namespace correctness_check_l807_807054

noncomputable def questionD (x y : ℝ) : Prop := 
  3 * x^2 * y - 2 * y * x^2 = x^2 * y

theorem correctness_check (x y : ℝ) : questionD x y :=
by 
  sorry

end correctness_check_l807_807054


namespace compare_values_l807_807178

noncomputable theory

-- Definitions of the given conditions
def a : ℝ := Real.exp (-3)
def b : ℝ := Real.log 1.02
def c : ℝ := Real.sin 0.04

-- The theorem to be proven
theorem compare_values : b < c ∧ c < a :=
  by
    sorry

end compare_values_l807_807178


namespace regular_octagon_interior_angle_l807_807231

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807231


namespace time_to_cross_bridge_l807_807979

-- Define the given conditions in Lean.
def length_of_train : ℝ := 140
def speed_of_train_kmph : ℝ := 45
def length_of_bridge : ℝ := 235

-- Define the conversion factor from km/hr to m/s.
def kmph_to_mps (speed: ℝ) : ℝ := speed * (1000 / 3600)

-- Define the speed in meters per second.
def speed_of_train_mps : ℝ := kmph_to_mps speed_of_train_kmph

-- Define the total distance the train needs to cover.
def total_distance : ℝ := length_of_train + length_of_bridge

-- The goal is to prove that the time to cross the bridge is 30 seconds.
theorem time_to_cross_bridge : total_distance / speed_of_train_mps = 30 := by
  sorry

end time_to_cross_bridge_l807_807979


namespace interior_angle_regular_octagon_l807_807492

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807492


namespace angle_complement_supplement_l807_807825

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l807_807825


namespace find_total_tests_l807_807621

noncomputable def total_tests (n : ℕ) (x : ℚ) : ℕ :=
if (90 * (n + 1) = nx + 97) ∧ (87 * (n + 1) = nx + 73) then n else 0

theorem find_total_tests (n : ℕ) (x : ℚ) :
  ((90 * (n + 1) = n * x + 97) ∧ (87 * (n + 1) = n * x + 73)) → n = 8 :=
by
  sorry

#eval find_total_tests

end find_total_tests_l807_807621


namespace x_minus_y_l807_807701

theorem x_minus_y :
  (∃ x y : ℝ, 3 = 0.20 * x ∧ 3 = 0.40 * y) →
  ∃ x y : ℝ, 3 = 0.20 * x ∧ 3 = 0.40 * y ∧ x - y = 7.5 :=
by
  intro h
  obtain ⟨x, y, hx, hy⟩ := h
  use [x, y]
  exact ⟨hx, hy, sorry⟩

end x_minus_y_l807_807701


namespace all_positive_rationals_in_X_l807_807022

def X : Set ℚ := {x | x ∈ Set.Icc 2021 2022 ∨ (∃ y z, y ∈ X ∧ z ∈ X ∧ x = y / z)}

theorem all_positive_rationals_in_X :
  ∀ (q : ℚ), 0 < q → q ∈ X :=
sorry

end all_positive_rationals_in_X_l807_807022


namespace angle_solution_l807_807883

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l807_807883


namespace probability_of_xyz_eq_72_l807_807915

open ProbabilityTheory Finset

def dice_values : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem probability_of_xyz_eq_72 :
  (∑ x in dice_values, ∑ y in dice_values, ∑ z in dice_values, 
   if x * y * z = 72 then 1 else 0) / (dice_values.card ^ 3) = 1 / 36 :=
by
  sorry -- Proof omitted

end probability_of_xyz_eq_72_l807_807915


namespace age_difference_l807_807092

-- Define the present age of the son as a constant
def S : ℕ := 22

-- Define the equation given by the problem
noncomputable def age_relation (M : ℕ) : Prop :=
  M + 2 = 2 * (S + 2)

-- The theorem to prove the man is 24 years older than his son
theorem age_difference (M : ℕ) (h_rel : age_relation M) : M - S = 24 :=
by {
  sorry
}

end age_difference_l807_807092


namespace regular_octagon_angle_l807_807319

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807319


namespace water_bill_payment_ratio_l807_807648

variables (electricity_bill gas_bill water_bill internet_bill amount_remaining : ℤ)
variables (paid_gas_bill_payments paid_internet_bill_payments additional_gas_payment : ℤ)

-- Define the given conditions
def stephanie_budget := 
  electricity_bill = 60 ∧
  gas_bill = 40 ∧
  water_bill = 40 ∧
  internet_bill = 25 ∧
  amount_remaining = 30 ∧
  paid_gas_bill_payments = 3 ∧ -- three-quarters
  paid_internet_bill_payments = 4 ∧ -- four payments of $5
  additional_gas_payment = 5

-- Define the given problem as a theorem
theorem water_bill_payment_ratio 
  (h : stephanie_budget electricity_bill gas_bill water_bill internet_bill amount_remaining paid_gas_bill_payments paid_internet_bill_payments additional_gas_payment) :
  ∃ (paid_water_bill : ℤ), paid_water_bill / water_bill = 1 / 2 :=
sorry

end water_bill_payment_ratio_l807_807648


namespace interior_angle_regular_octagon_l807_807362

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807362


namespace contradiction_problem_l807_807033

theorem contradiction_problem (a b c d : ℝ) (h1 : a + b = 1) (h2 : c + d = 1) (h3 : ac + bd > 1) : 
  (a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0) → False := 
by
  sorry

end contradiction_problem_l807_807033


namespace angle_solution_l807_807878

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l807_807878


namespace angle_measure_l807_807901

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807901


namespace regular_octagon_interior_angle_l807_807482

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807482


namespace interior_angle_regular_octagon_l807_807457

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807457


namespace odd_number_of_circles_l807_807611

-- Define the problem conditions
variable (n : ℕ)
variable (circles : Fin n → Set (Set (ℝ × ℝ)))

-- Conditions of the problem
-- 1. There are \( n \) circles.
-- 2. Any two circles intersect at exactly two points.
def two_point_intersection : Prop :=
  ∀ (i j : Fin n), i ≠ j →
  ∃! (p q : ℝ × ℝ), p ≠ q ∧ 
  (∃ a b, a ∈ circles i ∧ a ∈ circles j ∧ 
          b ∈ circles i ∧ b ∈ circles j)

-- 3. No three circles have a common point.
def no_three_circles_common : Prop :=
  ∀ (i j k : Fin n), i ≠ j ∧ j ≠ k ∧ i ≠ k →
  ∀ (p : ℝ × ℝ), p ∈ circles i ∧ p ∈ circles j → p ∉ circles k

-- The theorem to prove
theorem odd_number_of_circles (h1 : two_point_intersection n circles) (h2 : no_three_circles_common n circles) : 
  Odd n :=
sorry

end odd_number_of_circles_l807_807611


namespace count_integers_satisfying_inequality_l807_807134

theorem count_integers_satisfying_inequality :
  ∃ n : ℕ, n = (finset.filter (λ m : ℤ, (m + 1) * (m - 5) < 0) (finset.Icc (-1) 4)).card ∧ n = 5 :=
by
  sorry

end count_integers_satisfying_inequality_l807_807134


namespace clock_angle_at_7_15_l807_807501

theorem clock_angle_at_7_15 :
  let hour_hand := 210 + 30 / 4,
      minute_hand := 90 in
  |hour_hand - minute_hand| = 127.5 := 
by
  let hour_hand := 210 + 30 / 4
  let minute_hand := 90
  have : hour_hand = 217.5
  have : minute_hand = 90
  have : |hour_hand - minute_hand| = |217.5 - 90|
  have : |127.5| = 127.5
  sorry   -- proof omitted

end clock_angle_at_7_15_l807_807501


namespace andy_correct_answer_l807_807564

-- Let y be the number Andy is using
def y : ℕ := 13  -- Derived from the conditions

-- Given condition based on Andy's incorrect operation
def condition : Prop := 4 * y + 5 = 57

-- Statement of the proof problem
theorem andy_correct_answer : condition → ((y + 5) * 4 = 72) := by
  intros h
  sorry

end andy_correct_answer_l807_807564


namespace interior_angle_regular_octagon_l807_807434

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807434


namespace mn_bisects_ef_l807_807556

open Real

-- Definitions and assumptions based on problem conditions
variables {A B C D M N E F : Point}
variable [cyclic_quad: CyclicQuadrilateral A B C D]
variable [mid_M_AD: Midpoint M A D]
variable [perp_MN_BC: Perpendicular M N BC]
variable [perp_ME_AB: Perpendicular M E AB]
variable [perp_MF_CD: Perpendicular M F CD]

-- Lean theorem statement to prove the question
theorem mn_bisects_ef : midpoint (M N) (E F) :=
by
  -- Here would be your proof steps, which are omitted
  sorry

end mn_bisects_ef_l807_807556


namespace arithmetic_geometric_sequences_l807_807004

theorem arithmetic_geometric_sequences :
  ∃ (A B C D : ℤ), A < B ∧ B > 0 ∧ C > 0 ∧ -- Ensure A, B, C are positive
  (B - A) = (C - B) ∧  -- Arithmetic sequence condition
  B * (49 : ℚ) = C * (49 / 9 : ℚ) ∧ -- Geometric sequence condition written using fractional equality
  A + B + C + D = 76 := 
by {
  sorry -- Placeholder for actual proof
}

end arithmetic_geometric_sequences_l807_807004


namespace column_heights_achievable_l807_807139

open Int

noncomputable def number_of_column_heights (n : ℕ) (h₁ h₂ h₃ : ℕ) : ℕ :=
  let min_height := n * h₁
  let max_height := n * h₃
  max_height - min_height + 1

theorem column_heights_achievable :
  number_of_column_heights 80 3 8 15 = 961 := by
  -- Proof goes here.
  sorry

end column_heights_achievable_l807_807139


namespace angle_solution_l807_807885

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l807_807885


namespace regular_octagon_interior_angle_l807_807241

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807241


namespace original_number_multiple_of_8_l807_807965

theorem original_number_multiple_of_8 (x y : ℤ) (h : 14 * x = 112 * y) : ∃ k : ℤ, x = 8 * k :=
by
  use y
  have h' : 112 = 14 * 8 := rfl
  rw [h', mul_assoc, mul_comm 8 y, ← mul_assoc, mul_comm 14 x] at h
  rw [← mul_assoc, mul_eq_mul_left_iff] at h
  cases h; { use h; rfl }
  sorry

end original_number_multiple_of_8_l807_807965


namespace regular_octagon_interior_angle_l807_807479

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807479


namespace distinct_license_plates_l807_807087

theorem distinct_license_plates : 
  let choices_digits := 10^5 in
  let choices_letters := 26^3 in
  let positions := 6 in
  positions * choices_digits * choices_letters = 105456000 :=
by 
  let choices_digits := 100000
  let choices_letters := 17576
  let positions := 6
  sorry

end distinct_license_plates_l807_807087


namespace angle_measure_l807_807909

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807909


namespace regular_octagon_interior_angle_eq_135_l807_807403

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807403


namespace even_poly_iff_exists_Q_l807_807625

-- Define necessary terms and variables
variable {R : Type*}
variables [CommRing R] [IsDomain R]

-- Define what it means for a polynomial to be even
def is_even_poly (P : R[X]) : Prop :=
  ∀ (z : R), P.eval z = P.eval (-z)

-- State the theorem
theorem even_poly_iff_exists_Q (P : R[X]) :
  is_even_poly P ↔ ∃ Q : R[X], P = Q * (Q * -1) :=
sorry

end even_poly_iff_exists_Q_l807_807625


namespace problem_solution_l807_807986

theorem problem_solution : 
  (∀ (L : Line) (P : Point), ¬∃! L1 : Line, perpendicular L L1 ∧ through_point L1 P) ∧
  (∀ (P : Plane) (Q : Point), ∃! L1 : Line, perpendicular_plane L1 P ∧ through_point L1 Q) ∧
  (∀ (L : Line) (Q : Point), ∃! P1 : Plane, perpendicular L P1 ∧ through_point P1 Q) ∧
  (∀ (P1 P2 : Plane) (Q : Point), ¬∃! P3 : Plane, perpendicular_planes P1 P3 ∧ through_point P3 Q) := sorry

end problem_solution_l807_807986


namespace regular_octagon_interior_angle_l807_807263

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807263


namespace each_interior_angle_of_regular_octagon_l807_807340

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807340


namespace interior_angle_regular_octagon_l807_807453

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807453


namespace regular_octagon_interior_angle_l807_807287

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807287


namespace num_four_digit_integers_divisible_by_7_l807_807508

theorem num_four_digit_integers_divisible_by_7 :
  ∃ n : ℕ, n = 1286 ∧ ∀ k : ℕ, (1000 ≤ k ∧ k ≤ 9999) → (k % 7 = 0 ↔ ∃ m : ℕ, k = m * 7) :=
by {
  sorry
}

end num_four_digit_integers_divisible_by_7_l807_807508


namespace regular_octagon_interior_angle_l807_807268

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807268


namespace no_a_with_82_points_l807_807950

-- Define S(a) as the sum of the floor of a divided by each n from 1 to floor of square root of a
def S (a : ℕ) : ℕ := ∑ n in Finset.range (Int.toNat (Real.sqrt a).floor) + 1, (a / n)

-- The Lean 4 statement
theorem no_a_with_82_points : ¬ ∃ a : ℕ, S a = 82 :=
by
  sorry

end no_a_with_82_points_l807_807950


namespace tyler_can_buy_15_pears_l807_807523

theorem tyler_can_buy_15_pears (cost_10_apples_5_oranges cost_4_oranges_6_pears : ℝ) : 
    let cost_5_oranges := cost_10_apples_5_oranges / 2
    let cost_1_orange := cost_5_oranges / 5
    let cost_20_apples := 2 * cost_10_apples_5_oranges
    let cost_10_oranges := 2 * cost_5_oranges
    let cost_6_pears := cost_4_oranges_6_pears
    let cost_1_pear := cost_6_pears / 6
in cost_20_apples = cost_10_oranges ∧
   cost_10_oranges = 2.5 * cost_6_pears →
   20 * cost_10_apples_5_oranges / (15 * cost_1_pear) = 15 :=
by
  intros h h_eq
  sorry

end tyler_can_buy_15_pears_l807_807523


namespace bounded_sequence_l807_807687

def sequence_a (a : ℕ → ℚ) (h : ∀ n, ∃ p q : ℕ, p.coprime q ∧ a n = p / q ∧ 0 < q) : Prop :=
∀ n, ∃ p q : ℕ, p.coprime q ∧ a (n + 1) = (p^2 + 2015) / (p * q)

theorem bounded_sequence {a : ℕ → ℚ} (h : sequence_a a (λ _, ⟨1, 1, nat.coprime_one_right 1, by norm_num, by norm_num⟩)) :
  ∃ a1 > 2015, ∃ M : ℚ, ∀ n, a n < M :=
sorry

end bounded_sequence_l807_807687


namespace investment_duration_l807_807972

noncomputable def investment_period (P A r n : ℝ) : ℝ :=
  (Real.log (A / P)) / (n * Real.log (1 + r / n))

theorem investment_duration :
  let P := 6000
  let A := 6945.75
  let r := 0.10
  let n := 2
  investment_period P A r n ≈ 1.52 := 
sorry

end investment_duration_l807_807972


namespace card_probability_l807_807023

/-- 
The probability that the first card is a heart, the second card is a King, and the third card is a spade
when dealt successively without replacement from a standard deck of 52 cards is 13/2550.
-/
theorem card_probability : 
  let P := (1 / 4) * (4 / 51) * (13 / 50) in
  (P = 13 / 2550) :=
by
  sorry

end card_probability_l807_807023


namespace angle_measure_l807_807849

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807849


namespace reporters_not_involved_in_politics_or_subfields_l807_807538

theorem reporters_not_involved_in_politics_or_subfields :
  let P_X := 0.30
  let P_Y := 0.20
  let P_Z := 0.15
  let P_XY := 0.05
  let P_YZ := 0.03
  let P_XZ := 0.02
  let P_XYZ := 0.01
  let S_finance := 0.10
  let S_environment := 0.07
  let S_social := 0.05
  let total_political_coverage := P_X + P_Y + P_Z - P_XY - P_YZ - P_XZ + P_XY + P_YZ + P_XZ - P_XYZ
  let total_subfields_coverage := S_finance + S_environment + S_social
  let involved := total_political_coverage + total_subfields_coverage
  100% - involved = 0.27 :=
begin
  sorry
end

end reporters_not_involved_in_politics_or_subfields_l807_807538


namespace angle_measure_l807_807838

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l807_807838


namespace regular_octagon_interior_angle_deg_l807_807387

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807387


namespace find_phi_l807_807678

theorem find_phi (φ : ℝ) (h1 : abs φ < π) :
  ∃ k : ℤ, (φ = -π / 3 + k * π) ∧ (abs (-π / 3 + k * π) < π) :=
begin
  sorry
end

end find_phi_l807_807678


namespace each_interior_angle_of_regular_octagon_l807_807341

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807341


namespace problem1_problem2_problem3_l807_807953

-- Problem 1 Statement
theorem problem1 : (π - 3.14)^0 + (1 / 2)^(-1) + (-1)^(2023) = 2 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

-- Problem 2 Statement
theorem problem2 (b : ℝ) : (-b)^2 * b + 6 * b^4 / (2 * b) + (-2 * b)^3 = -4 * b^3 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

-- Problem 3 Statement
theorem problem3 (x : ℝ) : (x - 1)^2 - x * (x + 2) = -4 * x + 1 :=
by {
  -- use tactic mode to assist the proof
  sorry
}

end problem1_problem2_problem3_l807_807953


namespace probability_team_A_wins_best_of_three_l807_807658

noncomputable def prob_of_winning_single_game : ℚ := 2 / 3

theorem probability_team_A_wins_best_of_three :
  let P := prob_of_winning_single_game in
  P * P + (1 - P) * P * P = 20 / 27 := by
  sorry

end probability_team_A_wins_best_of_three_l807_807658


namespace interior_angle_regular_octagon_l807_807499

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807499


namespace decimal_multiplication_l807_807199

theorem decimal_multiplication (h : 268 * 74 = 19832) : 2.68 * 0.74 = 1.9832 :=
by sorry

end decimal_multiplication_l807_807199


namespace regular_octagon_interior_angle_eq_135_l807_807400

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807400


namespace regular_octagon_interior_angle_deg_l807_807390

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807390


namespace regular_octagon_interior_angle_l807_807421

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807421


namespace regular_octagon_interior_angle_l807_807420

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807420


namespace volume_of_rotated_solid_l807_807010

theorem volume_of_rotated_solid
  (d1 d2 : ℕ)
  (r1 r2 h1 h2 : ℕ)
  (h_d1 : d1 = 10)
  (h_d2 : d2 = 8)
  (h_r1 : r1 = 10)
  (h_r2 : r2 = 2)
  (h_h1 : h1 = 1)
  (h_h2 : h2 = 8)
  : real.pi * (r1 ^ 2) * h1 + real.pi * (r2 ^ 2) * h2 = 132 * real.pi :=
by
  sorry

end volume_of_rotated_solid_l807_807010


namespace angle_measure_l807_807860

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l807_807860


namespace restore_grid_values_l807_807772

def gridCondition (A B C D : Nat) : Prop :=
  A < 9 ∧ 1 < 9 ∧ 9 < 9 ∧
  3 < 9 ∧ 5 < 9 ∧ D < 9 ∧
  B < 9 ∧ C < 9 ∧ 7 < 9 ∧
  -- Sum of any two numbers in adjacent cells is less than 12
  A + 1 < 12 ∧ A + 3 < 12 ∧
  B + C < 12 ∧ B + 3 < 12 ∧
  D + 2 < 12 ∧ C + 4 < 12 ∧
  -- The grid values are from 1 to 9
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  C ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
  -- Even numbers erased are 2, 4, 6, and 8

theorem restore_grid_values :
  ∃ (A B C D : Nat), gridCondition A B C D ∧ A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  sorry

end restore_grid_values_l807_807772


namespace min_balls_to_ensure_three_colors_l807_807958

section BallDraw

variable (num_colors : ℕ) (balls_per_color : ℕ) (total_balls : ℕ)

def at_least_three_colors (drawn : ℕ) : Prop :=
  drawn >= balls_per_color * (num_colors - 1) + 1

theorem min_balls_to_ensure_three_colors :
  at_least_three_colors 27 :=
by
  let num_colors := 4
  let balls_per_color := 13
  let total_balls := num_colors * balls_per_color
  have h1 : total_balls = 52 := by
    calc
      num_colors * balls_per_color = 4 * 13 := rfl
      ... = 52 := by norm_num
  sorry

end BallDraw

end min_balls_to_ensure_three_colors_l807_807958


namespace interior_angle_regular_octagon_l807_807496

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807496


namespace regular_octagon_interior_angle_eq_135_l807_807409

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807409


namespace angle_supplement_complement_l807_807895

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l807_807895


namespace fourth_row_sudoku_correct_l807_807145

noncomputable def sudoku_sequence : List ℕ := [2, 1, 4, 3, 6]

theorem fourth_row_sudoku_correct :
  ∃ r : List ℕ, 
  r.length = 5 ∧ 
  (∀ i, r.nth i ∈ some (List.range (6 + 1))) ∧
  (∀ i j, i ≠ j → r.nth i ≠ r.nth j) ∧
  (∀ n, n ∈ r → ∃ m, m ∈ r ∧ (m = 2 * n ∨ n = 2 * m ∨ m = n + 1 ∨ n = m + 1)) ∧
  r = sudoku_sequence := 
by 
  existsi sudoku_sequence
  simp [sudoku_sequence]
  unfold List.range -- automate the basic set properties involving 1 to 6
  sorry

end fourth_row_sudoku_correct_l807_807145


namespace conjugate_of_z_l807_807524

-- Definitions for the conditions
def complex_add (a b : ℂ) : ℂ := a + b
def complex_neg (a : ℂ) : ℂ := -a
def complex_mul (a b : ℂ) : ℂ := a * b
def complex_div (a b : ℂ) : ℂ := a / b

-- Definitions of imaginary unit and complex conjugate
def complex_i : ℂ := complex.I
def complex_conj (a : ℂ) : ℂ := conj a

-- Statement of the problem
theorem conjugate_of_z 
  (z : ℂ) 
  (h : complex_mul (complex_add 2 (complex_neg complex_i)) z = complex_mul complex_i complex_i ^ 2023) :
  complex_conj z = complex_div (complex_add 1 (complex_mul 2 complex_i)) 5 :=
sorry

end conjugate_of_z_l807_807524


namespace interior_angle_regular_octagon_l807_807437

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807437


namespace each_interior_angle_of_regular_octagon_l807_807349

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807349


namespace third_competitor_eats_300_hotdogs_in_5_minutes_l807_807543

def first_competitor_rate : ℕ := 10
def second_competitor_rate : ℕ := 3 * first_competitor_rate
def third_competitor_rate : ℕ := 2 * second_competitor_rate

theorem third_competitor_eats_300_hotdogs_in_5_minutes : third_competitor_rate * 5 = 300 :=
by
  unfold first_competitor_rate second_competitor_rate third_competitor_rate
  sorry

end third_competitor_eats_300_hotdogs_in_5_minutes_l807_807543


namespace avg_class_weight_is_46_67_l807_807070

-- Define the total number of students in section A
def num_students_a : ℕ := 40

-- Define the average weight of students in section A
def avg_weight_a : ℚ := 50

-- Define the total number of students in section B
def num_students_b : ℕ := 20

-- Define the average weight of students in section B
def avg_weight_b : ℚ := 40

-- Calculate the total weight of section A
def total_weight_a : ℚ := num_students_a * avg_weight_a

-- Calculate the total weight of section B
def total_weight_b : ℚ := num_students_b * avg_weight_b

-- Calculate the total weight of the entire class
def total_weight_class : ℚ := total_weight_a + total_weight_b

-- Calculate the total number of students in the entire class
def total_students_class : ℕ := num_students_a + num_students_b

-- Calculate the average weight of the entire class
def avg_weight_class : ℚ := total_weight_class / total_students_class

-- Theorem to prove
theorem avg_class_weight_is_46_67 :
  avg_weight_class = 46.67 := sorry

end avg_class_weight_is_46_67_l807_807070


namespace regular_octagon_interior_angle_l807_807246

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807246


namespace average_number_of_glasses_per_box_l807_807990

-- Definitions and conditions
variables (S L : ℕ) -- S is the number of smaller boxes, L is the number of larger boxes

-- Condition 1: One box contains 12 glasses, and the other contains 16 glasses.
-- (This is implicitly understood in the equation for total glasses)

-- Condition 3: There are 16 more larger boxes than smaller smaller boxes
def condition_3 := L = S + 16

-- Condition 4: The total number of glasses is 480.
def condition_4 := 12 * S + 16 * L = 480

-- Proving the average number of glasses per box is 15
theorem average_number_of_glasses_per_box (h1 : condition_3 S L) (h2 : condition_4 S L) :
  (480 : ℝ) / (S + L) = 15 :=
by 
  -- Assuming S and L are natural numbers 
  sorry

end average_number_of_glasses_per_box_l807_807990


namespace angle_measure_l807_807857

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l807_807857


namespace square_area_l807_807105

theorem square_area (s : ℝ) (h : s = 13) : s * s = 169 :=
by 
  rw h
  sorry

end square_area_l807_807105


namespace polygon_intersection_l807_807580

-- Define the conditions in Lean
variables (P : SimplePolygon) (C : Circle) 
-- The radius of the circle C is 1
variable (radius_C : RealUnitCircle) -- RealUnitCircle denotes circles with radius 1
-- P is completely inside C and does not pass through the center of C
variable (inside_C : CompletelyInside P C) (exclude_center : ExcludesCenter P C) 
-- The perimeter of P is 36
variable (perimeter_P : Perimeter P = 36)

-- Prove the statement
theorem polygon_intersection (P : SimplePolygon) (C : Circle) 
  (inside_C : CompletelyInside P C) (exclude_center : ExcludesCenter P C) 
  (perimeter_P : Perimeter P = 36) :
  (∃ r : RealUnitCircle, ∃ t : ℝ, radialIntersections P C r t ≥ 6) ∨
  (∃ concentricC : Circle, ∃ p : Point, (p ∈ P) ∧ (p ∈ concentric C) ∧ (countCommonPoints P concentric C ≥ 6)) :=
  sorry

end polygon_intersection_l807_807580


namespace grid_solution_l807_807737

-- Define the grid and conditions
variables (A B C D : ℕ)

-- Grid values with the constraints
def grid_valid (A B C D : ℕ) :=
  A ≠ 1 ∧ A ≠ 9 ∧ A ≠ 3 ∧ A ≠ 5 ∧ A ≠ 7 ∧
  B ≠ 1 ∧ B ≠ 9 ∧ B ≠ 3 ∧ B ≠ 5 ∧ B ≠ 7 ∧
  C ≠ 1 ∧ C ≠ 9 ∧ C ≠ 3 ∧ C ≠ 5 ∧ C ≠ 7 ∧
  D ≠ 1 ∧ D ≠ 9 ∧ D ≠ 3 ∧ D ≠ 5 ∧ D ≠ 7 ∧
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ (A + 3 < 12) ∧
  (D + 9 < 12) ∧ (D + 5 < 12) ∧ 
  (B + 3 < 12) ∧ (B + C < 12) ∧
  (C + 7 < 12)

-- Given the conditions, we need to prove the values of A, B, C, and D
theorem grid_solution : 
  grid_valid 8 6 4 2 ∧ 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  split
  · -- to show grid_valid 8 6 4 2
    sorry
    
  · -- to show the values of A, B, C, and D
    repeat {split, assumption}

end grid_solution_l807_807737


namespace dot_product_range_l807_807554

-- Given conditions of the ellipse and points
def ellipse (x y : ℝ) : Prop := (x^2 / 4) + (y^2 / 2) = 1
def inside_ellipse (P : ℝ × ℝ) : Prop :=
  let (m, n) := P in (m^2 / 4) + (n^2 / 2) < 1
def form_geometric (P : ℝ × ℝ) : Prop :=
  let (m, n) := P in
    (m^2 + n^2)^2 = Real.sqrt ((m - 2)^2 + n^2) * Real.sqrt ((m + 2)^2 + n^2)

-- Dot product calculation
def dot_product (P : ℝ × ℝ) : ℝ :=
  let (m, n) := P in 2*n^2 - 2

theorem dot_product_range (P : ℝ × ℝ) (inside : inside_ellipse P) (geo : form_geometric P) :
  -2 ≤ dot_product P ∧ dot_product P < 0 :=
sorry

end dot_product_range_l807_807554


namespace tetrahedron_contains_center_l807_807039

noncomputable theory

-- Define the probability space and random selection of points on the sphere
def sphere : Type := {p : ℝ^3 // ∥p∥ = 1}

-- The probability that four randomly selected points on the sphere form a tetrahedron containing the center
def tetrahedron_contains_center_probability : ℝ := 1 / 8

-- Statement of the problem
theorem tetrahedron_contains_center : 
  ∀ (A X Y Z : sphere), 
  tetrahedron_contains_center_probability = 1 / 8 :=
sorry

end tetrahedron_contains_center_l807_807039


namespace angle_measure_l807_807799

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l807_807799


namespace angle_measure_l807_807830

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l807_807830


namespace interior_angle_regular_octagon_l807_807372

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807372


namespace find_s_l807_807214

-- Given conditions as definitions
def t := 3.75
def equation (s : ℝ) : ℝ := 15 * s^2

-- Statement to be proved
theorem find_s : (equation 0.5 = t) := by
  sorry

end find_s_l807_807214


namespace angle_supplement_complement_l807_807812

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l807_807812


namespace sum_of_integers_lcm_72_l807_807045

theorem sum_of_integers_lcm_72 :
  let S := {ν : ℕ | Nat.lcm ν 24 = 72} in
  ∑ ν in S, ν = 180 :=
by
  sorry

end sum_of_integers_lcm_72_l807_807045


namespace regular_octagon_interior_angle_l807_807290

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807290


namespace sum_x_coords_of_intersections_l807_807912

-- Definitions for the circles
def C₁ (x y : ℝ) : Prop := x^2 - 6 * x + y^2 - 8 * y + 24 = 0
def C₂ (x y : ℝ) : Prop := x^2 - 4 * x + y^2 - 8 * y + 16 = 0

-- Statement that the sum of the x-coordinates of all intersections of C₁ and C₂ is 4
theorem sum_x_coords_of_intersections : 
  ∃ (x y : ℝ), C₁ x y ∧ C₂ x y ∧ ((∑ (p : ℝ × ℝ) in {p | C₁ p.1 p.2 ∧ C₂ p.1 p.2}.toFinset, p.1) = 4) :=
by
  sorry

end sum_x_coords_of_intersections_l807_807912


namespace value_of_first_part_l807_807520

theorem value_of_first_part (total_amount : ℝ) (ratio1 ratio2 ratio3 : ℝ) 
  (h_total : total_amount = 782) 
  (h_ratio1 : ratio1 = 1/2) 
  (h_ratio2 : ratio2 = 1/3) 
  (h_ratio3 : ratio3 = 3/4) : 
  let common_denominator := 12 in
  let part1 := ratio1 * common_denominator / (ratio1 * common_denominator + ratio2 * common_denominator + ratio3 * common_denominator) * total_amount in
  part1 = 247 :=
by
  sorry

end value_of_first_part_l807_807520


namespace triangle_bge_is_right_l807_807550

-- Definitions of the conditions
variables {A B C D E F G H : Type}
variable [incidence_geometry A B C D E F G H]
variables (a b c : Triangle A B C) 
variables (ad : AngleBisector A D B C) 
variables (de : Perpendicular D E A C) 
variables (df : Perpendicular D F A B)
variables (be : Segment B E) 
variables (cf : Segment C F)
variables (h : Orthocenter H A B C)
variables (circ_afh : Circumcircle AFH) 
variable (g : Intersection G BE circ_afh)

-- The proof to show triangle BGE is a right triangle
theorem triangle_bge_is_right :
  right_triangle (Triangle B G E) :=
begin
  sorry,
end

end triangle_bge_is_right_l807_807550


namespace angle_measure_l807_807855

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l807_807855


namespace interior_angle_regular_octagon_l807_807493

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807493


namespace regular_octagon_interior_angle_deg_l807_807392

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807392


namespace grid_solution_l807_807740

-- Define the grid and conditions
variables (A B C D : ℕ)

-- Grid values with the constraints
def grid_valid (A B C D : ℕ) :=
  A ≠ 1 ∧ A ≠ 9 ∧ A ≠ 3 ∧ A ≠ 5 ∧ A ≠ 7 ∧
  B ≠ 1 ∧ B ≠ 9 ∧ B ≠ 3 ∧ B ≠ 5 ∧ B ≠ 7 ∧
  C ≠ 1 ∧ C ≠ 9 ∧ C ≠ 3 ∧ C ≠ 5 ∧ C ≠ 7 ∧
  D ≠ 1 ∧ D ≠ 9 ∧ D ≠ 3 ∧ D ≠ 5 ∧ D ≠ 7 ∧
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ (A + 3 < 12) ∧
  (D + 9 < 12) ∧ (D + 5 < 12) ∧ 
  (B + 3 < 12) ∧ (B + C < 12) ∧
  (C + 7 < 12)

-- Given the conditions, we need to prove the values of A, B, C, and D
theorem grid_solution : 
  grid_valid 8 6 4 2 ∧ 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  split
  · -- to show grid_valid 8 6 4 2
    sorry
    
  · -- to show the values of A, B, C, and D
    repeat {split, assumption}

end grid_solution_l807_807740


namespace trajectory_of_P_l807_807193

noncomputable def point := (ℝ × ℝ)

def distance (p1 p2 : point) : ℝ :=
  real.sqrt ((p1.1 - p2.1) ^ 2 + (p1.2 - p2.2) ^ 2)

def F1 : point := (3, 0)
def F2 : point := (-3, 0)
def P (x y : ℝ) : point := (x, y)

theorem trajectory_of_P (x y : ℝ) (h : distance (P x y) F1 + distance (P x y) F2 = 10) : 
  (x^2 / 25) + (y^2 / 16) = 1 :=
sorry

end trajectory_of_P_l807_807193


namespace regular_octagon_interior_angle_l807_807416

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807416


namespace correct_grid_l807_807730

def A := 8
def B := 6
def C := 4
def D := 2

def grid := [[A, 1, 9],
             [3, 5, D],
             [B, C, 7]]

theorem correct_grid :
  (A + 1 < 12) ∧ (A + 3 < 12) ∧ (1 + 9 < 12) ∧
  (1 + 5 < 12) ∧ (3 + 5 < 12) ∧ (3 + B < 12) ∧
  (5 + D < 12) ∧ (5 + C < 12) ∧ (9 + D < 12) ∧
  (B + C < 12) ∧ (C + 7 < 12) :=
by
  -- This is to provide a sketch dummy theorem, we'd prove each step here  
  sorry

end correct_grid_l807_807730


namespace lateral_surface_area_of_pyramid_l807_807689

theorem lateral_surface_area_of_pyramid
  (sin_alpha : ℝ)
  (A_section : ℝ)
  (h1 : sin_alpha = 15 / 17)
  (h2 : A_section = 3 * Real.sqrt 34) :
  ∃ A_lateral : ℝ, A_lateral = 68 :=
sorry

end lateral_surface_area_of_pyramid_l807_807689


namespace angle_measure_l807_807847

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807847


namespace angle_complement_supplement_l807_807818

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l807_807818


namespace odd_sum_numbers_l807_807111

theorem odd_sum_numbers : 
  let odd_digits := {1, 3, 5, 7, 9}
  let even_digits := {2, 4, 6, 8}
  let valid_numbers := { n | 10 * n.1 + n.2 | n.1 ∈ odd_digits ∧ n.2 ∈ even_digits }
  in valid_numbers.card = 20 :=
by
  sorry

end odd_sum_numbers_l807_807111


namespace polynomial_evaluation_l807_807657

-- Define the polynomial p(x) and the condition p(x) - p'(x) = x^2 + 2x + 1
variable (p : ℝ → ℝ)
variable (hp : ∀ x, p x - (deriv p x) = x^2 + 2 * x + 1)

-- Statement to prove p(5) = 50 given the conditions
theorem polynomial_evaluation : p 5 = 50 := 
sorry

end polynomial_evaluation_l807_807657


namespace angle_measure_l807_807842

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807842


namespace angle_measure_l807_807840

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807840


namespace candidate_percentage_marks_l807_807079

theorem candidate_percentage_marks (T P : ℝ) (h1 : 0.30 * T = P - 60) (h2 : P = 240) :
  let second_candidate_marks := P + 30 in
  (second_candidate_marks / T) * 100 = 45 := 
by
  let second_candidate_marks := P + 30
  have hT : T = 600 := by
    calc
      T = (240 - 60) / 0.30 := by field_simp [h2, h1]
      ... = 600          := by norm_num
  have h_percentage : (second_candidate_marks / T) * 100 = 45 := by
    calc
      (second_candidate_marks / T) * 100 = ((240 + 30) / 600) * 100 := by rw [second_candidate_marks, hT]
      ... = (270 / 600) * 100             := by norm_num
      ... = 45                            := by norm_num
  rw h_percentage
  sorry

end candidate_percentage_marks_l807_807079


namespace regular_octagon_interior_angle_l807_807236

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807236


namespace grid_solution_l807_807741

-- Define the grid and conditions
variables (A B C D : ℕ)

-- Grid values with the constraints
def grid_valid (A B C D : ℕ) :=
  A ≠ 1 ∧ A ≠ 9 ∧ A ≠ 3 ∧ A ≠ 5 ∧ A ≠ 7 ∧
  B ≠ 1 ∧ B ≠ 9 ∧ B ≠ 3 ∧ B ≠ 5 ∧ B ≠ 7 ∧
  C ≠ 1 ∧ C ≠ 9 ∧ C ≠ 3 ∧ C ≠ 5 ∧ C ≠ 7 ∧
  D ≠ 1 ∧ D ≠ 9 ∧ D ≠ 3 ∧ D ≠ 5 ∧ D ≠ 7 ∧
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ (A + 3 < 12) ∧
  (D + 9 < 12) ∧ (D + 5 < 12) ∧ 
  (B + 3 < 12) ∧ (B + C < 12) ∧
  (C + 7 < 12)

-- Given the conditions, we need to prove the values of A, B, C, and D
theorem grid_solution : 
  grid_valid 8 6 4 2 ∧ 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  split
  · -- to show grid_valid 8 6 4 2
    sorry
    
  · -- to show the values of A, B, C, and D
    repeat {split, assumption}

end grid_solution_l807_807741


namespace interior_angle_regular_octagon_l807_807438

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807438


namespace first_number_in_proportion_is_60_l807_807522

theorem first_number_in_proportion_is_60 : 
  ∀ (x : ℝ), (x / 6 = 2 / 0.19999999999999998) → x = 60 :=
by
  intros x hx
  sorry

end first_number_in_proportion_is_60_l807_807522


namespace principal_amount_l807_807093

theorem principal_amount (P : ℕ) (R : ℕ) (T : ℕ) (SI : ℕ) 
  (h1 : R = 12)
  (h2 : T = 10)
  (h3 : SI = 1500) 
  (h4 : SI = (P * R * T) / 100) : P = 1250 :=
by sorry

end principal_amount_l807_807093


namespace luis_can_make_sum_multiple_of_4_l807_807602

noncomputable def sum_of_dice (dice: List ℕ) : ℕ :=
  dice.sum 

theorem luis_can_make_sum_multiple_of_4 (d1 d2 d3: ℕ) 
  (h1: 1 ≤ d1 ∧ d1 ≤ 6) 
  (h2: 1 ≤ d2 ∧ d2 ≤ 6) 
  (h3: 1 ≤ d3 ∧ d3 ≤ 6) : 
  ∃ (dice: List ℕ), dice.length = 3 ∧ 
  sum_of_dice dice % 4 = 0 := 
by
  sorry

end luis_can_make_sum_multiple_of_4_l807_807602


namespace total_pizzas_served_l807_807098

-- Define the conditions
def pizzas_lunch : Nat := 9
def pizzas_dinner : Nat := 6

-- Define the theorem to prove
theorem total_pizzas_served : pizzas_lunch + pizzas_dinner = 15 := by
  sorry

end total_pizzas_served_l807_807098


namespace solve_determinant_l807_807987

-- Definitions based on the conditions
def determinant (a b c d : ℤ) : ℤ := a * d - b * c

-- The problem translated to Lean 4:
theorem solve_determinant (x : ℤ) 
  (h : determinant (x + 1) x (2 * x - 6) (2 * (x - 1)) = 10) :
  x = 2 :=
sorry -- Proof is skipped

end solve_determinant_l807_807987


namespace correct_grid_l807_807731

def A := 8
def B := 6
def C := 4
def D := 2

def grid := [[A, 1, 9],
             [3, 5, D],
             [B, C, 7]]

theorem correct_grid :
  (A + 1 < 12) ∧ (A + 3 < 12) ∧ (1 + 9 < 12) ∧
  (1 + 5 < 12) ∧ (3 + 5 < 12) ∧ (3 + B < 12) ∧
  (5 + D < 12) ∧ (5 + C < 12) ∧ (9 + D < 12) ∧
  (B + C < 12) ∧ (C + 7 < 12) :=
by
  -- This is to provide a sketch dummy theorem, we'd prove each step here  
  sorry

end correct_grid_l807_807731


namespace candy_division_l807_807174

def pieces_per_bag (total_candies : ℕ) (bags : ℕ) : ℕ :=
total_candies / bags

theorem candy_division : pieces_per_bag 42 2 = 21 :=
by
  sorry

end candy_division_l807_807174


namespace total_games_across_leagues_l807_807700

-- Defining the conditions for the leagues
def leagueA_teams := 20
def leagueB_teams := 25
def leagueC_teams := 30

-- Function to calculate the number of games in a round-robin tournament
def number_of_games (n : ℕ) := n * (n - 1) / 2

-- Proposition to prove total games across all leagues
theorem total_games_across_leagues :
  number_of_games leagueA_teams + number_of_games leagueB_teams + number_of_games leagueC_teams = 925 := by
  sorry

end total_games_across_leagues_l807_807700


namespace angle_measure_l807_807862

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l807_807862


namespace regular_octagon_interior_angle_l807_807259

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807259


namespace sufficient_but_not_necessary_condition_l807_807191

def f (a x : ℝ) : ℝ := 1 - 2 * Math.sin (a * x + Real.pi / 4)^2

theorem sufficient_but_not_necessary_condition (a : ℝ) :
  (∀ x ∈ Ioo (Real.pi / 12) (Real.pi / 6), f 1 x < f 1 (x + ε) - ε) ∧
  ¬(∀ a > 0, a < 3 / 2 → (∀ x ∈ Ioo (Real.pi / 12) (Real.pi / 6), f a x < f a (x + ε) - ε)) :=
sorry

end sufficient_but_not_necessary_condition_l807_807191


namespace problem_statement_l807_807582

noncomputable def A := {x : ℝ | log 1.5 (x - 1) > 0}
noncomputable def B := {x : ℝ | 2^x < 4}

theorem problem_statement : (A ∩ B) = ∅ := by
  sorry

end problem_statement_l807_807582


namespace regular_octagon_interior_angle_deg_l807_807391

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807391


namespace inequality_am_gm_l807_807168

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
sorry

end inequality_am_gm_l807_807168


namespace regular_octagon_interior_angle_deg_l807_807385

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807385


namespace angle_supplement_complement_l807_807806

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l807_807806


namespace probability_of_xyz_72_l807_807925

noncomputable def probability_product_is_72 : ℚ :=
  let dice := {1, 2, 3, 4, 5, 6}
  let outcomes := {e : ℕ × ℕ × ℕ | e.1 ∈ dice ∧ e.2.1 ∈ dice ∧ e.2.2 ∈ dice}
  let favourable_outcomes := {e : ℕ × ℕ × ℕ | e ∈ outcomes ∧ e.1 * e.2.1 * e.2.2 = 72}
  (favourable_outcomes.to_finset.card : ℚ) / outcomes.to_finset.card

theorem probability_of_xyz_72 :
  probability_product_is_72 = 1 / 24 :=
sorry

end probability_of_xyz_72_l807_807925


namespace total_souvenirs_distributed_l807_807119

theorem total_souvenirs_distributed (x y : ℕ) (h1 : 0.20 * x + 0.25 * y = 220) (h2 : y = 400) :
  x + y = 1000 :=
by
  sorry

end total_souvenirs_distributed_l807_807119


namespace cylindrical_tubes_distance_l807_807708

theorem cylindrical_tubes_distance 
(r1 r2 : ℝ) 
(h1 : r1 = 72)
(h2 : r2 = 24) 
(x : ℝ) 
(a b c : ℤ) 
(hx : x = (a : ℝ) * Real.pi + (b : ℝ) * Real.sqrt (c : ℝ)) 
(h_non_divisible : ¬ ∃ p : ℕ, p.prime ∧ p^2 ∣ c) 
(hx_val : x = 96 * Real.pi + 96 * Real.sqrt 3) :
a + b + c = 195 := 
by 
  sorry

end cylindrical_tubes_distance_l807_807708


namespace equal_sums_of_distances_l807_807183

noncomputable def is_non_touching_line (quad : Quadrilateral) (l : Line) : Prop := 
  -- Define when a line does not touch sides of quadrilateral
  sorry 
  
noncomputable def dist_to_line (v : Point) (l : Line) : ℝ := 
  -- Define the distance from a vertex to a line
  sorry

theorem equal_sums_of_distances {quad : Quadrilateral}
  (h1 : ∀ (l₁ : Line), is_non_touching_line quad l₁ → 
    let d_A := dist_to_line quad.A l₁ in
    let d_B := dist_to_line quad.B l₁ in
    let d_C := dist_to_line quad.C l₁ in
    let d_D := dist_to_line quad.D l₁ in
    d_A + d_C = d_B + d_D)
  (h2 : ∀ (l₂ : Line), is_non_touching_line quad l₂ → 
    let d'_A := dist_to_line quad.A l₂ in
    let d'_B := dist_to_line quad.B l₂ in
    let d'_C := dist_to_line quad.C l₂ in
    let d'_D := dist_to_line quad.D l₂ in
    d'_A + d'_C = d'_B + d'_D) :
  ∀ (l : Line), is_non_touching_line quad l → 
    let d_A := dist_to_line quad.A l in
    let d_B := dist_to_line quad.B l in
    let d_C := dist_to_line quad.C l in
    let d_D := dist_to_line quad.D l in
    d_A + d_C = d_B + d_D :=
by
  sorry

end equal_sums_of_distances_l807_807183


namespace correct_grid_l807_807733

def A := 8
def B := 6
def C := 4
def D := 2

def grid := [[A, 1, 9],
             [3, 5, D],
             [B, C, 7]]

theorem correct_grid :
  (A + 1 < 12) ∧ (A + 3 < 12) ∧ (1 + 9 < 12) ∧
  (1 + 5 < 12) ∧ (3 + 5 < 12) ∧ (3 + B < 12) ∧
  (5 + D < 12) ∧ (5 + C < 12) ∧ (9 + D < 12) ∧
  (B + C < 12) ∧ (C + 7 < 12) :=
by
  -- This is to provide a sketch dummy theorem, we'd prove each step here  
  sorry

end correct_grid_l807_807733


namespace Mildred_heavier_than_Carol_l807_807605

-- Definition of weights for Mildred and Carol
def weight_Mildred : ℕ := 59
def weight_Carol : ℕ := 9

-- Definition of how much heavier Mildred is than Carol
def weight_difference : ℕ := weight_Mildred - weight_Carol

-- The theorem stating the difference in weight
theorem Mildred_heavier_than_Carol : weight_difference = 50 := 
by 
  -- Just state the theorem without providing the actual steps (proof skipped)
  sorry

end Mildred_heavier_than_Carol_l807_807605


namespace interior_angle_regular_octagon_l807_807449

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807449


namespace angle_measure_l807_807794

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l807_807794


namespace stockings_total_cost_l807_807617

-- Defining the conditions
def total_stockings : ℕ := 9
def original_price_per_stocking : ℝ := 20
def discount_rate : ℝ := 0.10
def monogramming_cost_per_stocking : ℝ := 5

-- Calculate the total cost of stockings
theorem stockings_total_cost :
  total_stockings * ((original_price_per_stocking * (1 - discount_rate)) + monogramming_cost_per_stocking) = 207 := 
by
  sorry

end stockings_total_cost_l807_807617


namespace angle_supplement_complement_l807_807890

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l807_807890


namespace inequality_proof_l807_807160

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
by
  sorry

end inequality_proof_l807_807160


namespace find_grid_values_l807_807753

-- Define the grid and the conditions in Lean
variables (A B C D : ℕ)
assume h1: A, B, C, D ∈ {2, 4, 6, 8}
assume h2: A + 1 < 12
assume h3: A + 9 < 12
assume h4: A + 3 < 12
assume h5: A + 5 < 12
assume h6: A + D < 12
assume h7: B + C < 12
assume h8: B + 5 < 12
assume h9: C + 7 < 12
assume h10: D + 9 < 12

-- Prove the specific values for A, B, C, and D
theorem find_grid_values : 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by sorry

end find_grid_values_l807_807753


namespace regular_octagon_interior_angle_l807_807424

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807424


namespace angle_measure_is_60_l807_807783

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l807_807783


namespace angle_measure_l807_807911

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807911


namespace number_of_girls_l807_807548

theorem number_of_girls (total_pupils boys : ℕ) (hyp1 : total_pupils = 929) (hyp2 : boys = 387) :
  total_pupils - boys = 542 :=
by
  rw [hyp1, hyp2]
  norm_num

end number_of_girls_l807_807548


namespace calc_square_uncovered_area_l807_807534

theorem calc_square_uncovered_area :
  ∀ (side_length : ℕ) (circle_diameter : ℝ) (num_circles : ℕ),
    side_length = 16 →
    circle_diameter = (16 / 3) →
    num_circles = 9 →
    (side_length ^ 2) - num_circles * (Real.pi * (circle_diameter / 2) ^ 2) = 256 - 64 * Real.pi :=
by
  intros side_length circle_diameter num_circles h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  sorry

end calc_square_uncovered_area_l807_807534


namespace interior_angle_regular_octagon_l807_807433

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807433


namespace common_chord_length_is_correct_l807_807028

noncomputable def length_of_common_chord : ℝ := 
  let radiusA := 10
  let radiusB := 15
  let centerA := (0:ℝ, 0:ℝ)
  let centerB := (15:ℝ, 0:ℝ)
  have circleA_passes_through_centerB : (15^2 + 0^2) = 15^2 := by
    sorry
  let common_chord_length := 15 * real.sqrt 3
  common_chord_length

theorem common_chord_length_is_correct :
  let radiusA := 10
  let radiusB := 15
  let centerA := (0:ℝ, 0:ℝ)
  let centerB := (15:ℝ, 0:ℝ)
  (length_of_common_chord = 15 * real.sqrt 3) := 
by
  sorry

end common_chord_length_is_correct_l807_807028


namespace volume_tetrahedron_solution_l807_807562

noncomputable def volume_tetrahedron_problem (A B C D : ℝ³)
    (h_angles_sum : ∀ (P : ℝ³), P ∈ {A, B, C, D} → 
        ∑ (angle : ℝ) in (set_of (λ angle, angle ∈ plane_angles P {A, B, C, D})), angle = 180)
    (h_BC : dist B C = 4)
    (h_cos_BAC : cos (angle B A C) = 3/4)
    (h_sin_CBD : sin (angle C B D) = 5 * sqrt 7 / 16) :
    ℝ :=
  sorry

theorem volume_tetrahedron_solution (A B C D : ℝ³)
    (h_angles_sum : ∀ (P : ℝ³), P ∈ {A, B, C, D} → 
        ∑ (angle : ℝ) in (set_of (λ angle, angle ∈ plane_angles P {A, B, C, D})), angle = 180)
    (h_BC : dist B C = 4)
    (h_cos_BAC : cos (angle B A C) = 3/4)
    (h_sin_CBD : sin (angle C B D) = 5 * sqrt 7 / 16) :
    volume_tetrahedron_problem A B C D h_angles_sum h_BC h_cos_BAC h_sin_CBD = 15 * sqrt 6 / 4 :=
  sorry

end volume_tetrahedron_solution_l807_807562


namespace greatest_number_of_quarters_l807_807704

theorem greatest_number_of_quarters (q d : ℕ) (h1 : q = d) (h2 : 0.25 * q + 0.10 * d = 4.90) :
  q = 14 :=
by
  sorry

end greatest_number_of_quarters_l807_807704


namespace angle_supplement_complement_l807_807810

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l807_807810


namespace find_constants_l807_807992

theorem find_constants (a b : ℝ) (h1 : ∀ x, x = π → y = a * real.sec (b * x) - 1)
    (h2 : y = 3) : a = 4 ∧ b = 1 := by
  sorry

end find_constants_l807_807992


namespace regular_octagon_interior_angle_l807_807335

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807335


namespace inequality_proof_l807_807166

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + (3/(a * b + b * c + c * a)) ≥ 6/(a + b + c) := 
sorry

end inequality_proof_l807_807166


namespace slope_at_point_l807_807011

noncomputable def eval_derivative_at_point (x : ℝ) : ℝ :=
  3 * x^2

theorem slope_at_point : 
  let slope := eval_derivative_at_point 1 in
  let angle := 45 in -- Given the problem context
  floatEq (Math.atan slope).toDegrees 45 1 :=
by
  sorry

end slope_at_point_l807_807011


namespace find_some_number_l807_807969

theorem find_some_number (n m : ℕ) (h : (n / 20) * (n / m) = 1) (n_eq_40 : n = 40) : m = 2 :=
by
  sorry

end find_some_number_l807_807969


namespace regular_octagon_interior_angle_l807_807252

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807252


namespace angle_supplement_complement_l807_807898

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l807_807898


namespace angle_measure_l807_807834

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l807_807834


namespace angle_supplement_complement_l807_807870

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l807_807870


namespace regular_octagon_interior_angle_l807_807415

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807415


namespace regular_octagon_interior_angle_eq_135_l807_807405

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807405


namespace regular_octagon_interior_angle_l807_807326

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807326


namespace interior_angle_regular_octagon_l807_807373

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807373


namespace geometric_inequality_l807_807594

variables {A B C : Type} [metric_space A] [metric_space B] [metric_space C]
variables {P_A Q_A P_B Q_B P_C Q_C: A}
variables {S ω : set A}
variables (R : ℝ)

-- Assuming the conditions given in the problem
def acute_triangle (ABC : Type) [metric_space ABC] := sorry
def incircle (ω : set A) := sorry
def circumcircle (S : set A) := sorry
def circumradius (R : ℝ) := sorry
def circle_ω_A (ω_A : set A) := is_internal_tangent ω_A S A ∧ is_external_tangent ω_A ω
def circle_S_A (S_A : set A) := is_internal_tangent S_A S A ∧ is_internal_tangent S_A ω

-- Proving the required inequality
theorem geometric_inequality (h_acute : acute_triangle ABC) (h_incircle : incircle ω)
(h_circumcircle : circumcircle S) (h_circumradius : circumradius R)
(h_ω_A : ∀ A, circle_ω_A (P_A A) (Q_A A) (S_A A))
(h_S_A : ∀ A, circle_S_A (P_A A) (Q_A A) (S_A A))
(h_PAQA : ∀ A, P_A A = center_of ω_A ∧ Q_A A = center_of S_A)
(h_PBQB : ∀ B, P_B B = center_of ω_B ∧ Q_B B = center_of S_B)
(h_PCQC : ∀ C, P_C C = center_of ω_C ∧ Q_C C = center_of S_C)
: 8 * dist P_A Q_A * dist P_B Q_B * dist P_C Q_C ≤ R ^ 3 ∧ (8 * dist P_A Q_A * dist P_B Q_B * dist P_C Q_C = R ^ 3 ↔ is_equilateral_triangle ABC) :=
sorry

end geometric_inequality_l807_807594


namespace regular_octagon_interior_angle_eq_135_l807_807408

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807408


namespace interior_angle_regular_octagon_l807_807488

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807488


namespace m_range_l807_807226

def A : Set ℝ := {x | x^2 - 28 ≤ 0}
def B (m : ℝ) : Set ℝ := {x | 2 * x^2 - (5 + m) * x + 5 ≤ 0}
def R : Set ℝ := Set.Univ \ A

theorem m_range (m : ℝ) : B m ⊆ R ↔ m ∈ Set.Iio (-5 - 2 * Real.sqrt 10) ∪ Set.Ioi (-5 + 2 * Real.sqrt 10) :=
sorry

end m_range_l807_807226


namespace sqrt_plus_inv_sqrt_eq_three_implies_frac_eq_inv_2025_l807_807194

theorem sqrt_plus_inv_sqrt_eq_three_implies_frac_eq_inv_2025
  (x : ℝ) (h : sqrt x + 1 / sqrt x = 3) :
  x / (x^2 + 2018 * x + 1) = 1 / 2025 := by
  sorry

end sqrt_plus_inv_sqrt_eq_three_implies_frac_eq_inv_2025_l807_807194


namespace angle_complement_supplement_l807_807826

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l807_807826


namespace regular_octagon_interior_angle_l807_807238

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807238


namespace angle_complement_supplement_l807_807817

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l807_807817


namespace regular_octagon_interior_angle_deg_l807_807386

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807386


namespace exist_equilateral_triangle_l807_807185

theorem exist_equilateral_triangle (O A B C : Type) [point O] [point A] [point B] [point C] 
  (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (coplanar : coplanar O A B C) : 
  (∃ (ABC: equilateral_triangle A B C), 
    dist O A = x ∧ dist O B = y ∧ dist O C = z) ↔ 
    (x + y ≥ z ∧ y + z ≥ x ∧ z + x ≥ y) := 
by
  sorry

end exist_equilateral_triangle_l807_807185


namespace circle_area_irrational_if_rational_diameter_l807_807527

noncomputable def pi : ℝ := Real.pi

theorem circle_area_irrational_if_rational_diameter (d : ℚ) :
  ¬ ∃ (A : ℝ), A = pi * (d / 2)^2 ∧ (∃ (q : ℚ), A = q) :=
by
  sorry

end circle_area_irrational_if_rational_diameter_l807_807527


namespace tangent_line_at_P_no_zero_points_sum_of_zero_points_l807_807195

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.log x - a * x

/-- Given that f(x) = ln(x) - 2x, prove that the tangent line at point P(1, -2) has the equation x + y + 1 = 0. -/
theorem tangent_line_at_P (a : ℝ) (h : a = 2) : ∀ x y : ℝ, x + y + 1 = 0 :=
sorry

/-- Show that for f(x) = ln(x) - ax, the function f(x) has no zero points if a > 1/e. -/
theorem no_zero_points (a : ℝ) (h : a > 1 / Real.exp 1) : ¬∃ x : ℝ, f x a = 0 :=
sorry

/-- For f(x) = ln(x) - ax and x1 ≠ x2 such that f(x1) = f(x2) = 0, prove that x1 + x2 > 2 / a. -/
theorem sum_of_zero_points (a x₁ x₂ : ℝ) (h₁ : x₁ ≠ x₂) (h₂ : f x₁ a = 0) (h₃ : f x₂ a = 0) : x₁ + x₂ > 2 / a :=
sorry

end tangent_line_at_P_no_zero_points_sum_of_zero_points_l807_807195


namespace solution_exists_l807_807557

variable (A B C D : Type) [metric_space A] [metric_space B] [metric_space C] [metric_space D]
variables [inner_product_space ℝ A] [inner_product_space ℝ B] [inner_product_space ℝ C] [inner_product_space ℝ D]

noncomputable def trapezoid_problem : Prop :=
  ∃ (AB CD AC: ℝ),
    (CD = 15) ∧
    (tan D = 3/4) ∧
    (tan B = 3/5) ∧
    (AB = (11.25 / (3/5))) ∧ -- Assuming from intermediate steps
    (BC = real.sqrt (AB^2 + AC^2)) ∧ 
    (BC ≈ 21.864) -- Assuming approximate equality to handle floats

theorem solution_exists : trapezoid_problem A B C D :=
sorry

end solution_exists_l807_807557


namespace max_value_of_h_l807_807159

noncomputable def f (x : ℝ) : ℝ := -x + 3
noncomputable def g (x : ℝ) : ℝ := Real.log x / Real.log 2

noncomputable def h (x : ℝ) : ℝ := min (f x) (g x)

theorem max_value_of_h : ∃ x : ℝ, h x = 1 :=
by
  sorry

end max_value_of_h_l807_807159


namespace problem_statement_l807_807059

variables {a b y x : ℝ}

theorem problem_statement :
  (3 * a + 2 * b ≠ 5 * a * b) ∧
  (5 * y - 3 * y ≠ 2) ∧
  (7 * a + a ≠ 7 * a ^ 2) ∧
  (3 * x^2 * y - 2 * y * x^2 = x^2 * y) :=
by
  split
  · intro h
    have h₁ : 3 * a + 2 * b = 3 * a + 2 * b := rfl
    rw h₁ at h
    sorry
  split
  · intro h
    have h₁ : 5 * y - 3 * y = 2 * y := by ring
    rw h₁ at h
    sorry
  split
  · intro h
    have h₁ : 7 * a + a = 8 * a := by ring
    rw h₁ at h
    sorry
  · ring

end problem_statement_l807_807059


namespace product_evaluation_l807_807144

theorem product_evaluation (b : ℕ) (h : b = 3) :
  (b-12) * (b-11) * (b-10) * (b-9) * (b-8) * (b-7) * (b-6) * (b-5) * (b-4) * (b-3) * (b-2) *
  (b-1) * b = 0 :=
by {
  rw h,
  sorry
}

end product_evaluation_l807_807144


namespace angle_supplement_complement_l807_807893

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l807_807893


namespace sum_of_fourth_powers_l807_807512

theorem sum_of_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 18.5 :=
sorry

end sum_of_fourth_powers_l807_807512


namespace triangular_number_solution_l807_807150

noncomputable def floor (x : ℝ) : ℕ := Int.to_nat (Real.floor x)

theorem triangular_number_solution (n : ℕ) :
  (let x := floor (Real.sqrt (2 * n : ℝ)) in 1 + x ∣ 2 * n) →
  ∃ x : ℕ, n = x * (x + 1) / 2 :=
by
  sorry

end triangular_number_solution_l807_807150


namespace regular_octagon_interior_angle_eq_135_l807_807404

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807404


namespace regular_octagon_angle_l807_807310

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807310


namespace restore_grid_values_l807_807773

def gridCondition (A B C D : Nat) : Prop :=
  A < 9 ∧ 1 < 9 ∧ 9 < 9 ∧
  3 < 9 ∧ 5 < 9 ∧ D < 9 ∧
  B < 9 ∧ C < 9 ∧ 7 < 9 ∧
  -- Sum of any two numbers in adjacent cells is less than 12
  A + 1 < 12 ∧ A + 3 < 12 ∧
  B + C < 12 ∧ B + 3 < 12 ∧
  D + 2 < 12 ∧ C + 4 < 12 ∧
  -- The grid values are from 1 to 9
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  C ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
  -- Even numbers erased are 2, 4, 6, and 8

theorem restore_grid_values :
  ∃ (A B C D : Nat), gridCondition A B C D ∧ A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  sorry

end restore_grid_values_l807_807773


namespace regular_octagon_angle_l807_807307

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807307


namespace seniors_are_correct_l807_807547

noncomputable def number_of_seniors : ℕ :=
  let total_students := 800
  let juniors := 0.22 * total_students
  let sophomores := 0.25 * total_students
  let freshmen := sophomores + 64
  total_students - (juniors + sophomores + freshmen)

theorem seniors_are_correct :
  number_of_seniors = 160 := by
  sorry

end seniors_are_correct_l807_807547


namespace regular_octagon_interior_angle_deg_l807_807384

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807384


namespace problem_equivalent_l807_807995

-- Definitions for easy referencing
def sqrt81 : ℝ := Real.sqrt 81
def sqrt144 : ℝ := Real.sqrt 144

-- The theorem we need to prove:
theorem problem_equivalent :
  sqrt81 - sqrt144 = -7 :=
by sorry

end problem_equivalent_l807_807995


namespace correct_grid_l807_807728

def A := 8
def B := 6
def C := 4
def D := 2

def grid := [[A, 1, 9],
             [3, 5, D],
             [B, C, 7]]

theorem correct_grid :
  (A + 1 < 12) ∧ (A + 3 < 12) ∧ (1 + 9 < 12) ∧
  (1 + 5 < 12) ∧ (3 + 5 < 12) ∧ (3 + B < 12) ∧
  (5 + D < 12) ∧ (5 + C < 12) ∧ (9 + D < 12) ∧
  (B + C < 12) ∧ (C + 7 < 12) :=
by
  -- This is to provide a sketch dummy theorem, we'd prove each step here  
  sorry

end correct_grid_l807_807728


namespace regular_octagon_interior_angle_l807_807465

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807465


namespace regular_octagon_interior_angle_deg_l807_807382

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807382


namespace three_pow_m_plus_2n_l807_807176

theorem three_pow_m_plus_2n (m n : ℕ) (h1 : 3^m = 5) (h2 : 9^n = 10) : 3^(m + 2 * n) = 50 :=
by
  sorry

end three_pow_m_plus_2n_l807_807176


namespace inequality_proof_l807_807163

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
by
  sorry

end inequality_proof_l807_807163


namespace regular_octagon_interior_angle_l807_807477

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807477


namespace regular_octagon_angle_l807_807303

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807303


namespace probability_of_xyz_eq_72_l807_807916

open ProbabilityTheory Finset

def dice_values : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem probability_of_xyz_eq_72 :
  (∑ x in dice_values, ∑ y in dice_values, ∑ z in dice_values, 
   if x * y * z = 72 then 1 else 0) / (dice_values.card ^ 3) = 1 / 36 :=
by
  sorry -- Proof omitted

end probability_of_xyz_eq_72_l807_807916


namespace regular_octagon_interior_angle_l807_807244

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807244


namespace inequality_am_gm_l807_807171

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
sorry

end inequality_am_gm_l807_807171


namespace max_value_of_a_l807_807528

theorem max_value_of_a (a : ℝ) : (∀ x ∈ set.Ioo 1 2, x^2 - |a| * x + a - 1 > 0) → a ≤ 2 := 
by 
sorry

end max_value_of_a_l807_807528


namespace middle_card_is_five_l807_807024

section card_numbers

variables {a b c : ℕ}

-- Conditions
def distinct (a b c : ℕ) : Prop := a ≠ b ∧ b ≠ c ∧ a ≠ c
def sum_fifteen (a b c : ℕ) : Prop := a + b + c = 15
def sum_two_smallest_less_than_ten (a b : ℕ) : Prop := a + b < 10
def ascending_order (a b c : ℕ) : Prop := a < b ∧ b < c 

-- Main theorem statement
theorem middle_card_is_five 
  (h1 : distinct a b c)
  (h2 : sum_fifteen a b c)
  (h3 : sum_two_smallest_less_than_ten a b) 
  (h4 : ascending_order a b c)
  (h5 : ∀ x, (x = a → (∃ b c, sum_fifteen x b c ∧ distinct x b c ∧ sum_two_smallest_less_than_ten x b ∧ ascending_order x b c ∧ ¬ (b = 5 ∧ c = 10))) →
           (∃ b c, sum_fifteen x b c ∧ distinct x b c ∧ sum_two_smallest_less_than_ten b c ∧ ascending_order x b c ∧ ¬ (b = 2 ∧ c = 7)))
  (h6 : ∀ x, (x = c → (∃ a b, sum_fifteen a b x ∧ distinct a b x ∧ sum_two_smallest_less_than_ten a b ∧ ascending_order a b x ∧ ¬ (a = 1 ∧ b = 4))) →
           (∃ a b, sum_fifteen a b x ∧ distinct a b x ∧ sum_two_smallest_less_than_ten a b ∧ ascending_order a b x ∧ ¬ (a = 2 ∧ b = 6)))
  (h7 : ∀ x, (x = b → (∃ a c, sum_fifteen a x c ∧ distinct a x c ∧ sum_two_smallest_less_than_ten a c ∧ ascending_order a x c ∧ ¬ (a = 1 ∧ c = 9 ∨ a = 2 ∧ c = 8))) →
           (∃ a c, sum_fifteen a x c ∧ distinct a x c ∧ sum_two_smallest_less_than_ten a c ∧ ascending_order a x c ∧ ¬ (a = 1 ∧ c = 6 ∨ a = 2 ∧ c = 5)))
  : b = 5 := sorry

end card_numbers

end middle_card_is_five_l807_807024


namespace at_least_one_weight_more_than_35_l807_807179

variable {a : Fin 11 → ℕ}
variable (h_unique : ∀ i j, i ≠ j → a i ≠ a j)
variable (h_ordered : ∀ i j, i < j → a i < a j)
variable (h_condition : (∑ i in Finset.range 6, a i) > (∑ i in Finset.range 6 11, a i+6))

theorem at_least_one_weight_more_than_35 : ∃ i, a i > 35 := sorry

end at_least_one_weight_more_than_35_l807_807179


namespace angle_complement_supplement_l807_807819

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l807_807819


namespace smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m_l807_807504

noncomputable def smaller_angle_at_715 : ℝ :=
  let hour_position := 7 * 30 + 30 / 4
  let minute_position := 15 * (360 / 60)
  let angle_between := abs (hour_position - minute_position)
  if angle_between > 180 then 360 - angle_between else angle_between

theorem smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m :
  smaller_angle_at_715 = 127.5 := 
sorry

end smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m_l807_807504


namespace angle_measure_l807_807803

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l807_807803


namespace interior_angle_regular_octagon_l807_807461

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807461


namespace regular_octagon_interior_angle_l807_807411

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807411


namespace angle_supplement_complement_l807_807896

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l807_807896


namespace restore_grid_values_l807_807745

def is_adjacent_sum_lt_12 (A B C D : ℕ) : Prop :=
  (A + 1 < 12) ∧ (A + 9 < 12) ∧
  (3 + 1 < 12) ∧ (3 + 5 < 12) ∧
  (3 + D < 12) ∧ (5 + 1 < 12) ∧ 
  (5 + D < 12) ∧ (B + C < 12) ∧
  (B + 3 < 12) ∧ (B + 5 < 12) ∧
  (C + 5 < 12) ∧ (C + 7 < 12) ∧
  (D + 9 < 12) ∧ (D + 7 < 12)

theorem restore_grid_values : 
  ∃ (A B C D : ℕ), 
  (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) ∧ 
  is_adjacent_sum_lt_12 A B C D := 
by 
  exists 8
  exists 6
  exists 4
  exists 2
  split; [refl, split; [refl, split; [refl, split; [refl, sorry]]]] 

end restore_grid_values_l807_807745


namespace restore_grid_l807_807716

-- Definitions for each cell in the grid.
variable (A B C D : ℕ)

-- The given grid and conditions
noncomputable def grid : list (list ℕ) :=
  [[A, 1, 9],
   [3, 5, D],
   [B, C, 7]]

-- The condition that the sum of numbers in adjacent cells is less than 12.
def valid (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 3 < 12 ∧
  1 + 9 < 12 ∧ 5 + D < 12 ∧
  3 + 5 < 12 ∧  B + C < 12 ∧
  9 + D < 12 ∧ 7 + C < 12 ∧
  9 + 1 < 12 ∧ B + 7 < 12

-- Assertion to be proved.
theorem restore_grid (A B C D : ℕ) (valid : valid A B C D) :
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 := by
  sorry

end restore_grid_l807_807716


namespace expected_value_of_T_l807_807656

theorem expected_value_of_T :
  let boys := 10;
  let girls := 15;
  let total_people := boys + girls;
  let num_trios := 23;
  let prob_BGB := (10 / 25 : ℝ) * (15 / 24) * (9 / 23);
  let prob_GBG := (15 / 25 : ℝ) * (10 / 24) * (14 / 23);
  let prob_trio := prob_BGB + prob_GBG;
  let expected_T := num_trios * prob_trio
  in (expected_T : ℝ).round = 4 :=
by
  let boys := 10;
  let girls := 15;
  let total_people := boys + girls;
  let num_trios := 23;
  let prob_BGB := (10 / 25 : ℝ) * (15 / 24) * (9 / 23);
  let prob_GBG := (15 / 25 : ℝ) * (10 / 24) * (14 / 23);
  let prob_trio := prob_BGB + prob_GBG;
  let expected_T := num_trios * prob_trio;
  have h : (expected_T : ℝ).round = 4, sorry;
  exact h

end expected_value_of_T_l807_807656


namespace no_valid_k_exists_l807_807120

theorem no_valid_k_exists {k : ℕ} : ¬(∃ p q : ℕ, p.Prime ∧ q.Prime ∧ p + q = 41 ∧ p * q = k) :=
by
  sorry

end no_valid_k_exists_l807_807120


namespace quadratic_formula_solutions_l807_807643

variable (a b c : ℝ)

noncomputable def discriminant : ℝ := b^2 - 4 * a * c

noncomputable def quadratic_solutions : set ℝ :=
if h : discriminant a b c > 0 then
  {(-b + Real.sqrt (discriminant a b c)) / (2 * a), (-b - Real.sqrt (discriminant a b c)) / (2 * a)}
else if h : discriminant a b c = 0 then
  { -b / (2 * a) }
else
  ∅

theorem quadratic_formula_solutions (h : a ≠ 0) :
  quadratic_solutions a b c = 
    {(-b + Real.sqrt (b^2 - 4*a*c)) / (2*a), (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)} ∨ 
    quadratic_solutions a b c = { -b / (2 * a) } ∨ 
    quadratic_solutions a b c = ∅ :=
sorry

end quadratic_formula_solutions_l807_807643


namespace regular_octagon_interior_angle_l807_807476

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807476


namespace weeks_to_work_l807_807568

-- Definitions of conditions as per step a)
def isabelle_ticket_cost : ℕ := 20
def brother_ticket_cost : ℕ := 10
def brothers_total_savings : ℕ := 5
def isabelle_savings : ℕ := 5
def job_pay_per_week : ℕ := 3
def total_ticket_cost := isabelle_ticket_cost + 2 * brother_ticket_cost
def total_savings := isabelle_savings + brothers_total_savings
def remaining_amount := total_ticket_cost - total_savings

-- Theorem statement to match the question
theorem weeks_to_work : remaining_amount / job_pay_per_week = 10 := by
  -- Lean expects a proof here, replaced with sorry to skip it
  sorry

end weeks_to_work_l807_807568


namespace correct_average_calculation_l807_807662

theorem correct_average_calculation (n : ℕ) (incorrect_avg correct_num wrong_num : ℕ) (incorrect_avg_eq : incorrect_avg = 21) (n_eq : n = 10) (correct_num_eq : correct_num = 36) (wrong_num_eq : wrong_num = 26) :
  (incorrect_avg * n + (correct_num - wrong_num)) / n = 22 := by
  sorry

end correct_average_calculation_l807_807662


namespace angle_solution_l807_807884

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l807_807884


namespace problem_statement_l807_807057

variables {a b y x : ℝ}

theorem problem_statement :
  (3 * a + 2 * b ≠ 5 * a * b) ∧
  (5 * y - 3 * y ≠ 2) ∧
  (7 * a + a ≠ 7 * a ^ 2) ∧
  (3 * x^2 * y - 2 * y * x^2 = x^2 * y) :=
by
  split
  · intro h
    have h₁ : 3 * a + 2 * b = 3 * a + 2 * b := rfl
    rw h₁ at h
    sorry
  split
  · intro h
    have h₁ : 5 * y - 3 * y = 2 * y := by ring
    rw h₁ at h
    sorry
  split
  · intro h
    have h₁ : 7 * a + a = 8 * a := by ring
    rw h₁ at h
    sorry
  · ring

end problem_statement_l807_807057


namespace interior_angle_regular_octagon_l807_807361

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807361


namespace restore_grid_l807_807713

-- Definitions for each cell in the grid.
variable (A B C D : ℕ)

-- The given grid and conditions
noncomputable def grid : list (list ℕ) :=
  [[A, 1, 9],
   [3, 5, D],
   [B, C, 7]]

-- The condition that the sum of numbers in adjacent cells is less than 12.
def valid (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 3 < 12 ∧
  1 + 9 < 12 ∧ 5 + D < 12 ∧
  3 + 5 < 12 ∧  B + C < 12 ∧
  9 + D < 12 ∧ 7 + C < 12 ∧
  9 + 1 < 12 ∧ B + 7 < 12

-- Assertion to be proved.
theorem restore_grid (A B C D : ℕ) (valid : valid A B C D) :
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 := by
  sorry

end restore_grid_l807_807713


namespace distance_focus_parabola_to_asymptotes_hyperbola_l807_807154

theorem distance_focus_parabola_to_asymptotes_hyperbola :
  let focus : ℝ × ℝ := (2, 0)
  let asymptote1 : ℝ → ℝ := λ x => √3 * x
  let asymptote2 : ℝ → ℝ := λ x => -√3 * x
  let distance (p : ℝ × ℝ) (l : ℝ → ℝ) : ℝ :=
    abs (√3 * p.1 - p.2) / real.sqrt (1 + 3)

  distance focus asymptote1 = √3 ∧ distance focus asymptote2 = √3 :=
by
  sorry

end distance_focus_parabola_to_asymptotes_hyperbola_l807_807154


namespace determine_grid_numbers_l807_807723

theorem determine_grid_numbers (A B C D : ℕ) :
  (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧ (1 ≤ D ∧ D ≤ 9) ∧ 
  (A ≠ 1) ∧ (A ≠ 3) ∧ (A ≠ 5) ∧ (A ≠ 7) ∧ (A ≠ 9) ∧ 
  (B ≠ 1) ∧ (B ≠ 3) ∧ (B ≠ 5) ∧ (B ≠ 7) ∧ (B ≠ 9) ∧ 
  (C ≠ 1) ∧ (C ≠ 3) ∧ (C ≠ 5) ∧ (C ≠ 7) ∧ (C ≠ 9) ∧ 
  (D ≠ 1) ∧ (D ≠ 3) ∧ (D ≠ 5) ∧ (D ≠ 7) ∧ (D ≠ 9) ∧ 
  (A ≠ 2 ∧ A ≠ 4 ∧ A ≠ 6 ∧ A ≠ 8 ∨ B ≠ 2 ∧ B ≠ 4 ∧ B ≠ 6 ∧ B ≠ 8 ∨ C ≠ 2 ∧ C ≠ 4 ∧ C ≠ 6 ∧ C ≠ 8 ∨ D ≠ 2 ∧ D ≠ 4 ∧ D ≠ 6 ∧ D ≠ 8) ∧ 
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ 
  (3 + 5 < 12) ∧ (3 + D < 12) ∧ (B + 5 < 12) ∧ 
  (B + C < 12) ∧ (C + 7 < 12) ∧ (B + 7 < 12) 
  → (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) :=
by 
  intros A B C D
  exact sorry

end determine_grid_numbers_l807_807723


namespace angle_supplement_complement_l807_807899

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l807_807899


namespace vasya_plus_signs_l807_807034

theorem vasya_plus_signs (sum_val : ℕ) (num_of_threes : ℕ) (total_sum : ℕ) :
  ((nat.digits 10 sum_val).count(3)) = 20 →
  total_sum = 600 →
  ∃ k, sum_val = (10^k) ∧ 20 - 1 = 9 :=
by sorry

end vasya_plus_signs_l807_807034


namespace angle_supplement_complement_l807_807894

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l807_807894


namespace restore_grid_l807_807715

-- Definitions for each cell in the grid.
variable (A B C D : ℕ)

-- The given grid and conditions
noncomputable def grid : list (list ℕ) :=
  [[A, 1, 9],
   [3, 5, D],
   [B, C, 7]]

-- The condition that the sum of numbers in adjacent cells is less than 12.
def valid (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 3 < 12 ∧
  1 + 9 < 12 ∧ 5 + D < 12 ∧
  3 + 5 < 12 ∧  B + C < 12 ∧
  9 + D < 12 ∧ 7 + C < 12 ∧
  9 + 1 < 12 ∧ B + 7 < 12

-- Assertion to be proved.
theorem restore_grid (A B C D : ℕ) (valid : valid A B C D) :
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 := by
  sorry

end restore_grid_l807_807715


namespace number_of_divisors_of_n_l807_807180

theorem number_of_divisors_of_n :
  let n : ℕ := (7^3) * (11^2) * (13^4)
  ∃ d : ℕ, d = 60 ∧ ∀ m : ℕ, m ∣ n ↔ ∃ l₁ l₂ l₃ : ℕ, l₁ ≤ 3 ∧ l₂ ≤ 2 ∧ l₃ ≤ 4 ∧ m = 7^l₁ * 11^l₂ * 13^l₃ := 
by
  sorry

end number_of_divisors_of_n_l807_807180


namespace complex_product_polar_form_l807_807996

theorem complex_product_polar_form :
  ∃ r θ, r > 0 ∧ 0 ≤ θ ∧ θ < 360 ∧ 
  (r = 12 ∧ θ = 245) :=
by
  sorry

end complex_product_polar_form_l807_807996


namespace proof_problem_l807_807698

variable (a : ℝ)
def U := Set.univ
def M := {x : ℝ | x + a ≥ 0}
def N := {x : ℝ | x - 2 < 1}
def C_N := {x : ℝ | x ≥ 3} -- Complement of N in U

theorem proof_problem :
  (M ∩ C_N = {x : ℝ | x ≥ 3}) → a ≥ -3 :=
by
  sorry

end proof_problem_l807_807698


namespace sin_cos_sum_l807_807207

theorem sin_cos_sum (x y : ℝ) (h : x = 2 ∧ y = -1) :
  let r := Real.sqrt (x^2 + y^2)
  let sin_alpha := y / r
  let cos_alpha := x / r
  sin_alpha + cos_alpha = (Real.sqrt 5) / 5 :=
by
  obtain ⟨hx, hy⟩ := h
  calc
  sorry

end sin_cos_sum_l807_807207


namespace sequence_solution_l807_807561

theorem sequence_solution :
  ∃ (a : ℕ → ℕ), a 1 = 2 ∧ (∀ n : ℕ, n ∈ (Set.Icc 1 9) → 
    (n * a (n + 1) = (n + 1) * a n + 2)) ∧ a 10 = 38 :=
by
  sorry

end sequence_solution_l807_807561


namespace regular_octagon_interior_angle_l807_807239

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807239


namespace sum_symmetric_induction_added_terms_expression_l807_807622

theorem sum_symmetric_induction (n : ℕ) (h : n > 0) : 1 + 2 + ... + n + (n - 1) + ... + 2 + 1 = n^2 := by
  sorry

theorem added_terms_expression (k : ℕ) : k > 0 → 1 + 2 + ... + k + (k - 1) + ... + 2 + 1 = k^2 → 
  (1 + 2 + ... + k + (k - 1) + ... + 2 + 1) + (k + 1) + (k + 1) = (k + 1)^2 := by 
  sorry

end sum_symmetric_induction_added_terms_expression_l807_807622


namespace regular_octagon_interior_angle_eq_135_l807_807401

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807401


namespace regular_octagon_interior_angle_l807_807302

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807302


namespace angle_measure_is_60_l807_807787

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l807_807787


namespace positive_difference_balances_l807_807576

noncomputable def laura_balance (L_0 : ℝ) (L_r : ℝ) (L_n : ℕ) (t : ℕ) : ℝ :=
  L_0 * (1 + L_r / L_n) ^ (L_n * t)

noncomputable def mark_balance (M_0 : ℝ) (M_r : ℝ) (t : ℕ) : ℝ :=
  M_0 * (1 + M_r * t)

theorem positive_difference_balances :
  let L_0 := 10000
  let L_r := 0.04
  let L_n := 2
  let t := 20
  let M_0 := 10000
  let M_r := 0.06
  abs ((laura_balance L_0 L_r L_n t) - (mark_balance M_0 M_r t)) = 80.40 :=
by
  sorry

end positive_difference_balances_l807_807576


namespace regular_octagon_interior_angle_l807_807466

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807466


namespace fewer_cans_collected_today_than_yesterday_l807_807940

theorem fewer_cans_collected_today_than_yesterday :
  let sarah_yesterday := 50
  let lara_yesterday := sarah_yesterday + 30
  let sarah_today := 40
  let lara_today := 70
  let total_yesterday := sarah_yesterday + lara_yesterday
  let total_today := sarah_today + lara_today
  total_yesterday - total_today = 20 :=
by
  sorry

end fewer_cans_collected_today_than_yesterday_l807_807940


namespace correct_grid_l807_807727

def A := 8
def B := 6
def C := 4
def D := 2

def grid := [[A, 1, 9],
             [3, 5, D],
             [B, C, 7]]

theorem correct_grid :
  (A + 1 < 12) ∧ (A + 3 < 12) ∧ (1 + 9 < 12) ∧
  (1 + 5 < 12) ∧ (3 + 5 < 12) ∧ (3 + B < 12) ∧
  (5 + D < 12) ∧ (5 + C < 12) ∧ (9 + D < 12) ∧
  (B + C < 12) ∧ (C + 7 < 12) :=
by
  -- This is to provide a sketch dummy theorem, we'd prove each step here  
  sorry

end correct_grid_l807_807727


namespace angle_measure_l807_807904

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807904


namespace product_fractions_gt_one_over_eleven_l807_807624

theorem product_fractions_gt_one_over_eleven :
  (∏ i in finset.range 60, (2 * (i + 1) : ℚ) / (2 * (i + 1) + 1)) > 1 / 11 :=
by
  sorry

end product_fractions_gt_one_over_eleven_l807_807624


namespace regular_octagon_interior_angle_l807_807325

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807325


namespace nth_equation_l807_807016

theorem nth_equation (n : ℕ) : (2 * n + 1)^2 - (2 * n - 1)^2 = 8 * n := by
  sorry

end nth_equation_l807_807016


namespace largest_integer_power_of_five_factorial_sum_l807_807158

theorem largest_integer_power_of_five_factorial_sum : 
  (∀ N : ℕ, N > 0 → Fact (N.factorial)) →
  (∃ n : ℕ, ∀ k : ℕ, k > n → ¬(5^k ∣ (97.factorial + 98.factorial + 99.factorial))) ∧
  (∀ m : ℕ, ¬(5^(m + 1) ∣ (97.factorial + 98.factorial + 99.factorial)) ↔ m = 22) :=
sorry

end largest_integer_power_of_five_factorial_sum_l807_807158


namespace regular_octagon_interior_angle_l807_807292

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807292


namespace triangle_perimeter_proof_l807_807533

-- Non-computable definitions might be needed due to the use of real numbers and the sqrt function
noncomputable def triangle_perimeter (A B C : ℝ) (AB : ℝ) : ℝ :=
  AB + sqrt ((AB / sqrt 2) ^ 2) + sqrt ((AB / sqrt 2) ^ 2)

-- The formal statement in Lean
theorem triangle_perimeter_proof (A B C : ℝ) (AB : ℝ) (h1 : AB = 10) 
  (h2 : ∠ABC = 90) (h3 : squares_outside : true)
  (h4 : points_on_circle: true) : 
  triangle_perimeter A B C AB = 10 + 10 * sqrt 2 := 
  sorry

end triangle_perimeter_proof_l807_807533


namespace angle_supplement_complement_l807_807813

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l807_807813


namespace angle_supplement_complement_l807_807868

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l807_807868


namespace jenga_blocks_before_jess_turn_l807_807542

-- Definitions based on conditions
noncomputable def num_players : ℕ := 5
noncomputable def num_rounds_before_sixth : ℕ := 5
noncomputable def additional_blocks_each_round : ℕ → ℕ
| 1     := 1
| (n+1) := additional_blocks_each_round n + 1

noncomputable def blocks_removed_in_sixth_round : ℕ := 6

-- Proving the total number of blocks before Jess's turn in the sixth round
theorem jenga_blocks_before_jess_turn :
  let blocks_removed_before_sixth :=
    ∑ i in finset.range num_rounds_before_sixth, (num_players * additional_blocks_each_round (i + 1)) in
  blocks_removed_before_sixth + blocks_removed_in_sixth_round = 81 :=
sorry

end jenga_blocks_before_jess_turn_l807_807542


namespace each_interior_angle_of_regular_octagon_l807_807351

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807351


namespace angle_supplement_complement_l807_807805

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l807_807805


namespace euros_received_l807_807113

def exchange_rate : ℝ := 2000 / 1.80
def service_fee_rate : ℝ := 0.05
def dollars : ℝ := 2.00

theorem euros_received : 
  let gross_euros := exchange_rate * dollars
  let net_euros := gross_euros * (1 - service_fee_rate)
  net_euros = 2111 :=
by
  let gross_euros := exchange_rate * dollars
  let net_euros := gross_euros * (1 - service_fee_rate)
  have gross_eq : gross_euros = 2222.22 := by sorry
  have net_eq : net_euros = 2111 := by sorry
  exact net_eq

end euros_received_l807_807113


namespace odd_three_mn_l807_807655

theorem odd_three_mn (m n : ℕ) (hm : m % 2 = 1) (hn : n % 2 = 1) : (3 * m * n) % 2 = 1 :=
sorry

end odd_three_mn_l807_807655


namespace product_of_two_numbers_l807_807029

open Set

theorem product_of_two_numbers (x y : ℕ) (h1 : 1 ≤ x) (h2 : x < y) (h3 : y ≤ 34) (h4 : 23 * x - 21 * y = -595) : x * y = 416 := by
  sorry

end product_of_two_numbers_l807_807029


namespace prob_xyz_eq_72_l807_807927

-- Define the set of possible outcomes for a standard six-sided die
def dice_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define a predicate that checks if three dice rolls multiply to 72
def is_valid_combination (x y z : ℕ) : Prop := (x * y * z = 72)

-- Define the event space for three dice rolls
def event_space : Finset (ℕ × ℕ × ℕ) := Finset.product dice_outcomes (Finset.product dice_outcomes dice_outcomes)

-- Define the probability of an event
def probability {α : Type*} [Fintype α] (s : Finset α) (event : α → Prop) : ℚ :=
  (s.filter event).card.to_rat / s.card.to_rat

-- State the theorem
theorem prob_xyz_eq_72 : probability event_space (λ t, is_valid_combination t.1 t.2.1 t.2.2) = (7 / 216) := 
by { sorry }

end prob_xyz_eq_72_l807_807927


namespace angle_supplement_complement_l807_807869

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l807_807869


namespace profit_percentage_calculation_l807_807945

noncomputable def selling_price : ℝ := 850
noncomputable def profit : ℝ := 215
noncomputable def cost_price : ℝ := selling_price - profit
noncomputable def profit_percentage : ℝ := (profit / cost_price) * 100

theorem profit_percentage_calculation :
  profit_percentage ≈ 33.86 :=
sorry

end profit_percentage_calculation_l807_807945


namespace restore_grid_values_l807_807749

def is_adjacent_sum_lt_12 (A B C D : ℕ) : Prop :=
  (A + 1 < 12) ∧ (A + 9 < 12) ∧
  (3 + 1 < 12) ∧ (3 + 5 < 12) ∧
  (3 + D < 12) ∧ (5 + 1 < 12) ∧ 
  (5 + D < 12) ∧ (B + C < 12) ∧
  (B + 3 < 12) ∧ (B + 5 < 12) ∧
  (C + 5 < 12) ∧ (C + 7 < 12) ∧
  (D + 9 < 12) ∧ (D + 7 < 12)

theorem restore_grid_values : 
  ∃ (A B C D : ℕ), 
  (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) ∧ 
  is_adjacent_sum_lt_12 A B C D := 
by 
  exists 8
  exists 6
  exists 4
  exists 2
  split; [refl, split; [refl, split; [refl, split; [refl, sorry]]]] 

end restore_grid_values_l807_807749


namespace interior_angle_regular_octagon_l807_807487

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807487


namespace conjugate_of_z_l807_807525

-- Definitions for the conditions
def complex_add (a b : ℂ) : ℂ := a + b
def complex_neg (a : ℂ) : ℂ := -a
def complex_mul (a b : ℂ) : ℂ := a * b
def complex_div (a b : ℂ) : ℂ := a / b

-- Definitions of imaginary unit and complex conjugate
def complex_i : ℂ := complex.I
def complex_conj (a : ℂ) : ℂ := conj a

-- Statement of the problem
theorem conjugate_of_z 
  (z : ℂ) 
  (h : complex_mul (complex_add 2 (complex_neg complex_i)) z = complex_mul complex_i complex_i ^ 2023) :
  complex_conj z = complex_div (complex_add 1 (complex_mul 2 complex_i)) 5 :=
sorry

end conjugate_of_z_l807_807525


namespace range_of_a_l807_807211

noncomputable def complex_number_conditions (a : ℝ) : Prop :=
  let i : ℂ := complex.I
  let z1 : ℂ := (-1 + 5 * i) / (1 + i)
  let z2 : ℂ := a - 2 - i
  let z2_conj : ℂ := complex.conj (a - 2 - i)
  (abs (z1 - z2_conj)) < (abs z1)

theorem range_of_a (a : ℝ) (h : complex_number_conditions a) : 1 < a ∧ a < 7 :=
  sorry

end range_of_a_l807_807211


namespace repeating_decimal_fraction_sum_l807_807049

theorem repeating_decimal_fraction_sum (x : ℚ) (hx : x = 0.27272727) :
  (x.numerator + x.denominator) = 23 := by
  sorry

end repeating_decimal_fraction_sum_l807_807049


namespace restore_grid_values_l807_807768

def gridCondition (A B C D : Nat) : Prop :=
  A < 9 ∧ 1 < 9 ∧ 9 < 9 ∧
  3 < 9 ∧ 5 < 9 ∧ D < 9 ∧
  B < 9 ∧ C < 9 ∧ 7 < 9 ∧
  -- Sum of any two numbers in adjacent cells is less than 12
  A + 1 < 12 ∧ A + 3 < 12 ∧
  B + C < 12 ∧ B + 3 < 12 ∧
  D + 2 < 12 ∧ C + 4 < 12 ∧
  -- The grid values are from 1 to 9
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  C ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
  -- Even numbers erased are 2, 4, 6, and 8

theorem restore_grid_values :
  ∃ (A B C D : Nat), gridCondition A B C D ∧ A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  sorry

end restore_grid_values_l807_807768


namespace regular_octagon_interior_angle_l807_807289

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807289


namespace total_balloons_l807_807175

theorem total_balloons (F S M : ℕ) (hF : F = 5) (hS : S = 6) (hM : M = 7) : F + S + M = 18 :=
by 
  sorry

end total_balloons_l807_807175


namespace restore_grid_l807_807762

noncomputable def grid_values (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 9 < 12 ∧
  3 + 1 < 12 ∧ 3 + 5 < 12 ∧
  B + 3 < 12 ∧ B + 4 < 12 ∧
  B + 7 < 12 ∧ 
  C + 4 < 12 ∧
  C + 7 < 12 ∧ 
  D + 9 < 12 ∧
  D + 7 < 12 ∧
  D + 5 < 12

def even_numbers_erased (A B C D : ℕ) : Prop :=
  A ∈ {8} ∧ B ∈ {6} ∧ C ∈ {4} ∧ D ∈ {2}

theorem restore_grid :
  ∃ (A B C D : ℕ), grid_values A B C D ∧ even_numbers_erased A B C D :=
by {
  use [8, 6, 4, 2],
  dsimp [grid_values, even_numbers_erased],
  exact ⟨⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩, 
         ⟨rfl, rfl, rfl, rfl⟩⟩
}

end restore_grid_l807_807762


namespace average_speed_is_70_kmh_l807_807692

-- Define the given conditions
def distance1 : ℕ := 90
def distance2 : ℕ := 50
def time1 : ℕ := 1
def time2 : ℕ := 1

-- We need to prove that the average speed of the car is 70 km/h
theorem average_speed_is_70_kmh :
    ((distance1 + distance2) / (time1 + time2)) = 70 := 
by 
    -- This is the proof placeholder
    sorry

end average_speed_is_70_kmh_l807_807692


namespace integral_correct_l807_807935

theorem integral_correct : 
  ∫ x in 0..1, (x^2 + real.sqrt(1 - x^2)) = (real.pi / 4) + (1 / 3) :=
by
  sorry

end integral_correct_l807_807935


namespace angle_measure_is_60_l807_807782

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l807_807782


namespace find_λ_l807_807172

noncomputable def λ_solution : Real :=
  (1 + Real.sqrt 33) / 2

theorem find_λ (z : Complex) (λ : Real) (hz : Complex.abs z = 3) (hλ : λ > 1) 
  (equilateral : Triangle.equi (1 : Complex) z (λ * z)) : λ = λ_solution :=
by 
  sorry

end find_λ_l807_807172


namespace valid_votes_B_is_3159_l807_807551

noncomputable def V : ℕ := 9720
noncomputable def invalid_votes : ℕ := 0.2 * V
noncomputable def valid_votes : ℕ := V - invalid_votes
noncomputable def A_votes : ℕ := B_votes + 0.15 * V
noncomputable def B_votes : ℕ := 3159

theorem valid_votes_B_is_3159 :
  A_votes + B_votes = valid_votes → B_votes = 3159 :=
sorry

end valid_votes_B_is_3159_l807_807551


namespace angle_measure_l807_807833

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l807_807833


namespace interior_angle_regular_octagon_l807_807464

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807464


namespace angle_solution_l807_807876

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l807_807876


namespace interior_angle_regular_octagon_l807_807358

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807358


namespace each_interior_angle_of_regular_octagon_l807_807344

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807344


namespace regular_octagon_interior_angle_eq_135_l807_807393

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807393


namespace grid_solution_l807_807734

-- Define the grid and conditions
variables (A B C D : ℕ)

-- Grid values with the constraints
def grid_valid (A B C D : ℕ) :=
  A ≠ 1 ∧ A ≠ 9 ∧ A ≠ 3 ∧ A ≠ 5 ∧ A ≠ 7 ∧
  B ≠ 1 ∧ B ≠ 9 ∧ B ≠ 3 ∧ B ≠ 5 ∧ B ≠ 7 ∧
  C ≠ 1 ∧ C ≠ 9 ∧ C ≠ 3 ∧ C ≠ 5 ∧ C ≠ 7 ∧
  D ≠ 1 ∧ D ≠ 9 ∧ D ≠ 3 ∧ D ≠ 5 ∧ D ≠ 7 ∧
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ (A + 3 < 12) ∧
  (D + 9 < 12) ∧ (D + 5 < 12) ∧ 
  (B + 3 < 12) ∧ (B + C < 12) ∧
  (C + 7 < 12)

-- Given the conditions, we need to prove the values of A, B, C, and D
theorem grid_solution : 
  grid_valid 8 6 4 2 ∧ 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  split
  · -- to show grid_valid 8 6 4 2
    sorry
    
  · -- to show the values of A, B, C, and D
    repeat {split, assumption}

end grid_solution_l807_807734


namespace regular_octagon_interior_angle_l807_807293

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807293


namespace fewerCansCollected_l807_807937

-- Definitions for conditions
def cansCollectedYesterdaySarah := 50
def cansCollectedMoreYesterdayLara := 30
def cansCollectedTodaySarah := 40
def cansCollectedTodayLara := 70

-- Total cans collected yesterday
def totalCansCollectedYesterday := cansCollectedYesterdaySarah + (cansCollectedYesterdaySarah + cansCollectedMoreYesterdayLara)

-- Total cans collected today
def totalCansCollectedToday := cansCollectedTodaySarah + cansCollectedTodayLara

-- Proving the difference
theorem fewerCansCollected :
  totalCansCollectedYesterday - totalCansCollectedToday = 20 := by
  sorry

end fewerCansCollected_l807_807937


namespace regular_octagon_interior_angle_l807_807265

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807265


namespace cone_altitude_to_base_radius_l807_807100

theorem cone_altitude_to_base_radius (r h : ℝ) (h₁ : r > 0)
  (h₂ : (1 / 3) * Mathlib.pi * r^2 * h = (1 / 2) * (4 / 3) * Mathlib.pi * r^3) :
  h / r = 2 :=
by
  sorry

end cone_altitude_to_base_radius_l807_807100


namespace angle_measure_l807_807792

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l807_807792


namespace bob_cake_flour_fill_times_l807_807993

theorem bob_cake_flour_fill_times :
  let required_flour := 15 / 4
  let excess_flour := 2 / 3
  let container_capacity := 4 / 3
  let adjusted_flour := required_flour - excess_flour in
  (adjusted_flour / container_capacity).ceil = 3 := 
by
  sorry

end bob_cake_flour_fill_times_l807_807993


namespace regular_octagon_interior_angle_l807_807481

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807481


namespace perpendicular_line_equation_l807_807776

theorem perpendicular_line_equation (a b c : ℝ) : 
    (a = 3) → (b = -6) → (c = 9) → (x1 y1 : ℝ) → (x1 = -2) → (y1 = -3) →
    ∃ m d : ℝ, d = -7 ∧ m = -2 ∧ (∀ x y: ℝ, y = m * x + d ↔ y = -2 * x - 7) :=
by
  intros h₁ h₂ h₃ x1 y1 h₄ h₅
  use -2, -7
  simp
  intro x y
  split
  · intro h
    simp at *
    assumption
  · intro h
    simp at *
    assumption

end perpendicular_line_equation_l807_807776


namespace weighted_average_proof_l807_807539

noncomputable def weighted_average_of_class 
  (n : ℕ) -- number of students
  (n_math_100 : ℕ) (n_math_0 : ℕ) (avg_math_rest : ℝ) -- math exam conditions
  (n_science_100 : ℕ) (n_science_0 : ℕ) (avg_science_rest : ℝ) -- science exam conditions
  (w_math : ℝ) (w_science : ℝ) -- weights for exams
  : ℝ := 
  let total_math := n_math_100 * 100 + n_math_0 * 0 + (n - n_math_100 - n_math_0) * avg_math_rest in
  let total_science := n_science_100 * 100 + n_science_0 * 0 + (n - n_science_100 - n_science_0) * avg_science_rest in
  let weighted_math := total_math * w_math in
  let weighted_science := total_science * w_science in
  let total_weighted_marks := weighted_math + weighted_science in
  total_weighted_marks / n

theorem weighted_average_proof : 
  weighted_average_of_class 30 3 4 50 2 5 60 0.4 0.6 = 50.9333 :=
by
  sorry

end weighted_average_proof_l807_807539


namespace time_trains_cross_opposite_directions_l807_807032

-- Define the conditions
def length_of_each_train : ℝ := 120
def time_to_cross_post_train1 : ℝ := 10
def time_to_cross_post_train2 : ℝ := 15

-- Calculate speeds
def speed_train1 : ℝ := length_of_each_train / time_to_cross_post_train1
def speed_train2 : ℝ := length_of_each_train / time_to_cross_post_train2

-- Define the total distance to be covered when crossing each other
def total_distance : ℝ := 2 * length_of_each_train

-- Relative speed when traveling in opposite directions
def relative_speed : ℝ := speed_train1 + speed_train2

-- The time it takes for the two trains to cross each other traveling in opposite directions
def time_to_cross_each_other : ℝ := total_distance / relative_speed

-- Statement of the theorem to prove
theorem time_trains_cross_opposite_directions : time_to_cross_each_other = 12 := by
  sorry

end time_trains_cross_opposite_directions_l807_807032


namespace probability_xyz_72_l807_807919

noncomputable def dice_probability : ℚ :=
  let outcomes : Finset (ℕ × ℕ × ℕ) := 
    {(a, b, c) | a ∈ {1, 2, 3, 4, 5, 6}, b ∈ {1, 2, 3, 4, 5, 6}, c ∈ {1, 2, 3, 4, 5, 6}}.toFinset
  let favorable_outcomes := 
    outcomes.filter (λ (abc : ℕ × ℕ × ℕ), abc.1 * abc.2 * abc.3 = 72)
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ)

theorem probability_xyz_72 : dice_probability = 1/24 :=
  sorry

end probability_xyz_72_l807_807919


namespace interior_angle_regular_octagon_l807_807440

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807440


namespace regular_octagon_interior_angle_deg_l807_807377

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807377


namespace angle_measure_l807_807851

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807851


namespace angle_measure_l807_807793

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l807_807793


namespace inequality_proof_l807_807164

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + (3/(a * b + b * c + c * a)) ≥ 6/(a + b + c) := 
sorry

end inequality_proof_l807_807164


namespace identify_geometric_progression_l807_807932

def is_geometric_progression (seq : List ℕ) : Prop :=
  ∃ r : ℕ, ∀ (n : ℕ), n + 1 < seq.length → seq[n + 1] = r * seq[n]

theorem identify_geometric_progression :
  let seq_A := [1, 2, 3, 4, 5, 6]
  let seq_B := [1, 2, 4, 8, 16, 32]
  let seq_C := [0, 0, 0, 0, 0, 0]
  let seq_D := [1, -2, 3, -4, 5, -6]
  is_geometric_progression seq_B ∧ 
  ¬ is_geometric_progression seq_A ∧
  ¬ is_geometric_progression seq_C ∧
  ¬ is_geometric_progression seq_D :=
by
  sorry

end identify_geometric_progression_l807_807932


namespace interior_angle_regular_octagon_l807_807366

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807366


namespace restore_grid_l807_807712

-- Definitions for each cell in the grid.
variable (A B C D : ℕ)

-- The given grid and conditions
noncomputable def grid : list (list ℕ) :=
  [[A, 1, 9],
   [3, 5, D],
   [B, C, 7]]

-- The condition that the sum of numbers in adjacent cells is less than 12.
def valid (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 3 < 12 ∧
  1 + 9 < 12 ∧ 5 + D < 12 ∧
  3 + 5 < 12 ∧  B + C < 12 ∧
  9 + D < 12 ∧ 7 + C < 12 ∧
  9 + 1 < 12 ∧ B + 7 < 12

-- Assertion to be proved.
theorem restore_grid (A B C D : ℕ) (valid : valid A B C D) :
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 := by
  sorry

end restore_grid_l807_807712


namespace quadrilateral_cosine_law_l807_807661

variables {AB BC CD AD AC : ℝ}
variables {B C φ : ℝ}

-- Conditions in the problem
theorem quadrilateral_cosine_law (h1 : φ = φ)
  (h2 : AD^2 = AC^2 + CD^2 - 2 * AC * CD * Real.cos (A.1 : ℝ))
  (h3 : AC^2 = AB^2 + BC^2 - 2 * AB * BC * Real.cos (B : ℝ))
  (h4 : AC * Real.cos (A.1 : ℝ) = AB * Real.cos (φ : ℝ) + BC * Real.cos (C : ℝ)) :
  AD^2 = AB^2 + BC^2 + CD^2 - 2 * (AB * BC * Real.cos (B : ℝ) + BC * CD * Real.cos (C : ℝ) + CD * AB * Real.cos (φ : ℝ)) :=
sorry

end quadrilateral_cosine_law_l807_807661


namespace angle_supplement_complement_l807_807866

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l807_807866


namespace chord_length_of_line_and_circle_l807_807665

noncomputable def circle_eq : Real → Real → Prop :=
  λ x y, x^2 + y^2 + 4 * x - 4 * y + 6 = 0

noncomputable def line_eq : Real → Real → Prop :=
  λ x y, x - y + 4 = 0

theorem chord_length_of_line_and_circle :
  ∀ (x y : Real), circle_eq x y ∧ line_eq x y → 2 * Real.sqrt 2 = 2 * Real.sqrt 2 :=
by
  -- Proof omitted, use sorry
  sorry

end chord_length_of_line_and_circle_l807_807665


namespace exists_consecutive_non_primes_l807_807637

theorem exists_consecutive_non_primes (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℕ, ∀ i : ℕ, i < k → ¬Nat.Prime (n + i) := 
sorry

end exists_consecutive_non_primes_l807_807637


namespace correctness_check_l807_807055

noncomputable def questionD (x y : ℝ) : Prop := 
  3 * x^2 * y - 2 * y * x^2 = x^2 * y

theorem correctness_check (x y : ℝ) : questionD x y :=
by 
  sorry

end correctness_check_l807_807055


namespace projection_of_tetrahedron_can_be_square_l807_807998

theorem projection_of_tetrahedron_can_be_square
  {A B C D : Point}
  (h_eq_edges : dist A B = dist A C ∧ dist A B = dist A D ∧ dist B C = dist B D ∧ dist C D = dist A C)
  (h_angles : angle A C B = 90 ∧ angle A D B = 90 ∧ angle B D C = 90) :
  ∃ (P : Plane), is_projection_square P (Tetrahedron.mk A B C D) :=
sorry

end projection_of_tetrahedron_can_be_square_l807_807998


namespace three_gorges_scientific_notation_l807_807660

theorem three_gorges_scientific_notation :
  ∃a n : ℝ, (1 ≤ |a| ∧ |a| < 10) ∧ (798.5 * 10^1 = a * 10^n) ∧ a = 7.985 ∧ n = 2 :=
by
  sorry

end three_gorges_scientific_notation_l807_807660


namespace assign_roles_l807_807971

/-
  Given:
  1. Three distinct male roles.
  2. Three distinct female roles.
  3. Two roles that can be either gender.
  4. A man can only play a male role.
  5. A woman can only play a female role.
  6. Six men and seven women are available for audition.

  Prove:
  The number of ways to assign the eight roles is 1058400.
-/

theorem assign_roles :
  let male_roles := 3
  let female_roles := 3
  let either_gender_roles := 2
  (6.choose male_roles) * male_roles.factorial *
  (7.choose female_roles) * female_roles.factorial *
  ((6 - male_roles + 7 - female_roles).choose either_gender_roles) * either_gender_roles.factorial = 1058400 :=
by
  sorry

end assign_roles_l807_807971


namespace regular_octagon_angle_l807_807309

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807309


namespace regular_octagon_interior_angle_deg_l807_807379

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807379


namespace overall_percentage_profit_l807_807103

theorem overall_percentage_profit (A_buy_cheat_percent A_sell_cheat_percent B_buy_cheat_percent B_sell_cheat_percent : ℝ)
  (A_fraction B_fraction : ℝ) :
  A_buy_cheat_percent = 0.12 → A_sell_cheat_percent = 0.20 →
  B_buy_cheat_percent = 0.18 → B_sell_cheat_percent = 0.30 →
  A_fraction = 0.5 → B_fraction = 0.5 →
  let profit_A := (112 - 80) / 80 * 100,
      profit_B := (118 - 70) / 70 * 100,
      overall_profit := (profit_A + profit_B) / 2
  in overall_profit = 54.29 :=
by
  intros h1 h2 h3 h4 h5 h6,
  sorry

end overall_percentage_profit_l807_807103


namespace one_minus_repeat_eight_l807_807147

theorem one_minus_repeat_eight : 1 - (0.888888...) = 1 / 9 :=
by
  -- Assume repeating decimal representation
  let b := 0.888888...
  -- Given that b is equivalent to 8/9
  have h₁ : b = 8 / 9 := sorry
  -- Subtracting from 1
  have h₂ : 1 - b = 1 - 8 / 9 := sorry
  assume h₃ : 1 - 8 / 9 = 1 / 9
  -- Ultimately, these steps together show
  exact h₃

end one_minus_repeat_eight_l807_807147


namespace count_special_prime_dates_2024_l807_807608

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def prime_months := {m : ℕ | is_prime m ∧ m ≤ 12}
noncomputable def prime_days := {d : ℕ | is_prime d ∧ d ≤ 31}

noncomputable def is_weekend_2024 (day : ℕ) : Prop := 
  let jan1 := 1 -- 01/01/2024 is a Monday
  let weekend := (jan1 + day - 1) % 7 ∈ {5, 6} -- Saturday or Sunday
  weekend

def special_prime_dates_2024 : ℕ :=
  (finset.mesh prime_months prime_days).filter (λ (d : ℕ × ℕ), is_weekend_2024 (d.2 + 31 * (d.1 - 1))).card

theorem count_special_prime_dates_2024
  : special_prime_dates_2024 = 10 :=
sorry

end count_special_prime_dates_2024_l807_807608


namespace grid_solution_l807_807739

-- Define the grid and conditions
variables (A B C D : ℕ)

-- Grid values with the constraints
def grid_valid (A B C D : ℕ) :=
  A ≠ 1 ∧ A ≠ 9 ∧ A ≠ 3 ∧ A ≠ 5 ∧ A ≠ 7 ∧
  B ≠ 1 ∧ B ≠ 9 ∧ B ≠ 3 ∧ B ≠ 5 ∧ B ≠ 7 ∧
  C ≠ 1 ∧ C ≠ 9 ∧ C ≠ 3 ∧ C ≠ 5 ∧ C ≠ 7 ∧
  D ≠ 1 ∧ D ≠ 9 ∧ D ≠ 3 ∧ D ≠ 5 ∧ D ≠ 7 ∧
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ (A + 3 < 12) ∧
  (D + 9 < 12) ∧ (D + 5 < 12) ∧ 
  (B + 3 < 12) ∧ (B + C < 12) ∧
  (C + 7 < 12)

-- Given the conditions, we need to prove the values of A, B, C, and D
theorem grid_solution : 
  grid_valid 8 6 4 2 ∧ 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  split
  · -- to show grid_valid 8 6 4 2
    sorry
    
  · -- to show the values of A, B, C, and D
    repeat {split, assumption}

end grid_solution_l807_807739


namespace angle_supplement_complement_l807_807888

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l807_807888


namespace locus_hyperbola_min_area_OAB_l807_807184

-- Define the points and conditions
structure Point where
  x : ℝ
  y : ℝ

def M := Point.mk (-2) 0
def N := Point.mk 2 0
def O := Point.mk 0 0

-- Define the locus of point P
def locus_P (P : Point) : Prop :=
  (P.x ^ 2 - P.y ^ 2 = 2) ∧ (P.x > 0)

-- Define the distance function
def dist (P Q : Point) : ℝ :=
  real.sqrt ((P.x - Q.x) ^ 2 + (P.y - Q.y) ^ 2)

-- The given condition for P
def condition (P : Point) : Prop :=
  dist P M - dist P N = 2 * real.sqrt 2

-- Verify the locus of point P
theorem locus_hyperbola (P : Point) :
  condition P → locus_P P := sorry

-- Define the triangle area function
def area_triangle (O A B : Point) : ℝ :=
  abs (O.x * (A.y - B.y) + A.x * (B.y - O.y) + B.x * (O.y - A.y)) / 2

-- Define the minimal area problem
theorem min_area_OAB (A B : Point) :
  area_triangle O A B = 2 * real.sqrt 2 := sorry

end locus_hyperbola_min_area_OAB_l807_807184


namespace regular_octagon_interior_angle_l807_807294

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807294


namespace chloe_pawn_loss_l807_807645

theorem chloe_pawn_loss (sophia_lost : ℕ) (total_left : ℕ) (total_initial : ℕ) (each_start : ℕ) (sophia_initial : ℕ) :
  sophia_lost = 5 → total_left = 10 → each_start = 8 → total_initial = 16 → sophia_initial = 8 →
  ∃ (chloe_lost : ℕ), chloe_lost = 1 :=
by
  sorry

end chloe_pawn_loss_l807_807645


namespace eq_of_frac_sub_l807_807123

theorem eq_of_frac_sub (x : ℝ) (hx : x ≠ 1) : 
  (2 / (x^2 - 1) - 1 / (x - 1)) = - (1 / (x + 1)) := 
by sorry

end eq_of_frac_sub_l807_807123


namespace interior_angle_regular_octagon_l807_807489

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807489


namespace angle_measure_is_60_l807_807789

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l807_807789


namespace regular_octagon_interior_angle_deg_l807_807376

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807376


namespace largest_prime_value_of_quadratic_expression_l807_807777

theorem largest_prime_value_of_quadratic_expression : 
  ∃ n : ℕ, n > 0 ∧ Prime (n^2 - 12 * n + 27) ∧ ∀ m : ℕ, m > 0 → Prime (m^2 - 12 * m + 27) → (n^2 - 12 * n + 27) ≥ (m^2 - 12 * m + 27) := 
by
  sorry


end largest_prime_value_of_quadratic_expression_l807_807777


namespace probability_of_line_intersecting_circle_l807_807632

noncomputable def probability_intersection : ℚ :=
  let possible_pairs := [(1, 2), (1, 3), (1, 4), (1, 5), (1, 6),
                          (2, 3), (2, 4), (2, 5), (2, 6),
                          (3, 4), (3, 5), (3, 6),
                          (4, 5), (4, 6),
                          (5, 6) : List (ℕ × ℕ)] in
  (possible_pairs.length : ℚ) / 36

theorem probability_of_line_intersecting_circle :
  probability_intersection = 5 / 12 :=
by
  sorry

end probability_of_line_intersecting_circle_l807_807632


namespace convex_hull_polygon_l807_807018

theorem convex_hull_polygon (n : ℕ) (points : list (ℝ × ℝ))
  (h_points : points.length = n)
  (h_no_other_points_in_triangle : ∀ (p q r : (ℝ × ℝ)), p ∈ points → q ∈ points → r ∈ points → 
    (∀ (s : (ℝ × ℝ)), s ∈ points → s ≠ p → s ≠ q → s ≠ r → 
      ¬ inside_triangle s p q r ∧ ¬ on_edge_of_triangle s p q r)) :
  ∃ (A : list (ℝ × ℝ)), A.perm points ∧ A.forms_convex_polygon := 
sorry

end convex_hull_polygon_l807_807018


namespace angle_measure_l807_807903

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807903


namespace martha_total_payment_l807_807025

noncomputable def cheese_kg : ℝ := 1.5
noncomputable def meat_kg : ℝ := 0.55
noncomputable def pasta_kg : ℝ := 0.28
noncomputable def tomatoes_kg : ℝ := 2.2

noncomputable def cheese_price_per_kg : ℝ := 6.30
noncomputable def meat_price_per_kg : ℝ := 8.55
noncomputable def pasta_price_per_kg : ℝ := 2.40
noncomputable def tomatoes_price_per_kg : ℝ := 1.79

noncomputable def total_cost :=
  cheese_kg * cheese_price_per_kg +
  meat_kg * meat_price_per_kg +
  pasta_kg * pasta_price_per_kg +
  tomatoes_kg * tomatoes_price_per_kg

theorem martha_total_payment : total_cost = 18.76 := by
  sorry

end martha_total_payment_l807_807025


namespace angle_measure_l807_807854

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l807_807854


namespace angle_supplement_complement_l807_807873

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l807_807873


namespace regular_octagon_interior_angle_l807_807283

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807283


namespace regular_octagon_interior_angle_l807_807245

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807245


namespace mildred_heavier_than_carol_l807_807607

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9

theorem mildred_heavier_than_carol : mildred_weight - carol_weight = 50 := 
by
  sorry

end mildred_heavier_than_carol_l807_807607


namespace degree_of_polynomial_l807_807136

theorem degree_of_polynomial :
  ∀ (x : ℝ), 
  (degree ((X^3 + X + 1)^5 * (X^4 + X^2 + 1)^2) = 23) := by
sorry

end degree_of_polynomial_l807_807136


namespace angle_complement_supplement_l807_807820

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l807_807820


namespace inequality_proof_l807_807167

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + (3/(a * b + b * c + c * a)) ≥ 6/(a + b + c) := 
sorry

end inequality_proof_l807_807167


namespace angle_measure_l807_807843

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807843


namespace restore_grid_l807_807763

noncomputable def grid_values (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 9 < 12 ∧
  3 + 1 < 12 ∧ 3 + 5 < 12 ∧
  B + 3 < 12 ∧ B + 4 < 12 ∧
  B + 7 < 12 ∧ 
  C + 4 < 12 ∧
  C + 7 < 12 ∧ 
  D + 9 < 12 ∧
  D + 7 < 12 ∧
  D + 5 < 12

def even_numbers_erased (A B C D : ℕ) : Prop :=
  A ∈ {8} ∧ B ∈ {6} ∧ C ∈ {4} ∧ D ∈ {2}

theorem restore_grid :
  ∃ (A B C D : ℕ), grid_values A B C D ∧ even_numbers_erased A B C D :=
by {
  use [8, 6, 4, 2],
  dsimp [grid_values, even_numbers_erased],
  exact ⟨⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩, 
         ⟨rfl, rfl, rfl, rfl⟩⟩
}

end restore_grid_l807_807763


namespace number_of_friends_shared_with_l807_807634

-- Conditions and given data
def doughnuts_samuel : ℕ := 2 * 12
def doughnuts_cathy : ℕ := 3 * 12
def total_doughnuts : ℕ := doughnuts_samuel + doughnuts_cathy
def each_person_doughnuts : ℕ := 6
def total_people := total_doughnuts / each_person_doughnuts
def samuel_and_cathy : ℕ := 2

-- Statement to prove - Number of friends they shared with
theorem number_of_friends_shared_with : (total_people - samuel_and_cathy) = 8 := by
  sorry

end number_of_friends_shared_with_l807_807634


namespace solve_for_y_solve_for_x_l807_807213

variable (x y : ℝ)

theorem solve_for_y (h : 2 * x + 3 * y - 4 = 0) : y = (4 - 2 * x) / 3 := 
sorry

theorem solve_for_x (h : 2 * x + 3 * y - 4 = 0) : x = (4 - 3 * y) / 2 := 
sorry

end solve_for_y_solve_for_x_l807_807213


namespace restore_grid_l807_807760

noncomputable def grid_values (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 9 < 12 ∧
  3 + 1 < 12 ∧ 3 + 5 < 12 ∧
  B + 3 < 12 ∧ B + 4 < 12 ∧
  B + 7 < 12 ∧ 
  C + 4 < 12 ∧
  C + 7 < 12 ∧ 
  D + 9 < 12 ∧
  D + 7 < 12 ∧
  D + 5 < 12

def even_numbers_erased (A B C D : ℕ) : Prop :=
  A ∈ {8} ∧ B ∈ {6} ∧ C ∈ {4} ∧ D ∈ {2}

theorem restore_grid :
  ∃ (A B C D : ℕ), grid_values A B C D ∧ even_numbers_erased A B C D :=
by {
  use [8, 6, 4, 2],
  dsimp [grid_values, even_numbers_erased],
  exact ⟨⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩, 
         ⟨rfl, rfl, rfl, rfl⟩⟩
}

end restore_grid_l807_807760


namespace regular_octagon_interior_angle_l807_807270

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807270


namespace interior_angle_regular_octagon_l807_807452

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807452


namespace correct_statement_B_l807_807934

theorem correct_statement_B (α : ℝ) : 
  (∃ (α : ℝ), sin α = 0 ∧ (α = 0 ∨ α = π)) :=
by
  -- Given the conditions from the problem
  have h1 : ¬ (∀ θ : ℝ, θ ∈ set.Icc 0 180 → angle_tri θ ∈ {1, 2}) := sorry
  have h2 : ∃ α : ℝ, sin α = 0 ∧ (tan α = 0) := sorry
  have h3 : ∀ α β : ℝ, α ÷ β = 0 → α.Mod β = α ∧ ( ∀ k : ℤ, α = β + k * (2 * π)) := sorry
  have h4 : ∀ α : ℝ, α ∈ set.Icc (π/2) π → obtuse_angle α := sorry
  exact h2

end correct_statement_B_l807_807934


namespace slope_range_l807_807555

-- Definitions of points and lines
def pointA : ℝ × ℝ := (0, 4 / 3)
def pointB : ℝ × ℝ := (-1, 0)
def pointC : ℝ × ℝ := (1, 0)

-- Locus of point P satisfying geometric mean distance condition
def locusP_ext1 (P : ℝ × ℝ) :=
  P.1^2 + P.2^2 + (3/2) * P.2 - 1 = 0

def locusP_ext2 (P : ℝ × ℝ) :=
  8 * P.1^2 - 17 * P.2^2 + 12 * P.2 - 8 = 0

-- Locus definition
def is_locus_point (P : ℝ × ℝ) :=
  locusP_ext1 P ∨ locusP_ext2 P

-- Incenter of triangle ABC
def incenterD : ℝ × ℝ := (0, 1 / 2)

-- Range of slope k considering the given conditions
def valid_slopes : set ℝ := {0, 1/2, -1/2, (2 * real.sqrt 34) / 17, -(2 * real.sqrt 34) / 17, real.sqrt 2 / 2, - real.sqrt 2 / 2}

-- Proof of valid slopes given line l passes through incenter and intersects the locus at 3 points
theorem slope_range (k : ℝ) (H : ∃ P1 P2 P3 : ℝ × ℝ, 
                    P1 ≠ P2 ∧ P2 ≠ P3 ∧ P1 ≠ P3 ∧ 
                    (is_locus_point P1 ∧ is_locus_point P2 ∧ is_locus_point P3) ∧
                    ∀ P : ℝ × ℝ, incenterD.2 = k * incenterD.1 + (1/2) ∧ 
                    is_locus_point P) : 
  k ∈ valid_slopes :=
sorry

end slope_range_l807_807555


namespace number_of_solutions_is_two_l807_807130

noncomputable def count_solutions := 
  {x : ℝ // 2^(3*x^2 - 8*x + 2) = 1}

theorem number_of_solutions_is_two :
  (Fintype.card (count_solutions)) = 2 :=
sorry

end number_of_solutions_is_two_l807_807130


namespace pow_exponents_identity_l807_807514

theorem pow_exponents_identity (m n : ℝ) (h1 : 2^m = 3) (h2 : 2^n = 4) : 
  2^(3 * m - 2 * n) = 27 / 16 := by
  sorry

end pow_exponents_identity_l807_807514


namespace find_a_l807_807677

def quadratic_form {a : ℝ} (x : ℝ) : ℝ :=
  a * (x - 3) ^ 2 + 5

theorem find_a (a : ℝ) : 
  quadratic_form 0 = -20 → 
  a = -25/9 :=
by
  intros h
  sorry

end find_a_l807_807677


namespace pipe_weight_calc_l807_807064

-- Definitions of the conditions
def pipe_length : ℝ := 42 -- in centimeters
def external_diameter : ℝ := 8 -- in centimeters
def pipe_thickness : ℝ := 1 -- in centimeters
def iron_density : ℝ := 3 -- in grams per cubic centimeter

-- The Lean statement
theorem pipe_weight_calc : 
  let external_radius := external_diameter / 2 in
  let internal_diameter := external_diameter - 2 * pipe_thickness in
  let internal_radius := internal_diameter / 2 in 
  let volume_total := Real.pi * external_radius^2 * pipe_length in
  let volume_hollow := Real.pi * internal_radius^2 * pipe_length in
  let volume_iron := volume_total - volume_hollow in
  let weight := volume_iron * iron_density in
  | weight - 2764.74 | < 0.1 :=
by
  sorry

end pipe_weight_calc_l807_807064


namespace inverse_proposition_of_three_right_angles_is_rectangle_l807_807679

-- Define the condition: A quadrilateral with three right angles is a rectangle
def quadrilateral_with_three_right_angles_is_rectangle (Q : Type) [quadrilateral Q] : Prop :=
  ∀ (q : Q), has_three_right_angles q → is_rectangle q

-- State the inverse proposition: A rectangle has three right angles
def rectangle_has_three_right_angles (R : Type) [rectangle R] : Prop :=
  ∀ (r : R), is_rectangle r → has_three_right_angles r

-- The proof problem statement in Lean
theorem inverse_proposition_of_three_right_angles_is_rectangle (Q R : Type) [quadrilateral Q] [rectangle R] :
  (quadrilateral_with_three_right_angles_is_rectangle Q) →
  (rectangle_has_three_right_angles R) :=
begin
  intros h,
  sorry
end

end inverse_proposition_of_three_right_angles_is_rectangle_l807_807679


namespace fewerCansCollected_l807_807939

-- Definitions for conditions
def cansCollectedYesterdaySarah := 50
def cansCollectedMoreYesterdayLara := 30
def cansCollectedTodaySarah := 40
def cansCollectedTodayLara := 70

-- Total cans collected yesterday
def totalCansCollectedYesterday := cansCollectedYesterdaySarah + (cansCollectedYesterdaySarah + cansCollectedMoreYesterdayLara)

-- Total cans collected today
def totalCansCollectedToday := cansCollectedTodaySarah + cansCollectedTodayLara

-- Proving the difference
theorem fewerCansCollected :
  totalCansCollectedYesterday - totalCansCollectedToday = 20 := by
  sorry

end fewerCansCollected_l807_807939


namespace regular_octagon_interior_angle_l807_807425

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807425


namespace subcommittees_with_at_least_one_teacher_l807_807683

-- Define the total number of members and the count of teachers
def total_members : ℕ := 12
def teacher_count : ℕ := 5
def subcommittee_size : ℕ := 5

-- Define binomial coefficient calculation
def binom (n k : ℕ) : ℕ :=
  Nat.choose n k

-- Problem statement: number of five-person subcommittees with at least one teacher
theorem subcommittees_with_at_least_one_teacher :
  binom total_members subcommittee_size - binom (total_members - teacher_count) subcommittee_size = 771 := by
  sorry

end subcommittees_with_at_least_one_teacher_l807_807683


namespace regular_octagon_interior_angle_l807_807257

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807257


namespace angle_measure_is_60_l807_807785

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l807_807785


namespace angle_supplement_complement_l807_807814

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l807_807814


namespace regular_octagon_interior_angle_eq_135_l807_807402

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807402


namespace orange_jellybeans_count_l807_807086

theorem orange_jellybeans_count (total blue purple red : Nat)
  (h_total : total = 200)
  (h_blue : blue = 14)
  (h_purple : purple = 26)
  (h_red : red = 120) :
  ∃ orange : Nat, orange = total - (blue + purple + red) ∧ orange = 40 :=
by
  sorry

end orange_jellybeans_count_l807_807086


namespace interior_angle_regular_octagon_l807_807357

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807357


namespace directrix_of_parabola_l807_807682

-- Define the variables and constants
variables (x y a : ℝ) (h₁ : x^2 = 4 * a * y) (h₂ : x = -2) (h₃ : y = 1)

theorem directrix_of_parabola (h : (-2)^2 = 4 * a * 1) : y = -1 := 
by
  -- Our proof will happen here, but we omit the details
  sorry

end directrix_of_parabola_l807_807682


namespace one_minus_repeating_eight_l807_807148

theorem one_minus_repeating_eight : (1 - (0.8 + 0.08 + 0.008 + (λ n, 8/10 ^ n))) = 1 / 9 := by
    sorry

end one_minus_repeating_eight_l807_807148


namespace not_all_vertices_coincide_l807_807030

open Classical
noncomputable theory

structure Polygon (Point : Type) :=
(vertices : List Point)

def equal_polygons {Point : Type} (F F' : Polygon Point) : Prop :=
  ∃ (f : List Point → List Point), 
    Permutation.equiv f F.vertices F'.vertices
  
def vertices_belong {Point : Type} (F F' : Polygon Point) : Prop :=
  ∀ v ∈ F.vertices, v ∈ F'.vertices

theorem not_all_vertices_coincide {Point : Type}
  (F F' : Polygon Point)
  (Heq : equal_polygons F F')
  (Hbelong : vertices_belong F F') :
  ¬(∀ v ∈ F.vertices, ∃ u ∈ F'.vertices, v = u) :=
sorry

end not_all_vertices_coincide_l807_807030


namespace regular_octagon_interior_angle_l807_807276

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807276


namespace each_interior_angle_of_regular_octagon_l807_807346

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807346


namespace restore_grid_values_l807_807743

def is_adjacent_sum_lt_12 (A B C D : ℕ) : Prop :=
  (A + 1 < 12) ∧ (A + 9 < 12) ∧
  (3 + 1 < 12) ∧ (3 + 5 < 12) ∧
  (3 + D < 12) ∧ (5 + 1 < 12) ∧ 
  (5 + D < 12) ∧ (B + C < 12) ∧
  (B + 3 < 12) ∧ (B + 5 < 12) ∧
  (C + 5 < 12) ∧ (C + 7 < 12) ∧
  (D + 9 < 12) ∧ (D + 7 < 12)

theorem restore_grid_values : 
  ∃ (A B C D : ℕ), 
  (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) ∧ 
  is_adjacent_sum_lt_12 A B C D := 
by 
  exists 8
  exists 6
  exists 4
  exists 2
  split; [refl, split; [refl, split; [refl, split; [refl, sorry]]]] 

end restore_grid_values_l807_807743


namespace ship_speed_is_18_km_per_hr_l807_807102

noncomputable def speed_of_ship : ℝ := 
  let distance_m := 100.008 in
  let time_s := 20 in
  let distance_km := distance_m / 1000 in
  let time_hr := time_s / 3600 in
  distance_km / time_hr

theorem ship_speed_is_18_km_per_hr : speed_of_ship = 18 := by
  sorry

end ship_speed_is_18_km_per_hr_l807_807102


namespace place_5_in_cup_C_l807_807116

theorem place_5_in_cup_C :
  ∃ (slips : list ℝ) (cups : finset ℝ), 
    slips = [2.5, 2.5, 3, 3, 3, 3.5, 3.5, 4, 4, 4.5, 4.5, 5, 5.5, 6] ∧ 
    (∀ (A B C D E : finset ℝ), 
      cups = insert 8 (insert 9 (insert 10 (insert 11 (insert 12 ∅)))) ∧
      3.5 ∈ E ∧ 4.5 ∈ B →
      5 ∈ C
    ) :=
sorry

end place_5_in_cup_C_l807_807116


namespace split_payment_l807_807627

noncomputable def Rahul_work_per_day := (1 : ℝ) / 3
noncomputable def Rajesh_work_per_day := (1 : ℝ) / 2
noncomputable def Ritesh_work_per_day := (1 : ℝ) / 4

noncomputable def total_work_per_day := Rahul_work_per_day + Rajesh_work_per_day + Ritesh_work_per_day

noncomputable def Rahul_proportion := Rahul_work_per_day / total_work_per_day
noncomputable def Rajesh_proportion := Rajesh_work_per_day / total_work_per_day
noncomputable def Ritesh_proportion := Ritesh_work_per_day / total_work_per_day

noncomputable def total_payment := 510

noncomputable def Rahul_share := Rahul_proportion * total_payment
noncomputable def Rajesh_share := Rajesh_proportion * total_payment
noncomputable def Ritesh_share := Ritesh_proportion * total_payment

theorem split_payment :
  Rahul_share + Rajesh_share + Ritesh_share = total_payment :=
by
  sorry

end split_payment_l807_807627


namespace solution_set_of_inequality_l807_807012

theorem solution_set_of_inequality :
  {x : ℝ | 2 * x - x^2 > 0} = set.Ioo 0 2 :=
sorry

end solution_set_of_inequality_l807_807012


namespace sum_of_solutions_eq_minus_2_l807_807695

-- Defining the equation and the goal
theorem sum_of_solutions_eq_minus_2 (x1 x2 : ℝ) (floor : ℝ → ℤ) (h1 : floor (3 * x1 + 1) = 2 * x1 - 1 / 2)
(h2 : floor (3 * x2 + 1) = 2 * x2 - 1 / 2) :
  x1 + x2 = -2 :=
sorry

end sum_of_solutions_eq_minus_2_l807_807695


namespace avg_remaining_two_l807_807068

theorem avg_remaining_two (a b c d e : ℝ) (h1 : (a + b + c + d + e) / 5 = 8) (h2 : (a + b + c) / 3 = 4) :
  (d + e) / 2 = 14 := by
  sorry

end avg_remaining_two_l807_807068


namespace regular_octagon_angle_l807_807311

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807311


namespace angle_measure_l807_807795

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l807_807795


namespace interior_angle_regular_octagon_l807_807445

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807445


namespace range_of_A_max_f_A_l807_807222

-- Define the given conditions in Lean
def triangle (A B C : ℝ) (a b c S : ℝ) : Prop :=
  c * cos A = 4 / b ∧ S ≥ 2 ∧ S = 1/2 * b * c * sin A

-- Prove the range of angle A
theorem range_of_A (A B C a b c S : ℝ) (h_triangle : triangle A B C a b c S) :
  π / 4 ≤ A ∧ A < π / 2 :=
sorry

-- Define the function f(x):
def f (A : ℝ) : ℝ := cos A ^ 2 + sqrt 3 * sin (π / 2 + A / 2) ^ 2 - sqrt 3 / 2

-- Prove the maximum value of f(A)
theorem max_f_A (A B C a b c S : ℝ) (h_triangle : triangle A B C a b c S) :
  ∀ A, π / 4 ≤ A ∧ A ≤ π / 2 → f A ≤ 1 / 2 + sqrt 6 / 4 :=
sorry

end range_of_A_max_f_A_l807_807222


namespace product_of_distinct_divisors_of_200000_is_117_l807_807588

theorem product_of_distinct_divisors_of_200000_is_117 :
  let T := {d | d ∣ 200000 ∧ d > 0} in
  let numbers := {d | ∃ a b ∈ T, a ≠ b ∧ d = a * b} in
  numbers.card = 117 :=
by
  let T : set ℕ := {d | d ∣ 200000 ∧ d > 0}
  let numbers : set ℕ := {d | ∃ a b ∈ T, a ≠ b ∧ d = a * b}
  sorry

end product_of_distinct_divisors_of_200000_is_117_l807_807588


namespace hyperbola_eccentricity_l807_807202

def hyperbola_foci_and_conditions (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) : Prop :=
∃ e : ℝ, 
  (c = Real.sqrt(a^2 + b^2)) ∧ -- condition from hyperbola equation
  (N = (a * Real.sqrt(2 * c^2 - a^2) / c, (c^2 - a^2) / c)) ∧ -- point N
  (M = (a, b)) ∧ -- point M
  (line_parallel (MF₁ : Line) (ON : Line)) ∧ -- lines MF₁ and ON are parallel
  (f(e) = 2) -- the function f on eccentricity e equals 2
  
theorem hyperbola_eccentricity (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) :
  hyperbola_foci_and_conditions a b h_a h_b :=
sorry

end hyperbola_eccentricity_l807_807202


namespace conclusion1_conclusion2_l807_807933

section problem

def f1 (x : ℝ) : ℝ := 2^x + x^3

def f2 (x : ℝ) : ℝ := x^2 + 2*x + 3

theorem conclusion1 (x1 x2 : ℝ) (h : x1 ≠ x2) : (f1 x1 - f1 x2) / (x1 - x2) > 0 :=
sorry

theorem conclusion2 (x : ℝ) : f2 (-x) ≠ f2 x :=
sorry

end problem

end conclusion1_conclusion2_l807_807933


namespace ratio_of_tangent_and_parallel_lines_l807_807707

open EuclideanGeometry

-- Define the given problem in lean
theorem ratio_of_tangent_and_parallel_lines
  (σ₁ σ₂ : Circle)
  (A B P Q R S W : Point)
  (h₁ : A ∈ σ₁ ∧ A ∈ σ₂)
  (h₂ : B ∈ σ₁ ∧ B ∈ σ₂)
  (h₃ : is_tangent PQ σ₁ ∧ is_tangent PQ σ₂ ∧ P ∈ σ₁ ∧ Q ∈ σ₂)
  (h₄ : is_tangent RS σ₁ ∧ is_tangent RS σ₂ ∧ R ∈ σ₁ ∧ S ∈ σ₂)
  (h₅ : RB ∥ PQ)
  (h₆ : W ∈ σ₂)
  (h₇ : line_through R B intersects σ₂ at W)
  : RB / BW = 1 / 3 := 
by sorry

end ratio_of_tangent_and_parallel_lines_l807_807707


namespace boy_speed_on_first_day_l807_807078

/-- Define the conditions and the problem. -/
theorem boy_speed_on_first_day 
  (distance : ℝ := 3)
  (speed_on_second_day : ℝ := 12)
  (travel_time_diff_in_minutes : ℝ := 15)
  (first_day_lateness : ℝ := 7 / 60) -- converted to hours
  (second_day_earliness : ℝ := 8 / 60) -- converted to hours
  (v : ℝ) -- speed on the first day
  (h1 : travel_time_diff_in_minutes = 15)
  (distance_positive : distance > 0)
  (speed_pos : v > 0 ∧ speed_on_second_day > 0)
  (eq1 : fraction_of_time_saved := (first_day_lateness + second_day_earliness)) 
  (eq2 : fraction_of_time_saved = 1 / 4) -- converting seconds to hours and simplifying)  

: v = 6 :=
by 
  -- Define time taken with the unknown speed v
  have t_first := distance / v, 
  -- Define time taken on the second day
  have t_second := distance / speed_on_second_day, 
  -- Define the equality given by the difference in time
  have time_diff := t_first - t_second = fraction_of_time_saved, 
  -- Substitute the condition for fraction_of_time_saved
  have time_diff_with_condition := t_first - t_second = eq2, 
  -- solve for v
  have solve_eq := by linarith,
  sorry

end boy_speed_on_first_day_l807_807078


namespace fewer_cans_collected_today_than_yesterday_l807_807941

theorem fewer_cans_collected_today_than_yesterday :
  let sarah_yesterday := 50
  let lara_yesterday := sarah_yesterday + 30
  let sarah_today := 40
  let lara_today := 70
  let total_yesterday := sarah_yesterday + lara_yesterday
  let total_today := sarah_today + lara_today
  total_yesterday - total_today = 20 :=
by
  sorry

end fewer_cans_collected_today_than_yesterday_l807_807941


namespace appropriate_sampling_method_l807_807540

def total_families := 500
def high_income_families := 125
def middle_income_families := 280
def low_income_families := 95
def sample_size := 100
def influenced_by_income := True

theorem appropriate_sampling_method
  (htotal : total_families = 500)
  (hhigh : high_income_families = 125)
  (hmiddle : middle_income_families = 280)
  (hlow : low_income_families = 95)
  (hsample : sample_size = 100)
  (hinfluence : influenced_by_income = True) :
  ∃ method, method = "Stratified sampling method" :=
sorry

end appropriate_sampling_method_l807_807540


namespace interior_angle_regular_octagon_l807_807454

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807454


namespace david_profit_l807_807203

noncomputable def profit_calculation : ℝ :=
  let weight := 50 in
  let cost := 50 in
  let price_per_kg := 1.2 in
  let discount_percentage := 0.05 in
  let tax_percentage := 0.12 in
  let total_selling_price := price_per_kg * weight in
  let discount_amount := total_selling_price * discount_percentage in
  let selling_price_after_discount := total_selling_price - discount_amount in
  let tax_amount := selling_price_after_discount * tax_percentage in
  let final_selling_price := selling_price_after_discount + tax_amount in
  final_selling_price - cost

theorem david_profit :
  profit_calculation = 13.84 :=
by
  sorry

end david_profit_l807_807203


namespace angle_supplement_complement_l807_807871

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l807_807871


namespace range_of_a_l807_807212

noncomputable def complex_number_conditions (a : ℝ) : Prop :=
  let i : ℂ := complex.I
  let z1 : ℂ := (-1 + 5 * i) / (1 + i)
  let z2 : ℂ := a - 2 - i
  let z2_conj : ℂ := complex.conj (a - 2 - i)
  (abs (z1 - z2_conj)) < (abs z1)

theorem range_of_a (a : ℝ) (h : complex_number_conditions a) : 1 < a ∧ a < 7 :=
  sorry

end range_of_a_l807_807212


namespace first_three_digits_of_x_are_571_l807_807774

noncomputable def x : ℝ := (10^2003 + 1)^(11/7)

theorem first_three_digits_of_x_are_571 : 
  ∃ d₁ d₂ d₃ : ℕ, 
  (d₁, d₂, d₃) = (5, 7, 1) ∧ 
  ∃ k : ℤ, 
  (x - k : ℝ) * 1000 = d₁ * 100 + d₂ * 10 + d₃ := 
by
  sorry

end first_three_digits_of_x_are_571_l807_807774


namespace regular_octagon_interior_angle_l807_807417

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807417


namespace proof_a16_to_a20_l807_807696

noncomputable def geometric_sequence_sum (a q : ℝ) (n : ℕ) : ℝ :=
if q = 1 then a * n else a * (1 - q^n) / (1 - q)

variable {a1 q : ℝ} (h1 : q ≠ 1) (h1_5 : geometric_sequence_sum a1 q 5 = 2) (h1_10 : geometric_sequence_sum a1 q 10 = 6)
variable a16 : ℝ

def a_n (a1 q : ℝ) (n : ℕ) := a1 * q^(n-1)

def sum_a16_to_a20 (a16 : ℝ) (q : ℝ) : ℝ := a16 * (1 + q + q^2 + q^3 + q^4)

theorem proof_a16_to_a20 :
  a16 = a_n a1 q 16 →
  (q ^ 5 = 2) →
  sum_a16_to_a20 a16 q = 16 :=
begin
  intros h_a16 h_q5,
  sorry
end

end proof_a16_to_a20_l807_807696


namespace interior_angle_regular_octagon_l807_807374

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807374


namespace no_solutions_then_a_eq_zero_l807_807569

theorem no_solutions_then_a_eq_zero (a b : ℝ) :
  (∀ x y : ℝ, ¬ (y^2 = x^2 + a * x + b ∧ x^2 = y^2 + a * y + b)) → a = 0 :=
by
  sorry

end no_solutions_then_a_eq_zero_l807_807569


namespace grid_solution_l807_807735

-- Define the grid and conditions
variables (A B C D : ℕ)

-- Grid values with the constraints
def grid_valid (A B C D : ℕ) :=
  A ≠ 1 ∧ A ≠ 9 ∧ A ≠ 3 ∧ A ≠ 5 ∧ A ≠ 7 ∧
  B ≠ 1 ∧ B ≠ 9 ∧ B ≠ 3 ∧ B ≠ 5 ∧ B ≠ 7 ∧
  C ≠ 1 ∧ C ≠ 9 ∧ C ≠ 3 ∧ C ≠ 5 ∧ C ≠ 7 ∧
  D ≠ 1 ∧ D ≠ 9 ∧ D ≠ 3 ∧ D ≠ 5 ∧ D ≠ 7 ∧
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ (A + 3 < 12) ∧
  (D + 9 < 12) ∧ (D + 5 < 12) ∧ 
  (B + 3 < 12) ∧ (B + C < 12) ∧
  (C + 7 < 12)

-- Given the conditions, we need to prove the values of A, B, C, and D
theorem grid_solution : 
  grid_valid 8 6 4 2 ∧ 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  split
  · -- to show grid_valid 8 6 4 2
    sorry
    
  · -- to show the values of A, B, C, and D
    repeat {split, assumption}

end grid_solution_l807_807735


namespace determine_x_l807_807135

def line_slope (p1 p2 : ℝ × ℝ) : ℝ :=
  if p2.1 - p1.1 = 0 then 0 else (p2.2 - p1.2) / (p2.1 - p1.1)

def point_on_line (p1 p2 p : ℝ × ℝ) : Prop :=
  line_slope p1 p2 = line_slope p1 p

theorem determine_x :
  point_on_line (2, 1) (10, 7) (x, -1) → x = -2 / 3 :=
by
  intro h
  -- Proof goes here
  sorry

end determine_x_l807_807135


namespace find_grid_values_l807_807751

-- Define the grid and the conditions in Lean
variables (A B C D : ℕ)
assume h1: A, B, C, D ∈ {2, 4, 6, 8}
assume h2: A + 1 < 12
assume h3: A + 9 < 12
assume h4: A + 3 < 12
assume h5: A + 5 < 12
assume h6: A + D < 12
assume h7: B + C < 12
assume h8: B + 5 < 12
assume h9: C + 7 < 12
assume h10: D + 9 < 12

-- Prove the specific values for A, B, C, and D
theorem find_grid_values : 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by sorry

end find_grid_values_l807_807751


namespace regular_octagon_interior_angle_deg_l807_807388

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807388


namespace angle_solution_l807_807880

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l807_807880


namespace interior_angle_regular_octagon_l807_807429

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807429


namespace interior_angle_regular_octagon_l807_807483

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807483


namespace graph_of_g_neg_is_B_l807_807676

def g (x : ℝ) : ℝ :=
  if -4 ≤ x ∧ x ≤ 1 then -3 - x
  else if 1 < x ∧ x ≤ 5 then - Math.sqrt (9 - (x - 1)^2) - 3
  else if 5 < x ∧ x ≤ 8 then 3 * (x - 5) - 1
  else 0

def g_neg (x : ℝ) : ℝ :=
  g (-x)

theorem graph_of_g_neg_is_B :
  ∀ x, g_neg x =
    if -4 ≤ x ∧ x ≤ 1 then -3 + x
    else if 1 < x ∧ x ≤ 5 then - Math.sqrt (9 - (1 + x)^2) - 3
    else if 5 < x ∧ x ≤ 8 then 3 * (-x - 5) - 1
    else 0 :=
sorry

end graph_of_g_neg_is_B_l807_807676


namespace train_crossing_time_l807_807976

def train_length := 140
def train_speed_kmph := 45
def bridge_length := 235
def speed_to_mps (kmph : ℕ) : ℕ := (kmph * 1000) / 3600
def total_distance := train_length + bridge_length
def train_speed := speed_to_mps train_speed_kmph
def time_to_cross := total_distance / train_speed

theorem train_crossing_time : time_to_cross = 30 := by
  sorry

end train_crossing_time_l807_807976


namespace each_interior_angle_of_regular_octagon_l807_807348

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807348


namespace angle_measure_l807_807858

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l807_807858


namespace simplify_and_evaluate_expr_l807_807639

-- Define the expression
def expr (x : ℝ) := ((2 / (x - 1)) + (1 / (x + 1))) * (x^2 - 1)

-- The value x to be substituted
noncomputable def x_val : ℝ := (Real.sqrt 3 - 1) / 3

-- State the theorem to prove that the simplified expression evaluates to sqrt(3)
theorem simplify_and_evaluate_expr : expr x_val = Real.sqrt 3 := sorry

end simplify_and_evaluate_expr_l807_807639


namespace regular_octagon_interior_angle_l807_807296

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807296


namespace regular_octagon_interior_angle_l807_807278

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807278


namespace lcm_of_210_and_330_l807_807659

def hcf (a b : ℕ) : ℕ := Nat.gcd a b
def lcm (a b : ℕ) : ℕ := Nat.lcm a b

theorem lcm_of_210_and_330 (hcf_210_330 : hcf 210 330 = 30) : lcm 210 330 = 2310 := by
  sorry

end lcm_of_210_and_330_l807_807659


namespace total_pizzas_served_l807_807099

-- Define the conditions
def pizzas_lunch : Nat := 9
def pizzas_dinner : Nat := 6

-- Define the theorem to prove
theorem total_pizzas_served : pizzas_lunch + pizzas_dinner = 15 := by
  sorry

end total_pizzas_served_l807_807099


namespace regular_octagon_interior_angle_l807_807282

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807282


namespace regular_octagon_interior_angle_l807_807284

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807284


namespace angle_measure_l807_807861

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l807_807861


namespace integral_e_ax_sin_bx_correct_integral_e_ax_cos_bx_correct_l807_807128

noncomputable def integral_e_ax_sin_bx (a b x : ℝ) : ℝ :=
  (e^(a * x) / (a^2 + b^2)) * (a * sin (b * x) - b * cos (b * x))

noncomputable def integral_e_ax_cos_bx (a b x : ℝ) : ℝ :=
  (e^(a * x) / (a^2 + b^2)) * (a * cos (b * x) + b * sin (b * x))

theorem integral_e_ax_sin_bx_correct (a b C : ℝ) :
  ∫ e^(a * x) * sin (b * x) dx = integral_e_ax_sin_bx a b x + C :=
sorry

theorem integral_e_ax_cos_bx_correct (a b C : ℝ) :
  ∫ e^(a * x) * cos (b * x) dx = integral_e_ax_cos_bx a b x + C :=
sorry

end integral_e_ax_sin_bx_correct_integral_e_ax_cos_bx_correct_l807_807128


namespace sums_of_integers_have_same_remainder_l807_807985

theorem sums_of_integers_have_same_remainder (n : ℕ) (n_pos : 0 < n) : 
  ∃ (i j : ℕ), (1 ≤ i ∧ i ≤ 2 * n) ∧ (1 ≤ j ∧ j ≤ 2 * n) ∧ i ≠ j ∧ ((i + i) % (2 * n) = (j + j) % (2 * n)) :=
by
  sorry

end sums_of_integers_have_same_remainder_l807_807985


namespace regular_octagon_interior_angle_deg_l807_807380

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807380


namespace b_is_geometric_sum_T_n_l807_807205

-- Definition of the arithmetic sequence {a_n}
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- The conditions given in the problem
variable {a : ℕ → ℚ}
variable {b : ℕ → ℚ}
variable {T : ℕ → ℚ}

-- Given that {a_n} is an arithmetic sequence
axiom a_arithmetic : is_arithmetic_sequence a

-- Conditions of the sequence {a_n}
axiom a1 : a 1 = 3
axiom a2_plus_a4 : a 2 + a 4 = 18

-- Definition of the sequence {b_n}
def b (n : ℕ) : ℚ := (3^n) / n * (a n)

-- Proof (statement only) that {b_n} is a geometric sequence with common ratio 3
theorem b_is_geometric : ∃ r : ℚ, ∀ n : ℕ, b (n + 1) = r * (b n) :=
sorry

-- Definition of the sequence {a_n + b_n}
def c (n : ℕ) : ℚ := a n + b n

-- Proof (statement only) of the sum T_n of the first n terms of the sequence {a_n + b_n}
theorem sum_T_n (n : ℕ) : T n = (3 + 3 * ↑n) * ↑n / 2 + 9 * (3 ^ n - 1) / 2 :=
sorry

end b_is_geometric_sum_T_n_l807_807205


namespace common_number_is_six_l807_807088

theorem common_number_is_six 
  (sum_first_four : ℕ) (avg_first_four : sum_first_four = 4 * 7)
  (sum_last_four : ℕ) (avg_last_four : sum_last_four = 4 * 11)
  (total_sum : ℕ) (avg_all_seven : total_sum = 7 * (66 / 7)) :
  ∃ common_number, common_number = 6 :=
by
  have sum_first_four := 4 * 7
  have sum_last_four := 4 * 11
  have total_sum := 7 * (66 / 7)
  have combined_sum := sum_first_four + sum_last_four
  have common_number := combined_sum - total_sum
  exact ⟨common_number, rfl⟩
  sorry

end common_number_is_six_l807_807088


namespace stockings_total_cost_l807_807618

-- Define a function to compute the total cost given the conditions
def total_cost (n_grandchildren n_children : Nat) 
               (stocking_price discount monogram_cost : Nat) : Nat :=
  let total_stockings := n_grandchildren + n_children
  let discounted_price := stocking_price - (stocking_price * discount / 100)
  let total_stockings_cost := discounted_price * total_stockings
  let total_monogram_cost := monogram_cost * total_stockings
  total_stockings_cost + total_monogram_cost

-- Prove that the total cost calculation is correct given the conditions
theorem stockings_total_cost :
  total_cost 5 4 20 10 5 = 207 :=
by
  -- A placeholder for the proof
  sorry

end stockings_total_cost_l807_807618


namespace each_interior_angle_of_regular_octagon_l807_807350

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807350


namespace regular_octagon_interior_angle_eq_135_l807_807397

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807397


namespace sum_abs_gt_3_lt_8_l807_807694

theorem sum_abs_gt_3_lt_8 : 
  (Set.sum (Set.filter (λx : ℤ, 3 < Int.natAbs x ∧ Int.natAbs x < 8) (Set.univ)) : ℤ) = 0 :=
by
  sorry

end sum_abs_gt_3_lt_8_l807_807694


namespace regular_octagon_interior_angle_l807_807419

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807419


namespace range_of_a_l807_807074

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → x / (x^2 + 3 * x + 1) ≤ a) → a ≥ 1/5 :=
by
  intro h
  sorry

end range_of_a_l807_807074


namespace percentage_of_mixture_X_is_13_333_l807_807635

variable (X Y : ℝ) (P : ℝ)

-- Conditions
def mixture_X_contains_40_percent_ryegrass : Prop := X = 0.40
def mixture_Y_contains_25_percent_ryegrass : Prop := Y = 0.25
def final_mixture_contains_27_percent_ryegrass : Prop := 0.4 * P + 0.25 * (100 - P) = 27

-- The goal
theorem percentage_of_mixture_X_is_13_333
    (h1 : mixture_X_contains_40_percent_ryegrass X)
    (h2 : mixture_Y_contains_25_percent_ryegrass Y)
    (h3 : final_mixture_contains_27_percent_ryegrass P) :
  P = 200 / 15 := by
  sorry

end percentage_of_mixture_X_is_13_333_l807_807635


namespace regular_octagon_angle_l807_807304

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807304


namespace select_representatives_l807_807137

theorem select_representatives : 
  (Nat.choose 5 2 * Nat.choose 4 2) + (Nat.choose 5 3 * Nat.choose 4 1) = 100 := 
by
  sorry

end select_representatives_l807_807137


namespace interior_angle_regular_octagon_l807_807500

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807500


namespace stockings_total_cost_l807_807616

-- Defining the conditions
def total_stockings : ℕ := 9
def original_price_per_stocking : ℝ := 20
def discount_rate : ℝ := 0.10
def monogramming_cost_per_stocking : ℝ := 5

-- Calculate the total cost of stockings
theorem stockings_total_cost :
  total_stockings * ((original_price_per_stocking * (1 - discount_rate)) + monogramming_cost_per_stocking) = 207 := 
by
  sorry

end stockings_total_cost_l807_807616


namespace fewer_cans_collected_today_than_yesterday_l807_807942

theorem fewer_cans_collected_today_than_yesterday :
  let sarah_yesterday := 50
  let lara_yesterday := sarah_yesterday + 30
  let sarah_today := 40
  let lara_today := 70
  let total_yesterday := sarah_yesterday + lara_yesterday
  let total_today := sarah_today + lara_today
  total_yesterday - total_today = 20 :=
by
  sorry

end fewer_cans_collected_today_than_yesterday_l807_807942


namespace restore_grid_l807_807758

noncomputable def grid_values (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 9 < 12 ∧
  3 + 1 < 12 ∧ 3 + 5 < 12 ∧
  B + 3 < 12 ∧ B + 4 < 12 ∧
  B + 7 < 12 ∧ 
  C + 4 < 12 ∧
  C + 7 < 12 ∧ 
  D + 9 < 12 ∧
  D + 7 < 12 ∧
  D + 5 < 12

def even_numbers_erased (A B C D : ℕ) : Prop :=
  A ∈ {8} ∧ B ∈ {6} ∧ C ∈ {4} ∧ D ∈ {2}

theorem restore_grid :
  ∃ (A B C D : ℕ), grid_values A B C D ∧ even_numbers_erased A B C D :=
by {
  use [8, 6, 4, 2],
  dsimp [grid_values, even_numbers_erased],
  exact ⟨⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩, 
         ⟨rfl, rfl, rfl, rfl⟩⟩
}

end restore_grid_l807_807758


namespace angle_AOC_value_l807_807200

def angle_AOB : ℝ := 60 -- angle AOB is 60 degrees
def angle_BOC := (1/2) * angle_AOB -- angle BOC is half of angle AOB

def angle_AOC (outside : Bool) : ℝ :=
if outside then angle_AOB + angle_BOC else angle_BOC

theorem angle_AOC_value (outside : Bool) : 
  angle_AOC outside = if outside then 90 else 30 :=
by sorry

end angle_AOC_value_l807_807200


namespace solve_for_alpha_l807_807553

variables (α β γ δ : ℝ)

theorem solve_for_alpha (h : α + β + γ + δ = 360) : α = 360 - β - γ - δ :=
by sorry

end solve_for_alpha_l807_807553


namespace Elena_recipe_multiple_l807_807140

/-- 
Suppose Elena's bread recipe calls for 8 ounces of butter for each 14 cups of flour used. She needs to make a certain multiple of the original recipe. 
If 12 ounces of butter is used, then 56 cups of flour are needed. How many times the original recipe does she need to make?
-/
theorem Elena_recipe_multiple 
  (butter_original : ℚ) (flour_original : ℚ) 
  (butter_used : ℚ) (flour_needed : ℚ) 
  (h_butter_original : butter_original = 8) 
  (h_flour_original : flour_original = 14) 
  (h_butter_used : butter_used = 12)
  (h_flour_needed : flour_needed = 56) :
  (flour_needed / flour_original) / (butter_used / butter_original) = 2.666 :=
by
  rw [←h_butter_original, ←h_flour_original, ←h_butter_used, ←h_flour_needed]
  sorry

end Elena_recipe_multiple_l807_807140


namespace transformation_result_l807_807129

open Matrix

variables (S : Vector3 → Vector3)
variables (a b : ℝ) (u v : Vector3)

def linear_property := ∀ (a b : ℝ) (u v : Vector3), S (a • u + b • v) = a • S u + b • S v
def cross_product_property := ∀ (u v : Vector3), S (u × v) = S u × S v
def S_vector1 := S ⟨9, 3, 4⟩ = ⟨6, -2, 9⟩
def S_vector2 := S ⟨-3, 9, 4⟩ = ⟨6, 9, -2⟩

theorem transformation_result 
  (h1 : linear_property S)
  (h2 : cross_product_property S)
  (h3 : S_vector1 S)
  (h4 : S_vector2 S) :
  S ⟨6, 12, 8⟩ = ⟨12, 12, 12⟩ :=
sorry

end transformation_result_l807_807129


namespace common_tangent_l807_807526

-- Define the functions f and g
def f (x : ℝ) : ℝ := (1 / (2 * Real.exp 1)) * x^2
def g (a x : ℝ) : ℝ := a * Real.log x

-- Derivatives of the functions
def f' (x : ℝ) : ℝ := x / (Real.exp 1)
def g' (a x : ℝ) : ℝ := a / x

-- Statement of the problem
theorem common_tangent (a s : ℝ) (hs : 0 < s) (ha : a > 0) :
  (f' s = g' a s) ∧ (f s = g a s) → a = 1 :=
by
  sorry

end common_tangent_l807_807526


namespace trapezoid_sum_of_nonparallel_sides_equals_parallel_side_l807_807109

open EuclideanGeometry

-- Define the elements of the triangle ABC
variables {A B C : Point} (ABC_triangle : is_triangle A B C)

-- Define the incenter I and line KL parallel to AB
noncomputable def incenter (A B C : Point) [ABC_triangle : is_triangle A B C] : Point := sorry
noncomputable def line_through_incenter (A B C I : Point) (h : is_incenter I A B C) (AB_parallel : Line → Prop) : Line := sorry

theorem trapezoid_sum_of_nonparallel_sides_equals_parallel_side :
  ∀ (A B C : Point) (ABC_triangle : is_triangle A B C),
  let I := incenter A B C in
  let KL := line_through_incenter A B C I (is_incenter_of_triangle A B C) (parallel AB) in
  cuts_off_trapezoid ABC_triangle KL ∧
  (sum_non_parallel_sides ABC_triangle AK BL = side KL) := sorry

end trapezoid_sum_of_nonparallel_sides_equals_parallel_side_l807_807109


namespace determine_grid_numbers_l807_807722

theorem determine_grid_numbers (A B C D : ℕ) :
  (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧ (1 ≤ D ∧ D ≤ 9) ∧ 
  (A ≠ 1) ∧ (A ≠ 3) ∧ (A ≠ 5) ∧ (A ≠ 7) ∧ (A ≠ 9) ∧ 
  (B ≠ 1) ∧ (B ≠ 3) ∧ (B ≠ 5) ∧ (B ≠ 7) ∧ (B ≠ 9) ∧ 
  (C ≠ 1) ∧ (C ≠ 3) ∧ (C ≠ 5) ∧ (C ≠ 7) ∧ (C ≠ 9) ∧ 
  (D ≠ 1) ∧ (D ≠ 3) ∧ (D ≠ 5) ∧ (D ≠ 7) ∧ (D ≠ 9) ∧ 
  (A ≠ 2 ∧ A ≠ 4 ∧ A ≠ 6 ∧ A ≠ 8 ∨ B ≠ 2 ∧ B ≠ 4 ∧ B ≠ 6 ∧ B ≠ 8 ∨ C ≠ 2 ∧ C ≠ 4 ∧ C ≠ 6 ∧ C ≠ 8 ∨ D ≠ 2 ∧ D ≠ 4 ∧ D ≠ 6 ∧ D ≠ 8) ∧ 
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ 
  (3 + 5 < 12) ∧ (3 + D < 12) ∧ (B + 5 < 12) ∧ 
  (B + C < 12) ∧ (C + 7 < 12) ∧ (B + 7 < 12) 
  → (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) :=
by 
  intros A B C D
  exact sorry

end determine_grid_numbers_l807_807722


namespace interior_angle_regular_octagon_l807_807490

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807490


namespace range_of_a_l807_807209

noncomputable def z1 : ℂ := by {
  let z := (-1 + 5 * complex.I) / (1 + complex.I)
  exact z }

axiom a : ℝ

def z2 (a : ℝ) : ℂ := a - 2 - complex.I

axiom habs_lt : ∃ (a : ℝ), abs (z1 - complex.conj (z2 a)) < abs z1

theorem range_of_a (a : ℝ) (h : 1 < a ∧ a < 7) : 
  abs (z1 - complex.conj (z2 a)) < abs z1 := sorry

end range_of_a_l807_807209


namespace lowest_total_cost_l807_807021

theorem lowest_total_cost (x y z a b c: ℕ) (h1: x < y) (h2: y < z) (h3: a < b) (h4: b < c) : 
  min (ax + by + cz) (min (az + by + cx) (min (ay + bz + cx) (ay + bx + cz))) = az + by + cx := 
sorry

end lowest_total_cost_l807_807021


namespace cans_per_bag_l807_807633

theorem cans_per_bag (bags_on_Saturday bags_on_Sunday total_cans : ℕ) (h_saturday : bags_on_Saturday = 3) (h_sunday : bags_on_Sunday = 4) (h_total : total_cans = 63) :
  (total_cans / (bags_on_Saturday + bags_on_Sunday) = 9) :=
by {
  sorry
}

end cans_per_bag_l807_807633


namespace restore_grid_values_l807_807769

def gridCondition (A B C D : Nat) : Prop :=
  A < 9 ∧ 1 < 9 ∧ 9 < 9 ∧
  3 < 9 ∧ 5 < 9 ∧ D < 9 ∧
  B < 9 ∧ C < 9 ∧ 7 < 9 ∧
  -- Sum of any two numbers in adjacent cells is less than 12
  A + 1 < 12 ∧ A + 3 < 12 ∧
  B + C < 12 ∧ B + 3 < 12 ∧
  D + 2 < 12 ∧ C + 4 < 12 ∧
  -- The grid values are from 1 to 9
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  C ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
  -- Even numbers erased are 2, 4, 6, and 8

theorem restore_grid_values :
  ∃ (A B C D : Nat), gridCondition A B C D ∧ A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  sorry

end restore_grid_values_l807_807769


namespace cans_per_person_on_second_day_l807_807124

theorem cans_per_person_on_second_day :
  ∀ (initial_stock : ℕ) (people_first_day : ℕ) (cans_taken_first_day : ℕ)
    (restock_first_day : ℕ) (people_second_day : ℕ)
    (restock_second_day : ℕ) (total_cans_given : ℕ) (cans_per_person_second_day : ℚ),
    cans_taken_first_day = 1 →
    initial_stock = 2000 →
    people_first_day = 500 →
    restock_first_day = 1500 →
    people_second_day = 1000 →
    restock_second_day = 3000 →
    total_cans_given = 2500 →
    cans_per_person_second_day = total_cans_given / people_second_day →
    cans_per_person_second_day = 2.5 := by
  sorry

end cans_per_person_on_second_day_l807_807124


namespace gumballs_in_packages_l807_807230

theorem gumballs_in_packages (total_gumballs : ℕ) (gumballs_per_package : ℕ) (h1 : total_gumballs = 20) (h2 : gumballs_per_package = 5) :
  total_gumballs / gumballs_per_package = 4 :=
by {
  sorry
}

end gumballs_in_packages_l807_807230


namespace regular_octagon_interior_angle_l807_807412

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807412


namespace regular_octagon_interior_angle_l807_807272

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807272


namespace regular_octagon_interior_angle_l807_807475

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807475


namespace general_term_arithmetic_seq_l807_807101

variable {n : ℕ}

-- Definitions of the sequences and the "delicate sequence" property
def arithmetic_seq (b : ℕ → ℝ) (a : ℝ) (d : ℝ) : Prop :=
  ∀ n : ℕ, b (n + 1) = a + n * d

def is_delicate_seq (b : ℕ → ℝ) (k : ℝ) : Prop :=
  ∀ n : ℕ, let Sn := ∑ i in finset.range n, b (i + 1) in
           let S2n := ∑ i in finset.range (2 * n), b (i + 1) in
           Sn / S2n = k

-- The theorem statement
theorem general_term_arithmetic_seq (b : ℕ → ℝ) (d : ℝ) (k : ℝ) (a : ℝ)
  (h_arith : arithmetic_seq b a d)
  (h_delicate : is_delicate_seq b k)
  (h_a : a = 1) :
  b n = 2 * n.to_ℝ - 1 :=
sorry

end general_term_arithmetic_seq_l807_807101


namespace regular_octagon_interior_angle_l807_807427

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807427


namespace regular_octagon_interior_angle_l807_807330

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807330


namespace regular_octagon_interior_angle_eq_135_l807_807395

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807395


namespace base_of_256_is_6_l807_807955

theorem base_of_256_is_6 :
  ∃ b : ℕ, b^3 ≤ 256 ∧ 256 < b^4 ∧ b = 6 :=
by
  use 6
  split
  { norm_num }
  split
  { norm_num }
  trivial

end base_of_256_is_6_l807_807955


namespace interior_angle_regular_octagon_l807_807370

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807370


namespace regular_octagon_interior_angle_l807_807423

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807423


namespace smallest_initial_number_l807_807601

theorem smallest_initial_number (N : ℕ) (h₁ : N ≤ 999) (h₂ : 27 * N - 240 ≥ 1000) : N = 46 :=
by {
    sorry
}

end smallest_initial_number_l807_807601


namespace area_of_PQ_square_l807_807558

theorem area_of_PQ_square (a b c : ℕ)
  (h1 : a^2 = 144)
  (h2 : b^2 = 169)
  (h3 : a^2 + c^2 = b^2) :
  c^2 = 25 :=
by
  sorry

end area_of_PQ_square_l807_807558


namespace distance_P_Q_l807_807225

def P : (ℝ × ℝ × ℝ) := (-1, 2, -3)
def Q : (ℝ × ℝ × ℝ) := (3, -2, -1)

def distance_between_points (A B : ℝ × ℝ × ℝ) : ℝ :=
  real.sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2 + (B.3 - A.3) ^ 2)

theorem distance_P_Q : distance_between_points P Q = 6 :=
by
  sorry

end distance_P_Q_l807_807225


namespace greatest_integer_difference_l807_807519

theorem greatest_integer_difference (x y : ℤ) (hx : 4 < x ∧ x < 8) (hy : 8 < y ∧ y < 12) :
  ∃ d : ℤ, d = y - x ∧ ∀ z, 4 < z ∧ z < 8 ∧ 8 < y ∧ y < 12 → (y - z ≤ d) :=
sorry

end greatest_integer_difference_l807_807519


namespace regular_octagon_interior_angle_l807_807337

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807337


namespace regular_octagon_interior_angle_l807_807336

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807336


namespace spinner_prob_l807_807066

theorem spinner_prob:
  let sections := 4
  let prob := 1 / sections
  let prob_not_e := 1 - prob
  (prob_not_e * prob_not_e) = 9 / 16 :=
by
  sorry

end spinner_prob_l807_807066


namespace regular_octagon_interior_angle_l807_807237

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807237


namespace interior_angle_regular_octagon_l807_807369

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807369


namespace restore_grid_values_l807_807747

def is_adjacent_sum_lt_12 (A B C D : ℕ) : Prop :=
  (A + 1 < 12) ∧ (A + 9 < 12) ∧
  (3 + 1 < 12) ∧ (3 + 5 < 12) ∧
  (3 + D < 12) ∧ (5 + 1 < 12) ∧ 
  (5 + D < 12) ∧ (B + C < 12) ∧
  (B + 3 < 12) ∧ (B + 5 < 12) ∧
  (C + 5 < 12) ∧ (C + 7 < 12) ∧
  (D + 9 < 12) ∧ (D + 7 < 12)

theorem restore_grid_values : 
  ∃ (A B C D : ℕ), 
  (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) ∧ 
  is_adjacent_sum_lt_12 A B C D := 
by 
  exists 8
  exists 6
  exists 4
  exists 2
  split; [refl, split; [refl, split; [refl, split; [refl, sorry]]]] 

end restore_grid_values_l807_807747


namespace set_intersection_l807_807190

def A (x : ℝ) : Prop := x > 0
def B (x : ℝ) : Prop := x^2 < 4

theorem set_intersection : {x | A x} ∩ {x | B x} = {x | 0 < x ∧ x < 2} := by
  sorry

end set_intersection_l807_807190


namespace range_of_m_l807_807173

-- Definitions used to state conditions of the problem.
def fractional_equation (m x : ℝ) : Prop := (m / (2 * x - 1)) + 2 = 0
def positive_solution (x : ℝ) : Prop := x > 0

-- The Lean 4 theorem statement
theorem range_of_m (m x : ℝ) (h : fractional_equation m x) (hx : positive_solution x) : m < 2 ∧ m ≠ 0 :=
by
  sorry

end range_of_m_l807_807173


namespace break_time_is_30_l807_807702

-- Define constants for conditions
def time_between_stations := 120
def total_travel_time := 270

-- Define the theorem we need to prove
theorem break_time_is_30 :
  ∃ break_time, total_travel_time = time_between_stations + time_between_stations + break_time ∧ break_time = 30 :=
by
  have h: total_travel_time = time_between_stations + time_between_stations + 30 := sorry
  use 30
  split
  case 1 => exact h
  case 2 => refl

end break_time_is_30_l807_807702


namespace original_number_l807_807968

theorem original_number (x : ℕ) (h : ∃ k, 14 * x = 112 * k) : x = 8 :=
sorry

end original_number_l807_807968


namespace outfit_combinations_l807_807944

theorem outfit_combinations (shirts pants hats : ℕ) (h_shirts : shirts = 5) (h_pants : pants = 3) (h_hats : hats = 2) :
  shirts * pants * hats = 30 :=
by
  rw [h_shirts, h_pants, h_hats]
  norm_num
  sorry

end outfit_combinations_l807_807944


namespace regular_octagon_interior_angle_eq_135_l807_807410

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807410


namespace arithmetic_geometric_seq_l807_807007

theorem arithmetic_geometric_seq (A B C D : ℤ) (h_arith_seq : A + C = 2 * B)
  (h_geom_seq : B * D = C * C) (h_frac : 7 * B = 3 * C) (h_posB : B > 0)
  (h_posC : C > 0) (h_posD : D > 0) : A + B + C + D = 76 :=
sorry

end arithmetic_geometric_seq_l807_807007


namespace sufficient_but_not_necessary_condition_l807_807217

theorem sufficient_but_not_necessary_condition (b c : ℝ) :
  (∃ x0 : ℝ, (x0^2 + b * x0 + c) < 0) ↔ (c < 0) ∨ true :=
sorry

end sufficient_but_not_necessary_condition_l807_807217


namespace remaining_DVDs_l807_807957

theorem remaining_DVDs (A B : ℕ) (hA : A = 126) (hB : B = 81) : A - B = 45 :=
by
  rw [hA, hB]
  exact Nat.sub_self.symm

end remaining_DVDs_l807_807957


namespace ratio_of_areas_l807_807577

def S1 := { p : ℝ × ℝ | log (3 + p.1^2 + p.2^2) / log 10 ≤ 1 + log (p.1 + 2 * p.2) / log 10 }
def S2 := { p : ℝ × ℝ | log (4 + p.1^2 + p.2^2) / log 10 ≤ 3 + log (p.1 + 2 * p.2) / log 10 }

theorem ratio_of_areas : 
  (π * 1225^2) / (π * 11^2) = 12410 :=
sorry

end ratio_of_areas_l807_807577


namespace regular_octagon_interior_angle_eq_135_l807_807406

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807406


namespace perimeter_circumradius_ratio_neq_l807_807027

-- Define the properties for the equilateral triangle
def Triangle (A K R P : ℝ) : Prop :=
  P = 3 * A ∧ K = A^2 * Real.sqrt 3 / 4 ∧ R = A * Real.sqrt 3 / 3

-- Define the properties for the square
def Square (b k r p : ℝ) : Prop :=
  p = 4 * b ∧ k = b^2 ∧ r = b * Real.sqrt 2 / 2

-- Main statement to prove
theorem perimeter_circumradius_ratio_neq 
  (A b K R P k r p : ℝ)
  (hT : Triangle A K R P) 
  (hS : Square b k r p) :
  P / p ≠ R / r := 
by
  rcases hT with ⟨hP, hK, hR⟩
  rcases hS with ⟨hp, hk, hr⟩
  sorry

end perimeter_circumradius_ratio_neq_l807_807027


namespace area_of_inscribed_rectangle_l807_807628

-- Define the given data and prove the area of rectangle ABCD
theorem area_of_inscribed_rectangle :
  ∀ (ABCD PQR : Type) 
    (AD AB CD DC PR PQ QS PS : ℝ),
    AD = CD → AB = DC → AB = 2 * AD → PR = 12 → PQ = 8 →
    (PS + QS = PR) →
    let altitude_from_Q_to_PR := 8 in
    altitude_from_Q_to_PR = PQ →
    AD * AB = 18 :=
by
  intros ABCD PQR AD AB CD DC PR PQ QS PS h1 h2 h3 h4 h5 h6 h7
  sorry

end area_of_inscribed_rectangle_l807_807628


namespace probability_xyz_72_l807_807920

noncomputable def dice_probability : ℚ :=
  let outcomes : Finset (ℕ × ℕ × ℕ) := 
    {(a, b, c) | a ∈ {1, 2, 3, 4, 5, 6}, b ∈ {1, 2, 3, 4, 5, 6}, c ∈ {1, 2, 3, 4, 5, 6}}.toFinset
  let favorable_outcomes := 
    outcomes.filter (λ (abc : ℕ × ℕ × ℕ), abc.1 * abc.2 * abc.3 = 72)
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ)

theorem probability_xyz_72 : dice_probability = 1/24 :=
  sorry

end probability_xyz_72_l807_807920


namespace angle_solution_l807_807877

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l807_807877


namespace intersection_and_chord_length_l807_807684

noncomputable def curve1Cartesian : Prop := 
  ∀ (x y : ℝ), x^2 + y^2 - 4*x = 0 ↔ ∃ (θ : ℝ), x = 4 * Real.cos θ ∧ y = 4 * Real.sin θ

noncomputable def curve2Cartesian : Prop := 
  ∀ (x y t : ℝ), (x = 3 + 4*t ∧ y = 2 + 3*t) ↔ 3*x - 4*y - 1 = 0

def distance_from_center_to_line (x_center y_center : ℝ) (line : ℝ → ℝ → Prop) : ℝ :=
  abs (3*x_center - 4*y_center - 1) / Real.sqrt (3^2 + 4^2)

def is_intersecting (C1 C2 : ℝ → ℝ → Prop) : Prop :=
  let d := distance_from_center_to_line 2 0 (λ x y, 3*x - 4*y - 1) in
    d < 2

noncomputable def chord_length (radius distance : ℝ) : ℝ :=
  2 * Real.sqrt (radius^2 - distance^2)

theorem intersection_and_chord_length :
  (∀ (ρ θ : ℝ), ρ = 4 * Real.cos θ ↔ ∃ (x y : ℝ), curve1Cartesian x y) ∧
  (∀ (x y t : ℝ), (x = 3 + 4*t ∧ y = 2 + 3*t) ↔ curve2Cartesian x y t) ∧
  is_intersecting curve1Cartesian curve2Cartesian ∧
  chord_length 2 1 = 2 * Real.sqrt 3 :=
by sorry

end intersection_and_chord_length_l807_807684


namespace determine_grid_numbers_l807_807718

theorem determine_grid_numbers (A B C D : ℕ) :
  (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧ (1 ≤ D ∧ D ≤ 9) ∧ 
  (A ≠ 1) ∧ (A ≠ 3) ∧ (A ≠ 5) ∧ (A ≠ 7) ∧ (A ≠ 9) ∧ 
  (B ≠ 1) ∧ (B ≠ 3) ∧ (B ≠ 5) ∧ (B ≠ 7) ∧ (B ≠ 9) ∧ 
  (C ≠ 1) ∧ (C ≠ 3) ∧ (C ≠ 5) ∧ (C ≠ 7) ∧ (C ≠ 9) ∧ 
  (D ≠ 1) ∧ (D ≠ 3) ∧ (D ≠ 5) ∧ (D ≠ 7) ∧ (D ≠ 9) ∧ 
  (A ≠ 2 ∧ A ≠ 4 ∧ A ≠ 6 ∧ A ≠ 8 ∨ B ≠ 2 ∧ B ≠ 4 ∧ B ≠ 6 ∧ B ≠ 8 ∨ C ≠ 2 ∧ C ≠ 4 ∧ C ≠ 6 ∧ C ≠ 8 ∨ D ≠ 2 ∧ D ≠ 4 ∧ D ≠ 6 ∧ D ≠ 8) ∧ 
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ 
  (3 + 5 < 12) ∧ (3 + D < 12) ∧ (B + 5 < 12) ∧ 
  (B + C < 12) ∧ (C + 7 < 12) ∧ (B + 7 < 12) 
  → (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) :=
by 
  intros A B C D
  exact sorry

end determine_grid_numbers_l807_807718


namespace machine_B_multiple_l807_807063

theorem machine_B_multiple (x m : ℕ) (h1 : (x / 10)) (h2 : (m * x / 5)) (h3 : (x / 2)) :
  m = 2 :=
sorry

end machine_B_multiple_l807_807063


namespace simplify_fraction_product_l807_807638

theorem simplify_fraction_product : 
  (270 / 24) * (7 / 210) * (6 / 4) = 4.5 :=
by
  sorry

end simplify_fraction_product_l807_807638


namespace correct_grid_l807_807726

def A := 8
def B := 6
def C := 4
def D := 2

def grid := [[A, 1, 9],
             [3, 5, D],
             [B, C, 7]]

theorem correct_grid :
  (A + 1 < 12) ∧ (A + 3 < 12) ∧ (1 + 9 < 12) ∧
  (1 + 5 < 12) ∧ (3 + 5 < 12) ∧ (3 + B < 12) ∧
  (5 + D < 12) ∧ (5 + C < 12) ∧ (9 + D < 12) ∧
  (B + C < 12) ∧ (C + 7 < 12) :=
by
  -- This is to provide a sketch dummy theorem, we'd prove each step here  
  sorry

end correct_grid_l807_807726


namespace total_number_of_friends_l807_807668

-- definitions for the conditions
def total_bill : ℝ := 150
def silas_paying_half : ℝ := total_bill / 2
def remaining_bill : ℝ := total_bill - silas_paying_half
def tip_percent : ℝ := 0.10
def tip_amount : ℝ := tip_percent * total_bill
def total_with_tip : ℝ := remaining_bill + tip_amount
def amount_one_friend_pays : ℝ := 18
def number_of_friends_splitting : ℝ := total_with_tip / amount_one_friend_pays

-- Proof Problem Statement in Lean
theorem total_number_of_friends : ℝ :=
  number_of_friends_splitting + 1 = 6 := sorry

end total_number_of_friends_l807_807668


namespace interior_angle_regular_octagon_l807_807367

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807367


namespace solve_equation_l807_807642

/-- 
  Given the equation:
    ∀ x, (x = 2 ∨ (3 < x ∧ x < 4)) ↔ (⌊(1/x) * ⌊x⌋^2⌋ = 2),
  where ⌊u⌋ represents the greatest integer less than or equal to u.
-/
theorem solve_equation (x : ℝ) : (x = 2 ∨ (3 < x ∧ x < 4)) ↔ ⌊(1/x) * ⌊x⌋^2⌋ = 2 := 
sorry

end solve_equation_l807_807642


namespace train_speed_kmph_l807_807108

def length_of_train : ℝ := 120
def time_to_cross_bridge : ℝ := 17.39860811135109
def length_of_bridge : ℝ := 170

theorem train_speed_kmph : 
  (length_of_train + length_of_bridge) / time_to_cross_bridge * 3.6 = 60 := 
by
  sorry

end train_speed_kmph_l807_807108


namespace regular_octagon_interior_angle_l807_807274

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807274


namespace line_through_points_l807_807671

theorem line_through_points (x1 y1 x2 y2 : ℝ) (h1 : (x1, y1) = (2, 8)) (h2 : (x2, y2) = (5, 2)) :
  ∃ m b : ℝ, (∀ x, y = m * x + b → (x, y) = (2,8) ∨ (x, y) = (5, 2)) ∧ (m + b = 10) :=
by
  sorry

end line_through_points_l807_807671


namespace man_is_older_by_l807_807090

theorem man_is_older_by :
  ∀ (M S : ℕ), S = 22 → (M + 2) = 2 * (S + 2) → (M - S) = 24 :=
by
  intros M S h1 h2
  sorry

end man_is_older_by_l807_807090


namespace interior_angle_regular_octagon_l807_807446

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807446


namespace mildred_heavier_than_carol_l807_807606

def mildred_weight : ℕ := 59
def carol_weight : ℕ := 9

theorem mildred_heavier_than_carol : mildred_weight - carol_weight = 50 := 
by
  sorry

end mildred_heavier_than_carol_l807_807606


namespace sergeant_distance_travel_l807_807080

noncomputable def sergeant_distance (x k : ℝ) : ℝ :=
  let t₁ := 1 / (x * (k - 1))
  let t₂ := 1 / (x * (k + 1))
  let t := t₁ + t₂
  let d := k * 4 / 3
  d

theorem sergeant_distance_travel (x k : ℝ) (h1 : (4 * k) / (k^2 - 1) = 4 / 3) :
  sergeant_distance x k = 8 / 3 := by
  sorry

end sergeant_distance_travel_l807_807080


namespace cooling_time_l807_807114

theorem cooling_time 
  (drive_rate : ℕ) (cool_time : ℕ)
  (rate : drive_rate = 8)
  (t1 : 40 = 5 * drive_rate)
  (total_distance : 88)
  (total_time : 13)
  (drive_total_no_cool : total_time * drive_rate = 104) 
  (distance_with_cool : drive_total_no_cool - total_distance = 16)
  : cool_time = 2 := 
sorry

end cooling_time_l807_807114


namespace Q_value_ratio_l807_807131

noncomputable def g (x : ℂ) : ℂ := x^2009 + 19*x^2008 + 1

noncomputable def roots : Fin 2009 → ℂ := sorry -- Define distinct roots s1, s2, ..., s2009

noncomputable def Q (z : ℂ) : ℂ := sorry -- Define the polynomial Q of degree 2009

theorem Q_value_ratio :
  (∀ j : Fin 2009, Q (roots j + 2 / roots j) = 0) →
  (Q (2) / Q (-2) = 361 / 400) :=
sorry

end Q_value_ratio_l807_807131


namespace interior_angle_regular_octagon_l807_807431

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807431


namespace interior_angle_regular_octagon_l807_807462

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807462


namespace bernoulli_trial_theorem_l807_807115

noncomputable def bernoulli_trial_prob (p : ℝ) (n : ℕ) (m : ℕ) (h : m > n) : ℝ :=
  p * (1 - p)^(m - 1)

theorem bernoulli_trial_theorem (p : ℝ) (n : ℕ) (m : ℕ) (h : m > n) (h_p : 0 < p ∧ p < 1) :
  let q := 1 - p in
  (q = 1 - p) →
  (n ≥ 1) →
  (m > n) →
  bernoulli_trial_prob p n m h = p * (1 - p)^(m - 1) := 
sorry

end bernoulli_trial_theorem_l807_807115


namespace angle_measure_l807_807859

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l807_807859


namespace ellipse_standard_equation_l807_807693

theorem ellipse_standard_equation (a c : ℝ) (h1 : a^2 = 13) (h2 : c^2 = 12) :
  (∃ b : ℝ, b^2 = a^2 - c^2 ∧ 
    ((∀ x y : ℝ, (x^2 / 13 + y^2 = 1)) ∨ (∀ x y : ℝ, (x^2 + y^2 / 13 = 1)))) :=
by
  sorry

end ellipse_standard_equation_l807_807693


namespace vectors_perpendicular_angle_is_120_degrees_l807_807228

variable (a b : EuclideanSpace ℝ (Fin 3))

axiom norm_a : ∥a∥ = 2
axiom norm_b : ∥b∥ = 4
axiom dot_prod_cond : (a - b) ⬝ b = -20

noncomputable def angle_between_vectors : ℝ := real.acos ((a ⬝ b) / (∥a∥ * ∥b∥))

theorem vectors_perpendicular : (a + b) ⬝ a = 0 :=
by
  have h_dot : a ⬝ b = -4 := sorry
  calc
    (a + b) ⬝ a = a ⬝ a + a ⬝ b : by rw [inner_add_right]
    ... = ∥a∥^2 + a ⬝ b : by rw [real_inner_self_eq_norm_sq]
    ... = 2^2 - 4 : by rw [norm_a, h_dot]
    ... = 4 - 4 : by norm_num
    ... = 0 : by norm_num

theorem angle_is_120_degrees : angle_between_vectors a b = real.pi * 2 / 3 :=
by
  have h_dot : a ⬝ b = -4 := sorry
  have h_normab : ∥a∥ * ∥b∥ = 2 * 4 := by rw [norm_a, norm_b]
  have h_cos : real.cos (real.pi * 2 / 3) = -1 / 2 := by rw [real.cos_pi_mul_two_div_three]
  rw [angle_between_vectors, h_dot, h_normab, real.div_mul_eq_div_mul_one_div],
  exact h_cos

end vectors_perpendicular_angle_is_120_degrees_l807_807228


namespace min_value_2a_plus_b_l807_807515

theorem min_value_2a_plus_b {A B : Type} (h1 : P(A) = 1 / a) (h2 : P(B) = 2 / b)
    (h3 : a > 0) (h4 : b > 0) (h5 : P(A) + P(B) = 1) : 
    ∀ a b : ℝ, 2 * a + b ≥ 8 := by
    sorry

end min_value_2a_plus_b_l807_807515


namespace find_k_parallel_lines_l807_807224

theorem find_k_parallel_lines (k: ℝ) (x y : ℝ):
  ( (k-3)*x + (4-k)*y + 1 = 0) ∧ (2*(k-3)*x - 2*y + 3 = 0) ∧ 
  (λ k: ℝ, -2*(k-3) = 2*(4-k)*(k-3)) -> (k = 3 ∨ k = 5) := 
begin
    sorry
end

end find_k_parallel_lines_l807_807224


namespace each_interior_angle_of_regular_octagon_l807_807342

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807342


namespace angle_measure_l807_807906

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807906


namespace probability_greater_than_two_on_three_dice_l807_807549

theorem probability_greater_than_two_on_three_dice :
  (4 / 6 : ℚ) ^ 3 = (8 / 27 : ℚ) :=
by
  sorry

end probability_greater_than_two_on_three_dice_l807_807549


namespace regular_octagon_interior_angle_l807_807242

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807242


namespace regular_octagon_interior_angle_l807_807301

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807301


namespace probability_both_classes_l807_807603

theorem probability_both_classes (total_students: ℕ) (french_students: ℕ) (spanish_students: ℕ) :
  total_students = 30 ∧
  french_students = 20 ∧
  spanish_students = 24 →
  (let both_classes_students := french_students + spanish_students - total_students in
   let only_french_students := french_students - both_classes_students in
   let only_spanish_students := spanish_students - both_classes_students in
   let total_ways := Nat.choose total_students 2 in
   let ways_only_french := Nat.choose only_french_students 2 in
   let ways_only_spanish := Nat.choose only_spanish_students 2 in
   let unfavorable_ways := ways_only_french + ways_only_spanish in
   let favorable_prob := 1 - (unfavorable_ways / total_ways) in
   favorable_prob = 25 / 29)
:= by {
  sorry
}

end probability_both_classes_l807_807603


namespace correct_grid_l807_807729

def A := 8
def B := 6
def C := 4
def D := 2

def grid := [[A, 1, 9],
             [3, 5, D],
             [B, C, 7]]

theorem correct_grid :
  (A + 1 < 12) ∧ (A + 3 < 12) ∧ (1 + 9 < 12) ∧
  (1 + 5 < 12) ∧ (3 + 5 < 12) ∧ (3 + B < 12) ∧
  (5 + D < 12) ∧ (5 + C < 12) ∧ (9 + D < 12) ∧
  (B + C < 12) ∧ (C + 7 < 12) :=
by
  -- This is to provide a sketch dummy theorem, we'd prove each step here  
  sorry

end correct_grid_l807_807729


namespace find_grid_values_l807_807750

-- Define the grid and the conditions in Lean
variables (A B C D : ℕ)
assume h1: A, B, C, D ∈ {2, 4, 6, 8}
assume h2: A + 1 < 12
assume h3: A + 9 < 12
assume h4: A + 3 < 12
assume h5: A + 5 < 12
assume h6: A + D < 12
assume h7: B + C < 12
assume h8: B + 5 < 12
assume h9: C + 7 < 12
assume h10: D + 9 < 12

-- Prove the specific values for A, B, C, and D
theorem find_grid_values : 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by sorry

end find_grid_values_l807_807750


namespace interior_angle_regular_octagon_l807_807460

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807460


namespace regular_octagon_interior_angle_l807_807288

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807288


namespace problem_l807_807654

variable (x y z w : ℚ)

theorem problem
  (h1 : x / y = 7)
  (h2 : z / y = 5)
  (h3 : z / w = 3 / 4) :
  w / x = 20 / 21 :=
by sorry

end problem_l807_807654


namespace probability_xyz_72_l807_807921

noncomputable def dice_probability : ℚ :=
  let outcomes : Finset (ℕ × ℕ × ℕ) := 
    {(a, b, c) | a ∈ {1, 2, 3, 4, 5, 6}, b ∈ {1, 2, 3, 4, 5, 6}, c ∈ {1, 2, 3, 4, 5, 6}}.toFinset
  let favorable_outcomes := 
    outcomes.filter (λ (abc : ℕ × ℕ × ℕ), abc.1 * abc.2 * abc.3 = 72)
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ)

theorem probability_xyz_72 : dice_probability = 1/24 :=
  sorry

end probability_xyz_72_l807_807921


namespace grid_solution_l807_807736

-- Define the grid and conditions
variables (A B C D : ℕ)

-- Grid values with the constraints
def grid_valid (A B C D : ℕ) :=
  A ≠ 1 ∧ A ≠ 9 ∧ A ≠ 3 ∧ A ≠ 5 ∧ A ≠ 7 ∧
  B ≠ 1 ∧ B ≠ 9 ∧ B ≠ 3 ∧ B ≠ 5 ∧ B ≠ 7 ∧
  C ≠ 1 ∧ C ≠ 9 ∧ C ≠ 3 ∧ C ≠ 5 ∧ C ≠ 7 ∧
  D ≠ 1 ∧ D ≠ 9 ∧ D ≠ 3 ∧ D ≠ 5 ∧ D ≠ 7 ∧
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ (A + 3 < 12) ∧
  (D + 9 < 12) ∧ (D + 5 < 12) ∧ 
  (B + 3 < 12) ∧ (B + C < 12) ∧
  (C + 7 < 12)

-- Given the conditions, we need to prove the values of A, B, C, and D
theorem grid_solution : 
  grid_valid 8 6 4 2 ∧ 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  split
  · -- to show grid_valid 8 6 4 2
    sorry
    
  · -- to show the values of A, B, C, and D
    repeat {split, assumption}

end grid_solution_l807_807736


namespace solution_l807_807218

/-- Given the function f(x) = |x - a| - a/2* ln(x), where a ∈ ℝ, 
we aim to show that if the function has two zero points x1 and x2,
with x1 < x2, then 1 < x1 < a < x2 < a^2. --/

theorem solution (a x1 x2 : ℝ) 
  (h1 : 0 < a)
  (hx1 : x1 < x2)
  (h_eq1 : abs (x1 - a) - (a / 2) * log x1 = 0)
  (h_eq2 : abs (x2 - a) - (a / 2) * log x2 = 0) :
  1 < x1 ∧ x1 < a ∧ a < x2 ∧ x2 < a^2 :=
by
  sorry

end solution_l807_807218


namespace regular_octagon_interior_angle_l807_807280

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807280


namespace regular_octagon_interior_angle_l807_807258

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807258


namespace regular_octagon_interior_angle_l807_807250

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807250


namespace regular_octagon_interior_angle_l807_807232

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807232


namespace angle_supplement_complement_l807_807875

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l807_807875


namespace pipe_A_fill_time_l807_807709

theorem pipe_A_fill_time (x : ℝ) (h1 : ∀ t : ℝ, t = 45) (h2 : ∀ t : ℝ, t = 18) :
  (1/x + 1/45 = 1/18) → x = 30 :=
by {
  -- Proof is omitted
  sorry
}

end pipe_A_fill_time_l807_807709


namespace max_value_of_quadratic_exists_r_for_max_value_of_quadratic_l807_807778

theorem max_value_of_quadratic (r : ℝ) : -7 * r ^ 2 + 50 * r - 20 ≤ 5 :=
by sorry

theorem exists_r_for_max_value_of_quadratic : ∃ r : ℝ, -7 * r ^ 2 + 50 * r - 20 = 5 :=
by sorry

end max_value_of_quadratic_exists_r_for_max_value_of_quadratic_l807_807778


namespace problem_solution_l807_807126
open Complex

noncomputable def problem : ℂ := ∏ k in Finset.range 15 \set{0}, ∏ j in Finset.range 12 \set{0}, (exp (2 * π * Complex.I * j / 13) - exp (2 * π * Complex.I * k / 16))

theorem problem_solution : problem = 4096 := by
  sorry

end problem_solution_l807_807126


namespace angle_supplement_complement_l807_807872

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l807_807872


namespace find_x0_l807_807216

-- Define the function f piecewise
def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 3^(-x) - 2 else real.sqrt x

-- State the theorem to be proved
theorem find_x0 : ∃ x0 : ℝ, f x0 = 1 ∧ (x0 = 1 ∨ x0 = -1) := by
  sorry

end find_x0_l807_807216


namespace vector_computation_equiv_l807_807127

variables (u v w : ℤ × ℤ)

def vector_expr (u v w : ℤ × ℤ) :=
  2 • u + 4 • v - 3 • w

theorem vector_computation_equiv :
  u = (3, -5) →
  v = (-1, 6) →
  w = (2, -4) →
  vector_expr u v w = (-4, 26) :=
by
  intros hu hv hw
  rw [hu, hv, hw]
  dsimp [vector_expr]
  -- The actual proof goes here, but we use 'sorry' to skip it.
  sorry

end vector_computation_equiv_l807_807127


namespace correct_calculation_l807_807053

def calculation_is_correct (a b x y : ℝ) : Prop :=
  (3 * x^2 * y - 2 * y * x^2 = x^2 * y)

theorem correct_calculation :
  ∀ (a b x y : ℝ), calculation_is_correct a b x y :=
by
  intros a b x y
  sorry

end correct_calculation_l807_807053


namespace correct_transformation_l807_807936

theorem correct_transformation (x : ℝ) : 2 * x = 3 * x + 4 → 2 * x - 3 * x = 4 :=
by
  intro h
  exact h

end correct_transformation_l807_807936


namespace point_in_first_quadrant_l807_807003

-- Define the system of equations
def equations (x y : ℝ) : Prop :=
  x + y = 2 ∧ x - y = 1

-- State the theorem
theorem point_in_first_quadrant (x y : ℝ) (h : equations x y) : x > 0 ∧ y > 0 := by
  sorry

end point_in_first_quadrant_l807_807003


namespace angle_measure_is_60_l807_807784

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l807_807784


namespace lateral_surface_area_l807_807691

open Real

-- The sine of the dihedral angle at the lateral edge of a regular quadrilateral pyramid
def sin_dihedral_angle : ℝ := 15 / 17

-- The area of the pyramid's diagonal section
def area_diagonal_section : ℝ := 3 * sqrt 34

-- The statement that we need to prove
theorem lateral_surface_area (sin_dihedral_angle = 15 / 17) (area_diagonal_section = 3 * sqrt 34) : 
  lateral_surface_area = 68 :=
sorry

end lateral_surface_area_l807_807691


namespace locus_points_Z_l807_807615

variables {R : Type*} [LinearOrderR R] [Ring R] [HasOrderedSub R]
          {triangle : Triangle R} {P Q R A B C D E F Z Z0 : Point R}

variables (AB PQ QR RP CD EF : R)
variables (hAB : AB / PQ = CD / QR ∧ AB / PQ = EF / RP)
variables (hPQR : PQR = {P, Q, R})
variables (hSegments : triangle.segmentP = AB ∧ triangle.segmentQ = CD ∧ triangle.segmentR = EF)
variables (hZ0 : triangle.interior)
variables (area_triangle : ∀ {X Y Z : Point R}, R)

-- The main statement
theorem locus_points_Z :
  ∃ L : Set (Point R), 
    (∀ Z ∈ L, 
        area_triangle Z A B + area_triangle Z C D + area_triangle Z E F = 
        area_triangle Z0 A B + area_triangle Z0 C D + area_triangle Z0 E F) 
    ∧ (hAB → L = {Z : Point R | Z lies_on (line_through D1 E1 parallel_to Z0)} )
    ∧ (¬hAB → L = {Z : Point R | Z lies_in triangle} )
:= sorry

end locus_points_Z_l807_807615


namespace regular_octagon_interior_angle_l807_807467

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807467


namespace bella_items_l807_807991

theorem bella_items (M F D : ℕ) 
  (h1 : M = 60)
  (h2 : M = 2 * F)
  (h3 : F = D + 20) :
  (7 * M + 7 * F + 7 * D) / 5 = 140 := 
by
  sorry

end bella_items_l807_807991


namespace find_d_bound_hundred_d_rounded_l807_807593

noncomputable def b : ℕ → ℝ
| 0     := 3 / 5
| (n+1) := 2 * b n ^ 2 - 1

def sequence_product (n : ℕ) : ℝ :=
  (List.range n).map b |>.foldr (· * ·) 1

theorem find_d_bound (d : ℝ) : 
  (∀ n : ℕ, 0 < n → |sequence_product n| ≤ d / (3 : ℝ) ^ n) →
  d = 171 / 125 := by
  sorry

theorem hundred_d_rounded : 
  100 * (171 / 125) = 136 := by
  norm_num

end find_d_bound_hundred_d_rounded_l807_807593


namespace regular_octagon_interior_angle_l807_807275

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807275


namespace factorial_div_l807_807192

theorem factorial_div (h : fact 7 = 5040) : fact 7 / fact 4 = 210 := by
  sorry

end factorial_div_l807_807192


namespace more_permutations_with_P_l807_807599

open Finset

-- Define the set of all permutations of {1, 2, ..., 2n}
noncomputable def permutations_set (n : ℕ) : Finset (Fin n.succ_perms) :=
  Finset.univ

-- Define property P
def has_property_P (n : ℕ) (π : Fin n.succ.succ_perms) : Prop :=
  ∃ i : Fin (2 * n - 1), |(π : ℕ) - (π.succ : ℕ)| = n

-- Define sets A, B
def set_A (n : ℕ) : Finset (Fin (2 * n) -> ℕ) := {π ∈ permutations_set n | has_property_P n π}
def set_B (n : ℕ) : Finset (Fin (2 * n) -> ℕ) := {π ∈ permutations_set n | ¬ has_property_P n π}

theorem more_permutations_with_P (n : ℕ) (h : 0 < n) : |set_A n| > |set_B n| :=
begin
  sorry
end

end more_permutations_with_P_l807_807599


namespace multiplication_is_valid_l807_807653

-- Define that the three-digit number n = 306
def three_digit_number := 306

-- The multiplication by 1995 should result in the defined product
def valid_multiplication (n : ℕ) := 1995 * n

theorem multiplication_is_valid : valid_multiplication three_digit_number = 1995 * 306 := by
  -- Since we only need the statement, we use sorry here
  sorry

end multiplication_is_valid_l807_807653


namespace angle_supplement_complement_l807_807809

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l807_807809


namespace regular_octagon_interior_angle_l807_807262

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807262


namespace regular_octagon_interior_angle_deg_l807_807381

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807381


namespace Mildred_heavier_than_Carol_l807_807604

-- Definition of weights for Mildred and Carol
def weight_Mildred : ℕ := 59
def weight_Carol : ℕ := 9

-- Definition of how much heavier Mildred is than Carol
def weight_difference : ℕ := weight_Mildred - weight_Carol

-- The theorem stating the difference in weight
theorem Mildred_heavier_than_Carol : weight_difference = 50 := 
by 
  -- Just state the theorem without providing the actual steps (proof skipped)
  sorry

end Mildred_heavier_than_Carol_l807_807604


namespace determine_grid_numbers_l807_807725

theorem determine_grid_numbers (A B C D : ℕ) :
  (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧ (1 ≤ D ∧ D ≤ 9) ∧ 
  (A ≠ 1) ∧ (A ≠ 3) ∧ (A ≠ 5) ∧ (A ≠ 7) ∧ (A ≠ 9) ∧ 
  (B ≠ 1) ∧ (B ≠ 3) ∧ (B ≠ 5) ∧ (B ≠ 7) ∧ (B ≠ 9) ∧ 
  (C ≠ 1) ∧ (C ≠ 3) ∧ (C ≠ 5) ∧ (C ≠ 7) ∧ (C ≠ 9) ∧ 
  (D ≠ 1) ∧ (D ≠ 3) ∧ (D ≠ 5) ∧ (D ≠ 7) ∧ (D ≠ 9) ∧ 
  (A ≠ 2 ∧ A ≠ 4 ∧ A ≠ 6 ∧ A ≠ 8 ∨ B ≠ 2 ∧ B ≠ 4 ∧ B ≠ 6 ∧ B ≠ 8 ∨ C ≠ 2 ∧ C ≠ 4 ∧ C ≠ 6 ∧ C ≠ 8 ∨ D ≠ 2 ∧ D ≠ 4 ∧ D ≠ 6 ∧ D ≠ 8) ∧ 
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ 
  (3 + 5 < 12) ∧ (3 + D < 12) ∧ (B + 5 < 12) ∧ 
  (B + C < 12) ∧ (C + 7 < 12) ∧ (B + 7 < 12) 
  → (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) :=
by 
  intros A B C D
  exact sorry

end determine_grid_numbers_l807_807725


namespace P_inter_Q_empty_l807_807586

def P := {y : ℝ | ∃ x : ℝ, y = x^2}
def Q := {(x, y) : ℝ × ℝ | y = x^2}

theorem P_inter_Q_empty : P ∩ Q = ∅ :=
by sorry

end P_inter_Q_empty_l807_807586


namespace interior_angle_regular_octagon_l807_807430

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807430


namespace solve_matrix_eq_l807_807153

noncomputable def matrixP (i j : ℕ): ℝ :=
  match i, j with
  | 0, 0 => 2
  | 0, 1 => -2/3
  | 1, 0 => 3
  | 1, 1 => -4
  | _, _ => 0

theorem solve_matrix_eq :
  let P := λ i j, matrixP i j,
  v1 := (4, 0),
  v2 := (2, -3),
  target1 := (8, 12),
  target2 := (2, -6) in
  (P 0 0 * v1.1 + P 0 1 * v1.2 = target1.1) ∧
  (P 1 0 * v1.1 + P 1 1 * v1.2 = target1.2) ∧
  (P 0 0 * v2.1 + P 0 1 * v2.2 = target2.1) ∧
  (P 1 0 * v2.1 + P 1 1 * v2.2 = target2.2) :=
by {
  sorry -- proof goes here
}

end solve_matrix_eq_l807_807153


namespace gdp_scientific_notation_l807_807560

theorem gdp_scientific_notation (gdp : ℝ) (h : gdp = 338.8 * 10^9) : gdp = 3.388 * 10^10 :=
by sorry

end gdp_scientific_notation_l807_807560


namespace meet_time_opposite_directions_catch_up_time_same_direction_l807_807609

def length_of_track := 440
def speed_A := 5
def speed_B := 6

theorem meet_time_opposite_directions :
  (length_of_track / (speed_A + speed_B)) = 40 :=
by
  sorry

theorem catch_up_time_same_direction :
  (length_of_track / (speed_B - speed_A)) = 440 :=
by
  sorry

end meet_time_opposite_directions_catch_up_time_same_direction_l807_807609


namespace angle_complement_supplement_l807_807827

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l807_807827


namespace regular_octagon_interior_angle_l807_807480

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807480


namespace regular_octagon_interior_angle_l807_807426

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807426


namespace lateral_surface_area_of_pyramid_l807_807688

theorem lateral_surface_area_of_pyramid
  (sin_alpha : ℝ)
  (A_section : ℝ)
  (h1 : sin_alpha = 15 / 17)
  (h2 : A_section = 3 * Real.sqrt 34) :
  ∃ A_lateral : ℝ, A_lateral = 68 :=
sorry

end lateral_surface_area_of_pyramid_l807_807688


namespace interior_angle_regular_octagon_l807_807450

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807450


namespace matrix_vector_scaling_l807_807590

variables (B : Matrix (Fin 2) (Fin 2) ℝ)

def v : ℝ × ℝ := (3, -1)
def w : ℝ × ℝ := (12, -4)
noncomputable def pow_four (B : Matrix (Fin 2) (Fin 2) ℝ) := B ^ 4

theorem matrix_vector_scaling (h : B.vecMul v = w) :
  pow_four B.vecMul v = (768, -256) := sorry

end matrix_vector_scaling_l807_807590


namespace regular_octagon_angle_l807_807320

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807320


namespace tax_rate_on_clothing_l807_807613

variable (T : ℝ) -- Total amount spent before taxes
variable (C : ℝ) -- Tax rate on clothing

-- Conditions
def spent_on_clothing := 0.60 * T
def spent_on_food := 0.10 * T
def spent_on_other := 0.30 * T
def tax_on_clothing := (C / 100) * spent_on_clothing
def tax_on_other := 0.08 * spent_on_other
def total_tax := 0.048 * T

theorem tax_rate_on_clothing :
  tax_on_clothing + tax_on_other = total_tax → C = 4 :=
by
  intro h
  sorry

end tax_rate_on_clothing_l807_807613


namespace angle_measure_l807_807797

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l807_807797


namespace angle2_degree_l807_807204

def degrees := Real

def vertical_angles (a1 a2 : degrees) : Prop := a1 = a2

def complementary_angle (a : degrees) : degrees := 180 - a

theorem angle2_degree (a1 a2 : degrees)
  (h1 : vertical_angles a1 a2)
  (h2 : complementary_angle a1 = 79 + 32/60) :
  a2 = 100 + 28/60 :=
by
  sorry

end angle2_degree_l807_807204


namespace david_more_pushups_l807_807133

theorem david_more_pushups (d z : ℕ) (h1 : d = 51) (h2 : d + z = 53) : d - z = 49 := by
  sorry

end david_more_pushups_l807_807133


namespace interior_angle_regular_octagon_l807_807432

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807432


namespace solve_equation_l807_807641

theorem solve_equation:
  ∀ x y z : ℝ, x^2 + 5 * y^2 + 5 * z^2 - 4 * x * z - 2 * y - 4 * y * z + 1 = 0 → 
    x = 4 ∧ y = 1 ∧ z = 2 :=
by
  intros x y z h
  sorry

end solve_equation_l807_807641


namespace image_square_transformation_l807_807585

-- Definitions for points and transformations
structure Point (α : Type) [Add α] [Mul α] :=
  (x y : α)

def O : Point ℝ := ⟨0, 0⟩
def A : Point ℝ := ⟨1, 0⟩
def B : Point ℝ := ⟨1, 1⟩
def C : Point ℝ := ⟨0, 1⟩

-- Transformation definitions
def u (p : Point ℝ) : ℝ := p.x ^ 2 + p.y ^ 2
def v (p : Point ℝ) : ℝ := p.x - p.y

-- Image points in uv-plane
def O_uv := ⟨u O, v O⟩
def A_uv := ⟨u A, v A⟩
def B_uv := ⟨u B, v B⟩
def C_uv := ⟨u C, v C⟩

-- Target points in uv-plane to be proved bounded
def P₀_uv := (0, 0) : ℝ × ℝ
def P₁_uv := (1, 1) : ℝ × ℝ
def P₂_uv := (2, 0) : ℝ × ℝ
def P₃_uv := (1, -1) : ℝ × ℝ

-- Main theorem statement
theorem image_square_transformation :
  {O_uv, A_uv, B_uv, C_uv} = {P₀_uv, P₁_uv, P₂_uv, P₃_uv} :=
by
  sorry

end image_square_transformation_l807_807585


namespace regular_octagon_interior_angle_l807_807468

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807468


namespace sequence_sum_50_l807_807220

/--
Given the sequence \(S_n = 1-2+3-4+\cdots+(-1)^{n-1} \cdot n\),
we want to prove that \(S_{50} = -25\).
-/
theorem sequence_sum_50 : 
  let S : ℕ → ℤ := λ n, ∑ i in finset.range n, (-1) ^ (i + 1) * (i + 1)
  S 50 = -25 :=
begin
  sorry  -- Placeholder for the proof
end

end sequence_sum_50_l807_807220


namespace angle_solution_l807_807881

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l807_807881


namespace regular_octagon_interior_angle_l807_807474

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807474


namespace friends_attended_l807_807081

theorem friends_attended (total_guests bride_couples groom_couples : ℕ)
                         (bride_guests groom_guests family_guests friends : ℕ)
                         (h1 : total_guests = 300)
                         (h2 : bride_couples = 30)
                         (h3 : groom_couples = 30)
                         (h4 : bride_guests = bride_couples * 2)
                         (h5 : groom_guests = groom_couples * 2)
                         (h6 : family_guests = bride_guests + groom_guests)
                         (h7 : friends = total_guests - family_guests) :
  friends = 180 :=
by sorry

end friends_attended_l807_807081


namespace inequality_proof_l807_807165

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + (3/(a * b + b * c + c * a)) ≥ 6/(a + b + c) := 
sorry

end inequality_proof_l807_807165


namespace interior_angle_regular_octagon_l807_807359

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807359


namespace scientific_notation_000073_l807_807664

theorem scientific_notation_000073 : 0.000073 = 7.3 * 10^(-5) := by
  sorry

end scientific_notation_000073_l807_807664


namespace doughnut_machine_completion_time_l807_807076

theorem doughnut_machine_completion_time :
  ∀ (start_time : ℕ) (end_time : ℕ) (maintenance_time : ℕ) (fraction_completed : ℕ),
  start_time = 6 ∧ end_time = 9 ∧ maintenance_time = 45 ∧ fraction_completed = 1/4 →
  let time_taken := end_time - start_time in
  let total_time_without_stop := (1 / fraction_completed) * time_taken in
  let total_time_with_stop := total_time_without_stop + maintenance_time / 60 in
  let completion_time := start_time + total_time_with_stop in
  completion_time = 18 +  3 / 4 :=
sorry

end doughnut_machine_completion_time_l807_807076


namespace restore_grid_values_l807_807767

def gridCondition (A B C D : Nat) : Prop :=
  A < 9 ∧ 1 < 9 ∧ 9 < 9 ∧
  3 < 9 ∧ 5 < 9 ∧ D < 9 ∧
  B < 9 ∧ C < 9 ∧ 7 < 9 ∧
  -- Sum of any two numbers in adjacent cells is less than 12
  A + 1 < 12 ∧ A + 3 < 12 ∧
  B + C < 12 ∧ B + 3 < 12 ∧
  D + 2 < 12 ∧ C + 4 < 12 ∧
  -- The grid values are from 1 to 9
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  C ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
  -- Even numbers erased are 2, 4, 6, and 8

theorem restore_grid_values :
  ∃ (A B C D : Nat), gridCondition A B C D ∧ A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  sorry

end restore_grid_values_l807_807767


namespace circle_tangent_to_line_l807_807686

variable {A B C D B' : Type}

-- Geometry setup: Points, reflections, and circles.
axiom cyclic_quadrilateral (A B C D : Type) : Prop
axiom product_of_opposite_sides (AB CD BC DA : ℝ) : Prop
axiom reflection (B : Type) (AC : Type) : Type

-- Tangency condition.
axiom touches (circle : Type) (line : Type) : Prop

-- Given data.
variables (AC : Type) [cyclic_quadrilateral A B C D]
variables [product_of_opposite_sides AB CD BC DA]
variables (B' := reflection B AC)

-- Statement to be proven.
theorem circle_tangent_to_line (circle_A_B'_D : Type) : touches circle_A_B'_D AC := by sorry

end circle_tangent_to_line_l807_807686


namespace angle_measure_l807_807836

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l807_807836


namespace sin_product_exact_value_l807_807042

open Real

theorem sin_product_exact_value :
  (sin (10 * π / 180) * sin (30 * π / 180) * sin (50 * π / 180) * sin (70 * π / 180)) = 1 / 16 := 
by
  sorry

end sin_product_exact_value_l807_807042


namespace regular_octagon_interior_angle_l807_807470

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807470


namespace confidence_level_for_relationship_l807_807531

-- Define the problem conditions and the target question.
def chi_squared_value : ℝ := 8.654
def critical_value : ℝ := 6.635
def confidence_level : ℝ := 99

theorem confidence_level_for_relationship (h : chi_squared_value > critical_value) : confidence_level = 99 :=
sorry

end confidence_level_for_relationship_l807_807531


namespace process_terminates_with_one_element_in_each_list_final_elements_are_different_l807_807038

-- Define the initial lists
def List1 := [1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56, 61, 66, 71, 76, 81, 86, 91, 96]
def List2 := [4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 99]

-- Predicate to state the termination of the process with exactly one element in each list
theorem process_terminates_with_one_element_in_each_list (List1 List2 : List ℕ):
  ∃ n m, List.length List1 = n ∧ List.length List2 = m ∧ (n = 1 ∧ m = 1) :=
sorry

-- Predicate to state that the final elements in the lists are different
theorem final_elements_are_different (List1 List2 : List ℕ) :
  ∀ a b, a ∈ List1 → b ∈ List2 → (a % 5 = 1 ∧ b % 5 = 4) → a ≠ b :=
sorry

end process_terminates_with_one_element_in_each_list_final_elements_are_different_l807_807038


namespace angle_measure_l807_807850

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807850


namespace angle_complement_supplement_l807_807822

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l807_807822


namespace part_a_part_b_l807_807187
-- Import necessary Lean 4 libraries

-- Definitions for Part (a)
def angle_AOB : ℝ := sorry  -- Assume we have the given angle as a real number
def angle_AOC (angle_AOB : ℝ) : ℝ := 2 * angle_AOB

-- Theorem stating the construction for Part (a)
theorem part_a (angle_AOB : ℝ) : angle_AOC angle_AOB = 2 * angle_AOB := by
  sorry

-- Definitions for Part (b)
def angle_AB1B (angle_AOB : ℝ) : ℝ := angle_AOB / 2

-- Theorem stating the construction for Part (b)
theorem part_b (angle_AOB : ℝ) : angle_AB1B angle_AOB = angle_AOB / 2 := by
  sorry

end part_a_part_b_l807_807187


namespace angle_complement_supplement_l807_807816

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l807_807816


namespace probability_valid_roll_l807_807031

-- Definitions for the problem conditions
def sides_eight_sided : List ℕ := List.range' 1 9  -- Values from 1 to 8
def sides_four_sided : List ℕ := List.range' 1 5  -- Values from 1 to 4

-- Function to calculate the total number of outcomes
def total_outcomes : ℕ := List.length sides_eight_sided * List.length sides_four_sided * List.length sides_eight_sided

-- Function to check if a roll of three dice satisfies the condition
def is_valid_roll (d1 d2 : ℕ) (d3 : ℕ) : Bool :=
  d1 + d2 = d3

-- Count all the favorable outcomes
def count_favorable_outcomes : ℕ :=
  sides_eight_sided.product (sides_four_sided.product sides_eight_sided)
    |>.count (λ roll, is_valid_roll roll.1 roll.2.1 roll.2.2)

-- The statement of the main theorem
theorem probability_valid_roll : count_favorable_outcomes / total_outcomes = 9 / 64 :=
by 
  -- Placeholder for the proof
  sorry

end probability_valid_roll_l807_807031


namespace sumNats_l807_807775

-- Define the set of natural numbers between 29 and 31 inclusive
def NatRange : List ℕ := [29, 30, 31]

-- Define the condition that checks the elements in the range
def isValidNumbers (n : ℕ) : Prop := n ≤ 31 ∧ n > 28

-- Check if all numbers in NatRange are valid
def allValidNumbers : Prop := ∀ n, n ∈ NatRange → isValidNumbers n

-- Define the sum function for the list
def sumList (lst : List ℕ) : ℕ := lst.foldr (.+.) 0

-- The main theorem
theorem sumNats : (allValidNumbers → (sumList NatRange) = 90) :=
by
  sorry

end sumNats_l807_807775


namespace can_partition_l807_807181

theorem can_partition (a : List ℕ) (h1 : a.head = 1)
  (h2 : ∀ i, i < a.length - 1 → a.get i ≤ a.get (i + 1) ∧ a.get (i + 1) ≤ 2 * a.get i)
  (h3 : a.sum % 2 = 0) : ∃ (s1 s2 : List ℕ), s1.sum = s2.sum := 
sorry

end can_partition_l807_807181


namespace card_operation_impossibility_l807_807000

theorem card_operation_impossibility :
  let m₁ := 1037
  let n₁ := 1159
  let m₂ := 611
  let n₂ := 1081
  let gcd := Nat.gcd
  let swap (x y : ℕ) := (y, x)
  let sum_first (x y : ℕ) := (x + y, y)
  let diff_second (x y : ℕ) := (x, (x - y).natAbs)
  -- Define allowed operations
  let operations := [λ p : ℕ × ℕ, swap p.1 p.2,
                          λ p : ℕ × ℕ, sum_first p.1 p.2,
                          λ p : ℕ × ℕ, diff_second p.1 p.2]
  -- Prove impossibility of achieving (611, 1081) starting from (1037, 1159) using allowed operations
  ¬ (∃ (f : List (ℕ × ℕ → ℕ × ℕ)), f ⊆ operations ∧ (f.foldl (λ p op, op p) (m₁, n₁) = (m₂, n₂))) :=
by
  let m₁ := 1037
  let n₁ := 1159
  let m₂ := 611
  let n₂ := 1081
  let gcd := Nat.gcd
  have gcd_m₁_n₁ : gcd m₁ n₁ = 61 := by sorry
  have gcd_m₂_n₂ : gcd m₂ n₂ = 47 := by sorry
  have Hdiff_gcd : gcd m₁ n₁ ≠ gcd m₂ n₂ := by sorry
  assume H : ∃ (f : List (ℕ × ℕ → ℕ × ℕ)), f ⊆ operations ∧ (f.foldl (λ p op, op p) (m₁, n₁) = (m₂, n₂))
  cases H with f Hf
  have : gcd (f.foldl (λ p op, op p) (m₁, n₁)).1 (f.foldl (λ p op, op p) (m₁, n₁)).2 = gcd m₁ n₁ := by sorry
  rw [Hf.2] at this
  exact Hdiff_gcd this

end card_operation_impossibility_l807_807000


namespace one_minus_repeat_eight_l807_807146

theorem one_minus_repeat_eight : 1 - (0.888888...) = 1 / 9 :=
by
  -- Assume repeating decimal representation
  let b := 0.888888...
  -- Given that b is equivalent to 8/9
  have h₁ : b = 8 / 9 := sorry
  -- Subtracting from 1
  have h₂ : 1 - b = 1 - 8 / 9 := sorry
  assume h₃ : 1 - 8 / 9 = 1 / 9
  -- Ultimately, these steps together show
  exact h₃

end one_minus_repeat_eight_l807_807146


namespace regular_octagon_interior_angle_l807_807269

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807269


namespace emily_can_see_emerson_for_20_minutes_l807_807142

-- Define the speeds of Emily and Emerson
def emily_speed : ℝ := 16  -- in miles per hour
def emerson_speed : ℝ := 10  -- in miles per hour

-- Define the initial distances
def initial_distance : ℝ := 1  -- in miles, Emerson ahead of Emily
def final_distance : ℝ := 1  -- in miles, Emerson behind Emily

-- Calculate the total time Emily can see Emerson in hours
def total_time_visible : ℝ := initial_distance / (emily_speed - emerson_speed) + final_distance / (emily_speed - emerson_speed)

-- Convert the total time from hours to minutes
def total_time_visible_minutes : ℝ := total_time_visible * 60

-- The proof statement we need to establish
theorem emily_can_see_emerson_for_20_minutes :
  total_time_visible_minutes = 20 := by
  sorry

end emily_can_see_emerson_for_20_minutes_l807_807142


namespace regular_octagon_interior_angle_l807_807254

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807254


namespace angle_measure_l807_807910

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807910


namespace sin_cos_triple_angle_l807_807201

theorem sin_cos_triple_angle (α : ℝ) :
  (sin α - cos α = 1 / 3) → sin (3 * α) + cos (3 * α) = -25 / 27 :=
by
  intro h
  sorry

end sin_cos_triple_angle_l807_807201


namespace cube_volume_given_face_perimeter_l807_807002

-- Define the perimeter condition
def is_face_perimeter (perimeter : ℝ) (side_length : ℝ) : Prop :=
  4 * side_length = perimeter

-- Define volume computation
def cube_volume (side_length : ℝ) : ℝ :=
  side_length^3

-- Theorem stating the relationship between face perimeter and cube volume
theorem cube_volume_given_face_perimeter : 
  ∀ (side_length perimeter : ℝ), is_face_perimeter 40 side_length → cube_volume side_length = 1000 :=
by
  intros side_length perimeter h
  sorry

end cube_volume_given_face_perimeter_l807_807002


namespace interior_angle_regular_octagon_l807_807486

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807486


namespace four_digit_numbers_divisible_by_7_l807_807509

theorem four_digit_numbers_divisible_by_7 :
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_n := Nat.ceil (lower_bound / 7)
  let largest_n := upper_bound / 7
  ∃ n : ℕ, smallest_n = 143 ∧ largest_n = 1428 ∧ (largest_n - smallest_n + 1 = 1286) :=
by
  let lower_bound := 1000
  let upper_bound := 9999
  let smallest_n := Nat.ceil (lower_bound / 7)
  let largest_n := upper_bound / 7
  use smallest_n, largest_n
  have h1 : smallest_n = 143 := sorry
  have h2 : largest_n = 1428 := sorry
  have h3 : largest_n - smallest_n + 1 = 1286 := sorry
  exact ⟨h1, h2, h3⟩

end four_digit_numbers_divisible_by_7_l807_807509


namespace perimeter_of_square_C_l807_807646

theorem perimeter_of_square_C (a b : ℝ) 
  (hA : 4 * a = 16) 
  (hB : 4 * b = 32) : 
  4 * (a + b) = 48 := by
  sorry

end perimeter_of_square_C_l807_807646


namespace no_primes_in_seq_l807_807943

def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

noncomputable def Q : ℕ := (List.range (53+1)).filter (λ k, Nat.prime k).foldr (*) 1

def seq_term (n : ℕ) : ℕ := Q + n

def range_n : List ℕ := List.range' 4 (37)

theorem no_primes_in_seq : (range_n.filter (λ n, is_prime (seq_term n))).length = 0 :=
by
  sorry

end no_primes_in_seq_l807_807943


namespace angle_measure_l807_807846

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807846


namespace interior_angle_regular_octagon_l807_807441

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807441


namespace restore_grid_values_l807_807744

def is_adjacent_sum_lt_12 (A B C D : ℕ) : Prop :=
  (A + 1 < 12) ∧ (A + 9 < 12) ∧
  (3 + 1 < 12) ∧ (3 + 5 < 12) ∧
  (3 + D < 12) ∧ (5 + 1 < 12) ∧ 
  (5 + D < 12) ∧ (B + C < 12) ∧
  (B + 3 < 12) ∧ (B + 5 < 12) ∧
  (C + 5 < 12) ∧ (C + 7 < 12) ∧
  (D + 9 < 12) ∧ (D + 7 < 12)

theorem restore_grid_values : 
  ∃ (A B C D : ℕ), 
  (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) ∧ 
  is_adjacent_sum_lt_12 A B C D := 
by 
  exists 8
  exists 6
  exists 4
  exists 2
  split; [refl, split; [refl, split; [refl, split; [refl, sorry]]]] 

end restore_grid_values_l807_807744


namespace restore_grid_values_l807_807748

def is_adjacent_sum_lt_12 (A B C D : ℕ) : Prop :=
  (A + 1 < 12) ∧ (A + 9 < 12) ∧
  (3 + 1 < 12) ∧ (3 + 5 < 12) ∧
  (3 + D < 12) ∧ (5 + 1 < 12) ∧ 
  (5 + D < 12) ∧ (B + C < 12) ∧
  (B + 3 < 12) ∧ (B + 5 < 12) ∧
  (C + 5 < 12) ∧ (C + 7 < 12) ∧
  (D + 9 < 12) ∧ (D + 7 < 12)

theorem restore_grid_values : 
  ∃ (A B C D : ℕ), 
  (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) ∧ 
  is_adjacent_sum_lt_12 A B C D := 
by 
  exists 8
  exists 6
  exists 4
  exists 2
  split; [refl, split; [refl, split; [refl, split; [refl, sorry]]]] 

end restore_grid_values_l807_807748


namespace regular_octagon_interior_angle_l807_807322

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807322


namespace distance_is_6_l807_807077

noncomputable def distance_between_house_and_school : ℝ :=
  let D := 6 in D

theorem distance_is_6
  (speed_going : ℝ := 3)
  (speed_returning : ℝ := 2)
  (total_time : ℝ := 5) :
  (D / speed_going + D / speed_returning = total_time) → 
  D = 6 :=
by
  intro h
  -- actual proof would be here, omitted with sorry
  sorry

end distance_is_6_l807_807077


namespace regular_octagon_interior_angle_l807_807249

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807249


namespace angle_measure_l807_807908

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807908


namespace area_of_square_eq_seventeen_l807_807041

open Real

-- Define the points in ℝ²
def P : ℝ × ℝ := (1, 2)
def Q : ℝ × ℝ := (-3, 3)
def R : ℝ × ℝ := (-2, 8)
def S : ℝ × ℝ := (2, 7)

-- Define the distance formula between two points in ℝ²
def distance (A B : ℝ × ℝ) : ℝ :=
  sqrt ((B.1 - A.1) ^ 2 + (B.2 - A.2) ^ 2)

-- Define the lengths of the sides
def PQ : ℝ := distance P Q
def QR : ℝ := distance Q R
def RS : ℝ := distance R S
def SP : ℝ := distance S P

-- Define the property that the shape is a square
def is_square : Prop :=
  PQ = QR ∧ QR = RS ∧ RS = SP ∧ SP = PQ

-- Calculate the area of the square
def area_square (side_length : ℝ) : ℝ :=
  side_length ^ 2

-- The statement of the proof problem
theorem area_of_square_eq_seventeen (h : is_square) : area_square PQ = 17 :=
by
  sorry

end area_of_square_eq_seventeen_l807_807041


namespace determine_grid_numbers_l807_807720

theorem determine_grid_numbers (A B C D : ℕ) :
  (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧ (1 ≤ D ∧ D ≤ 9) ∧ 
  (A ≠ 1) ∧ (A ≠ 3) ∧ (A ≠ 5) ∧ (A ≠ 7) ∧ (A ≠ 9) ∧ 
  (B ≠ 1) ∧ (B ≠ 3) ∧ (B ≠ 5) ∧ (B ≠ 7) ∧ (B ≠ 9) ∧ 
  (C ≠ 1) ∧ (C ≠ 3) ∧ (C ≠ 5) ∧ (C ≠ 7) ∧ (C ≠ 9) ∧ 
  (D ≠ 1) ∧ (D ≠ 3) ∧ (D ≠ 5) ∧ (D ≠ 7) ∧ (D ≠ 9) ∧ 
  (A ≠ 2 ∧ A ≠ 4 ∧ A ≠ 6 ∧ A ≠ 8 ∨ B ≠ 2 ∧ B ≠ 4 ∧ B ≠ 6 ∧ B ≠ 8 ∨ C ≠ 2 ∧ C ≠ 4 ∧ C ≠ 6 ∧ C ≠ 8 ∨ D ≠ 2 ∧ D ≠ 4 ∧ D ≠ 6 ∧ D ≠ 8) ∧ 
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ 
  (3 + 5 < 12) ∧ (3 + D < 12) ∧ (B + 5 < 12) ∧ 
  (B + C < 12) ∧ (C + 7 < 12) ∧ (B + 7 < 12) 
  → (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) :=
by 
  intros A B C D
  exact sorry

end determine_grid_numbers_l807_807720


namespace restore_grid_l807_807761

noncomputable def grid_values (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 9 < 12 ∧
  3 + 1 < 12 ∧ 3 + 5 < 12 ∧
  B + 3 < 12 ∧ B + 4 < 12 ∧
  B + 7 < 12 ∧ 
  C + 4 < 12 ∧
  C + 7 < 12 ∧ 
  D + 9 < 12 ∧
  D + 7 < 12 ∧
  D + 5 < 12

def even_numbers_erased (A B C D : ℕ) : Prop :=
  A ∈ {8} ∧ B ∈ {6} ∧ C ∈ {4} ∧ D ∈ {2}

theorem restore_grid :
  ∃ (A B C D : ℕ), grid_values A B C D ∧ even_numbers_erased A B C D :=
by {
  use [8, 6, 4, 2],
  dsimp [grid_values, even_numbers_erased],
  exact ⟨⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩, 
         ⟨rfl, rfl, rfl, rfl⟩⟩
}

end restore_grid_l807_807761


namespace interior_angle_regular_octagon_l807_807365

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807365


namespace interior_angle_regular_octagon_l807_807497

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807497


namespace angle_supplement_complement_l807_807808

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l807_807808


namespace sum_of_areas_of_triangles_l807_807989

theorem sum_of_areas_of_triangles 
  (AB BG GE DE : ℕ) 
  (A₁ A₂ : ℕ)
  (H1 : AB = 2) 
  (H2 : BG = 3) 
  (H3 : GE = 4) 
  (H4 : DE = 5) 
  (H5 : 3 * A₁ + 4 * A₂ = 48)
  (H6 : 9 * A₁ + 5 * A₂ = 102) : 
  1 * AB * A₁ / 2 + 1 * DE * A₂ / 2 = 23 :=
by
  sorry

end sum_of_areas_of_triangles_l807_807989


namespace solve_for_asterisk_l807_807050

theorem solve_for_asterisk (asterisk : ℝ) : 
  ((60 / 20) * (60 / asterisk) = 1) → asterisk = 180 :=
by
  sorry

end solve_for_asterisk_l807_807050


namespace inequality_am_gm_l807_807169

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
sorry

end inequality_am_gm_l807_807169


namespace regular_octagon_angle_l807_807312

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807312


namespace perp_line_OP_QR_l807_807620

-- Define all points and respective angles/relations
variables {O A B C M N Q R P : Type*}
variables [MetricSpace ℝ Type*]
variables [Circle ℝ Type*]
variables (ω : circle O)
variables (l : Line ℝ)

-- Given conditions
variable (h1 : A ∈ ω)
variable (h2 : B ∈ ω)
variable (h3 : (π / 3 : ℝ) < ∠ A O B)
variable (h4 : ∠ A O B < 2 * π / 3)
variable (h5 : IsCircumcenter C A B O)
variable (h6 : l.PassesThrough C)
variable (h7 : l.MakesAngleWith (OC) (π / 3))
variable (h8 : IntersectionWithTangentsAt A B l M N)
variable (h9 : Circumcircle A M C IntersectsAt Q ω)
variable (h10 : Circumcircle B N C IntersectsAt R ω)
variable (h11 : Circumcircle A M C IntersectsCircumcircle B N C At P (P ≠ C))

-- Prove statement
theorem perp_line_OP_QR 
  (A B C M N Q R P : Point ℝ) (ω : circle O) (l : Line ℝ) :
  A ∈ ω → B ∈ ω →
  (π / 3 < ∠ A O B) → (∠ A O B < 2 * π / 3) →
  IsCircumcenter C A B O →
  l.PassesThrough(C) →
  MakesAngleWith l (OC) (π / 3) →
  IntersectionWithTangentsAt A B l M N →
  CircumcircleIntersects A M C Q ω →
  CircumcircleIntersects B N C R ω →
  IntersectsCircumcircle A M C B N C P (P ≠ C) →
  Perpendicular (LineThrough O P) (LineThrough Q R) :=
sorry

end perp_line_OP_QR_l807_807620


namespace count_even_integers_l807_807506

def belongs_to_set (n : ℕ) : Prop :=
  ∀ (d : ℕ), d ∈ digits 10 n → d ∈ {1, 3, 4, 6, 7, 8}

def all_digits_different (n : ℕ) : Prop :=
  ∀ (d1 d2 : ℕ), d1 ∈ digits 10 n → d2 ∈ digits 10 n → d1 ≠ d2 → d1 ≠ d2

def is_even (n : ℕ) : Prop := n % 2 = 0

def in_range (n : ℕ) : Prop := 300 ≤ n ∧ n ≤ 900

theorem count_even_integers : 
  {n // in_range n ∧ is_even n ∧ belongs_to_set n ∧ all_digits_different n}.card = 36 := 
by
  sorry

end count_even_integers_l807_807506


namespace regular_octagon_interior_angle_l807_807251

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807251


namespace find_grid_values_l807_807752

-- Define the grid and the conditions in Lean
variables (A B C D : ℕ)
assume h1: A, B, C, D ∈ {2, 4, 6, 8}
assume h2: A + 1 < 12
assume h3: A + 9 < 12
assume h4: A + 3 < 12
assume h5: A + 5 < 12
assume h6: A + D < 12
assume h7: B + C < 12
assume h8: B + 5 < 12
assume h9: C + 7 < 12
assume h10: D + 9 < 12

-- Prove the specific values for A, B, C, and D
theorem find_grid_values : 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by sorry

end find_grid_values_l807_807752


namespace find_abc_sum_l807_807196

theorem find_abc_sum (a b c : ℤ) (h1 : a - 2 * b = 4) (h2 : a * b + c^2 - 1 = 0) :
  a + b + c = 5 ∨ a + b + c = 3 ∨ a + b + c = -1 ∨ a + b + c = -3 :=
  sorry

end find_abc_sum_l807_807196


namespace regular_octagon_interior_angle_l807_807323

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807323


namespace num_students_yes_R_l807_807975

noncomputable def num_students_total : ℕ := 800
noncomputable def num_students_yes_only_M : ℕ := 150
noncomputable def num_students_no_to_both : ℕ := 250

theorem num_students_yes_R : (num_students_total - num_students_no_to_both) - num_students_yes_only_M = 400 :=
by
  sorry

end num_students_yes_R_l807_807975


namespace interior_angle_regular_octagon_l807_807455

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807455


namespace interior_angle_regular_octagon_l807_807360

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807360


namespace regular_octagon_interior_angle_eq_135_l807_807394

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807394


namespace angle_AED_not_acute_l807_807981

theorem angle_AED_not_acute 
  (A B C D E : Point)
  (ABC_is_trapezoid : is_trapezoid A B C D)
  (circumscribed_about_circle : circumscribed A B C D)
  (diagonals_intersect : intersect (diagonal A C) (diagonal B D) E) 
  (equal_opposite_sides : distance A B + distance C D = distance A D + distance B C) :
  ¬ acute_angle (angle A E D) :=
by
  -- Let the proof proceed here
  sorry

end angle_AED_not_acute_l807_807981


namespace solid_volume_correct_l807_807132

-- Definitions of the geometric shapes and their properties
noncomputable def equilateral_triangle_side_length := 1
noncomputable def square_side_length := 1

-- Volumes according to the solution
noncomputable def volume_triangular_prism :=
  (Math.sqrt 3 / 4)  -- Base area * height

noncomputable def volume_regular_tetrahedron :=
  (Math.sqrt 2 / 12)

-- Total volume is the sum of individual volumes
noncomputable def total_volume :=
  volume_triangular_prism + 2 * volume_regular_tetrahedron

-- Prove that the total volume is as stated
theorem solid_volume_correct :
  total_volume = (Math.sqrt 3 + Math.sqrt 2) / 4 :=
sorry

end solid_volume_correct_l807_807132


namespace regular_octagon_interior_angle_l807_807469

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807469


namespace ozverin_concentration_after_5_times_l807_807703

noncomputable def ozverin_concentration (V : ℝ) (C₀ : ℝ) (v : ℝ) (n : ℕ) : ℝ :=
  C₀ * (1 - v / V) ^ n

theorem ozverin_concentration_after_5_times :
  ∀ (V : ℝ) (C₀ : ℝ) (v : ℝ) (n : ℕ), V = 0.5 → C₀ = 0.4 → v = 50 → n = 5 →
  ozverin_concentration V C₀ v n = 0.236196 :=
by
  intros V C₀ v n hV hC₀ hv hn
  rw [hV, hC₀, hv, hn]
  simp only [ozverin_concentration]
  norm_num
  sorry

end ozverin_concentration_after_5_times_l807_807703


namespace reach_one_l807_807674

theorem reach_one (n : ℕ) (hn : 0 < n) : ∃ k : ℕ, (n = 2^k) ∨ (n = 3^k * 2^m + 1) ∧ (∃ m : ℕ, 0 < m ∧ pow 2 m > n) :=
sorry

end reach_one_l807_807674


namespace restore_grid_l807_807717

-- Definitions for each cell in the grid.
variable (A B C D : ℕ)

-- The given grid and conditions
noncomputable def grid : list (list ℕ) :=
  [[A, 1, 9],
   [3, 5, D],
   [B, C, 7]]

-- The condition that the sum of numbers in adjacent cells is less than 12.
def valid (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 3 < 12 ∧
  1 + 9 < 12 ∧ 5 + D < 12 ∧
  3 + 5 < 12 ∧  B + C < 12 ∧
  9 + D < 12 ∧ 7 + C < 12 ∧
  9 + 1 < 12 ∧ B + 7 < 12

-- Assertion to be proved.
theorem restore_grid (A B C D : ℕ) (valid : valid A B C D) :
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 := by
  sorry

end restore_grid_l807_807717


namespace initial_weight_of_load_l807_807107

variable (W : ℝ)
variable (h : 0.8 * 0.9 * W = 36000)

theorem initial_weight_of_load :
  W = 50000 :=
by
  sorry

end initial_weight_of_load_l807_807107


namespace kitten_weight_l807_807963

theorem kitten_weight :
  ∃ (x y z : ℝ), x + y + z = 36 ∧ x + z = 3 * y ∧ x + y = 1 / 2 * z ∧ x = 3 := 
by
  sorry

end kitten_weight_l807_807963


namespace find_grid_values_l807_807756

-- Define the grid and the conditions in Lean
variables (A B C D : ℕ)
assume h1: A, B, C, D ∈ {2, 4, 6, 8}
assume h2: A + 1 < 12
assume h3: A + 9 < 12
assume h4: A + 3 < 12
assume h5: A + 5 < 12
assume h6: A + D < 12
assume h7: B + C < 12
assume h8: B + 5 < 12
assume h9: C + 7 < 12
assume h10: D + 9 < 12

-- Prove the specific values for A, B, C, and D
theorem find_grid_values : 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by sorry

end find_grid_values_l807_807756


namespace each_interior_angle_of_regular_octagon_l807_807339

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807339


namespace interior_angle_regular_octagon_l807_807447

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807447


namespace each_interior_angle_of_regular_octagon_l807_807353

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807353


namespace interior_angle_regular_octagon_l807_807364

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807364


namespace maximum_constant_c_l807_807600

theorem maximum_constant_c (λ : ℝ) (hλ : λ > 0) (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) : 
  let c₁ := 1
  let c₂ := (2 + λ) / 4
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x^2 + y^2 + λ * x * y ≥ c₁ * (x + y)^2) ∧ λ ≥ 2 ∨ 
  (∀ x y : ℝ, x ≥ 0 → y ≥ 0 → x^2 + y^2 + λ * x * y ≥ c₂ * (x + y)^2) ∧ 0 < λ ∧ λ < 2 :=
sorry

end maximum_constant_c_l807_807600


namespace regular_octagon_interior_angle_l807_807286

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807286


namespace angle_complement_supplement_l807_807821

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l807_807821


namespace two_circles_tangent_lines_parallel_l807_807073

theorem two_circles_tangent_lines_parallel
    (O1 O2 A B K L : Point)
    (circle1 : Circle O1)
    (circle2 : Circle O2)
    (h_intersect : A ∈ circle1 ∧ A ∈ circle2 ∧ B ∈ circle1 ∧ B ∈ circle2)
    (h_tangents : TangentAt A circle1 = Line K A ∧ TangentAt A circle2 = Line L A)
    (h_line_intersect : K ∈ Line B O1 ∧ L ∈ Line B O2) :
    Parallel (Line K L) (Line O1 O2) :=
sorry

end two_circles_tangent_lines_parallel_l807_807073


namespace angle_measure_l807_807828

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l807_807828


namespace circles_externally_tangent_l807_807008

noncomputable def radius_O1 : ℝ := 3
noncomputable def radius_O2 : ℝ := 5
noncomputable def distance_centers : ℝ := 8

theorem circles_externally_tangent :
  radius_O1 + radius_O2 = distance_centers → "externally tangent" := by
  sorry

end circles_externally_tangent_l807_807008


namespace area_triangle_OAB_l807_807583

theorem area_triangle_OAB :
  let C := {p : ℝ × ℝ | p.2^2 = 3 * p.1}
  let focus_F := (3 / 4, 0)
  let line_through_F := {p : ℝ × ℝ | p.2 = (Real.sqrt 3 / 3) * (p.1 - 3/4)}
  let O := (0, 0)
  (∃ A B ∈ C, A ∈ line_through_F ∧ B ∈ line_through_F ∧ 
  let y1 := A.2 in
  let y2 := B.2 in
  let sum_y := y1 + y2 in
  let prod_y := y1 * y2 in
  abs ((sum_y ^ 2 / (2 * sum_y + 1) - prod_y / (2 * sum_y + 1)) - 
  (sum_y ^ 2 / (2 * sum_y + 1) + prod_y / (2 * sum_y + 1))) = 
  3 * abs ((sqrt ((sum_y ^ 2 + 3 * prod_y)))) / 8)
    ↔ (9 / 4) := sorry

end area_triangle_OAB_l807_807583


namespace not_pythagorean_C_l807_807061

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem not_pythagorean_C :
  ¬ is_pythagorean_triple 8 12 16 :=
by
  unfold is_pythagorean_triple
  rw [pow_two, pow_two, pow_two]
  norm_num
  sorry

end not_pythagorean_C_l807_807061


namespace interior_angle_regular_octagon_l807_807459

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807459


namespace parabola_vertex_correct_l807_807138

noncomputable def parabola_vertex (p q : ℝ) : ℝ × ℝ :=
  let a := -1
  let b := p
  let c := q
  let x_vertex := -b / (2 * a)
  let y_vertex := a * x_vertex^2 + b * x_vertex + c
  (x_vertex, y_vertex)

theorem parabola_vertex_correct (p q : ℝ) :
  (parabola_vertex 2 24 = (1, 25)) :=
  sorry

end parabola_vertex_correct_l807_807138


namespace tan_theta_infinite_l807_807982

/-- Given a triangle with sides 6, 6, and 8, let θ be the acute angle between the two lines 
that bisect the area of the triangle. Prove that tan θ = ∞. -/
theorem tan_theta_infinite:
  ∃ (θ : ℝ), 
   (∀ (A B C : ℝ), A = 6 ∧ B = 6 ∧ C = 8 → 
    ∃ (PQ RS : ℝ → Prop),
      (∀ (x : ℝ), PQ x ↔ RS x → x = θ) ∧ 
      (tan θ = ∞)) :=
begin
  sorry
end

end tan_theta_infinite_l807_807982


namespace regular_octagon_interior_angle_l807_807473

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807473


namespace regular_octagon_interior_angle_l807_807471

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807471


namespace domain_of_log_function_l807_807669

theorem domain_of_log_function :
  {x : ℝ | ∃ y, y = real.log (x + 1) } = {x : ℝ | x > -1} :=
sorry

end domain_of_log_function_l807_807669


namespace inequality_proof_l807_807162

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
by
  sorry

end inequality_proof_l807_807162


namespace regular_octagon_interior_angle_l807_807235

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807235


namespace regular_octagon_interior_angle_l807_807297

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807297


namespace regular_octagon_interior_angle_l807_807266

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807266


namespace max_value_x_l807_807155

theorem max_value_x : ∃ x, x ^ 2 = 38 ∧ x = Real.sqrt 38 := by
  sorry

end max_value_x_l807_807155


namespace CTIY_concyclic_I_center_of_excircle_l807_807223

variables {O₁ O₂ T A B X S C Y I : Point}
variables (h_tangent_tangent : TangentCirc O₁ O₂ T)
variables (h_tangent_X : TangentLineCirc X O₂)
variables (h_intersects_O₁ : IntersectsCircLine A B O₁)
variables (h_order : lies_between B A X)
variables (h_intersects_XT : IntersectsLineCirc S O₁ (line XT))
variables (h_lies_TS : lies_between C T S)
variables (h_not_on_AB : ¬ lies_on_line C (line AB))
variables (h_tangent_Y : TangentLineCirc Y O₂ (line_through C))
variables (h_no_intersect : ¬ IntersectsLineLine (segment CY) (segment ST))
variables (h_intersects_SI : IntersectsLineLine S I (line XY))

/-- Points C, T, I, and Y are concyclic. -/
theorem CTIY_concyclic (h : tangent_circles O₁ O₂ T) (h : tangent_at X O₂) 
  (h : intersects_A B O₁) (h : ordered_triplet A B X) (h : XT_intersects_S S O₁)
  (h : C_in_TS (segment TS) T S) (h : C_not_in_AB (segment AB) C)
  (h : tangent_at_Y O₂ Y (line_through C)) (h : no_CY_vs_ST (segment CY) (segment ST))
  (h : SI_in_XY S I (line_through XY))
  : concyclic C T I Y := 
sorry

/-- Point I is the center of the excircle of triangle ABC opposite to angle A. --/
theorem I_center_of_excircle (h : tangent_circles O₁ O₂ T) (h : tangent_at X O₂) 
  (h : intersects_A B O₁) (h : ordered_triplet A B X) (h : XT_intersects_S S O₁)
  (h : C_in_TS (segment TS) T S) (h : C_not_in_AB (segment AB) C)
  (h : tangent_at_Y O₂ Y (line_through C)) (h : no_CY_vs_ST (segment CY) (segment ST))
  (h : SI_in_XY S I (line_through XY))
  : center_of_excircle I (triangle ABC) :=
sorry

end CTIY_concyclic_I_center_of_excircle_l807_807223


namespace sixty_third_digit_of_one_div_seventeen_l807_807040

theorem sixty_third_digit_of_one_div_seventeen : 
  (fractional_part (1 / 17) 63) = 6 := by
sorry

end sixty_third_digit_of_one_div_seventeen_l807_807040


namespace angle_measure_l807_807845

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807845


namespace regular_octagon_interior_angle_l807_807271

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807271


namespace equation_of_ellipse_l807_807672

-- Defining the given conditions in the problem
def F1 : ℝ × ℝ := (-4, 0)
def F2 : ℝ × ℝ := (4, 0)
def max_area (P : ℝ × ℝ) : ℝ := 
  let [f1x, f1y] := F1 
  let [f2x, f2y] := F2 
  abs ((f1x - f2x) * (f1y - f2y) + (P.1 - f1x) * (P.2 - f1y)) / 2

axiom max_area_eq_12 : ∀ (P : ℝ × ℝ), max_area P = 12

-- Proving that the equation of the ellipse is given by the equation stated
theorem equation_of_ellipse : 
  ∀ (x y : ℝ),  (x^2 / 25) + (y^2 / 9) = 1 := 
    sorry

end equation_of_ellipse_l807_807672


namespace lateral_surface_area_angle_between_faces_l807_807545

noncomputable def lateralSurfaceArea (a : ℝ) : ℝ := 3 * sqrt 15 * (sqrt 5 + 1)

theorem lateral_surface_area_angle_between_faces
  (r : ℝ) (h : ℝ)
  (has_sphere_mid_height : r = 1)
  (hemisphere_support : h = 1) :
  exists a : ℝ,
    a = 2 * sqrt 3 * (sqrt 5 + 1) ∧
    lateralSurfaceArea a = 3 * sqrt 15 * (sqrt 5 + 1) ∧
    angle_between_faces a = arcsin (sqrt (2 / 5)) :=
by
  sorry

end lateral_surface_area_angle_between_faces_l807_807545


namespace monotonically_increasing_sequence_l807_807973

theorem monotonically_increasing_sequence (λ : ℝ) :
  (∀ n : ℕ, 0 < n → (n^2 - 3 * λ * n) < ((n + 1)^2 - 3 * λ * (n + 1))) ↔ (λ < 1) :=
by
  sorry

end monotonically_increasing_sequence_l807_807973


namespace rearrange_2xn_table_l807_807946

theorem rearrange_2xn_table (n : ℕ) (h : n > 2) (table : Fin 2 → Fin n → ℕ)
  (h_col_diff : ∀ i j : Fin n, i ≠ j → (table 0 i + table 1 i) ≠ (table 0 j + table 1 j)) :
  ∃ (new_table : Fin 2 → Fin n → ℕ),
    (∀ i j : Fin n, i ≠ j → (new_table 0 i + new_table 1 i) ≠ (new_table 0 j + new_table 1 j)) ∧
    (new_table 0 0 + new_table 0 1 + ... + new_table 0 (n-1)) ≠ (new_table 1 0 + new_table 1 1 + ... + new_table 1 (n-1)) :=
sorry

end rearrange_2xn_table_l807_807946


namespace regular_octagon_angle_l807_807308

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807308


namespace total_weight_of_10_moles_l807_807044

theorem total_weight_of_10_moles
  (molecular_weight : ℕ)
  (moles : ℕ)
  (h_molecular_weight : molecular_weight = 2670)
  (h_moles : moles = 10) :
  moles * molecular_weight = 26700 := by
  -- By substituting the values from the hypotheses:
  -- We will get:
  -- 10 * 2670 = 26700
  sorry

end total_weight_of_10_moles_l807_807044


namespace interior_angle_regular_octagon_l807_807435

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807435


namespace angle_measure_l807_807831

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l807_807831


namespace difference_between_s_and_p_l807_807106

variable (p q r s : ℕ)

def total_amount := 10000
def condition1 := p = 2 * q
def condition2 := s = 4 * r
def condition3 := q = r
def condition4 := s = p / 2
def all_conditions := condition1 ∧ condition2 ∧ condition3 ∧ condition4 ∧ (p + q + r + s = total_amount)

theorem difference_between_s_and_p 
  (hpq : condition1) 
  (hsr : condition2) 
  (hqr : condition3) 
  (hsp : condition4) 
  (htotal : p + q + r + s = total_amount) : p - s = 2000 := by
  sorry

end difference_between_s_and_p_l807_807106


namespace bobby_candy_total_l807_807994

-- Definitions for the conditions
def initial_candy : Nat := 20
def first_candy_eaten : Nat := 34
def second_candy_eaten : Nat := 18

-- Theorem to prove the total pieces of candy Bobby ate
theorem bobby_candy_total : first_candy_eaten + second_candy_eaten = 52 := by
  sorry

end bobby_candy_total_l807_807994


namespace steve_needs_28_feet_of_wood_l807_807650

-- Define the required lengths
def lengths_4_feet : Nat := 6
def lengths_2_feet : Nat := 2

-- Define the wood length in feet for each type
def wood_length_4 : Nat := 4
def wood_length_2 : Nat := 2

-- Total feet of wood required
def total_wood : Nat := lengths_4_feet * wood_length_4 + lengths_2_feet * wood_length_2

-- The theorem to prove that the total amount of wood required is 28 feet
theorem steve_needs_28_feet_of_wood : total_wood = 28 :=
by
  sorry

end steve_needs_28_feet_of_wood_l807_807650


namespace locus_of_A_l807_807578

/-- Given fixed points B and C in the plane, and a point A outside the line BC,
the locus of points A such that the sum of angles ∠BAC and ∠BGC equals 180 degrees
is a circle centered at the midpoint of BC such that the triangle ABC is equilateral. -/
theorem locus_of_A (B C : EuclideanGeometry.Point ℝ)
  (A : EuclideanGeometry.Point ℝ)
  (hA : ¬Collinear B C A) (G : EuclideanGeometry.Point ℝ)
  (hG : EuclideanGeometry.Barycenter (triangle B C A) = G) :
    ∃ (D : EuclideanGeometry.Point ℝ), (D = midpoint ℝ B C) ∧
    ∀ A', EuclideanGeometry.InCircle A' B C D ∧ triangle A' B C .Equilateral :=
sorry

end locus_of_A_l807_807578


namespace regular_octagon_interior_angle_eq_135_l807_807398

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807398


namespace probability_of_getting_a_prize_l807_807949

theorem probability_of_getting_a_prize {prizes blanks : ℕ} (h_prizes : prizes = 10) (h_blanks : blanks = 25) :
  (prizes / (prizes + blanks) : ℚ) = 2 / 7 :=
by
  sorry

end probability_of_getting_a_prize_l807_807949


namespace probability_xyz_72_l807_807918

noncomputable def dice_probability : ℚ :=
  let outcomes : Finset (ℕ × ℕ × ℕ) := 
    {(a, b, c) | a ∈ {1, 2, 3, 4, 5, 6}, b ∈ {1, 2, 3, 4, 5, 6}, c ∈ {1, 2, 3, 4, 5, 6}}.toFinset
  let favorable_outcomes := 
    outcomes.filter (λ (abc : ℕ × ℕ × ℕ), abc.1 * abc.2 * abc.3 = 72)
  (favorable_outcomes.card : ℚ) / (outcomes.card : ℚ)

theorem probability_xyz_72 : dice_probability = 1/24 :=
  sorry

end probability_xyz_72_l807_807918


namespace angle_measure_l807_807900

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807900


namespace correctness_check_l807_807056

noncomputable def questionD (x y : ℝ) : Prop := 
  3 * x^2 * y - 2 * y * x^2 = x^2 * y

theorem correctness_check (x y : ℝ) : questionD x y :=
by 
  sorry

end correctness_check_l807_807056


namespace angle_measure_l807_807802

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l807_807802


namespace regular_octagon_interior_angle_l807_807299

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807299


namespace interior_angle_regular_octagon_l807_807371

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807371


namespace sum_x_coords_intersections_l807_807667

def g (x : ℝ) : ℝ :=
if x ≤ -5 then -5
else if x ≤ -3 then x + 4
else if x ≤ -1 then -3
else if x ≤ 0 then x
else if x ≤ 2 then -2
else if x ≤ 3 then x - 2
else 3

theorem sum_x_coords_intersections :
  let points := {x | g x = x - 1}
  let x_coords : list ℝ := points.toList.map id
  list.sum x_coords = 0 :=
by
  sorry

end sum_x_coords_intersections_l807_807667


namespace book_pages_l807_807573

def time_to_lunch := 4
def pages_per_hour := 250
def total_reading_time := 2 * time_to_lunch

theorem book_pages : (pages_per_hour * total_reading_time) = 2000 :=
by
  let total_pages := pages_per_hour * total_reading_time
  show total_pages = 2000
  from sorry

end book_pages_l807_807573


namespace correct_calculation_l807_807052

def calculation_is_correct (a b x y : ℝ) : Prop :=
  (3 * x^2 * y - 2 * y * x^2 = x^2 * y)

theorem correct_calculation :
  ∀ (a b x y : ℝ), calculation_is_correct a b x y :=
by
  intros a b x y
  sorry

end correct_calculation_l807_807052


namespace angle_measure_l807_807905

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807905


namespace triangle_AB_plus_AC_eq_2BC_l807_807186

theorem triangle_AB_plus_AC_eq_2BC
  (A B C G I : Point)
  (h1 : G = centroid A B C)
  (h2 : I = incenter A B C)
  (h3 : GI_parallel_BC : parallel (line_through G I) (line_through B C)) :
  AB + AC = 2 * BC := 
sorry

end triangle_AB_plus_AC_eq_2BC_l807_807186


namespace regular_octagon_interior_angle_l807_807329

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807329


namespace factorial_expression_equiv_l807_807121

theorem factorial_expression_equiv :
  6 * Nat.factorial 6 + 5 * Nat.factorial 5 + 3 * Nat.factorial 4 + Nat.factorial 4 = 1416 := 
sorry

end factorial_expression_equiv_l807_807121


namespace points_in_rectangle_distance_l807_807536

/-- In a 3x4 rectangle, if 4 points are randomly located, 
    then the distance between at least two of them is at most 25/8. -/
theorem points_in_rectangle_distance (a b : ℝ) (h₁ : a = 3) (h₂ : b = 4)
  {points : Fin 4 → ℝ × ℝ}
  (h₃ : ∀ i, 0 ≤ (points i).1 ∧ (points i).1 ≤ a)
  (h₄ : ∀ i, 0 ≤ (points i).2 ∧ (points i).2 ≤ b) :
  ∃ i j, i ≠ j ∧ dist (points i) (points j) ≤ 25 / 8 := 
by
  sorry

end points_in_rectangle_distance_l807_807536


namespace find_b_squared_l807_807083

theorem find_b_squared (a b : ℝ) (f : ℂ → ℂ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) 
  (h_def_f : ∀ z : ℂ, f z = (a + b * complex.I) * z)
  (h_equidistant : ∀ z : ℂ, complex.abs (f z - z) = complex.abs (f z))
  (h_norm : complex.abs (a + b * complex.I) = 8) :
  b^2 = 255 / 4 :=
by
  sorry

end find_b_squared_l807_807083


namespace clock_angle_at_7_15_l807_807502

theorem clock_angle_at_7_15 :
  let hour_hand := 210 + 30 / 4,
      minute_hand := 90 in
  |hour_hand - minute_hand| = 127.5 := 
by
  let hour_hand := 210 + 30 / 4
  let minute_hand := 90
  have : hour_hand = 217.5
  have : minute_hand = 90
  have : |hour_hand - minute_hand| = |217.5 - 90|
  have : |127.5| = 127.5
  sorry   -- proof omitted

end clock_angle_at_7_15_l807_807502


namespace angle_measure_l807_807837

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l807_807837


namespace regular_octagon_interior_angle_l807_807295

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807295


namespace steve_needs_28_feet_of_wood_l807_807649

-- Define the required lengths
def lengths_4_feet : Nat := 6
def lengths_2_feet : Nat := 2

-- Define the wood length in feet for each type
def wood_length_4 : Nat := 4
def wood_length_2 : Nat := 2

-- Total feet of wood required
def total_wood : Nat := lengths_4_feet * wood_length_4 + lengths_2_feet * wood_length_2

-- The theorem to prove that the total amount of wood required is 28 feet
theorem steve_needs_28_feet_of_wood : total_wood = 28 :=
by
  sorry

end steve_needs_28_feet_of_wood_l807_807649


namespace percentage_multiplication_l807_807047

theorem percentage_multiplication :
  (0.15 * 0.20 * 0.25) * 100 = 0.75 := 
by
  sorry

end percentage_multiplication_l807_807047


namespace no_intersection_l807_807595

-- Definitions of the sets M1 and M2 based on parameters A, B, C and integer x
def M1 (A B : ℤ) : Set ℤ := {y | ∃ x : ℤ, y = x^2 + A * x + B}
def M2 (C : ℤ) : Set ℤ := {y | ∃ x : ℤ, y = 2 * x^2 + 2 * x + C}

-- The statement of the theorem
theorem no_intersection (A B : ℤ) : ∃ C : ℤ, M1 A B ∩ M2 C = ∅ :=
sorry

end no_intersection_l807_807595


namespace interior_angle_regular_octagon_l807_807494

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807494


namespace regular_octagon_angle_l807_807318

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807318


namespace probability_of_xyz_72_l807_807922

noncomputable def probability_product_is_72 : ℚ :=
  let dice := {1, 2, 3, 4, 5, 6}
  let outcomes := {e : ℕ × ℕ × ℕ | e.1 ∈ dice ∧ e.2.1 ∈ dice ∧ e.2.2 ∈ dice}
  let favourable_outcomes := {e : ℕ × ℕ × ℕ | e ∈ outcomes ∧ e.1 * e.2.1 * e.2.2 = 72}
  (favourable_outcomes.to_finset.card : ℚ) / outcomes.to_finset.card

theorem probability_of_xyz_72 :
  probability_product_is_72 = 1 / 24 :=
sorry

end probability_of_xyz_72_l807_807922


namespace correct_formula_l807_807060

theorem correct_formula : ∀ (m : ℝ), (m + 1) * (m - 1) = m^2 - 1 :=
by
  intro m
  -- Applying the difference of squares formula
  calc
    (m + 1) * (m - 1) = m^2 - 1 : by sorry

end correct_formula_l807_807060


namespace final_sale_price_proof_l807_807009

-- Define the constants and conditions
def original_price : ℝ := 1200
def discount1 : ℝ := 0.25
def discount2 : ℝ := 0.20
def discount3 : ℝ := 0.15
def tax : ℝ := 0.10
def custom_fee : ℝ := 100

-- Define the successive discount application function
def apply_discount (price : ℝ) (discount : ℝ) : ℝ := price * (1 - discount)

-- Calculate price after each discount
def price_after_first_discount := apply_discount original_price discount1
def price_after_second_discount := apply_discount price_after_first_discount discount2
def price_after_third_discount := apply_discount price_after_second_discount discount3

-- Calculate price after tax
def price_after_tax := price_after_third_discount * (1 + tax)

-- Calculate final price
def final_sale_price := price_after_tax + custom_fee

theorem final_sale_price_proof : final_sale_price = 773.2 :=
by
  unfold original_price discount1 discount2 discount3 tax custom_fee
  unfold apply_discount price_after_first_discount price_after_second_discount price_after_third_discount price_after_tax final_sale_price
  simp [original_price, discount1, discount2, discount3, tax, custom_fee]
  sorry  -- Proof can be provided here if necessary

end final_sale_price_proof_l807_807009


namespace regular_octagon_interior_angle_l807_807418

-- Define the number of sides of a regular octagon
def num_sides : ℕ := 8

-- Define the formula for the sum of interior angles of a polygon
def sum_of_interior_angles (n : ℕ) : ℕ := 180 * (n - 2)

-- Define the value of the sum of interior angles for an octagon
def sum_of_interior_angles_of_octagon : ℕ := sum_of_interior_angles num_sides

-- Define the measure of each interior angle of a regular polygon
def interior_angle_of_regular_polygon (n : ℕ) : ℕ := sum_of_interior_angles n / n

-- Define the value for each interior angle of the regular octagon
def interior_angle_of_regular_octagon : ℕ := interior_angle_of_regular_polygon num_sides

-- Prove that each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_interior_angle :
  interior_angle_of_regular_octagon = 135 :=
by
  sorry

end regular_octagon_interior_angle_l807_807418


namespace each_interior_angle_of_regular_octagon_l807_807355

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807355


namespace angle_measure_l807_807801

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by
  sorry

end angle_measure_l807_807801


namespace fewerCansCollected_l807_807938

-- Definitions for conditions
def cansCollectedYesterdaySarah := 50
def cansCollectedMoreYesterdayLara := 30
def cansCollectedTodaySarah := 40
def cansCollectedTodayLara := 70

-- Total cans collected yesterday
def totalCansCollectedYesterday := cansCollectedYesterdaySarah + (cansCollectedYesterdaySarah + cansCollectedMoreYesterdayLara)

-- Total cans collected today
def totalCansCollectedToday := cansCollectedTodaySarah + cansCollectedTodayLara

-- Proving the difference
theorem fewerCansCollected :
  totalCansCollectedYesterday - totalCansCollectedToday = 20 := by
  sorry

end fewerCansCollected_l807_807938


namespace find_grid_values_l807_807754

-- Define the grid and the conditions in Lean
variables (A B C D : ℕ)
assume h1: A, B, C, D ∈ {2, 4, 6, 8}
assume h2: A + 1 < 12
assume h3: A + 9 < 12
assume h4: A + 3 < 12
assume h5: A + 5 < 12
assume h6: A + D < 12
assume h7: B + C < 12
assume h8: B + 5 < 12
assume h9: C + 7 < 12
assume h10: D + 9 < 12

-- Prove the specific values for A, B, C, and D
theorem find_grid_values : 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by sorry

end find_grid_values_l807_807754


namespace CD_twice_AK_l807_807541

-- Definitions and conditions
variables (A B C D E K : Point)
variables (hPentagon : ConvexPentagon A B C D E)
variables (hAE_AD : dist A E = dist A D)
variables (hAC_AB : dist A C = dist A B)
variables (hAngle : ∠ D A C = ∠ A E B + ∠ A B E)
variables (hAK_median : Median A K B E)

-- The theorem statement
theorem CD_twice_AK : dist C D = 2 * dist A K := sorry

end CD_twice_AK_l807_807541


namespace coefficient_of_x3_in_expansion_l807_807666

theorem coefficient_of_x3_in_expansion :
  ∃ c : ℤ, c = -192 ∧ (x^2 * (x - 2)^6).coeff 3 = c :=
by
  sorry

end coefficient_of_x3_in_expansion_l807_807666


namespace find_triples_l807_807152

theorem find_triples (a b c : ℕ) :
  (∃ n : ℕ, 2^a + 2^b + 2^c + 3 = n^2) ↔ (a = 1 ∧ b = 1 ∧ c = 1) ∨ (a = 3 ∧ b = 2 ∧ c = 1) :=
by
  sorry

end find_triples_l807_807152


namespace regular_octagon_interior_angle_l807_807255

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807255


namespace regular_octagon_interior_angle_eq_135_l807_807399

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807399


namespace ratio_pentagon_side_length_to_rectangle_width_l807_807094

def pentagon_side_length (p : ℕ) (n : ℕ) := p / n
def rectangle_width (p : ℕ) (ratio : ℕ) := p / (2 * (1 + ratio))

theorem ratio_pentagon_side_length_to_rectangle_width :
  pentagon_side_length 60 5 / rectangle_width 80 3 = (6 : ℚ) / 5 :=
by {
  sorry
}

end ratio_pentagon_side_length_to_rectangle_width_l807_807094


namespace original_number_multiple_of_8_l807_807966

theorem original_number_multiple_of_8 (x y : ℤ) (h : 14 * x = 112 * y) : ∃ k : ℤ, x = 8 * k :=
by
  use y
  have h' : 112 = 14 * 8 := rfl
  rw [h', mul_assoc, mul_comm 8 y, ← mul_assoc, mul_comm 14 x] at h
  rw [← mul_assoc, mul_eq_mul_left_iff] at h
  cases h; { use h; rfl }
  sorry

end original_number_multiple_of_8_l807_807966


namespace average_sale_over_six_months_l807_807085

theorem average_sale_over_six_months : 
  let s1 := 3435
  let s2 := 3920
  let s3 := 3855
  let s4 := 4230
  let s5 := 3560
  let s6 := 2000
  let total_sale := s1 + s2 + s3 + s4 + s5 + s6
  let average_sale := total_sale / 6
  average_sale = 3500 :=
by
  let s1 := 3435
  let s2 := 3920
  let s3 := 3855
  let s4 := 4230
  let s5 := 3560
  let s6 := 2000
  let total_sale := s1 + s2 + s3 + s4 + s5 + s6
  let average_sale := total_sale / 6
  show average_sale = 3500
  sorry

end average_sale_over_six_months_l807_807085


namespace find_p_of_equilateral_triangle_l807_807673

-- Define the problem conditions and required definitions

def parabola_focus (p : ℝ) (hp : 0 < p) : ℝ × ℝ := (p / 2, 0)

def directrix_x (p : ℝ) : ℝ := -p / 2

def hyperbola (y x : ℝ) : Prop := y^2 - x^2 = 1

def equilateral_triangle (A B F : ℝ × ℝ) : Prop :=
let d (P Q : ℝ × ℝ) := real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) in
d A B = d B F ∧ d B F = d F A ∧ d F A = d A B

-- Statement of the problem to be proved
theorem find_p_of_equilateral_triangle :
  ∀ (p : ℝ) (hp : 0 < p),
    let F := parabola_focus p hp,
    let directrix := directrix_x p,
    let A := (directrix, real.sqrt (1 + (p / 2)^2)),
    let B := (directrix, -real.sqrt (1 + (p / 2)^2)) in
    hyperbola A.2 A.1 →
    hyperbola B.2 B.1 →
    equilateral_triangle A B F →
    p = 2 * real.sqrt 3 :=
by
  intros p hp F directrix A B hA hB h_equilateral
  sorry -- the proof is omitted as instructed

end find_p_of_equilateral_triangle_l807_807673


namespace regular_octagon_interior_angle_l807_807324

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807324


namespace one_minus_repeating_eight_l807_807149

theorem one_minus_repeating_eight : (1 - (0.8 + 0.08 + 0.008 + (λ n, 8/10 ^ n))) = 1 / 9 := by
    sorry

end one_minus_repeating_eight_l807_807149


namespace PA_squared_QB_eq_QA_squared_PB_l807_807559

theorem PA_squared_QB_eq_QA_squared_PB 
  (ABC : Triangle)
  (h_equilateral : is_equilateral ABC)
  (M N : Point)
  (hM_midpoint : is_midpoint M ABC.A ABC.B)
  (hN_midpoint : is_midpoint N ABC.A ABC.C)
  (K L : Point)
  (circumcircle : Circle)
  (hcircumcircle : is_circumcircle circumcircle ABC)
  (h_KL_intersect : intersects_at circumcircle (line_through M N) K L)
  (P Q : Point)
  (hP_on_AB : lies_on P (line_through ABC.A ABC.B))
  (hQ_on_AB : lies_on Q (line_through ABC.A ABC.B))
  (h_CK_P : intersects_at (line_through ABC.C K) (line_through ABC.A ABC.B) P)
  (h_CL_Q : intersects_at (line_through ABC.C L) (line_through ABC.A ABC.B) Q) :
  length (segment P ABC.A) ^ 2 * length (segment Q ABC.B) = length (segment Q ABC.A) ^ 2 * length (segment P ABC.B) :=
sorry

end PA_squared_QB_eq_QA_squared_PB_l807_807559


namespace angle_measure_l807_807907

theorem angle_measure (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807907


namespace alice_flips_heads_probability_l807_807984

def prob_heads (n k : ℕ) (p q : ℚ) : ℚ :=
  (Nat.choose n k) * (p ^ k) * (q ^ (n - k))

theorem alice_flips_heads_probability :
  prob_heads 8 3 (1/3 : ℚ) (2/3 : ℚ) = 1792 / 6561 :=
by
  sorry

end alice_flips_heads_probability_l807_807984


namespace boys_girls_points_impossible_l807_807546

-- Define conditions and the equation based on the problem
variable {P_m P_d : ℕ} -- points scored by boys (P_m) and girls (P_d)
variable (total_points : ℕ := 15) -- total points distributed

theorem boys_girls_points_impossible 
  (h₀ : P_m + P_d = total_points) 
  (h₁ : P_m = 2 * P_d)
  (h₂ : total_points = 15) : 
  false := 
by 
  -- Since total points are fixed and distributed  
  -- Mathematically, the boys cannot score twice the points as the girls
  have h₃ : 3 * P_d = total_points := 
    by rw [h₁, two_mul, add_assoc, ←two_mul]; assumption
  
  have h₄ : P_d = 5 := 
    by linarith[h₃, h₂]

  -- This would imply P_m = 10, boys' points are 10
  have h₅ : P_m = 10 := 
    by rw [h₄, nat.mul_succ_one]

  -- However, in a round-robin tournament, with 2 boys each playing 5 games
  -- it is impossible for both boys to score 5 points each
  -- It leads to a contradiction as each boy cannot win all games
  contradiction

-- Provide the full theorem statement

end boys_girls_points_impossible_l807_807546


namespace correct_grid_l807_807732

def A := 8
def B := 6
def C := 4
def D := 2

def grid := [[A, 1, 9],
             [3, 5, D],
             [B, C, 7]]

theorem correct_grid :
  (A + 1 < 12) ∧ (A + 3 < 12) ∧ (1 + 9 < 12) ∧
  (1 + 5 < 12) ∧ (3 + 5 < 12) ∧ (3 + B < 12) ∧
  (5 + D < 12) ∧ (5 + C < 12) ∧ (9 + D < 12) ∧
  (B + C < 12) ∧ (C + 7 < 12) :=
by
  -- This is to provide a sketch dummy theorem, we'd prove each step here  
  sorry

end correct_grid_l807_807732


namespace angle_measure_l807_807853

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_measure_l807_807853


namespace math_problem_solution_l807_807680

noncomputable def least_positive_integer_with_2023_divisors (n j : ℕ) : Prop :=
  ∃ (a b c : ℕ), 
    (2023 = a * b * c ∧ 
     a = 7 ∧ 
     b = 17 ∧ 
     c = 17 ∧ 
     (∃ k l : ℕ, k = 2 ^ 10 ∧ l = 3 ^ 10 ∧ n = k * l) ∧ 
     j = 6 ∧ 
     6 ∣ (2 ^ j) ∧
     6 ∣ (3 ^ j) ∧
     n * 6 ^ j ~ (2 ^ 6 * 3 ^ 6 * 2 ^ 10 * 3 ^ 10))

theorem math_problem_solution : ∃ (n j : ℕ),
  least_positive_integer_with_2023_divisors n j ∧ (n + j = 60466182) :=
begin
  sorry
end

end math_problem_solution_l807_807680


namespace regular_octagon_interior_angle_l807_807331

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807331


namespace regular_octagon_interior_angle_l807_807298

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807298


namespace interior_angle_regular_octagon_l807_807495

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807495


namespace angle_complement_supplement_l807_807824

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l807_807824


namespace steve_needs_28_feet_of_wood_l807_807651

theorem steve_needs_28_feet_of_wood :
  (6 * 4) + (2 * 2) = 28 := by
  sorry

end steve_needs_28_feet_of_wood_l807_807651


namespace find_a_l807_807075

theorem find_a (a : ℕ) : a * 24 = 173 * 240 → a = 1730 :=
by
  intro h,
  sorry

end find_a_l807_807075


namespace lateral_surface_area_l807_807690

open Real

-- The sine of the dihedral angle at the lateral edge of a regular quadrilateral pyramid
def sin_dihedral_angle : ℝ := 15 / 17

-- The area of the pyramid's diagonal section
def area_diagonal_section : ℝ := 3 * sqrt 34

-- The statement that we need to prove
theorem lateral_surface_area (sin_dihedral_angle = 15 / 17) (area_diagonal_section = 3 * sqrt 34) : 
  lateral_surface_area = 68 :=
sorry

end lateral_surface_area_l807_807690


namespace angle_measure_is_60_l807_807790

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l807_807790


namespace proof_problem_l807_807013

-- Define the problem conditions
def condition (a b x : ℝ) : Prop :=
  (ax + 1) / (x + b) > 1 

-- Define the inequality to be proved under the conditions
def inequality (a b x : ℝ) : Prop :=
  x^2 + a * x - 2 * b < 0

-- Define the given solution set for the fraction inequality
def solution_set_fraction : set ℝ :=
  {x | x < -1} ∪ {x | x > 3}

-- Define the solution set for the quadratic inequality to be proved
def solution_set_quadratic : set ℝ :=
  {x | -3 < x ∧ x < -2}

-- State the main theorem
theorem proof_problem (a b : ℝ) (h : ∀ x, condition a b x → x ∈ solution_set_fraction) :
  ∀ x, inequality a b x → x ∈ solution_set_quadratic :=
by
  sorry

end proof_problem_l807_807013


namespace cans_restocked_after_second_day_l807_807999

theorem cans_restocked_after_second_day :
  let initial_cans := 2000
  let first_day_taken := 500 
  let first_day_restock := 1500
  let second_day_taken := 1000 * 2
  let total_given_away := 2500
  let remaining_after_second_day_before_restock := initial_cans - first_day_taken + first_day_restock - second_day_taken
  (total_given_away - remaining_after_second_day_before_restock) = 1500 := 
by {
  sorry
}

end cans_restocked_after_second_day_l807_807999


namespace solution_sets_equal_l807_807596

noncomputable def f : ℝ → ℝ := sorry

def P : set ℝ := {x | f x = x}

def Q : set ℝ := {x | f (f x) = x}

theorem solution_sets_equal (hf_bij : Function.Bijective f) (hf_inc : ∀ x y, x < y → f x < f y) :
  P = Q :=
by
  sorry

end solution_sets_equal_l807_807596


namespace restore_grid_values_l807_807771

def gridCondition (A B C D : Nat) : Prop :=
  A < 9 ∧ 1 < 9 ∧ 9 < 9 ∧
  3 < 9 ∧ 5 < 9 ∧ D < 9 ∧
  B < 9 ∧ C < 9 ∧ 7 < 9 ∧
  -- Sum of any two numbers in adjacent cells is less than 12
  A + 1 < 12 ∧ A + 3 < 12 ∧
  B + C < 12 ∧ B + 3 < 12 ∧
  D + 2 < 12 ∧ C + 4 < 12 ∧
  -- The grid values are from 1 to 9
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  C ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
  -- Even numbers erased are 2, 4, 6, and 8

theorem restore_grid_values :
  ∃ (A B C D : Nat), gridCondition A B C D ∧ A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  sorry

end restore_grid_values_l807_807771


namespace find_width_of_room_l807_807571

variable (length : ℕ) (total_carpet_owned : ℕ) (additional_carpet_needed : ℕ)
variable (total_area : ℕ) (width : ℕ)

theorem find_width_of_room
  (h1 : length = 11) 
  (h2 : total_carpet_owned = 16) 
  (h3 : additional_carpet_needed = 149)
  (h4 : total_area = total_carpet_owned + additional_carpet_needed) 
  (h5 : total_area = length * width) :
  width = 15 := by
    sorry

end find_width_of_room_l807_807571


namespace restore_grid_l807_807764

noncomputable def grid_values (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 9 < 12 ∧
  3 + 1 < 12 ∧ 3 + 5 < 12 ∧
  B + 3 < 12 ∧ B + 4 < 12 ∧
  B + 7 < 12 ∧ 
  C + 4 < 12 ∧
  C + 7 < 12 ∧ 
  D + 9 < 12 ∧
  D + 7 < 12 ∧
  D + 5 < 12

def even_numbers_erased (A B C D : ℕ) : Prop :=
  A ∈ {8} ∧ B ∈ {6} ∧ C ∈ {4} ∧ D ∈ {2}

theorem restore_grid :
  ∃ (A B C D : ℕ), grid_values A B C D ∧ even_numbers_erased A B C D :=
by {
  use [8, 6, 4, 2],
  dsimp [grid_values, even_numbers_erased],
  exact ⟨⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩, 
         ⟨rfl, rfl, rfl, rfl⟩⟩
}

end restore_grid_l807_807764


namespace percentage_of_girls_l807_807020

theorem percentage_of_girls :
  let classA_boys := 15
  let classA_girls := 20
  let classB_boys := 25
  let classB_girls := 35
  let classC_boys := 30
  let classC_girls := 40
  let classD_boys := 35
  let classD_girls := 45

  let classA_new_boys := classA_boys - 5 + 3 + 3
  let classA_new_girls := classA_girls - 4 + 7 + 6
  let classB_new_boys := classB_boys + 5 - 2 + 3
  let classB_new_girls := classB_girls + 4 - 5 + 6
  let classC_new_boys := classC_boys + 2 - 4
  let classC_new_girls := classC_girls + 5 - 6
  let classD_new_boys := classD_boys + 4 - 3
  let classD_new_girls := classD_girls + 6 - 7

  let total_boys := classA_new_boys + classB_new_boys + classC_new_boys + classD_new_boys
  let total_girls := classA_new_girls + classB_new_girls + classC_new_girls + classD_new_girls
  let total_students := total_boys + total_girls
  let percentage_girls := (total_girls : ℚ) / (total_students : ℚ) * 100

  percentage_girls ≈ 57.79 := 
by 
-- sorry to skip the proof
sorry

end percentage_of_girls_l807_807020


namespace angle_supplement_complement_l807_807889

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l807_807889


namespace final_amount_after_bets_l807_807970

theorem final_amount_after_bets
  (M0 : ℕ)
  (win_first_increase : ℕ → ℕ := λ x, x * 16 / 10)
  (loss_decrease : ℕ → ℕ := λ x, x / 2)
  (subsequent_win_increase : ℕ → ℕ := λ x, x * 15 / 10)
  (order : list (ℕ → ℕ)) :
  (M0 = 100) →
  (order = [win_first_increase, loss_decrease, subsequent_win_increase, loss_decrease]) →
  (order.foldl (λ m f, f m) M0 = 60) :=
begin
  intros,
  sorry
end

end final_amount_after_bets_l807_807970


namespace constant_sum_of_polynomials_l807_807597

theorem constant_sum_of_polynomials {R : Type*} [CommRing R] {h f g : R[X]} 
  (h_non_const : h.degree > 0) 
  (h_neq : f ≠ g)
  (h_eq : h.eval₂ f = h.eval₂ g) : 
  ∃ c : R, f + g = c :=
sorry

end constant_sum_of_polynomials_l807_807597


namespace circle_equation_l807_807157

theorem circle_equation 
  (P : ℝ × ℝ)
  (h1 : ∀ a : ℝ, (1 - a) * 2 + (P.snd) + 2 * a - 1 = 0)
  (h2 : P = (2, -1)) :
  ∀ x y : ℝ, (x - 2)^2 + (y + 1)^2 = 4 ↔ x^2 + y^2 - 4*x + 2*y + 1 = 0 :=
by sorry

end circle_equation_l807_807157


namespace interior_angle_regular_octagon_l807_807442

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807442


namespace tan_addition_identity_l807_807699

theorem tan_addition_identity 
  (tan_30 : Real := Real.tan (Real.pi / 6))
  (tan_15 : Real := 2 - Real.sqrt 3) : 
  tan_15 + tan_30 + tan_15 * tan_30 = 1 := 
by
  have h1 : tan_30 = Real.sqrt 3 / 3 := sorry
  have h2 : tan_15 = 2 - Real.sqrt 3 := sorry
  sorry

end tan_addition_identity_l807_807699


namespace regular_octagon_interior_angle_l807_807240

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807240


namespace smallest_brownie_pan_size_l807_807229

theorem smallest_brownie_pan_size :
  ∃ s : ℕ, (s - 2) ^ 2 = 4 * s - 4 ∧ ∀ t : ℕ, (t - 2) ^ 2 = 4 * t - 4 → s <= t :=
by
  sorry

end smallest_brownie_pan_size_l807_807229


namespace each_interior_angle_of_regular_octagon_l807_807352

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807352


namespace regular_octagon_interior_angle_l807_807261

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807261


namespace probability_of_xyz_eq_72_l807_807917

open ProbabilityTheory Finset

def dice_values : Finset ℕ := {1, 2, 3, 4, 5, 6}

theorem probability_of_xyz_eq_72 :
  (∑ x in dice_values, ∑ y in dice_values, ∑ z in dice_values, 
   if x * y * z = 72 then 1 else 0) / (dice_values.card ^ 3) = 1 / 36 :=
by
  sorry -- Proof omitted

end probability_of_xyz_eq_72_l807_807917


namespace angle_measure_l807_807839

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l807_807839


namespace regular_octagon_interior_angle_deg_l807_807375

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807375


namespace each_interior_angle_of_regular_octagon_l807_807347

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807347


namespace regular_octagon_angle_l807_807313

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807313


namespace train_crossing_time_l807_807948

-- Definitions based on conditions
def train_length : ℝ := 110 -- in meters
def bridge_length : ℝ := 150 -- in meters
def speed_kmph : ℝ := 36 -- in kilometers per hour

-- Conversion factor from kmph to m/s
def kmph_to_mps (v : ℝ) : ℝ := v * (1000/3600)

-- Speed in m/s
def speed_mps : ℝ := kmph_to_mps speed_kmph

-- Total distance to be covered
def total_distance : ℝ := train_length + bridge_length

-- Time to cross the bridge
def crossing_time (distance : ℝ) (speed : ℝ) : ℝ := distance / speed

-- Known results for verification
def expected_time : ℝ := 26

-- The statement we want to prove
theorem train_crossing_time :
  crossing_time total_distance speed_mps = expected_time :=
by
  -- Skipping the proof here
  sorry

end train_crossing_time_l807_807948


namespace find_result_l807_807227

variables {a b : ℝ^3}
#check ℝ

-- Setting up conditions
def is_unit_vector (v : ℝ^3) : Prop := ∥v∥ = 1
def given_condition : Prop := is_unit_vector a ∧ is_unit_vector b ∧ ∥3 • a - 2 • b∥ = 3

-- Main statement to be proven
theorem find_result (h : given_condition) : ∥3 • a + b∥ = 2 * real.sqrt 3 :=
sorry

end find_result_l807_807227


namespace regular_octagon_interior_angle_l807_807478

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807478


namespace smallest_positive_x_for_max_g_l807_807997

theorem smallest_positive_x_for_max_g :
  ∃ x : ℝ, 
    (∀ y : ℝ, (g y = (Real.sin (y / 4)) + (Real.sin (y / 9)) → g x ≥ g y)) ∧ x = 13050 :=
by
  let g := λ x : ℝ, Real.sin (x / 4) + Real.sin (x / 9)
  have h_max : ∀ x, g x ≤ 2 := sorry  -- Using the property of sin function 
  use 13050
  split
  · intro y h_eq
    sorry
  · rfl

end smallest_positive_x_for_max_g_l807_807997


namespace interior_angle_regular_octagon_l807_807484

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807484


namespace sum_xi_sq_lt_l807_807951

theorem sum_xi_sq_lt (n : ℕ) (x : Fin n → ℝ) (h1 : 2 ≤ n) 
  (h2 : ∑ i, x i = 0) 
  (h3 : ∀ t > 0, ∃ m ≤ (1/t : ℝ), (x = 1 / t (1 : ℝ), ∀ (i j : Fin n), (| x i - x j | < t))):
  ∑ i, (x i)^2 < (1 / n) * (Finset.univ.max' _ (λ i, x i) - Finset.univ.min' _ (λ i, x i)) := 
by 
  sorry

end sum_xi_sq_lt_l807_807951


namespace sum_series_l807_807122

theorem sum_series :
  3 * (List.sum (List.map (λ n => n - 1) (List.range' 2 14))) = 273 :=
by
  sorry

end sum_series_l807_807122


namespace interior_angle_regular_octagon_l807_807451

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807451


namespace steve_needs_28_feet_of_wood_l807_807652

theorem steve_needs_28_feet_of_wood :
  (6 * 4) + (2 * 2) = 28 := by
  sorry

end steve_needs_28_feet_of_wood_l807_807652


namespace max_value_of_quadratic_exists_r_for_max_value_of_quadratic_l807_807779

theorem max_value_of_quadratic (r : ℝ) : -7 * r ^ 2 + 50 * r - 20 ≤ 5 :=
by sorry

theorem exists_r_for_max_value_of_quadratic : ∃ r : ℝ, -7 * r ^ 2 + 50 * r - 20 = 5 :=
by sorry

end max_value_of_quadratic_exists_r_for_max_value_of_quadratic_l807_807779


namespace number_of_terms_l807_807206

variable {α : Type} [LinearOrderedField α]

def sum_of_arithmetic_sequence (a₁ aₙ d : α) (n : ℕ) : α :=
  n * (a₁ + aₙ) / 2

theorem number_of_terms (a₁ aₙ : α) (d : α) (n : ℕ)
  (h₀ : 4 * (2 * a₁ + 3 * d) / 2 = 21)
  (h₁ : 4 * (2 * aₙ - 3 * d) / 2 = 67)
  (h₂ : sum_of_arithmetic_sequence a₁ aₙ d n = 286) :
  n = 26 :=
sorry

end number_of_terms_l807_807206


namespace restore_grid_values_l807_807746

def is_adjacent_sum_lt_12 (A B C D : ℕ) : Prop :=
  (A + 1 < 12) ∧ (A + 9 < 12) ∧
  (3 + 1 < 12) ∧ (3 + 5 < 12) ∧
  (3 + D < 12) ∧ (5 + 1 < 12) ∧ 
  (5 + D < 12) ∧ (B + C < 12) ∧
  (B + 3 < 12) ∧ (B + 5 < 12) ∧
  (C + 5 < 12) ∧ (C + 7 < 12) ∧
  (D + 9 < 12) ∧ (D + 7 < 12)

theorem restore_grid_values : 
  ∃ (A B C D : ℕ), 
  (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) ∧ 
  is_adjacent_sum_lt_12 A B C D := 
by 
  exists 8
  exists 6
  exists 4
  exists 2
  split; [refl, split; [refl, split; [refl, split; [refl, sorry]]]] 

end restore_grid_values_l807_807746


namespace regular_octagon_interior_angle_l807_807243

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807243


namespace regular_octagon_interior_angle_l807_807277

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807277


namespace angle_solution_l807_807882

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l807_807882


namespace largest_multiple_of_9_negation_gt_neg_100_l807_807043

theorem largest_multiple_of_9_negation_gt_neg_100 : ∃ n : ℤ, (n % 9 = 0) ∧ (-n > -100) ∧ (∀ m : ℤ, (m % 9 = 0) ∧ (-m > -100) → m ≤ n) :=
by {
  use 99,
  split,
  { exact dec_trivial },
  split,
  { norm_num },
  { intros m hm h_neg,
    rw [int.dvd_iff_mod_eq_zero, ←int.neg_lt_neg_iff, neg_neg] at hm h_neg,
    have : m ≤ 99 := sorry, -- Here should be the verification step
    exact this }
}

end largest_multiple_of_9_negation_gt_neg_100_l807_807043


namespace workers_read_saramago_l807_807535

theorem workers_read_saramago (s : ℚ)
    (h1 : 72 = 72)
    (h2 : 45 = 5 * 72 / 8)
    (h3 : 4 = 4)
    (h4 : 72 * s - 5 = 72 - 72 * s - 45)
    : s = 1 / 4 :=
begin
  sorry
end

end workers_read_saramago_l807_807535


namespace interior_angle_regular_octagon_l807_807363

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807363


namespace sum_of_fourth_powers_l807_807513

theorem sum_of_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 2)
  (h2 : a^2 + b^2 + c^2 = 5)
  (h3 : a^3 + b^3 + c^3 = 8) :
  a^4 + b^4 + c^4 = 18.5 :=
sorry

end sum_of_fourth_powers_l807_807513


namespace distance_between_cars_l807_807706

theorem distance_between_cars (d : ℝ) :
  ∃ d = 350,
  (∀ (t : ℝ) (v_a v_b : ℝ), t = 2 → v_a = 60 → v_b = 45 →
  let distance_traveled := (v_a + v_b) * t in
  let fraction := 3 / 5 in
    distance_traveled / fraction = d) := 
sorry

end distance_between_cars_l807_807706


namespace regular_octagon_interior_angle_l807_807267

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807267


namespace lottery_ticket_no_guarantee_l807_807529

theorem lottery_ticket_no_guarantee
  (p : ℝ)
  (tickets : ℕ)
  (prob_win : p = 1/100000)
  (enough_tickets : ∃ n : ℕ, n ≥ tickets) :
  ¬ (tickets = 100000 → (∀ n, n ≤ tickets → n ≠ 0 → p ≠ 1/n)) :=
begin
  sorry
end

end lottery_ticket_no_guarantee_l807_807529


namespace regular_octagon_interior_angle_l807_807233

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807233


namespace joanie_loan_difference_l807_807572

theorem joanie_loan_difference:
  let P := 6000
  let r := 0.12
  let t := 4
  let n_quarterly := 4
  let n_annually := 1
  let A_quarterly := P * (1 + r / n_quarterly)^(n_quarterly * t)
  let A_annually := P * (1 + r / n_annually)^t
  A_quarterly - A_annually = 187.12 := sorry

end joanie_loan_difference_l807_807572


namespace determine_grid_numbers_l807_807721

theorem determine_grid_numbers (A B C D : ℕ) :
  (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧ (1 ≤ D ∧ D ≤ 9) ∧ 
  (A ≠ 1) ∧ (A ≠ 3) ∧ (A ≠ 5) ∧ (A ≠ 7) ∧ (A ≠ 9) ∧ 
  (B ≠ 1) ∧ (B ≠ 3) ∧ (B ≠ 5) ∧ (B ≠ 7) ∧ (B ≠ 9) ∧ 
  (C ≠ 1) ∧ (C ≠ 3) ∧ (C ≠ 5) ∧ (C ≠ 7) ∧ (C ≠ 9) ∧ 
  (D ≠ 1) ∧ (D ≠ 3) ∧ (D ≠ 5) ∧ (D ≠ 7) ∧ (D ≠ 9) ∧ 
  (A ≠ 2 ∧ A ≠ 4 ∧ A ≠ 6 ∧ A ≠ 8 ∨ B ≠ 2 ∧ B ≠ 4 ∧ B ≠ 6 ∧ B ≠ 8 ∨ C ≠ 2 ∧ C ≠ 4 ∧ C ≠ 6 ∧ C ≠ 8 ∨ D ≠ 2 ∧ D ≠ 4 ∧ D ≠ 6 ∧ D ≠ 8) ∧ 
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ 
  (3 + 5 < 12) ∧ (3 + D < 12) ∧ (B + 5 < 12) ∧ 
  (B + C < 12) ∧ (C + 7 < 12) ∧ (B + 7 < 12) 
  → (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) :=
by 
  intros A B C D
  exact sorry

end determine_grid_numbers_l807_807721


namespace tan_of_second_quadrant_angle_l807_807208

theorem tan_of_second_quadrant_angle (α : ℝ) (h1 : π / 2 < α ∧ α < π) (h2 : sin (π - α) = 3 / 5) : tan α = -3 / 4 :=
by
  sorry

end tan_of_second_quadrant_angle_l807_807208


namespace angle_measure_l807_807835

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 :=
by {
  sorry
}

end angle_measure_l807_807835


namespace sin_double_angle_l807_807516

theorem sin_double_angle (α : ℝ) (h : cos (π / 4 - α) = 3 / 5) : sin (2 * α) = -7 / 25 :=
by sorry

end sin_double_angle_l807_807516


namespace scientific_notation_l807_807118

theorem scientific_notation :
  56.9 * 10^9 = 5.69 * 10^(10 - 1) :=
by
  sorry

end scientific_notation_l807_807118


namespace interior_angle_regular_octagon_l807_807439

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807439


namespace initial_members_count_l807_807663

theorem initial_members_count (n : ℕ) (W : ℕ)
  (h1 : W = n * 48)
  (h2 : W + 171 = (n + 2) * 51) : 
  n = 23 :=
by sorry

end initial_members_count_l807_807663


namespace arithmetic_geometric_seq_l807_807006

theorem arithmetic_geometric_seq (A B C D : ℤ) (h_arith_seq : A + C = 2 * B)
  (h_geom_seq : B * D = C * C) (h_frac : 7 * B = 3 * C) (h_posB : B > 0)
  (h_posC : C > 0) (h_posD : D > 0) : A + B + C + D = 76 :=
sorry

end arithmetic_geometric_seq_l807_807006


namespace elsie_no_refill_l807_807141

theorem elsie_no_refill (initial_wipes : ℕ) (wipes_left : ℕ) (wipes_used : ℕ) 
  (initial_wipes_eq : initial_wipes = 70) 
  (wipes_left_eq : wipes_left = 60)
  (wipes_used_eq : wipes_used = initial_wipes - wipes_left) 
  (wipes_used_le_20 : wipes_used ≤ 20):
  wipes_used = 10 → 0 = 0 :=
by
  intros h
  exact rfl
  sorry

end elsie_no_refill_l807_807141


namespace parallelogram_area_sum_l807_807215

-- Define the quadratic equations
def quad1 (z : ℂ) : Prop := z^2 = 4 + 4 * complex.I * real.sqrt 15
def quad2 (z : ℂ) : Prop := z^2 = 2 + 2 * complex.I * real.sqrt 3

-- Define the area of the parallelogram
noncomputable def parallelogram_area (z1 z2 z3 z4 : ℂ) : ℝ :=
  complex.abs ((z1 * complex.conj z3) - (z2 * complex.conj z4)).im

-- This is the statement that needs to be proven
theorem parallelogram_area_sum :
  let z1 := √10 + √6 * complex.I
  let z2 := -√10 - √6 * complex.I
  let z3 := √3 + complex.I
  let z4 := -√3 - complex.I
  let S := parallelogram_area z1 z2 z3 z4
  ∃ (p q r s : ℕ), S = p * real.sqrt q - r * real.sqrt s ∧
  p + q + r + s = 20 :=
sorry

end parallelogram_area_sum_l807_807215


namespace triangle_area_ab_l807_807521

theorem triangle_area_ab (a b : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (hline : ∀ x y : ℝ, 2 * a * x + 3 * b * y = 12) (harea : (1/2) * (6 / a) * (4 / b) = 9) : 
    a * b = 4 / 3 :=
by 
  sorry

end triangle_area_ab_l807_807521


namespace time_to_cross_bridge_l807_807978

-- Define the given conditions in Lean.
def length_of_train : ℝ := 140
def speed_of_train_kmph : ℝ := 45
def length_of_bridge : ℝ := 235

-- Define the conversion factor from km/hr to m/s.
def kmph_to_mps (speed: ℝ) : ℝ := speed * (1000 / 3600)

-- Define the speed in meters per second.
def speed_of_train_mps : ℝ := kmph_to_mps speed_of_train_kmph

-- Define the total distance the train needs to cover.
def total_distance : ℝ := length_of_train + length_of_bridge

-- The goal is to prove that the time to cross the bridge is 30 seconds.
theorem time_to_cross_bridge : total_distance / speed_of_train_mps = 30 := by
  sorry

end time_to_cross_bridge_l807_807978


namespace regular_octagon_interior_angle_l807_807234

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807234


namespace each_interior_angle_of_regular_octagon_l807_807354

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807354


namespace kylie_apple_picking_l807_807575

theorem kylie_apple_picking : 
  let first_hour := 66 in
  let fibonacci_sum := 1 + 1 + 2 in
  let second_hour := fibonacci_sum * first_hour in
  let a1 := first_hour in
  let d := 10 in
  let a2 := a1 + d in
  let a3 := a2 + d in
  let third_hour := a1 + a2 + a3 in
  first_hour + second_hour + third_hour = 558 := 
by
  sorry

end kylie_apple_picking_l807_807575


namespace regular_octagon_angle_l807_807305

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807305


namespace perimeter_inequality_l807_807952

variable (A B C P : Point)

-- Definitions of the sides of the triangle ABC
def AB := distance A B
def BC := distance B C
def CA := distance C A

-- Definition of the sum of the sides of the triangle ABC being equal to 2s
def s : ℝ := (AB + BC + CA) / 2

-- Definitions of the distances from P to the vertices of the triangle ABC
def AP := distance A P
def BP := distance B P
def CP := distance C P

-- For any point P inside the triangle ABC, prove that s < AP + BP + CP < 2s
theorem perimeter_inequality (h : P ∈ triangle ABC) :
  s < AP + BP + CP ∧ AP + BP + CP < 2 * s := by
  sorry

end perimeter_inequality_l807_807952


namespace paint_all_stones_black_l807_807019

def can_paint_all_black (k : ℕ) : Prop :=
  (k = 1) ∨ (k ≠ 4 * m + 1) ∀ (m : ℕ), (1 ≤ m ∧ m ≤ 12)

theorem paint_all_stones_black (k : ℕ) : 
  1 ≤ k ∧ k ≤ 50 → can_paint_all_black k :=
sorry

end paint_all_stones_black_l807_807019


namespace angle_supplement_complement_l807_807807

theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by sorry

end angle_supplement_complement_l807_807807


namespace inequality_am_gm_l807_807170

theorem inequality_am_gm (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
sorry

end inequality_am_gm_l807_807170


namespace angle_supplement_complement_l807_807891

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l807_807891


namespace regular_octagon_interior_angle_l807_807333

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807333


namespace regular_octagon_interior_angle_l807_807256

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807256


namespace prob_xyz_eq_72_l807_807926

-- Define the set of possible outcomes for a standard six-sided die
def dice_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define a predicate that checks if three dice rolls multiply to 72
def is_valid_combination (x y z : ℕ) : Prop := (x * y * z = 72)

-- Define the event space for three dice rolls
def event_space : Finset (ℕ × ℕ × ℕ) := Finset.product dice_outcomes (Finset.product dice_outcomes dice_outcomes)

-- Define the probability of an event
def probability {α : Type*} [Fintype α] (s : Finset α) (event : α → Prop) : ℚ :=
  (s.filter event).card.to_rat / s.card.to_rat

-- State the theorem
theorem prob_xyz_eq_72 : probability event_space (λ t, is_valid_combination t.1 t.2.1 t.2.2) = (7 / 216) := 
by { sorry }

end prob_xyz_eq_72_l807_807926


namespace regular_octagon_interior_angle_l807_807273

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807273


namespace angle_measure_is_60_l807_807791

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l807_807791


namespace regular_octagon_interior_angle_l807_807327

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807327


namespace angle_measure_is_60_l807_807780

theorem angle_measure_is_60 (x : ℝ)
  (h1 : 180 - x = 4 * (90 - x)) : 
  x = 60 := 
by 
  sorry

end angle_measure_is_60_l807_807780


namespace percentage_reduction_hours_l807_807947

-- To define the percentage decrease we first set up our constants and variables
variables (W H : ℝ) -- Original hourly wage and hours worked
variable (H' : ℝ) -- New number of hours worked
variable (weekly_income : ℝ) -- Total weekly income
noncomputable def hourly_wage_increase : ℝ := 1.25 * W -- 25% increase

-- Conditions that the weekly income should remain the same
axiom unchanged_income : weekly_income = W * H
axiom unchanged_income_after_increase : weekly_income = hourly_wage_increase * H'

-- Define the percentage reduction
noncomputable def percentage_reduction (H H' : ℝ) : ℝ := 100 * (H - H') / H

-- The statement of the theorem representing the proof problem
theorem percentage_reduction_hours :
  percentage_reduction H H' = 20
:=
sorry

end percentage_reduction_hours_l807_807947


namespace vessel_capacity_is_10_l807_807112

-- Given conditions
structure Vessel :=
  (capacity : ℝ)
  (alcohol_concentration : ℝ)

constant vessel1 : Vessel := ⟨2, 0.30⟩
constant vessel2 : Vessel := ⟨6, 0.45⟩
constant total_mixture : ℝ := 8
constant new_concentration : ℝ := 0.33

-- Definition of total volume V
def total_volume (V : ℝ) : Prop :=
  let alcohol_content_vessel1 := vessel1.alcohol_concentration * vessel1.capacity in
  let alcohol_content_vessel2 := vessel2.alcohol_concentration * vessel2.capacity in
  let total_alcohol_content := alcohol_content_vessel1 + alcohol_content_vessel2 in
  new_concentration * V = total_alcohol_content

-- The statement to prove
theorem vessel_capacity_is_10 : ∃ V : ℝ, total_volume V ∧ V = 10 := by
  let V := 10
  have total_alcohol_content := 0.30 * 2 + 0.45 * 6
  have total_volume_eq := new_concentration * V = total_alcohol_content
  show ∃ V, total_volume V ∧ V = 10 from
    exists.intro V (and.intro total_volume_eq rfl)

end vessel_capacity_is_10_l807_807112


namespace interior_angle_regular_octagon_l807_807436

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807436


namespace regular_octagon_interior_angle_l807_807472

theorem regular_octagon_interior_angle:
  ∀ (n : ℕ) (h : n = 8), (sum_of_interior_angles_deg (n - 2) / n = 135) :=
begin
  intros n h,
  rw h,
  unfold sum_of_interior_angles_deg,
  sorry
end

end regular_octagon_interior_angle_l807_807472


namespace angle_measure_l807_807844

theorem angle_measure (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := by
  sorry

end angle_measure_l807_807844


namespace regular_octagon_interior_angle_l807_807281

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (∀ i, 1 ≤ i ∧ i ≤ n → 135 = (180 * (n - 2)) / n) := by
  intros n h1 i h2
  sorry

end regular_octagon_interior_angle_l807_807281


namespace smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m_l807_807503

noncomputable def smaller_angle_at_715 : ℝ :=
  let hour_position := 7 * 30 + 30 / 4
  let minute_position := 15 * (360 / 60)
  let angle_between := abs (hour_position - minute_position)
  if angle_between > 180 then 360 - angle_between else angle_between

theorem smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m :
  smaller_angle_at_715 = 127.5 := 
sorry

end smaller_angle_formed_by_hour_and_minute_hands_at_7_15_p_m_l807_807503


namespace prob_xyz_eq_72_l807_807928

-- Define the set of possible outcomes for a standard six-sided die
def dice_outcomes : Finset ℕ := {1, 2, 3, 4, 5, 6}

-- Define a predicate that checks if three dice rolls multiply to 72
def is_valid_combination (x y z : ℕ) : Prop := (x * y * z = 72)

-- Define the event space for three dice rolls
def event_space : Finset (ℕ × ℕ × ℕ) := Finset.product dice_outcomes (Finset.product dice_outcomes dice_outcomes)

-- Define the probability of an event
def probability {α : Type*} [Fintype α] (s : Finset α) (event : α → Prop) : ℚ :=
  (s.filter event).card.to_rat / s.card.to_rat

-- State the theorem
theorem prob_xyz_eq_72 : probability event_space (λ t, is_valid_combination t.1 t.2.1 t.2.2) = (7 / 216) := 
by { sorry }

end prob_xyz_eq_72_l807_807928


namespace difference_of_profit_share_l807_807065

theorem difference_of_profit_share (a b c : ℕ) (pa pb pc : ℕ) (profit_b : ℕ) 
  (a_capital : a = 8000) (b_capital : b = 10000) (c_capital : c = 12000) 
  (b_profit_share : profit_b = 1600)
  (investment_ratio : pa / 4 = pb / 5 ∧ pb / 5 = pc / 6) :
  pa - pc = 640 := 
sorry

end difference_of_profit_share_l807_807065


namespace num_four_digit_integers_divisible_by_7_l807_807507

theorem num_four_digit_integers_divisible_by_7 :
  ∃ n : ℕ, n = 1286 ∧ ∀ k : ℕ, (1000 ≤ k ∧ k ≤ 9999) → (k % 7 = 0 ↔ ∃ m : ℕ, k = m * 7) :=
by {
  sorry
}

end num_four_digit_integers_divisible_by_7_l807_807507


namespace restore_grid_values_l807_807770

def gridCondition (A B C D : Nat) : Prop :=
  A < 9 ∧ 1 < 9 ∧ 9 < 9 ∧
  3 < 9 ∧ 5 < 9 ∧ D < 9 ∧
  B < 9 ∧ C < 9 ∧ 7 < 9 ∧
  -- Sum of any two numbers in adjacent cells is less than 12
  A + 1 < 12 ∧ A + 3 < 12 ∧
  B + C < 12 ∧ B + 3 < 12 ∧
  D + 2 < 12 ∧ C + 4 < 12 ∧
  -- The grid values are from 1 to 9
  A ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  B ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  C ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9} ∧
  D ∈ {1, 2, 3, 4, 5, 6, 7, 8, 9}
  -- Even numbers erased are 2, 4, 6, and 8

theorem restore_grid_values :
  ∃ (A B C D : Nat), gridCondition A B C D ∧ A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  sorry

end restore_grid_values_l807_807770


namespace solve_for_x_l807_807046

theorem solve_for_x (x : ℝ) (y : ℝ) (z : ℝ) (h1 : y = 1) (h2 : z = 3) (h3 : x^2 * y * z - x * y * z^2 = 6) :
  x = -2 / 3 ∨ x = 3 :=
by sorry

end solve_for_x_l807_807046


namespace restore_grid_l807_807710

-- Definitions for each cell in the grid.
variable (A B C D : ℕ)

-- The given grid and conditions
noncomputable def grid : list (list ℕ) :=
  [[A, 1, 9],
   [3, 5, D],
   [B, C, 7]]

-- The condition that the sum of numbers in adjacent cells is less than 12.
def valid (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 3 < 12 ∧
  1 + 9 < 12 ∧ 5 + D < 12 ∧
  3 + 5 < 12 ∧  B + C < 12 ∧
  9 + D < 12 ∧ 7 + C < 12 ∧
  9 + 1 < 12 ∧ B + 7 < 12

-- Assertion to be proved.
theorem restore_grid (A B C D : ℕ) (valid : valid A B C D) :
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 := by
  sorry

end restore_grid_l807_807710


namespace smallest_a_condition_l807_807591

theorem smallest_a_condition
  (a b : ℝ)
  (h_nonneg_a : 0 ≤ a)
  (h_nonneg_b : 0 ≤ b)
  (h_eq : ∀ x : ℝ, Real.sin (a * x + b) = Real.sin (15 * x)) :
  a = 15 :=
sorry

end smallest_a_condition_l807_807591


namespace planes_touch_three_spheres_count_l807_807104

-- Declare the conditions as definitions
def square_side_length : ℝ := 10
def radii : Fin 4 → ℝ
| 0 => 1
| 1 => 2
| 2 => 4
| 3 => 3

-- The proof problem statement
theorem planes_touch_three_spheres_count :
    ∃ (planes_that_touch_three_spheres : ℕ) (planes_that_intersect_fourth_sphere : ℕ),
    planes_that_touch_three_spheres = 26 ∧ planes_that_intersect_fourth_sphere = 8 := 
by
  -- sorry skips the proof
  sorry

end planes_touch_three_spheres_count_l807_807104


namespace quadrilateral_area_is_correct_l807_807544

structure Point (α : Type _) :=
  (x : α)
  (y : α)

def area_of_quadrilateral (A B C D : Point ℝ) : ℝ :=
  let area_triangle (P Q R : Point ℝ) : ℝ :=
    |((P.x * Q.y + Q.x * R.y + R.x * P.y) - (P.y * Q.x + Q.y * R.x + R.y * P.x)) / 2|
  area_triangle A B C + area_triangle A C D

noncomputable def quad_area : ℝ :=
  let A := Point.mk 3.5 (-2.7)
  let B := Point.mk 4.2 8.5
  let C := Point.mk 12.9 5.3
  let D := Point.mk 11.6 (-3.4)
  area_of_quadrilateral A B C D

theorem quadrilateral_area_is_correct :
  quad_area = 106.03 := 
sorry

end quadrilateral_area_is_correct_l807_807544


namespace regular_octagon_interior_angle_l807_807338

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807338


namespace find_grid_values_l807_807755

-- Define the grid and the conditions in Lean
variables (A B C D : ℕ)
assume h1: A, B, C, D ∈ {2, 4, 6, 8}
assume h2: A + 1 < 12
assume h3: A + 9 < 12
assume h4: A + 3 < 12
assume h5: A + 5 < 12
assume h6: A + D < 12
assume h7: B + C < 12
assume h8: B + 5 < 12
assume h9: C + 7 < 12
assume h10: D + 9 < 12

-- Prove the specific values for A, B, C, and D
theorem find_grid_values : 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by sorry

end find_grid_values_l807_807755


namespace train_crossing_time_l807_807977

def train_length := 140
def train_speed_kmph := 45
def bridge_length := 235
def speed_to_mps (kmph : ℕ) : ℕ := (kmph * 1000) / 3600
def total_distance := train_length + bridge_length
def train_speed := speed_to_mps train_speed_kmph
def time_to_cross := total_distance / train_speed

theorem train_crossing_time : time_to_cross = 30 := by
  sorry

end train_crossing_time_l807_807977


namespace interior_angle_regular_octagon_l807_807463

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807463


namespace angle_supplement_complement_l807_807892

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l807_807892


namespace shorter_diagonal_of_trapezoid_l807_807705

-- Define the lengths of the sides of the trapezoid
def AB : ℝ := 40
def CD : ℝ := 28
def AD : ℝ := 13
def BC : ℝ := 17

-- Define the length of the shorter diagonal
def shorter_diagonal_length : ℝ := 30.35

-- State the main theorem which asserts that the length of the shorter diagonal of the trapezoid is 30.35
theorem shorter_diagonal_of_trapezoid : 
  ∀ {AB CD AD BC : ℝ},
  AB = 40 ∧ CD = 28 ∧ AD = 13 ∧ BC = 17 →
  (AC ≤ BD) ∧ (AC = 30.35 ∨ BD = 30.35) :=
by {
  intro _,
  sorry
}

end shorter_diagonal_of_trapezoid_l807_807705


namespace age_difference_l807_807091

-- Define the present age of the son as a constant
def S : ℕ := 22

-- Define the equation given by the problem
noncomputable def age_relation (M : ℕ) : Prop :=
  M + 2 = 2 * (S + 2)

-- The theorem to prove the man is 24 years older than his son
theorem age_difference (M : ℕ) (h_rel : age_relation M) : M - S = 24 :=
by {
  sorry
}

end age_difference_l807_807091


namespace isabelle_work_weeks_l807_807566

-- Define the costs and savings
def isabelle_ticket_cost := 20
def brother_ticket_cost := 10
def brothers_count := 2
def brothers_savings := 5
def isabelle_savings := 5
def weekly_earnings := 3

-- Calculate total required work weeks
theorem isabelle_work_weeks :
  let total_ticket_cost := isabelle_ticket_cost + brother_ticket_cost * brothers_count in
  let total_savings := isabelle_savings + brothers_savings in
  let required_savings := total_ticket_cost - total_savings in
  required_savings / weekly_earnings = 10 :=
by
  sorry

end isabelle_work_weeks_l807_807566


namespace correct_calculation_l807_807051

def calculation_is_correct (a b x y : ℝ) : Prop :=
  (3 * x^2 * y - 2 * y * x^2 = x^2 * y)

theorem correct_calculation :
  ∀ (a b x y : ℝ), calculation_is_correct a b x y :=
by
  intros a b x y
  sorry

end correct_calculation_l807_807051


namespace interior_angle_regular_octagon_l807_807368

theorem interior_angle_regular_octagon (n : ℕ) (h1 : n = 8)
  (h2 : ∀ n, (∑ i in finset.range n, interior_angle i) = 180 * (n - 2)) :
  interior_angle 8 = 135 :=
by
  -- sorry is inserted here to skip the proof as instructed
  sorry

end interior_angle_regular_octagon_l807_807368


namespace cube_sum_minus_triple_product_l807_807518

theorem cube_sum_minus_triple_product (x y z : ℝ) (h1 : x + y + z = 8) (h2 : xy + yz + zx = 20) :
  x^3 + y^3 + z^3 - 3 * x * y * z = 32 :=
sorry

end cube_sum_minus_triple_product_l807_807518


namespace regular_octagon_interior_angle_l807_807260

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807260


namespace angle_supplement_complement_l807_807865

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l807_807865


namespace angle_solution_l807_807879

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l807_807879


namespace regular_octagon_interior_angle_l807_807300

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → let sum_interior_angles := 180 * (n - 2) in
  ∀ interior_angle (angle := sum_interior_angles / n), angle = 135 := by
  intros n h₁ sum_interior_angles interior_angle angle
  rw h₁ at sum_interior_angles
  simp at sum_interior_angles
  have h₂ : sum_interior_angles = 1080 := by simp [sum_interior_angles]
  have h₃ : angle = 1080 / 8 := by simp [angle, h₂]
  simp at h₃
  exact h₃


end regular_octagon_interior_angle_l807_807300


namespace regular_octagon_interior_angle_deg_l807_807389

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807389


namespace least_squares_principle_l807_807685

variable (n : ℕ)
variable (a b : ℝ)
variable (x y : Fin n → ℝ)

theorem least_squares_principle :
  (∑ i, (y i - (a + b * x i))^2) = (∑ j, [minimization condition in least squares method]):
  sorry

end least_squares_principle_l807_807685


namespace regular_octagon_interior_angle_l807_807264

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (sum_of_interior_angles n) / n = 135 :=
by
  -- Define the sum of interior angles for a polygon.
  let sum_of_interior_angles := λ n : ℕ, 180 * (n - 2)
  sorry

end regular_octagon_interior_angle_l807_807264


namespace number_of_ways_to_represent_5030_l807_807584

theorem number_of_ways_to_represent_5030 :
  let even := {x : ℕ | x % 2 = 0}
  let in_range := {x : ℕ | x ≤ 98}
  let valid_b := even ∩ in_range
  ∃ (M : ℕ), M = 150 ∧ ∀ (b3 b2 b1 b0 : ℕ), 
    b3 ∈ valid_b ∧ b2 ∈ valid_b ∧ b1 ∈ valid_b ∧ b0 ∈ valid_b →
    5030 = b3 * 10 ^ 3 + b2 * 10 ^ 2 + b1 * 10 + b0 → 
    M = 150 :=
  sorry

end number_of_ways_to_represent_5030_l807_807584


namespace regular_octagon_angle_l807_807306

-- Define the number of sides of the polygon
def n := 8

-- Define the sum of interior angles of an n-sided polygon
def S (n : ℕ) : ℝ := 180 * (n - 2)

-- Define the measure of each interior angle of a regular n-sided polygon
def interior_angle (n : ℕ) : ℝ := S n / n

-- State the theorem: each interior angle of a regular octagon is 135 degrees
theorem regular_octagon_angle : interior_angle 8 = 135 :=
by
  sorry

end regular_octagon_angle_l807_807306


namespace regular_octagon_interior_angle_l807_807328

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807328


namespace event_with_highest_probability_l807_807930

-- Define the set of outcomes for a fair die
def outcomes := {1, 2, 3, 4, 5, 6}

-- Define the events
def is_odd (n : ℕ) := n ∈ {1, 3, 5}
def is_multiple_of_3 (n : ℕ) := n ∈ {3, 6}
def is_greater_than_5 (n : ℕ) := n ∈ {6}
def is_less_than_5 (n : ℕ) := n ∈ {1, 2, 3, 4}

-- Define probabilities
def probability (event : Finset ℕ) : ℚ :=
  (event.card : ℚ) / (outcomes.card : ℚ)

-- Probabilities of the events
def probability_of_odd := probability {1, 3, 5}
def probability_of_multiple_of_3 := probability {3, 6}
def probability_of_greater_than_5 := probability {6}
def probability_of_less_than_5 := probability {1, 2, 3, 4}

-- The statement to prove
theorem event_with_highest_probability :
  max (max probability_of_odd probability_of_multiple_of_3)
      (max probability_of_greater_than_5 probability_of_less_than_5) =
  probability_of_less_than_5 :=
begin
  sorry
end

end event_with_highest_probability_l807_807930


namespace binary_square_only_1_l807_807151

theorem binary_square_only_1 (n : ℕ) : (∃ k, n = k^2) ∧ (∀ (d : ℕ), (n.bits d = dimp (d < n.bits_length) true false)) → n = 1 := by
  sorry

end binary_square_only_1_l807_807151


namespace regular_octagon_interior_angle_deg_l807_807383

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807383


namespace interior_angle_regular_octagon_l807_807485

theorem interior_angle_regular_octagon : 
  ∀ (n : ℕ), (n = 8) → ∀ S : ℕ, (S = 180 * (n - 2)) → (S / n = 135) :=
by
  intros n hn S hS
  rw [hn, hS]
  norm_num
  sorry

end interior_angle_regular_octagon_l807_807485


namespace angle_complement_supplement_l807_807823

theorem angle_complement_supplement (x : ℝ) (h1 : 180 - x = 4 * (90 - x)) : x = 60 :=
sorry

end angle_complement_supplement_l807_807823


namespace probability_of_xyz_72_l807_807924

noncomputable def probability_product_is_72 : ℚ :=
  let dice := {1, 2, 3, 4, 5, 6}
  let outcomes := {e : ℕ × ℕ × ℕ | e.1 ∈ dice ∧ e.2.1 ∈ dice ∧ e.2.2 ∈ dice}
  let favourable_outcomes := {e : ℕ × ℕ × ℕ | e ∈ outcomes ∧ e.1 * e.2.1 * e.2.2 = 72}
  (favourable_outcomes.to_finset.card : ℚ) / outcomes.to_finset.card

theorem probability_of_xyz_72 :
  probability_product_is_72 = 1 / 24 :=
sorry

end probability_of_xyz_72_l807_807924


namespace find_divisor_l807_807067

theorem find_divisor 
  (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  (dividend = 172) → (quotient = 10) → (remainder = 2) → (dividend = (divisor * quotient) + remainder) → divisor = 17 :=
by 
  sorry

end find_divisor_l807_807067


namespace kendra_packs_l807_807574

/-- Kendra has some packs of pens. Tony has 2 packs of pens. There are 3 pens in each pack. 
Kendra and Tony decide to keep two pens each and give the remaining pens to their friends 
one pen per friend. They give pens to 14 friends. Prove that Kendra has 4 packs of pens. --/
theorem kendra_packs : ∀ (kendra_pens tony_pens pens_per_pack pens_kept pens_given friends : ℕ),
  tony_pens = 2 →
  pens_per_pack = 3 →
  pens_kept = 2 →
  pens_given = 14 →
  tony_pens * pens_per_pack - pens_kept + kendra_pens - pens_kept = pens_given →
  kendra_pens / pens_per_pack = 4 :=
by
  intros kendra_pens tony_pens pens_per_pack pens_kept pens_given friends
  intro h1
  intro h2
  intro h3
  intro h4
  intro h5
  sorry

end kendra_packs_l807_807574


namespace mass_of_added_ice_l807_807037

/-- Given the specific heat capacity of water, latent heat of fusion of ice, density of water,
initial temperature and volume of water, time taken to bring water to boil, 
and the time taken for water to boil again after adding ice,
prove the mass of the added ice. -/
theorem mass_of_added_ice 
  (c_B : ℝ) (λ : ℝ) (ρ : ℝ) (t_init : ℝ) (V : ℝ) (t_first : ℝ) (t_second : ℝ) : 
  c_B = 4200 ∧ λ = 3.3 * 10^5 ∧ ρ = 1000 ∧ t_init = 0 ∧ V = 2 * real.pi ∧ t_first = 600 ∧ t_second = 900 
  → let m_init := ρ * V,
         P_first := (c_B * m_init * (100 - t_init)) / t_first,
         P_second := (λ * (m_J : ℝ) + c_B * (m_J : ℝ) * 100) / t_second 
  in m_J = 1.68 :=
by
  intros h,
  sorry

end mass_of_added_ice_l807_807037


namespace each_interior_angle_of_regular_octagon_l807_807356

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807356


namespace regular_octagon_interior_angle_l807_807247

theorem regular_octagon_interior_angle :
  ∀ (n : ℕ), n = 8 → (n - 2) * 180 / n = 135 :=
by 
  intro n h
  rw [h]
  norm_num

end regular_octagon_interior_angle_l807_807247


namespace probability_of_xyz_72_l807_807923

noncomputable def probability_product_is_72 : ℚ :=
  let dice := {1, 2, 3, 4, 5, 6}
  let outcomes := {e : ℕ × ℕ × ℕ | e.1 ∈ dice ∧ e.2.1 ∈ dice ∧ e.2.2 ∈ dice}
  let favourable_outcomes := {e : ℕ × ℕ × ℕ | e ∈ outcomes ∧ e.1 * e.2.1 * e.2.2 = 72}
  (favourable_outcomes.to_finset.card : ℚ) / outcomes.to_finset.card

theorem probability_of_xyz_72 :
  probability_product_is_72 = 1 / 24 :=
sorry

end probability_of_xyz_72_l807_807923


namespace quad_circle_area_ratio_sum_four_l807_807626

theorem quad_circle_area_ratio_sum_four
  (r : ℝ)
  (AC_is_diameter : ∀ (O : Point2D), Circle.circle O r → ∃ (A C : Point2D), dist A C = 2 * r ∧ is_on_circle A ∧ is_on_circle C ∧ AC_is_diameter)  -- AC is a diameter
  (∠DAC_60 : ∀ (A D C : Point2D), angle A D C = 60)  -- ∠DAC = 60 degrees
  (∠BAC_30 : ∀ (A B C : Point2D), angle A B C = 30)  -- ∠BAC = 30 degrees
  : (let a : ℤ := 0
         b : ℤ := 3
         c : ℤ := 1
         ratio : ℝ := (r^2 * sqrt 3)/(π * r^2)
     in (ratio = (sqrt 3)/π) ∧ (a + b + c) = 4) :=
begin
  sorry
end

end quad_circle_area_ratio_sum_four_l807_807626


namespace regular_octagon_interior_angle_l807_807334

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807334


namespace restore_grid_l807_807765

noncomputable def grid_values (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 9 < 12 ∧
  3 + 1 < 12 ∧ 3 + 5 < 12 ∧
  B + 3 < 12 ∧ B + 4 < 12 ∧
  B + 7 < 12 ∧ 
  C + 4 < 12 ∧
  C + 7 < 12 ∧ 
  D + 9 < 12 ∧
  D + 7 < 12 ∧
  D + 5 < 12

def even_numbers_erased (A B C D : ℕ) : Prop :=
  A ∈ {8} ∧ B ∈ {6} ∧ C ∈ {4} ∧ D ∈ {2}

theorem restore_grid :
  ∃ (A B C D : ℕ), grid_values A B C D ∧ even_numbers_erased A B C D :=
by {
  use [8, 6, 4, 2],
  dsimp [grid_values, even_numbers_erased],
  exact ⟨⟨by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num, by norm_num⟩, 
         ⟨rfl, rfl, rfl, rfl⟩⟩
}

end restore_grid_l807_807765


namespace stickers_on_first_day_l807_807614

theorem stickers_on_first_day (s e total : ℕ) (h1 : e = 22) (h2 : total = 61) (h3 : total = s + e) : s = 39 :=
by
  sorry

end stickers_on_first_day_l807_807614


namespace angle_solution_l807_807887

noncomputable theory

def angle (x : ℝ) :=
  (180 - x = 4 * (90 - x))

theorem angle_solution (x : ℝ) : angle x → x = 60 :=
by
  intros h
  sorry

end angle_solution_l807_807887


namespace regular_octagon_interior_angle_deg_l807_807378

theorem regular_octagon_interior_angle_deg :
  ∀ n : ℕ, n = 8 → (∀ i < n, angle i = (n - 2) * 180 / n) :=
by
  sorry

end regular_octagon_interior_angle_deg_l807_807378


namespace determine_grid_numbers_l807_807719

theorem determine_grid_numbers (A B C D : ℕ) :
  (1 ≤ A ∧ A ≤ 9) ∧ (1 ≤ B ∧ B ≤ 9) ∧ (1 ≤ C ∧ C ≤ 9) ∧ (1 ≤ D ∧ D ≤ 9) ∧ 
  (A ≠ 1) ∧ (A ≠ 3) ∧ (A ≠ 5) ∧ (A ≠ 7) ∧ (A ≠ 9) ∧ 
  (B ≠ 1) ∧ (B ≠ 3) ∧ (B ≠ 5) ∧ (B ≠ 7) ∧ (B ≠ 9) ∧ 
  (C ≠ 1) ∧ (C ≠ 3) ∧ (C ≠ 5) ∧ (C ≠ 7) ∧ (C ≠ 9) ∧ 
  (D ≠ 1) ∧ (D ≠ 3) ∧ (D ≠ 5) ∧ (D ≠ 7) ∧ (D ≠ 9) ∧ 
  (A ≠ 2 ∧ A ≠ 4 ∧ A ≠ 6 ∧ A ≠ 8 ∨ B ≠ 2 ∧ B ≠ 4 ∧ B ≠ 6 ∧ B ≠ 8 ∨ C ≠ 2 ∧ C ≠ 4 ∧ C ≠ 6 ∧ C ≠ 8 ∨ D ≠ 2 ∧ D ≠ 4 ∧ D ≠ 6 ∧ D ≠ 8) ∧ 
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ 
  (3 + 5 < 12) ∧ (3 + D < 12) ∧ (B + 5 < 12) ∧ 
  (B + C < 12) ∧ (C + 7 < 12) ∧ (B + 7 < 12) 
  → (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) :=
by 
  intros A B C D
  exact sorry

end determine_grid_numbers_l807_807719


namespace angle_supplement_complement_l807_807867

-- Definitions of conditions as hypotheses
def is_complement (a: ℝ) := 90 - a
def is_supplement (a: ℝ) := 180 - a

-- The theorem we want to prove
theorem angle_supplement_complement (x : ℝ) (h : 180 - x = 4 * (90 - x)) : x = 60 := 
by {
  sorry
}

end angle_supplement_complement_l807_807867


namespace find_third_month_sale_l807_807961

def sales_first_month : ℕ := 3435
def sales_second_month : ℕ := 3927
def sales_fourth_month : ℕ := 4230
def sales_fifth_month : ℕ := 3562
def sales_sixth_month : ℕ := 1991
def required_average_sale : ℕ := 3500

theorem find_third_month_sale (S3 : ℕ) :
  (sales_first_month + sales_second_month + S3 + sales_fourth_month + sales_fifth_month + sales_sixth_month) / 6 = required_average_sale →
  S3 = 3855 := by
  sorry

end find_third_month_sale_l807_807961


namespace true_statements_count_l807_807629

def f (x : ℝ) : ℝ := sqrt 3 * (Real.cos x)^2 + 2 * (Real.sin x) * (Real.cos x) - sqrt 3 * (Real.sin x)^2

theorem true_statements_count :
  let cond1 := (f (Real.pi / 12) = 2)
  let cond2 := ∀ x : ℝ, f (Real.pi / 3 + x) = - f (Real.pi / 3 - x)
  let cond3 := ¬(∀ x : ℝ, f (x + Real.pi / 3) = - f (x - Real.pi / 3))
  let cond4 := ∃ (x1 x2 : ℝ), |f x1 - f x2| ≥ 4
  let count := (if cond1 then 1 else 0) + (if cond2 then 1 else 0) + (if cond3 then 1 else 0) + (if cond4 then 1 else 0)
  count = 3 :=
by {
  sorry
}

end true_statements_count_l807_807629


namespace inequality_proof_l807_807161

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  1 + 3 / (a * b + b * c + c * a) ≥ 6 / (a + b + c) :=
by
  sorry

end inequality_proof_l807_807161


namespace interior_angle_regular_octagon_l807_807448

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) 
  (h_regular : ∀ (i j : ℕ) (h_i : 0 ≤ i ∧ i < n) (h_j : 0 ≤ j ∧ j < n), i ≠ j → ∠_i = ∠_j) : 
  ∃ angle : ℝ, angle = 135 := by
  sorry

end interior_angle_regular_octagon_l807_807448


namespace prove_a_range_l807_807189

-- Defining the propositions p and q
def p (a : ℝ) : Prop := ∃ x ∈ Set.Icc (-1 : ℝ) 1, a^2 * x^2 + a * x - 2 = 0
def q (a : ℝ) : Prop := ∃! x : ℝ, x^2 + 2 * a * x + 2 * a ≤ 0

-- The proposition to prove
theorem prove_a_range (a : ℝ) (hpq : ¬(p a ∨ q a)) : a ∈ Set.Ioo (-1 : ℝ) 0 ∪ Set.Ioo 0 1 :=
by
  sorry

end prove_a_range_l807_807189


namespace regular_octagon_interior_angle_l807_807332

theorem regular_octagon_interior_angle (n : ℕ) (h : n = 8) : (180 * (n - 2)) / n = 135 := by
  sorry

end regular_octagon_interior_angle_l807_807332


namespace angle_supplement_complement_l807_807897

theorem angle_supplement_complement (x : ℝ) 
  (hsupp : 180 - x = 4 * (90 - x)) : 
  x = 60 :=
by
  sorry

end angle_supplement_complement_l807_807897


namespace calculate_adults_in_play_l807_807644

theorem calculate_adults_in_play :
  ∃ A : ℕ, (11 * A = 49 + 50) := sorry

end calculate_adults_in_play_l807_807644


namespace triangle_eum_similar_fmc_l807_807125

open EuclideanGeometry

variables {A B C E F M U : Point} [Circle α]
variables (bc : Chord α B C) (ef : Chord α E F)
-- Condition 1: EF is the perpendicular bisector of BC at M.
axiom perp_bisector_ef_bc_at_m : PerpendicularBisectorAtM ef bc M
-- Condition 2: AU is perpendicular to FC.
axiom au_perp_fc : Perpendicular AU FC

theorem triangle_eum_similar_fmc 
    (h1 : perp_bisector_ef_bc_at_m)
    (h2 : au_perp_fc) :
    Similar (Triangle E U M) (Triangle F M C) :=
sorry

end triangle_eum_similar_fmc_l807_807125


namespace correct_comparison_l807_807931

-- Definitions of conditions based on the problem 
def hormones_participate : Prop := false 
def enzymes_produced_by_living_cells : Prop := true 
def hormones_produced_by_endocrine : Prop := true 
def endocrine_can_produce_both : Prop := true 
def synthesize_enzymes_not_nec_hormones : Prop := true 
def not_all_proteins : Prop := true 

-- Statement of the equivalence between the correct answer and its proof
theorem correct_comparison :  (¬hormones_participate ∧ enzymes_produced_by_living_cells ∧ hormones_produced_by_endocrine ∧ endocrine_can_produce_both ∧ synthesize_enzymes_not_nec_hormones ∧ not_all_proteins) → (endocrine_can_produce_both) :=
by
  sorry

end correct_comparison_l807_807931


namespace each_interior_angle_of_regular_octagon_l807_807343

/-- A regular polygon with n sides has (n-2) * 180 degrees as the sum of its interior angles. -/
def sum_of_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- A regular octagon has each interior angle equal to its sum of interior angles divided by its number of sides -/
theorem each_interior_angle_of_regular_octagon : sum_of_interior_angles 8 / 8 = 135 :=
by
  sorry

end each_interior_angle_of_regular_octagon_l807_807343


namespace probability_inside_five_spheres_l807_807095

noncomputable def edge_length (s : ℝ) : ℝ := s

-- Define the circumradius R and inradius r
def circumradius (s : ℝ) : ℝ := (s * real.sqrt 6) / 4
def inradius (s : ℝ) : ℝ := (s * real.sqrt 6) / 12

-- Define the radius of the smaller spheres tangent to the faces
def tangent_sphere_radius (s : ℝ) : ℝ := inradius s

-- Volumes of spheres
def volume_of_sphere (r : ℝ) : ℝ := (4 / 3) * real.pi * (r ^ 3)

def volume_large_sphere (s : ℝ) : ℝ := volume_of_sphere (circumradius s)
def volume_small_sphere (s : ℝ) : ℝ := volume_of_sphere (inradius s)

-- Total volume of the 5 smaller spheres
def total_small_spheres_volume (s : ℝ) : ℝ := 5 * volume_small_sphere s

-- Probability calculation
def probability (s : ℝ) : ℝ := total_small_spheres_volume s / volume_large_sphere s

-- The theorem to prove
theorem probability_inside_five_spheres (s : ℝ) :
  probability s = 5 / 27 := sorry

end probability_inside_five_spheres_l807_807095


namespace twentieth_prime_is_71_l807_807913

/- Define what it means to be the nth prime number -/
def isNthPrime (n : ℕ) (p : ℕ) : Prop :=
  Nat.Prime p ∧ (finset.filter Nat.Prime (finset.range p)).card = n

/- Now we state the problem: proving that the 20th prime number is 71 -/
theorem twentieth_prime_is_71 : isNthPrime 20 71 := 
  sorry

end twentieth_prime_is_71_l807_807913


namespace division_problem_l807_807048

theorem division_problem :
  ∃ A : ℕ, (11 = (A * 3) + 2) ∧ A = 3 :=
by {
  use 3,
  split,
  { sorry }, -- This will prove 11 = (3 * 3) + 2
  { refl }
}

end division_problem_l807_807048


namespace grid_solution_l807_807738

-- Define the grid and conditions
variables (A B C D : ℕ)

-- Grid values with the constraints
def grid_valid (A B C D : ℕ) :=
  A ≠ 1 ∧ A ≠ 9 ∧ A ≠ 3 ∧ A ≠ 5 ∧ A ≠ 7 ∧
  B ≠ 1 ∧ B ≠ 9 ∧ B ≠ 3 ∧ B ≠ 5 ∧ B ≠ 7 ∧
  C ≠ 1 ∧ C ≠ 9 ∧ C ≠ 3 ∧ C ≠ 5 ∧ C ≠ 7 ∧
  D ≠ 1 ∧ D ≠ 9 ∧ D ≠ 3 ∧ D ≠ 5 ∧ D ≠ 7 ∧
  (A + 1 < 12) ∧ (A + 9 < 12) ∧ (A + 3 < 12) ∧
  (D + 9 < 12) ∧ (D + 5 < 12) ∧ 
  (B + 3 < 12) ∧ (B + C < 12) ∧
  (C + 7 < 12)

-- Given the conditions, we need to prove the values of A, B, C, and D
theorem grid_solution : 
  grid_valid 8 6 4 2 ∧ 
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 :=
by
  split
  · -- to show grid_valid 8 6 4 2
    sorry
    
  · -- to show the values of A, B, C, and D
    repeat {split, assumption}

end grid_solution_l807_807738


namespace range_of_a_l807_807210

noncomputable def z1 : ℂ := by {
  let z := (-1 + 5 * complex.I) / (1 + complex.I)
  exact z }

axiom a : ℝ

def z2 (a : ℝ) : ℂ := a - 2 - complex.I

axiom habs_lt : ∃ (a : ℝ), abs (z1 - complex.conj (z2 a)) < abs z1

theorem range_of_a (a : ℝ) (h : 1 < a ∧ a < 7) : 
  abs (z1 - complex.conj (z2 a)) < abs z1 := sorry

end range_of_a_l807_807210


namespace find_set_A_correct_l807_807592

noncomputable def find_set_A (a1 a2 a3 a4 a5 : ℕ) (A B : set ℕ) : Prop :=
  a1 < a2 ∧ a2 < a3 ∧ a3 < a4 ∧ a4 < a5 ∧
  A = {a1, a2, a3, a4, a5} ∧
  B = {a1^2, a2^2, a3^2, a4^2, a5^2} ∧
  A ∩ B = {a1, a4} ∧
  a1 + a4 ≠ 10 ∧
  A.union B.sum = 256

theorem find_set_A_correct :
  ∃ (a1 a2 a3 a4 a5 : ℕ),
  ∃ (A B : set ℕ),
  find_set_A a1 a2 a3 a4 a5 A B ∧
  (A = {1, 2, 3, 9, 12} ∨ A = {1, 3, 5, 9, 11}) :=
by
  sorry

end find_set_A_correct_l807_807592


namespace find_d_l807_807001

theorem find_d (d : ℕ) : (1059 % d = 1417 % d) ∧ (1059 % d = 2312 % d) ∧ (1417 % d = 2312 % d) ∧ (d > 1) → d = 179 :=
by
  sorry

end find_d_l807_807001


namespace area_of_pentagon_correct_l807_807631

noncomputable def area_of_pentagon : ℝ :=
  let AB := 5
  let BC := 3
  let BD := 3
  let AC := Real.sqrt (AB^2 - BC^2)
  let AD := Real.sqrt (AB^2 - BD^2)
  let EC := 1
  let FD := 2
  let AE := AC - EC
  let AF := AD - FD
  let sin_alpha := BC / AB
  let cos_alpha := AC / AB
  let sin_2alpha := 2 * sin_alpha * cos_alpha
  let area_ABC := 0.5 * AB * BC
  let area_AEF := 0.5 * AE * AF * sin_2alpha
  2 * area_ABC - area_AEF

theorem area_of_pentagon_correct :
  area_of_pentagon = 9.12 := sorry

end area_of_pentagon_correct_l807_807631


namespace restore_grid_l807_807711

-- Definitions for each cell in the grid.
variable (A B C D : ℕ)

-- The given grid and conditions
noncomputable def grid : list (list ℕ) :=
  [[A, 1, 9],
   [3, 5, D],
   [B, C, 7]]

-- The condition that the sum of numbers in adjacent cells is less than 12.
def valid (A B C D : ℕ) : Prop :=
  A + 1 < 12 ∧ A + 3 < 12 ∧
  1 + 9 < 12 ∧ 5 + D < 12 ∧
  3 + 5 < 12 ∧  B + C < 12 ∧
  9 + D < 12 ∧ 7 + C < 12 ∧
  9 + 1 < 12 ∧ B + 7 < 12

-- Assertion to be proved.
theorem restore_grid (A B C D : ℕ) (valid : valid A B C D) :
  A = 8 ∧ B = 6 ∧ C = 4 ∧ D = 2 := by
  sorry

end restore_grid_l807_807711


namespace regular_octagon_interior_angle_eq_135_l807_807396

theorem regular_octagon_interior_angle_eq_135 :
  ∀ (n : ℕ), n = 8 → (180 * (n - 2)) / n = 135 :=
by
  intro n h
  rw h
  norm_num
  sorry

end regular_octagon_interior_angle_eq_135_l807_807396


namespace interior_angle_regular_octagon_l807_807443

theorem interior_angle_regular_octagon (n : ℕ) (h_n : n = 8) :
  let sum_of_interior_angles := 180 * (n - 2)
  let each_angle := sum_of_interior_angles / n
  each_angle = 135 :=
by
  sorry

end interior_angle_regular_octagon_l807_807443


namespace restore_grid_values_l807_807742

def is_adjacent_sum_lt_12 (A B C D : ℕ) : Prop :=
  (A + 1 < 12) ∧ (A + 9 < 12) ∧
  (3 + 1 < 12) ∧ (3 + 5 < 12) ∧
  (3 + D < 12) ∧ (5 + 1 < 12) ∧ 
  (5 + D < 12) ∧ (B + C < 12) ∧
  (B + 3 < 12) ∧ (B + 5 < 12) ∧
  (C + 5 < 12) ∧ (C + 7 < 12) ∧
  (D + 9 < 12) ∧ (D + 7 < 12)

theorem restore_grid_values : 
  ∃ (A B C D : ℕ), 
  (A = 8) ∧ (B = 6) ∧ (C = 4) ∧ (D = 2) ∧ 
  is_adjacent_sum_lt_12 A B C D := 
by 
  exists 8
  exists 6
  exists 4
  exists 2
  split; [refl, split; [refl, split; [refl, split; [refl, sorry]]]] 

end restore_grid_values_l807_807742
