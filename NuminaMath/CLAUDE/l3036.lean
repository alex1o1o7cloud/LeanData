import Mathlib

namespace NUMINAMATH_CALUDE_initial_cheerleaders_initial_cheerleaders_correct_l3036_303652

theorem initial_cheerleaders (initial_football_players : ℕ) 
                             (quit_football_players : ℕ) 
                             (quit_cheerleaders : ℕ) 
                             (remaining_total : ℕ) : ℕ :=
  let initial_cheerleaders := 16
  have h1 : initial_football_players = 13 := by sorry
  have h2 : quit_football_players = 10 := by sorry
  have h3 : quit_cheerleaders = 4 := by sorry
  have h4 : remaining_total = 15 := by sorry
  have h5 : initial_football_players - quit_football_players + 
            (initial_cheerleaders - quit_cheerleaders) = remaining_total := by sorry
  initial_cheerleaders

theorem initial_cheerleaders_correct : initial_cheerleaders 13 10 4 15 = 16 := by sorry

end NUMINAMATH_CALUDE_initial_cheerleaders_initial_cheerleaders_correct_l3036_303652


namespace NUMINAMATH_CALUDE_cubic_root_sum_squares_l3036_303675

/-- Given a cubic polynomial x^3 - 3x - 2 = 0 with roots a, b, and c,
    prove that a(b + c)^2 + b(c + a)^2 + c(a + b)^2 = 6 -/
theorem cubic_root_sum_squares (a b c : ℝ) : 
  a^3 - 3*a - 2 = 0 → 
  b^3 - 3*b - 2 = 0 → 
  c^3 - 3*c - 2 = 0 → 
  a*(b + c)^2 + b*(c + a)^2 + c*(a + b)^2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_squares_l3036_303675


namespace NUMINAMATH_CALUDE_team_total_score_l3036_303690

/-- Represents a basketball player with their score -/
structure Player where
  name : String
  score : ℕ

/-- The school basketball team -/
def team : List Player := [
  { name := "Daniel", score := 7 },
  { name := "Ramon", score := 8 },
  { name := "Ian", score := 2 },
  { name := "Bernardo", score := 11 },
  { name := "Tiago", score := 6 },
  { name := "Pedro", score := 12 },
  { name := "Ed", score := 1 },
  { name := "André", score := 7 }
]

/-- The total score of the team is the sum of individual player scores -/
def totalScore (team : List Player) : ℕ :=
  team.map (·.score) |>.sum

/-- Theorem: The total score of the team is 54 -/
theorem team_total_score : totalScore team = 54 := by
  sorry

end NUMINAMATH_CALUDE_team_total_score_l3036_303690


namespace NUMINAMATH_CALUDE_no_real_solutions_l3036_303663

theorem no_real_solutions : 
  ¬∃ (x : ℝ), (x ≠ 2) ∧ ((x^3 - 8) / (x - 2) = 3*x) := by sorry

end NUMINAMATH_CALUDE_no_real_solutions_l3036_303663


namespace NUMINAMATH_CALUDE_product_of_binomials_with_sqrt_two_l3036_303616

theorem product_of_binomials_with_sqrt_two : (2 + Real.sqrt 2) * (2 - Real.sqrt 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_of_binomials_with_sqrt_two_l3036_303616


namespace NUMINAMATH_CALUDE_bicycle_inventory_problem_l3036_303694

/-- Represents the bicycle inventory problem for Hank's store over three days --/
theorem bicycle_inventory_problem 
  (B : ℤ) -- Initial number of bicycles
  (S : ℤ) -- Number of bicycles sold on Friday
  (h1 : S ≥ 0) -- Number of bicycles sold is non-negative
  (h2 : B - S + 15 - 12 + 8 - 9 + 11 = B + 3) -- Net increase equation
  : S = 10 := by
  sorry


end NUMINAMATH_CALUDE_bicycle_inventory_problem_l3036_303694


namespace NUMINAMATH_CALUDE_percent_decrease_l3036_303657

theorem percent_decrease (original_price sale_price : ℝ) 
  (h1 : original_price = 100)
  (h2 : sale_price = 55) : 
  (original_price - sale_price) / original_price * 100 = 45 := by
  sorry

end NUMINAMATH_CALUDE_percent_decrease_l3036_303657


namespace NUMINAMATH_CALUDE_collinear_points_solution_l3036_303614

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The theorem states that if points A(a,2), B(5,1), and C(-4,2a) are collinear, 
    then a = 5 ± √21 -/
theorem collinear_points_solution (a : ℝ) :
  collinear a 2 5 1 (-4) (2*a) → a = 5 + Real.sqrt 21 ∨ a = 5 - Real.sqrt 21 :=
by sorry

end NUMINAMATH_CALUDE_collinear_points_solution_l3036_303614


namespace NUMINAMATH_CALUDE_seaweed_livestock_amount_l3036_303685

-- Define the total amount of seaweed harvested
def total_seaweed : ℝ := 400

-- Define the percentage of seaweed used for fires
def fire_percentage : ℝ := 0.5

-- Define the percentage of remaining seaweed for human consumption
def human_percentage : ℝ := 0.25

-- Function to calculate the amount of seaweed fed to livestock
def seaweed_for_livestock : ℝ :=
  let remaining_after_fire := total_seaweed * (1 - fire_percentage)
  let for_humans := remaining_after_fire * human_percentage
  remaining_after_fire - for_humans

-- Theorem stating the amount of seaweed fed to livestock
theorem seaweed_livestock_amount : seaweed_for_livestock = 150 := by
  sorry

end NUMINAMATH_CALUDE_seaweed_livestock_amount_l3036_303685


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l3036_303645

theorem fraction_to_decimal : (11 : ℚ) / 125 = (88 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l3036_303645


namespace NUMINAMATH_CALUDE_chips_division_l3036_303644

theorem chips_division (total_chips : ℕ) (ratio_small : ℕ) (ratio_large : ℕ) :
  total_chips = 100 →
  ratio_small = 4 →
  ratio_large = 6 →
  (ratio_large : ℚ) / (ratio_small + ratio_large : ℚ) * 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_chips_division_l3036_303644


namespace NUMINAMATH_CALUDE_roots_equation_result_l3036_303630

theorem roots_equation_result (γ δ : ℝ) : 
  γ^2 - 3*γ - 2 = 0 → δ^2 - 3*δ - 2 = 0 → 7*γ^4 + 10*δ^3 = 1363 := by
  sorry

end NUMINAMATH_CALUDE_roots_equation_result_l3036_303630


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3036_303623

theorem imaginary_part_of_complex_fraction :
  Complex.im ((1 + 2*Complex.I) / (1 + Complex.I)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l3036_303623


namespace NUMINAMATH_CALUDE_ratio_equality_l3036_303668

theorem ratio_equality (a b c u v w : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_pos_u : 0 < u) (h_pos_v : 0 < v) (h_pos_w : 0 < w)
  (h_sum_abc : a^2 + b^2 + c^2 = 9)
  (h_sum_uvw : u^2 + v^2 + w^2 = 49)
  (h_dot_product : a*u + b*v + c*w = 21) :
  (a + b + c) / (u + v + w) = 3/7 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l3036_303668


namespace NUMINAMATH_CALUDE_absolute_value_square_equivalence_l3036_303611

theorem absolute_value_square_equivalence (m n : ℝ) :
  (|m| > |n| → m^2 > n^2) ∧
  (m^2 > n^2 → |m| > |n|) ∧
  (|m| ≤ |n| → m^2 ≤ n^2) ∧
  (m^2 ≤ n^2 → |m| ≤ |n|) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_square_equivalence_l3036_303611


namespace NUMINAMATH_CALUDE_total_spent_is_correct_l3036_303625

-- Define the prices and discounts
def skateboard_price : ℝ := 9.46
def marbles_price : ℝ := 9.56
def shorts_price : ℝ := 14.50
def action_figures_price : ℝ := 12.60
def skateboard_marbles_discount : ℝ := 0.10
def action_figures_discount : ℝ := 0.20
def video_game_price_eur : ℝ := 20.50
def exchange_rate : ℝ := 1.12

-- Define the calculation of discounted prices
def discounted_price (price : ℝ) (discount : ℝ) : ℝ :=
  price * (1 - discount)

-- Define the total spent
def total_spent : ℝ :=
  discounted_price skateboard_price skateboard_marbles_discount +
  discounted_price marbles_price skateboard_marbles_discount +
  shorts_price +
  discounted_price action_figures_price action_figures_discount +
  video_game_price_eur * exchange_rate

-- Theorem statement
theorem total_spent_is_correct :
  total_spent = 64.658 := by sorry

end NUMINAMATH_CALUDE_total_spent_is_correct_l3036_303625


namespace NUMINAMATH_CALUDE_hyperbola_other_asymptote_l3036_303600

/-- Represents a hyperbola -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- X-coordinate of the foci -/
  foci_x : ℝ

/-- Given a hyperbola, returns its other asymptote -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ :=
  fun x ↦ 2 * x + 16

theorem hyperbola_other_asymptote (h : Hyperbola) 
  (h1 : h.asymptote1 = fun x ↦ -2 * x + 4)
  (h2 : h.foci_x = -3) :
  other_asymptote h = fun x ↦ 2 * x + 16 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_other_asymptote_l3036_303600


namespace NUMINAMATH_CALUDE_find_other_number_l3036_303696

theorem find_other_number (x y : ℤ) : 
  (3 * x + 2 * y = 130) → 
  ((x = 35 ∨ y = 35) → 
  ((x ≠ 35 → y = 35 ∧ x = 20) ∧ 
   (y ≠ 35 → x = 35 ∧ y = 20))) := by
sorry

end NUMINAMATH_CALUDE_find_other_number_l3036_303696


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3036_303629

theorem complex_equation_solution (a b : ℝ) (i : ℂ) :
  i * i = -1 →
  (a + i) * i = b + i →
  a = 1 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3036_303629


namespace NUMINAMATH_CALUDE_downstream_speed_l3036_303615

/-- 
Theorem: Given a man's upstream rowing speed and still water speed, 
we can determine his downstream rowing speed.
-/
theorem downstream_speed 
  (upstream_speed : ℝ) 
  (still_water_speed : ℝ) 
  (h1 : upstream_speed = 22) 
  (h2 : still_water_speed = 32) : 
  ∃ downstream_speed : ℝ, 
    downstream_speed = 2 * still_water_speed - upstream_speed ∧ 
    downstream_speed = 42 := by
  sorry

end NUMINAMATH_CALUDE_downstream_speed_l3036_303615


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_square_plus_x_l3036_303682

theorem negation_of_forall_positive_square_plus_x :
  (¬ ∀ x : ℝ, x^2 + x > 0) ↔ (∃ x : ℝ, x^2 + x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_square_plus_x_l3036_303682


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3036_303642

-- Define the conditions
def p (x y : ℝ) : Prop := (x - 1) * (y - 2) = 0
def q (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 0

-- Theorem stating that p is necessary but not sufficient for q
theorem p_necessary_not_sufficient_for_q :
  (∀ x y : ℝ, q x y → p x y) ∧ (∃ x y : ℝ, p x y ∧ ¬(q x y)) := by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3036_303642


namespace NUMINAMATH_CALUDE_solution_range_l3036_303672

-- Define the system of inequalities
def has_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, a * x > -1 ∧ x + a > 0

-- Define the range of a
def a_range (a : ℝ) : Prop :=
  a < -1 ∨ a ≥ 0

-- Theorem statement
theorem solution_range :
  ∀ a : ℝ, has_solution a ↔ a_range a := by sorry

end NUMINAMATH_CALUDE_solution_range_l3036_303672


namespace NUMINAMATH_CALUDE_nine_sequence_sum_to_1989_l3036_303601

theorem nine_sequence_sum_to_1989 : ∃ (a b c : ℕ), 
  a + b + c = 9999999 ∧ 
  a ≤ 999 ∧ b ≤ 999 ∧ c ≤ 999 ∧
  a + b - c = 1989 := by
sorry

end NUMINAMATH_CALUDE_nine_sequence_sum_to_1989_l3036_303601


namespace NUMINAMATH_CALUDE_bees_after_six_days_l3036_303656

/-- Calculates the number of bees in the beehive after n days -/
def bees_in_hive (n : ℕ) : ℕ :=
  let a₁ : ℕ := 4  -- Initial term (1 original bee + 3 companions)
  let q : ℕ := 3   -- Common ratio (each bee brings 3 companions)
  a₁ * (q^n - 1) / (q - 1)

/-- Theorem stating that the number of bees after 6 days is 1456 -/
theorem bees_after_six_days :
  bees_in_hive 6 = 1456 := by
  sorry

end NUMINAMATH_CALUDE_bees_after_six_days_l3036_303656


namespace NUMINAMATH_CALUDE_circle_configuration_implies_zero_area_l3036_303621

-- Define the circle structure
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the line structure
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def CircleTangentToLine (c : Circle) (l : Line) : Prop :=
  sorry

def CirclesExternallyTangent (c1 c2 : Circle) : Prop :=
  sorry

def PointBetween (p1 p2 p3 : ℝ × ℝ) : Prop :=
  sorry

def TriangleArea (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  sorry

theorem circle_configuration_implies_zero_area 
  (P Q R : Circle)
  (l : Line)
  (P' Q' R' : ℝ × ℝ)
  (h1 : P.radius = 2)
  (h2 : Q.radius = 3)
  (h3 : R.radius = 4)
  (h4 : CircleTangentToLine P l)
  (h5 : CircleTangentToLine Q l)
  (h6 : CircleTangentToLine R l)
  (h7 : CirclesExternallyTangent Q P)
  (h8 : CirclesExternallyTangent Q R)
  (h9 : PointBetween P' Q' R')
  (h10 : P' = (P.center.1, l.a * P.center.1 + l.b))
  (h11 : Q' = (Q.center.1, l.a * Q.center.1 + l.b))
  (h12 : R' = (R.center.1, l.a * R.center.1 + l.b)) :
  TriangleArea P.center Q.center R.center = 0 :=
sorry

end NUMINAMATH_CALUDE_circle_configuration_implies_zero_area_l3036_303621


namespace NUMINAMATH_CALUDE_monomial_sum_condition_l3036_303610

theorem monomial_sum_condition (a b : ℕ) (m n : ℕ) : 
  (∃ k : ℕ, 2 * a^(m+2) * b^(2*n+2) + a^3 * b^8 = k * a^(m+2) * b^(2*n+2)) → 
  m = 1 ∧ n = 3 :=
by sorry

end NUMINAMATH_CALUDE_monomial_sum_condition_l3036_303610


namespace NUMINAMATH_CALUDE_sixth_salary_l3036_303633

def salary_problem (salaries : List ℝ) (mean : ℝ) : Prop :=
  let n : ℕ := salaries.length + 1
  let total : ℝ := salaries.sum
  salaries.length = 5 ∧
  mean * n = total + (n - salaries.length) * (mean * n - total)

theorem sixth_salary :
  ∀ (salaries : List ℝ) (mean : ℝ),
  salary_problem salaries mean →
  (mean * (salaries.length + 1) - salaries.sum) = 2500 :=
by sorry

#check sixth_salary

end NUMINAMATH_CALUDE_sixth_salary_l3036_303633


namespace NUMINAMATH_CALUDE_basketball_tryouts_l3036_303617

theorem basketball_tryouts (girls : ℕ) (called_back : ℕ) (didnt_make_cut : ℕ) : girls = 39 → called_back = 26 → didnt_make_cut = 17 → girls + (called_back + didnt_make_cut - girls) = 43 := by
  sorry

end NUMINAMATH_CALUDE_basketball_tryouts_l3036_303617


namespace NUMINAMATH_CALUDE_cousins_arrangement_l3036_303666

/-- The number of ways to arrange cousins in rooms -/
def arrange_cousins (n : ℕ) (m : ℕ) : ℕ :=
  -- n is the number of cousins
  -- m is the number of rooms
  sorry

/-- Theorem: Arranging 5 cousins in 4 rooms with at least one empty room -/
theorem cousins_arrangement :
  arrange_cousins 5 4 = 56 :=
by sorry

end NUMINAMATH_CALUDE_cousins_arrangement_l3036_303666


namespace NUMINAMATH_CALUDE_triangle_exists_l3036_303646

/-- Triangle inequality theorem for a triangle with sides a, b, and c -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A theorem stating that a triangle can be formed with side lengths 6, 8, and 13 -/
theorem triangle_exists : triangle_inequality 6 8 13 := by
  sorry

end NUMINAMATH_CALUDE_triangle_exists_l3036_303646


namespace NUMINAMATH_CALUDE_total_hours_worked_l3036_303650

/-- Represents the hours worked by Thomas, Toby, and Rebecca in one week -/
structure WorkHours where
  thomas : ℕ
  toby : ℕ
  rebecca : ℕ

/-- Calculates the total hours worked by all three people -/
def totalHours (h : WorkHours) : ℕ :=
  h.thomas + h.toby + h.rebecca

/-- Theorem stating the total hours worked given the conditions -/
theorem total_hours_worked :
  ∀ h : WorkHours,
    (∃ x : ℕ, h.thomas = x ∧
              h.toby = 2 * x - 10 ∧
              h.rebecca = h.toby - 8 ∧
              h.rebecca = 56) →
    totalHours h = 157 := by
  sorry

end NUMINAMATH_CALUDE_total_hours_worked_l3036_303650


namespace NUMINAMATH_CALUDE_polynomial_ratio_l3036_303606

-- Define the polynomial coefficients
variable (a₀ a₁ a₂ a₃ a₄ a₅ : ℚ)

-- Define the main equation
def main_equation (x : ℚ) : Prop :=
  (2 - x)^5 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5

-- State the theorem
theorem polynomial_ratio :
  (∀ x, main_equation a₀ a₁ a₂ a₃ a₄ a₅ x) →
  (a₀ + a₂ + a₄) / (a₁ + a₃) = -61 / 60 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_ratio_l3036_303606


namespace NUMINAMATH_CALUDE_possible_m_values_l3036_303683

def A : Set ℝ := {x | x^2 + x - 6 = 0}
def B (m : ℝ) : Set ℝ := {x | m * x + 1 = 0}

theorem possible_m_values : 
  {m : ℝ | B m ⊆ A} = {-1/2, 0, 1/3} := by sorry

end NUMINAMATH_CALUDE_possible_m_values_l3036_303683


namespace NUMINAMATH_CALUDE_curve_is_circle_implies_a_eq_neg_one_l3036_303679

/-- A curve is a circle if and only if its equation can be written in the form
    (x - h)^2 + (y - k)^2 = r^2, where (h, k) is the center and r is the radius. -/
def is_circle (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ h k r, r > 0 ∧ ∀ x y, f x y = 0 ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The equation of the curve -/
def curve_equation (a : ℝ) (x y : ℝ) : ℝ :=
  a^2 * x^2 + (a + 2) * y^2 + 2 * a * x + a

/-- Theorem: If the curve represented by a^2x^2 + (a+2)y^2 + 2ax + a = 0 is a circle, then a = -1 -/
theorem curve_is_circle_implies_a_eq_neg_one (a : ℝ) :
  is_circle (curve_equation a) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_curve_is_circle_implies_a_eq_neg_one_l3036_303679


namespace NUMINAMATH_CALUDE_chicken_pieces_needed_l3036_303631

/-- Represents the number of pieces of chicken used in different orders -/
structure ChickenPieces where
  pasta : ℕ
  barbecue : ℕ
  friedDinner : ℕ

/-- Represents the number of orders for each type of dish -/
structure Orders where
  pasta : ℕ
  barbecue : ℕ
  friedDinner : ℕ

/-- Calculates the total number of chicken pieces needed for all orders -/
def totalChickenPieces (pieces : ChickenPieces) (orders : Orders) : ℕ :=
  pieces.pasta * orders.pasta +
  pieces.barbecue * orders.barbecue +
  pieces.friedDinner * orders.friedDinner

/-- Theorem stating that given the specific conditions, 37 pieces of chicken are needed -/
theorem chicken_pieces_needed :
  let pieces := ChickenPieces.mk 2 3 8
  let orders := Orders.mk 6 3 2
  totalChickenPieces pieces orders = 37 := by
  sorry


end NUMINAMATH_CALUDE_chicken_pieces_needed_l3036_303631


namespace NUMINAMATH_CALUDE_owls_on_fence_l3036_303620

theorem owls_on_fence (initial_owls joining_owls : ℕ) :
  initial_owls = 12 → joining_owls = 7 → initial_owls + joining_owls = 19 := by
  sorry

end NUMINAMATH_CALUDE_owls_on_fence_l3036_303620


namespace NUMINAMATH_CALUDE_sam_total_dimes_l3036_303698

def initial_dimes : ℕ := 9
def given_dimes : ℕ := 7

theorem sam_total_dimes : initial_dimes + given_dimes = 16 := by
  sorry

end NUMINAMATH_CALUDE_sam_total_dimes_l3036_303698


namespace NUMINAMATH_CALUDE_cubic_equation_root_range_l3036_303609

theorem cubic_equation_root_range (m : ℝ) :
  (∃ x : ℝ, x ∈ Set.Icc 0 1 ∧ x^3 - 3*x - m = 0) ↔ m ∈ Set.Icc (-2) 0 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_root_range_l3036_303609


namespace NUMINAMATH_CALUDE_range_of_m_l3036_303641

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 2*x

-- Define sets A and B
variable (A B : Set ℝ)

-- State the theorem
theorem range_of_m (h : ∃ (x₁ x₂ : ℝ), x₁ ∈ A ∧ x₂ ∈ A ∧ x₁ ≠ x₂ ∧ f x₁ = f x₂ ∧ f x₁ ∈ B) :
  ∀ m ∈ B, m > -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l3036_303641


namespace NUMINAMATH_CALUDE_sphere_surface_area_l3036_303648

/-- Given a sphere with volume 4√3π, its surface area is 12π -/
theorem sphere_surface_area (V : ℝ) (R : ℝ) (S : ℝ) : 
  V = 4 * Real.sqrt 3 * Real.pi → 
  V = (4 / 3) * Real.pi * R^3 →
  S = 4 * Real.pi * R^2 →
  S = 12 * Real.pi := by
sorry


end NUMINAMATH_CALUDE_sphere_surface_area_l3036_303648


namespace NUMINAMATH_CALUDE_stream_current_rate_l3036_303654

/-- Represents the man's usual rowing speed in still water -/
def r : ℝ := sorry

/-- Represents the speed of the stream's current -/
def w : ℝ := sorry

/-- The distance traveled downstream and upstream -/
def distance : ℝ := 24

/-- Theorem stating the conditions and the conclusion about the stream's current -/
theorem stream_current_rate :
  (distance / (r + w) + 6 = distance / (r - w)) ∧
  (distance / (3*r + w) + 2 = distance / (3*r - w)) →
  w = 2 := by sorry

end NUMINAMATH_CALUDE_stream_current_rate_l3036_303654


namespace NUMINAMATH_CALUDE_sum_of_coefficients_cubic_expansion_l3036_303691

theorem sum_of_coefficients_cubic_expansion :
  ∃ (a b c d e : ℝ), 
    (∀ x, 27 * x^3 + 64 = (a*x + b) * (c*x^2 + d*x + e)) ∧
    (a + b + c + d + e = 20) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_cubic_expansion_l3036_303691


namespace NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3036_303604

/-- Two 2D vectors are perpendicular if their dot product is zero -/
def perpendicular (v w : ℝ × ℝ) : Prop :=
  v.1 * w.1 + v.2 * w.2 = 0

/-- Given vectors a and b, prove that if they are perpendicular, then x = 6 -/
theorem perpendicular_vectors_x_value :
  let a : ℝ × ℝ := (4, 2)
  let b : ℝ × ℝ := (x, 3)
  perpendicular a b → x = 6 :=
by
  sorry

#check perpendicular_vectors_x_value

end NUMINAMATH_CALUDE_perpendicular_vectors_x_value_l3036_303604


namespace NUMINAMATH_CALUDE_girls_in_circle_l3036_303664

theorem girls_in_circle (total : ℕ) (holding_boys_hand : ℕ) (holding_girls_hand : ℕ) 
  (h1 : total = 40)
  (h2 : holding_boys_hand = 22)
  (h3 : holding_girls_hand = 30) :
  ∃ (girls : ℕ), girls = 24 ∧ 
    girls * 2 = holding_girls_hand * 2 + holding_boys_hand + holding_girls_hand - total :=
by
  sorry

end NUMINAMATH_CALUDE_girls_in_circle_l3036_303664


namespace NUMINAMATH_CALUDE_jessica_seashells_l3036_303613

theorem jessica_seashells (initial_seashells : ℕ) (given_seashells : ℕ) :
  initial_seashells = 8 →
  given_seashells = 6 →
  initial_seashells - given_seashells = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_jessica_seashells_l3036_303613


namespace NUMINAMATH_CALUDE_evaluation_of_expression_l3036_303677

theorem evaluation_of_expression : (4^4 - 4*(4-1)^4)^4 = 21381376 := by
  sorry

end NUMINAMATH_CALUDE_evaluation_of_expression_l3036_303677


namespace NUMINAMATH_CALUDE_regular_pentagon_not_seamless_l3036_303697

def interior_angle (n : ℕ) : ℚ := (n - 2) * 180 / n

def is_divisor_of_360 (angle : ℚ) : Prop := ∃ k : ℕ, 360 = k * angle

theorem regular_pentagon_not_seamless :
  ¬(is_divisor_of_360 (interior_angle 5)) ∧
  (is_divisor_of_360 (interior_angle 3)) ∧
  (is_divisor_of_360 (interior_angle 4)) ∧
  (is_divisor_of_360 (interior_angle 6)) :=
sorry

end NUMINAMATH_CALUDE_regular_pentagon_not_seamless_l3036_303697


namespace NUMINAMATH_CALUDE_even_shifted_implies_equality_l3036_303607

def is_even_shifted (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (x + 1) = f (1 - x)

theorem even_shifted_implies_equality (f : ℝ → ℝ) 
  (h : is_even_shifted f) : f 0 = f 2 := by
  sorry

end NUMINAMATH_CALUDE_even_shifted_implies_equality_l3036_303607


namespace NUMINAMATH_CALUDE_stream_speed_l3036_303605

/-- Proves that given a boat with a speed of 13 km/hr in still water,
    traveling 68 km downstream in 4 hours, the speed of the stream is 4 km/hr. -/
theorem stream_speed (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_speed : ℝ) : 
  boat_speed = 13 →
  distance = 68 →
  time = 4 →
  distance = (boat_speed + stream_speed) * time →
  stream_speed = 4 := by
sorry

end NUMINAMATH_CALUDE_stream_speed_l3036_303605


namespace NUMINAMATH_CALUDE_factory_employee_count_l3036_303624

/-- Given a factory with three workshops and stratified sampling information, 
    prove the total number of employees. -/
theorem factory_employee_count 
  (x : ℕ) -- number of employees in Workshop A
  (y : ℕ) -- number of employees in Workshop C
  (h1 : x + 300 + y = 900) -- total employees
  (h2 : 20 + 15 + 10 = 45) -- stratified sample
  : x + 300 + y = 900 := by
  sorry

#check factory_employee_count

end NUMINAMATH_CALUDE_factory_employee_count_l3036_303624


namespace NUMINAMATH_CALUDE_emily_trivia_score_l3036_303640

/-- Emily's trivia game score calculation -/
theorem emily_trivia_score (first_round : ℤ) (last_round : ℤ) (final_score : ℤ) 
  (h1 : first_round = 16)
  (h2 : last_round = -48)
  (h3 : final_score = 1) :
  ∃ second_round : ℤ, first_round + second_round + last_round = final_score ∧ second_round = 33 := by
  sorry

end NUMINAMATH_CALUDE_emily_trivia_score_l3036_303640


namespace NUMINAMATH_CALUDE_derivative_of_product_l3036_303678

-- Define the function f(x) = (x+4)(x-7)
def f (x : ℝ) : ℝ := (x + 4) * (x - 7)

-- State the theorem
theorem derivative_of_product (x : ℝ) : 
  deriv f x = 2 * x - 3 := by
  sorry

end NUMINAMATH_CALUDE_derivative_of_product_l3036_303678


namespace NUMINAMATH_CALUDE_average_after_discarding_l3036_303670

theorem average_after_discarding (numbers : Finset ℕ) (sum : ℕ) (n : ℕ) :
  Finset.card numbers = 50 →
  sum = Finset.sum numbers id →
  sum / 50 = 38 →
  45 ∈ numbers →
  55 ∈ numbers →
  (sum - 45 - 55) / 48 = 75/2 :=
by
  sorry

end NUMINAMATH_CALUDE_average_after_discarding_l3036_303670


namespace NUMINAMATH_CALUDE_angle_B_is_pi_third_b_range_l3036_303622

open Real

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition in the problem --/
def condition (t : Triangle) : Prop :=
  cos t.C + (cos t.A - Real.sqrt 3 * sin t.A) * cos t.B = 0

/-- Theorem 1: Given the condition, angle B is π/3 --/
theorem angle_B_is_pi_third (t : Triangle) (h : condition t) : t.B = π / 3 := by
  sorry

/-- Additional condition for part 2 --/
def sum_sides_is_one (t : Triangle) : Prop :=
  t.a + t.c = 1

/-- Theorem 2: Given sum_sides_is_one and B = π/3, b is in [1/2, 1) --/
theorem b_range (t : Triangle) (h1 : sum_sides_is_one t) (h2 : t.B = π / 3) :
  1 / 2 ≤ t.b ∧ t.b < 1 := by
  sorry

end NUMINAMATH_CALUDE_angle_B_is_pi_third_b_range_l3036_303622


namespace NUMINAMATH_CALUDE_root_sum_reciprocal_minus_two_l3036_303626

-- Define the polynomial
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - x + 3

-- State the theorem
theorem root_sum_reciprocal_minus_two (p q r : ℝ) : 
  f p = 0 → f q = 0 → f r = 0 → 
  1 / (p - 2) + 1 / (q - 2) + 1 / (r - 2) = 1 := by sorry

end NUMINAMATH_CALUDE_root_sum_reciprocal_minus_two_l3036_303626


namespace NUMINAMATH_CALUDE_constant_function_operation_l3036_303660

-- Define the function g
def g : ℝ → ℝ := fun _ ↦ 5

-- State the theorem
theorem constant_function_operation (x : ℝ) : 3 * g (x - 3) + 1 = 16 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_operation_l3036_303660


namespace NUMINAMATH_CALUDE_jeff_tennis_time_l3036_303667

/-- Proves that Jeff played tennis for 2 hours given the conditions -/
theorem jeff_tennis_time (
  points_per_match : ℕ) 
  (minutes_per_point : ℕ) 
  (matches_won : ℕ) 
  (h1 : points_per_match = 8)
  (h2 : minutes_per_point = 5)
  (h3 : matches_won = 3)
  : (points_per_match * matches_won * minutes_per_point) / 60 = 2 := by
  sorry

end NUMINAMATH_CALUDE_jeff_tennis_time_l3036_303667


namespace NUMINAMATH_CALUDE_football_tournament_score_product_l3036_303693

/-- Represents a football team's score in the tournament -/
structure TeamScore where
  points : ℕ

/-- Represents the scores of all teams in the tournament -/
structure TournamentResult where
  scores : Finset TeamScore
  team_count : ℕ
  is_round_robin : Bool
  consecutive_scores : Bool

/-- The main theorem about the tournament results -/
theorem football_tournament_score_product (result : TournamentResult) :
  result.team_count = 4 ∧
  result.is_round_robin = true ∧
  result.consecutive_scores = true ∧
  result.scores.card = 4 →
  (result.scores.toList.map (λ s => s.points)).prod = 120 := by
  sorry

end NUMINAMATH_CALUDE_football_tournament_score_product_l3036_303693


namespace NUMINAMATH_CALUDE_gum_cost_800_l3036_303684

/-- The cost of gum pieces with a bulk discount -/
def gum_cost (pieces : ℕ) : ℚ :=
  let base_cost := pieces
  let discount_threshold := 500
  let discount_rate := 1 / 10
  let total_cents :=
    if pieces > discount_threshold
    then base_cost * (1 - discount_rate)
    else base_cost
  total_cents / 100

/-- The cost of 800 pieces of gum is $7.20 -/
theorem gum_cost_800 : gum_cost 800 = 72 / 10 := by
  sorry

end NUMINAMATH_CALUDE_gum_cost_800_l3036_303684


namespace NUMINAMATH_CALUDE_triangle_height_theorem_l3036_303673

-- Define the triangle ABC
theorem triangle_height_theorem (A B C : ℝ) (a b c : ℝ) :
  -- Conditions
  (0 < A ∧ A < π) →
  (0 < B ∧ B < π) →
  (0 < C ∧ C < π) →
  A + B + C = π →
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  1 + 2 * Real.cos (B + C) = 0 →
  a / Real.sin A = b / Real.sin B →
  a / Real.sin A = c / Real.sin C →
  -- Conclusion
  b * Real.sin C = (Real.sqrt 3 + 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_height_theorem_l3036_303673


namespace NUMINAMATH_CALUDE_mango_price_proof_l3036_303638

/-- The cost of a single lemon in dollars -/
def lemon_cost : ℚ := 2

/-- The cost of a single papaya in dollars -/
def papaya_cost : ℚ := 1

/-- The number of fruits required to get a discount -/
def fruits_for_discount : ℕ := 4

/-- The discount amount in dollars -/
def discount_amount : ℚ := 1

/-- The number of lemons Tom bought -/
def lemons_bought : ℕ := 6

/-- The number of papayas Tom bought -/
def papayas_bought : ℕ := 4

/-- The number of mangos Tom bought -/
def mangos_bought : ℕ := 2

/-- The total amount Tom paid in dollars -/
def total_paid : ℚ := 21

/-- The cost of a single mango in dollars -/
def mango_cost : ℚ := 4

theorem mango_price_proof :
  let total_fruits := lemons_bought + papayas_bought + mangos_bought
  let total_discounts := (total_fruits / fruits_for_discount : ℚ)
  let total_discount_amount := total_discounts * discount_amount
  let total_cost_before_discount := lemon_cost * lemons_bought + papaya_cost * papayas_bought + mango_cost * mangos_bought
  total_cost_before_discount - total_discount_amount = total_paid :=
sorry

end NUMINAMATH_CALUDE_mango_price_proof_l3036_303638


namespace NUMINAMATH_CALUDE_parallel_squares_theorem_l3036_303655

/-- Two squares with parallel sides -/
structure ParallelSquares where
  a : ℝ  -- Side length of the first square
  b : ℝ  -- Side length of the second square
  a_pos : 0 < a
  b_pos : 0 < b

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of an equilateral triangle -/
def is_equilateral (p q r : Point) : Prop :=
  (p.x - q.x)^2 + (p.y - q.y)^2 = (q.x - r.x)^2 + (q.y - r.y)^2 ∧
  (q.x - r.x)^2 + (q.y - r.y)^2 = (r.x - p.x)^2 + (r.y - p.y)^2

/-- The set of points M satisfying the condition -/
def valid_points (squares : ParallelSquares) : Set Point :=
  {m : Point | ∀ p : Point, p.x ∈ [-squares.a/2, squares.a/2] ∧ p.y ∈ [-squares.a/2, squares.a/2] →
    ∃ q : Point, q.x ∈ [-squares.b/2, squares.b/2] ∧ q.y ∈ [-squares.b/2, squares.b/2] ∧
    is_equilateral m p q}

/-- The main theorem -/
theorem parallel_squares_theorem (squares : ParallelSquares) :
  (valid_points squares).Nonempty ↔ squares.b ≥ (squares.a / 2) * (Real.sqrt 3 + 1) :=
sorry

end NUMINAMATH_CALUDE_parallel_squares_theorem_l3036_303655


namespace NUMINAMATH_CALUDE_quirkyville_reading_paradox_l3036_303687

/-- Represents the student population at Quirkyville College -/
structure StudentPopulation where
  total : ℕ
  enjoy_reading : ℕ
  claim_enjoy : ℕ
  claim_not_enjoy : ℕ

/-- The fraction of students who say they don't enjoy reading but actually do -/
def fraction_false_negative (pop : StudentPopulation) : ℚ :=
  (pop.enjoy_reading - pop.claim_enjoy) / pop.claim_not_enjoy

/-- Theorem stating the fraction of students who say they don't enjoy reading but actually do -/
theorem quirkyville_reading_paradox (pop : StudentPopulation) : 
  pop.total > 0 ∧ 
  pop.enjoy_reading = (70 * pop.total) / 100 ∧
  pop.claim_enjoy = (75 * pop.enjoy_reading) / 100 ∧
  pop.claim_not_enjoy = pop.total - pop.claim_enjoy →
  fraction_false_negative pop = 35 / 83 := by
  sorry

#eval (35 : ℚ) / 83

end NUMINAMATH_CALUDE_quirkyville_reading_paradox_l3036_303687


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3036_303634

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_length := 1.6 * L
  let new_area := 6 * new_length^2
  (new_area - original_area) / original_area * 100 = 156 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l3036_303634


namespace NUMINAMATH_CALUDE_equation_solutions_l3036_303608

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = (1 + Real.sqrt 5) / 4 ∧ x2 = (1 - Real.sqrt 5) / 4 ∧
    4 * x1^2 - 2 * x1 - 1 = 0 ∧ 4 * x2^2 - 2 * x2 - 1 = 0) ∧
  (∃ y1 y2 : ℝ, y1 = 1 ∧ y2 = 0 ∧
    (y1 + 1)^2 = (3 * y1 - 1)^2 ∧ (y2 + 1)^2 = (3 * y2 - 1)^2) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3036_303608


namespace NUMINAMATH_CALUDE_tickets_for_pesos_l3036_303689

/-- Given that T tickets cost R dollars and 10 pesos is worth 40 dollars,
    this theorem proves that the number of tickets that can be purchased
    for P pesos is 4PT/R. -/
theorem tickets_for_pesos (T R P : ℝ) (h1 : T > 0) (h2 : R > 0) (h3 : P > 0) :
  let dollars_per_peso : ℝ := 40 / 10
  let pesos_in_dollars : ℝ := P * dollars_per_peso
  let tickets_per_dollar : ℝ := T / R
  tickets_per_dollar * pesos_in_dollars = 4 * P * T / R :=
by sorry

end NUMINAMATH_CALUDE_tickets_for_pesos_l3036_303689


namespace NUMINAMATH_CALUDE_even_triple_composition_l3036_303695

/-- A function is even if f(-x) = f(x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The main theorem: if f is even, then f ∘ f ∘ f is even -/
theorem even_triple_composition {f : ℝ → ℝ} (hf : IsEven f) : IsEven (f ∘ f ∘ f) := by
  sorry

end NUMINAMATH_CALUDE_even_triple_composition_l3036_303695


namespace NUMINAMATH_CALUDE_petes_flag_problem_l3036_303619

theorem petes_flag_problem (us_stars : Nat) (us_stripes : Nat) (total_shapes : Nat) :
  us_stars = 50 →
  us_stripes = 13 →
  total_shapes = 54 →
  ∃ (circles squares : Nat),
    circles < us_stars / 2 ∧
    squares = 2 * us_stripes + 6 ∧
    circles + squares = total_shapes ∧
    us_stars / 2 - circles = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_petes_flag_problem_l3036_303619


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l3036_303653

def A : Set Int := {0, 1, 2}
def B : Set Int := {-1, 0, 1}

theorem union_of_A_and_B : A ∪ B = {-1, 0, 1, 2} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l3036_303653


namespace NUMINAMATH_CALUDE_power_of_power_l3036_303635

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_power_l3036_303635


namespace NUMINAMATH_CALUDE_division_calculation_l3036_303643

theorem division_calculation : (6 : ℚ) / (-1/2 + 1/3) = -36 := by sorry

end NUMINAMATH_CALUDE_division_calculation_l3036_303643


namespace NUMINAMATH_CALUDE_horner_first_step_for_f_l3036_303681

def f (x : ℝ) : ℝ := 0.5 * x^6 + 4 * x^5 - x^4 + 3 * x^3 - 5 * x

def horner_first_step (a₆ a₅ : ℝ) (x : ℝ) : ℝ := a₆ * x + a₅

theorem horner_first_step_for_f :
  horner_first_step 0.5 4 3 = 5.5 :=
sorry

end NUMINAMATH_CALUDE_horner_first_step_for_f_l3036_303681


namespace NUMINAMATH_CALUDE_problem_solution_l3036_303651

theorem problem_solution :
  (∃ x1 x2 : ℝ, x1 = 2 + Real.sqrt 7 ∧ x2 = 2 - Real.sqrt 7 ∧
    x1^2 - 4*x1 - 3 = 0 ∧ x2^2 - 4*x2 - 3 = 0) ∧
  (abs (-3) - 4 * Real.sin (π/4) + Real.sqrt 8 + (π - 3)^0 = 4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3036_303651


namespace NUMINAMATH_CALUDE_symmetric_points_line_equation_l3036_303676

/-- Given two points are symmetric about a line, prove the equation of the line -/
theorem symmetric_points_line_equation (O A : ℝ × ℝ) (l : Set (ℝ × ℝ)) :
  O = (0, 0) →
  A = (-4, 2) →
  (∀ p : ℝ × ℝ, p ∈ l ↔ (p.1 - O.1) * (A.1 - O.1) + (p.2 - O.2) * (A.2 - O.2) = 0) →
  (∀ x y : ℝ, (x, y) ∈ l ↔ 2*x - y + 5 = 0) :=
by sorry

end NUMINAMATH_CALUDE_symmetric_points_line_equation_l3036_303676


namespace NUMINAMATH_CALUDE_power_tower_mod_500_l3036_303639

theorem power_tower_mod_500 : 7^(7^(7^7)) ≡ 543 [ZMOD 500] := by
  sorry

end NUMINAMATH_CALUDE_power_tower_mod_500_l3036_303639


namespace NUMINAMATH_CALUDE_yolandas_walking_rate_l3036_303637

/-- Proves that Yolanda's walking rate is 5 miles per hour given the problem conditions -/
theorem yolandas_walking_rate
  (total_distance : ℝ)
  (bobs_rate : ℝ)
  (time_difference : ℝ)
  (bobs_distance : ℝ)
  (h1 : total_distance = 60)
  (h2 : bobs_rate = 6)
  (h3 : time_difference = 1)
  (h4 : bobs_distance = 30) :
  (total_distance - bobs_distance) / (bobs_distance / bobs_rate + time_difference) = 5 :=
by sorry

end NUMINAMATH_CALUDE_yolandas_walking_rate_l3036_303637


namespace NUMINAMATH_CALUDE_lucas_change_l3036_303699

/-- Represents the shopping scenario and calculates the change --/
def calculate_change (initial_amount : ℝ) 
  (avocado_costs : List ℝ) 
  (water_cost : ℝ) 
  (water_quantity : ℕ) 
  (apple_cost : ℝ) 
  (apple_quantity : ℕ) : ℝ :=
  let total_cost := (avocado_costs.sum + water_cost * water_quantity + apple_cost * apple_quantity)
  initial_amount - total_cost

/-- Theorem stating that Lucas brings home $6.75 in change --/
theorem lucas_change : 
  calculate_change 20 [1.50, 2.25, 3.00] 1.75 2 0.75 4 = 6.75 := by
  sorry

#eval calculate_change 20 [1.50, 2.25, 3.00] 1.75 2 0.75 4

end NUMINAMATH_CALUDE_lucas_change_l3036_303699


namespace NUMINAMATH_CALUDE_train_speed_l3036_303632

theorem train_speed (train_length : Real) (crossing_time : Real) (h1 : train_length = 1600) (h2 : crossing_time = 40) :
  (train_length / 1000) / (crossing_time / 3600) = 144 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l3036_303632


namespace NUMINAMATH_CALUDE_desk_purchase_price_l3036_303671

/-- Proves that the purchase price of a desk is $100 given the specified conditions -/
theorem desk_purchase_price (purchase_price selling_price : ℝ) : 
  selling_price = purchase_price + 0.5 * selling_price →
  selling_price - purchase_price = 100 →
  purchase_price = 100 := by
sorry

end NUMINAMATH_CALUDE_desk_purchase_price_l3036_303671


namespace NUMINAMATH_CALUDE_square_difference_252_248_l3036_303612

theorem square_difference_252_248 : 252^2 - 248^2 = 2000 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_252_248_l3036_303612


namespace NUMINAMATH_CALUDE_triangle_area_l3036_303665

/-- Given a triangle with perimeter 32 cm and inradius 3.5 cm, its area is 56 cm². -/
theorem triangle_area (p r A : ℝ) (h1 : p = 32) (h2 : r = 3.5) (h3 : A = r * p / 2) : A = 56 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3036_303665


namespace NUMINAMATH_CALUDE_rectangle_diagonal_l3036_303602

theorem rectangle_diagonal (l w : ℝ) (h1 : l = 8) (h2 : 2 * l + 2 * w = 46) :
  Real.sqrt (l^2 + w^2) = 17 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_diagonal_l3036_303602


namespace NUMINAMATH_CALUDE_voting_change_l3036_303680

theorem voting_change (total_members : ℕ) 
  (h_total : total_members = 400)
  (initial_for initial_against : ℕ) 
  (h_initial_sum : initial_for + initial_against = total_members)
  (h_initial_reject : initial_against > initial_for)
  (second_for second_against : ℕ) 
  (h_second_sum : second_for + second_against = total_members)
  (h_second_pass : second_for > second_against)
  (h_margin : second_for - second_against = 3 * (initial_against - initial_for))
  (h_proportion : second_for = (10 * initial_against) / 9) :
  second_for - initial_for = 48 := by
sorry

end NUMINAMATH_CALUDE_voting_change_l3036_303680


namespace NUMINAMATH_CALUDE_cost_operation_l3036_303688

theorem cost_operation (t : ℝ) (b b' : ℝ) : 
  (∀ C, C = t * b^4) →
  (∃ e, e = 16 * t * b^4) →
  (∃ e, e = t * b'^4) →
  b' = 2 * b :=
sorry

end NUMINAMATH_CALUDE_cost_operation_l3036_303688


namespace NUMINAMATH_CALUDE_last_non_zero_digit_30_factorial_l3036_303603

/-- The last non-zero digit of a natural number -/
def lastNonZeroDigit (n : ℕ) : ℕ :=
  n % 10 -- Definition, not from solution steps

/-- Factorial function -/
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

theorem last_non_zero_digit_30_factorial :
  lastNonZeroDigit (factorial 30) = 8 := by
  sorry

end NUMINAMATH_CALUDE_last_non_zero_digit_30_factorial_l3036_303603


namespace NUMINAMATH_CALUDE_cosine_function_phi_range_l3036_303661

/-- The cosine function -/
noncomputable def f (ω : ℝ) (φ : ℝ) (x : ℝ) : ℝ := 2 * Real.cos (ω * x + φ) + 1

/-- The theorem statement -/
theorem cosine_function_phi_range 
  (ω : ℝ) 
  (φ : ℝ) 
  (h_ω : ω > 0) 
  (h_φ : |φ| < π/2) 
  (h_period : ∃ (x₁ x₂ : ℝ), x₂ - x₁ = 2*π/3 ∧ f ω φ x₁ = 3 ∧ f ω φ x₂ = 3)
  (h_range : ∀ x ∈ Set.Ioo (-π/12) (π/6), f ω φ x > 1) :
  φ ∈ Set.Icc (-π/4) 0 :=
sorry

end NUMINAMATH_CALUDE_cosine_function_phi_range_l3036_303661


namespace NUMINAMATH_CALUDE_odd_periodic_function_difference_l3036_303674

theorem odd_periodic_function_difference (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_periodic : ∀ x, f (x + 5) = f x)
  (h_f1 : f 1 = 1)
  (h_f2 : f 2 = 2) :
  f 3 - f 4 = -1 := by
  sorry

end NUMINAMATH_CALUDE_odd_periodic_function_difference_l3036_303674


namespace NUMINAMATH_CALUDE_geometric_sequence_207th_term_l3036_303659

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r ^ (n - 1)

theorem geometric_sequence_207th_term :
  let a₁ := 8
  let a₂ := -8
  let r := a₂ / a₁
  geometric_sequence a₁ r 207 = 8 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_207th_term_l3036_303659


namespace NUMINAMATH_CALUDE_glitched_clock_correct_time_fraction_l3036_303618

/-- Represents a 24-hour digital clock with a glitch that displays '2' as '7' -/
structure GlitchedClock where
  /-- The number of hours in a day -/
  hours_per_day : ℕ := 24
  /-- The number of minutes per hour -/
  minutes_per_hour : ℕ := 60
  /-- The digit that is displayed incorrectly -/
  glitch_digit : ℕ := 2

/-- Calculates the fraction of the day the clock displays the correct time -/
def correct_time_fraction (clock : GlitchedClock) : ℚ :=
  let correct_hours := clock.hours_per_day - 6  -- Hours without '2'
  let correct_minutes_per_hour := clock.minutes_per_hour - 16  -- Minutes without '2' per hour
  (correct_hours : ℚ) / clock.hours_per_day * correct_minutes_per_hour / clock.minutes_per_hour

theorem glitched_clock_correct_time_fraction :
  ∀ (clock : GlitchedClock), correct_time_fraction clock = 11 / 20 := by
  sorry

end NUMINAMATH_CALUDE_glitched_clock_correct_time_fraction_l3036_303618


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l3036_303692

open Set

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {1, 4}

theorem complement_of_A_in_U : 
  (U \ A) = {2, 3, 5} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l3036_303692


namespace NUMINAMATH_CALUDE_inevitable_same_color_l3036_303636

/-- Represents the color of a ball -/
inductive Color
| Red
| Yellow

/-- Represents a bag of balls -/
structure Bag :=
  (red : Nat)
  (yellow : Nat)

/-- Represents a draw of balls -/
structure Draw :=
  (red : Nat)
  (yellow : Nat)

/-- Checks if a draw has at least two balls of the same color -/
def hasAtLeastTwoSameColor (d : Draw) : Prop :=
  d.red ≥ 2 ∨ d.yellow ≥ 2

/-- The theorem stating that drawing 3 balls from a bag with 3 red and 3 yellow balls
    will inevitably result in at least 2 balls of the same color -/
theorem inevitable_same_color (b : Bag) (d : Draw) 
    (h1 : b.red = 3)
    (h2 : b.yellow = 3)
    (h3 : d.red + d.yellow = 3)
    (h4 : d.red ≤ b.red ∧ d.yellow ≤ b.yellow) :
    hasAtLeastTwoSameColor d :=
  sorry


end NUMINAMATH_CALUDE_inevitable_same_color_l3036_303636


namespace NUMINAMATH_CALUDE_jeffreys_farm_chickens_total_chickens_is_76_l3036_303647

/-- Calculates the total number of chickens on Jeffrey's farm -/
theorem jeffreys_farm_chickens (num_hens : ℕ) (hen_rooster_ratio : ℕ) (chicks_per_hen : ℕ) : ℕ :=
  let num_roosters := num_hens / hen_rooster_ratio
  let num_chicks := num_hens * chicks_per_hen
  num_hens + num_roosters + num_chicks

/-- Proves that the total number of chickens on Jeffrey's farm is 76 -/
theorem total_chickens_is_76 :
  jeffreys_farm_chickens 12 3 5 = 76 := by
  sorry

end NUMINAMATH_CALUDE_jeffreys_farm_chickens_total_chickens_is_76_l3036_303647


namespace NUMINAMATH_CALUDE_m_plus_n_equals_plus_minus_one_l3036_303649

theorem m_plus_n_equals_plus_minus_one (m n : ℤ) 
  (hm : |m| = 3) 
  (hn : |n| = 2) 
  (hmn : m * n < 0) : 
  m + n = 1 ∨ m + n = -1 := by
sorry

end NUMINAMATH_CALUDE_m_plus_n_equals_plus_minus_one_l3036_303649


namespace NUMINAMATH_CALUDE_total_card_cost_l3036_303658

def christmas_cards : ℕ := 20
def birthday_cards : ℕ := 15
def card_cost : ℕ := 2

theorem total_card_cost : christmas_cards * card_cost + birthday_cards * card_cost = 70 := by
  sorry

end NUMINAMATH_CALUDE_total_card_cost_l3036_303658


namespace NUMINAMATH_CALUDE_min_distance_squared_l3036_303628

/-- The minimum squared distance between a curve and a line -/
theorem min_distance_squared (a b m n : ℝ) : 
  a > 0 → 
  b = -1/2 * a^2 + 3 * Real.log a → 
  n = 2 * m + 1/2 → 
  ∃ (min_dist : ℝ), 
    (∀ (x y : ℝ), y = -1/2 * x^2 + 3 * Real.log x → 
      (x - m)^2 + (y - n)^2 ≥ min_dist) ∧
    min_dist = 9/5 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_squared_l3036_303628


namespace NUMINAMATH_CALUDE_max_chickens_and_chicks_max_chicks_no_chickens_l3036_303662

/-- Represents the chicken coop problem -/
structure ChickenCoop where
  area : ℝ
  chicken_space : ℝ
  chick_space : ℝ
  chicken_feed : ℝ
  chick_feed : ℝ
  max_feed : ℝ

/-- Defines the specific chicken coop instance -/
def our_coop : ChickenCoop :=
  { area := 240
  , chicken_space := 4
  , chick_space := 2
  , chicken_feed := 160
  , chick_feed := 40
  , max_feed := 8000 }

/-- Theorem stating the maximum number of chickens and chicks -/
theorem max_chickens_and_chicks (coop : ChickenCoop) :
  ∃ (x y : ℕ), 
    x * coop.chicken_space + y * coop.chick_space = coop.area ∧
    x * coop.chicken_feed + y * coop.chick_feed ≤ coop.max_feed ∧
    x = 40 ∧ y = 40 ∧
    ∀ (a b : ℕ), 
      a * coop.chicken_space + b * coop.chick_space = coop.area →
      a * coop.chicken_feed + b * coop.chick_feed ≤ coop.max_feed →
      a ≤ x :=
by sorry

/-- Theorem stating the maximum number of chicks with no chickens -/
theorem max_chicks_no_chickens (coop : ChickenCoop) :
  ∃ (y : ℕ),
    y * coop.chick_space = coop.area ∧
    y * coop.chick_feed ≤ coop.max_feed ∧
    y = 120 ∧
    ∀ (b : ℕ),
      b * coop.chick_space = coop.area →
      b * coop.chick_feed ≤ coop.max_feed →
      b ≤ y :=
by sorry

end NUMINAMATH_CALUDE_max_chickens_and_chicks_max_chicks_no_chickens_l3036_303662


namespace NUMINAMATH_CALUDE_empty_set_implies_a_range_l3036_303686

theorem empty_set_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, a * x^2 - |x + 1| + 2 * a ≥ 0) → 
  a > (Real.sqrt 3 + 1) / 4 := by
sorry

end NUMINAMATH_CALUDE_empty_set_implies_a_range_l3036_303686


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3036_303627

theorem cubic_root_sum (n : ℤ) (p q r : ℤ) : 
  (∀ x : ℤ, x^3 - 2024*x + n = 0 ↔ x = p ∨ x = q ∨ x = r) →
  |p| + |q| + |r| = 100 := by
  sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3036_303627


namespace NUMINAMATH_CALUDE_solution_exists_in_interval_l3036_303669

def f (x : ℝ) := x^3 + x - 5

theorem solution_exists_in_interval :
  (f 1 < 0) → (f 2 > 0) → (f 1.5 < 0) →
  ∃ x, x ∈ Set.Ioo 1.5 2 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_solution_exists_in_interval_l3036_303669
