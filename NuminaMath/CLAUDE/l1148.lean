import Mathlib

namespace NUMINAMATH_CALUDE_cross_product_result_l1148_114857

def a : ℝ × ℝ × ℝ := (4, 2, -1)
def b : ℝ × ℝ × ℝ := (3, -5, 6)

def cross_product (v w : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  (v.2.1 * w.2.2 - v.2.2 * w.2.1,
   v.2.2 * w.1 - v.1 * w.2.2,
   v.1 * w.2.1 - v.2.1 * w.1)

theorem cross_product_result :
  cross_product a b = (7, -27, -26) := by
  sorry

end NUMINAMATH_CALUDE_cross_product_result_l1148_114857


namespace NUMINAMATH_CALUDE_duo_ball_playing_time_l1148_114802

theorem duo_ball_playing_time (num_children : ℕ) (total_time : ℕ) (players_per_game : ℕ) :
  num_children = 8 →
  total_time = 120 →
  players_per_game = 2 →
  (total_time * players_per_game) / num_children = 30 :=
by
  sorry

end NUMINAMATH_CALUDE_duo_ball_playing_time_l1148_114802


namespace NUMINAMATH_CALUDE_marble_selection_probability_l1148_114813

def total_marbles : ℕ := 3 + 2 + 2
def red_marbles : ℕ := 3
def blue_marbles : ℕ := 2
def green_marbles : ℕ := 2
def selected_marbles : ℕ := 4

theorem marble_selection_probability :
  (Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1) /
  (Nat.choose total_marbles selected_marbles) = 12 / 35 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_probability_l1148_114813


namespace NUMINAMATH_CALUDE_inequality_proof_l1148_114882

theorem inequality_proof (x a : ℝ) (hx : x > 0 ∧ x ≠ 1) (ha : a < 1) :
  (1 - x^a) / (1 - x) < (1 + x)^(a - 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1148_114882


namespace NUMINAMATH_CALUDE_tetrahedron_edge_assignment_l1148_114860

/-- Represents a tetrahedron with face areas -/
structure Tetrahedron where
  s : ℝ  -- smallest face area
  S : ℝ  -- largest face area
  a : ℝ  -- area of another face
  b : ℝ  -- area of the fourth face
  h_s_smallest : s ≤ a ∧ s ≤ b ∧ s ≤ S
  h_S_largest : S ≥ a ∧ S ≥ b ∧ S ≥ s
  h_positive : s > 0 ∧ S > 0 ∧ a > 0 ∧ b > 0

/-- Represents the edge values of a tetrahedron -/
structure TetrahedronEdges where
  e1 : ℝ  -- edge common to smallest and largest face
  e2 : ℝ  -- edge of smallest face
  e3 : ℝ  -- edge of smallest face
  e4 : ℝ  -- edge of largest face
  e5 : ℝ  -- edge of largest face
  e6 : ℝ  -- remaining edge

/-- Checks if the edge values satisfy the face area conditions -/
def satisfies_conditions (t : Tetrahedron) (e : TetrahedronEdges) : Prop :=
  e.e1 ≥ 0 ∧ e.e2 ≥ 0 ∧ e.e3 ≥ 0 ∧ e.e4 ≥ 0 ∧ e.e5 ≥ 0 ∧ e.e6 ≥ 0 ∧
  e.e1 + e.e2 + e.e3 = t.s ∧
  e.e1 + e.e4 + e.e5 = t.S ∧
  e.e2 + e.e5 + e.e6 = t.a ∧
  e.e3 + e.e4 + e.e6 = t.b

theorem tetrahedron_edge_assignment (t : Tetrahedron) :
  ∃ e : TetrahedronEdges, satisfies_conditions t e := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_edge_assignment_l1148_114860


namespace NUMINAMATH_CALUDE_circular_motion_angle_l1148_114823

theorem circular_motion_angle (θ : Real) : 
  (0 < θ) ∧ (θ < π) ∧                        -- 0 < θ < π
  (π < 2*θ) ∧ (2*θ < 3*π/2) ∧                -- Reaches third quadrant in 2 minutes
  (∃ (n : ℤ), 14*θ = n * (2*π)) →            -- Returns to original position in 14 minutes
  (θ = 4*π/7) ∨ (θ = 5*π/7) := by
sorry

end NUMINAMATH_CALUDE_circular_motion_angle_l1148_114823


namespace NUMINAMATH_CALUDE_initial_investment_l1148_114856

/-- Proves that the initial investment is 8000 given the specified conditions -/
theorem initial_investment (x : ℝ) : 
  (0.05 * x + 0.08 * 4000 = 0.06 * (x + 4000)) → x = 8000 :=
by
  sorry

end NUMINAMATH_CALUDE_initial_investment_l1148_114856


namespace NUMINAMATH_CALUDE_smallest_n_with_1981_zeros_l1148_114875

def count_trailing_zeros (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625) + (n / 3125)

theorem smallest_n_with_1981_zeros :
  ∃ (n : ℕ), count_trailing_zeros n = 1981 ∧
    ∀ (m : ℕ), m < n → count_trailing_zeros m < 1981 :=
by
  use 7935
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_1981_zeros_l1148_114875


namespace NUMINAMATH_CALUDE_sin_equality_810_degrees_l1148_114899

theorem sin_equality_810_degrees (n : ℤ) :
  -180 ≤ n ∧ n ≤ 180 ∧ Real.sin (n * π / 180) = Real.sin (810 * π / 180) → n = 90 := by
  sorry

end NUMINAMATH_CALUDE_sin_equality_810_degrees_l1148_114899


namespace NUMINAMATH_CALUDE_kevin_six_hops_l1148_114809

def kevin_hop (n : ℕ) : ℚ :=
  2 * (1 - (3/4)^n)

theorem kevin_six_hops :
  kevin_hop 6 = 3367 / 2048 := by
  sorry

end NUMINAMATH_CALUDE_kevin_six_hops_l1148_114809


namespace NUMINAMATH_CALUDE_workshop_salary_theorem_l1148_114884

def workshop_problem (total_workers : ℕ) (num_technicians : ℕ) (avg_salary_technicians : ℚ) (avg_salary_others : ℚ) : Prop :=
  let num_others : ℕ := total_workers - num_technicians
  let total_salary_technicians : ℚ := num_technicians * avg_salary_technicians
  let total_salary_others : ℚ := num_others * avg_salary_others
  let total_salary : ℚ := total_salary_technicians + total_salary_others
  let avg_salary_all : ℚ := total_salary / total_workers
  avg_salary_all = 6750

theorem workshop_salary_theorem :
  workshop_problem 56 7 12000 6000 := by
  sorry

end NUMINAMATH_CALUDE_workshop_salary_theorem_l1148_114884


namespace NUMINAMATH_CALUDE_fruit_distribution_l1148_114838

/-- Given 30 pieces of fruit to be distributed equally among 4 friends,
    the smallest number of pieces to remove for equal distribution is 2. -/
theorem fruit_distribution (total_fruit : Nat) (friends : Nat) (pieces_to_remove : Nat) : 
  total_fruit = 30 →
  friends = 4 →
  pieces_to_remove = 2 →
  (total_fruit - pieces_to_remove) % friends = 0 ∧
  ∀ n : Nat, n < pieces_to_remove → (total_fruit - n) % friends ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_fruit_distribution_l1148_114838


namespace NUMINAMATH_CALUDE_subset_implies_a_greater_than_half_l1148_114800

-- Define the sets M and N
def M : Set ℝ := {x | -2 * x + 1 ≥ 0}
def N (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem subset_implies_a_greater_than_half (a : ℝ) :
  M ⊆ N a → a > 1/2 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_a_greater_than_half_l1148_114800


namespace NUMINAMATH_CALUDE_museum_revenue_calculation_l1148_114831

/-- Revenue calculation for The Metropolitan Museum of Art --/
theorem museum_revenue_calculation 
  (total_visitors : ℕ) 
  (nyc_resident_ratio : ℚ)
  (college_student_ratio : ℚ)
  (college_ticket_price : ℕ) :
  total_visitors = 200 →
  nyc_resident_ratio = 1/2 →
  college_student_ratio = 3/10 →
  college_ticket_price = 4 →
  (total_visitors : ℚ) * nyc_resident_ratio * college_student_ratio * college_ticket_price = 120 := by
  sorry

#check museum_revenue_calculation

end NUMINAMATH_CALUDE_museum_revenue_calculation_l1148_114831


namespace NUMINAMATH_CALUDE_age_double_time_l1148_114889

/-- Given two brothers with current ages 15 and 5, this theorem proves that
    it will take 5 years for the older brother's age to be twice the younger brother's age. -/
theorem age_double_time (older_age younger_age : ℕ) (h1 : older_age = 15) (h2 : younger_age = 5) :
  ∃ (years : ℕ), years = 5 ∧ older_age + years = 2 * (younger_age + years) :=
sorry

end NUMINAMATH_CALUDE_age_double_time_l1148_114889


namespace NUMINAMATH_CALUDE_photo_gallery_problem_l1148_114871

/-- The total number of photos in a gallery after a two-day trip -/
def total_photos (initial : ℕ) (first_day : ℕ) (second_day : ℕ) : ℕ :=
  initial + first_day + second_day

/-- Theorem: Given the conditions of the photo gallery problem, the total number of photos is 920 -/
theorem photo_gallery_problem :
  let initial := 400
  let first_day := initial / 2
  let second_day := first_day + 120
  total_photos initial first_day second_day = 920 := by
  sorry

end NUMINAMATH_CALUDE_photo_gallery_problem_l1148_114871


namespace NUMINAMATH_CALUDE_otts_money_fraction_l1148_114858

theorem otts_money_fraction (moe loki nick ott : ℚ) : 
  moe > 0 → loki > 0 → nick > 0 → ott = 0 →
  ∃ (x : ℚ), x > 0 ∧ 
    x = moe / 3 ∧ 
    x = loki / 5 ∧ 
    x = nick / 4 →
  (3 * x) / (moe + loki + nick + 3 * x) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_otts_money_fraction_l1148_114858


namespace NUMINAMATH_CALUDE_cloak_change_theorem_l1148_114890

/-- Represents the price and change for buying an invisibility cloak -/
structure CloakTransaction where
  silver_paid : ℕ
  gold_change : ℕ

/-- Calculates the change in silver coins when buying a cloak with gold coins -/
def calculate_silver_change (t1 t2 : CloakTransaction) (gold_paid : ℕ) : ℕ :=
  sorry

theorem cloak_change_theorem (t1 t2 : CloakTransaction) 
  (h1 : t1.silver_paid = 20 ∧ t1.gold_change = 4)
  (h2 : t2.silver_paid = 15 ∧ t2.gold_change = 1) :
  calculate_silver_change t1 t2 14 = 10 :=
sorry

end NUMINAMATH_CALUDE_cloak_change_theorem_l1148_114890


namespace NUMINAMATH_CALUDE_smallest_integer_gcd_lcm_problem_l1148_114836

theorem smallest_integer_gcd_lcm_problem (x : ℕ) (a b : ℕ) : 
  x > 0 →
  a > 0 →
  b > 0 →
  a = 72 →
  Nat.gcd a b = x + 6 →
  Nat.lcm a b = x * (x + 6) →
  ∃ m : ℕ, (∀ n : ℕ, n > 0 ∧ 
    Nat.gcd 72 n = x + 6 ∧ 
    Nat.lcm 72 n = x * (x + 6) → 
    m ≤ n) ∧ 
    m = 12 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_gcd_lcm_problem_l1148_114836


namespace NUMINAMATH_CALUDE_sequence_theorem_l1148_114829

/-- A sequence whose reciprocal forms an arithmetic sequence -/
def IsReciprocalArithmetic (a : ℕ → ℝ) : Prop :=
  ∀ n, 2 / a (n + 1) = 1 / a n + 1 / a (n + 2)

/-- The main theorem -/
theorem sequence_theorem (x : ℕ → ℝ) (a : ℕ → ℝ) 
    (h_pos : ∀ n, x n > 0)
    (h_recip_arith : IsReciprocalArithmetic a)
    (h_x1 : x 1 = 3)
    (h_sum : x 1 + x 2 + x 3 = 39)
    (h_power : ∀ n, (x n) ^ (a n) = (x (n + 1)) ^ (a (n + 1)) ∧ 
                    (x n) ^ (a n) = (x (n + 2)) ^ (a (n + 2))) : 
  ∀ n, x n = 3^n := by
  sorry

end NUMINAMATH_CALUDE_sequence_theorem_l1148_114829


namespace NUMINAMATH_CALUDE_two_different_color_chips_probability_l1148_114873

def blue_chips : ℕ := 5
def yellow_chips : ℕ := 3
def red_chips : ℕ := 4

def total_chips : ℕ := blue_chips + yellow_chips + red_chips

def probability_different_colors : ℚ :=
  (blue_chips * yellow_chips + blue_chips * red_chips + yellow_chips * red_chips) * 2 /
  (total_chips * total_chips)

theorem two_different_color_chips_probability :
  probability_different_colors = 47 / 72 := by
  sorry

end NUMINAMATH_CALUDE_two_different_color_chips_probability_l1148_114873


namespace NUMINAMATH_CALUDE_highway_project_deadline_l1148_114810

/-- Represents the initial deadline for completing the highway project --/
def initial_deadline : ℝ := 37.5

/-- The number of initial workers --/
def initial_workers : ℕ := 100

/-- The number of additional workers hired --/
def additional_workers : ℕ := 60

/-- The initial daily work hours --/
def initial_hours : ℕ := 8

/-- The new daily work hours after hiring additional workers --/
def new_hours : ℕ := 10

/-- The number of days worked before hiring additional workers --/
def days_worked : ℕ := 25

/-- The fraction of work completed before hiring additional workers --/
def work_completed : ℚ := 1/3

/-- Theorem stating that the initial deadline is correct given the conditions --/
theorem highway_project_deadline :
  ∃ (total_work : ℝ),
    total_work = initial_workers * days_worked * initial_hours ∧
    (2/3 : ℝ) * total_work = (initial_workers + additional_workers) * (initial_deadline - days_worked) * new_hours :=
by sorry

end NUMINAMATH_CALUDE_highway_project_deadline_l1148_114810


namespace NUMINAMATH_CALUDE_solve_for_d_l1148_114892

theorem solve_for_d (n k c d : ℝ) (h : n = (2 * k * c * d) / (c + d)) (h_nonzero : 2 * k * c ≠ n) :
  d = (n * c) / (2 * k * c - n) := by
  sorry

end NUMINAMATH_CALUDE_solve_for_d_l1148_114892


namespace NUMINAMATH_CALUDE_modular_congruence_solution_l1148_114814

theorem modular_congruence_solution : ∃! n : ℤ, 0 ≤ n ∧ n ≤ 12 ∧ n ≡ -867 [ZMOD 13] ∧ n = 9 := by
  sorry

end NUMINAMATH_CALUDE_modular_congruence_solution_l1148_114814


namespace NUMINAMATH_CALUDE_wario_field_goals_l1148_114879

theorem wario_field_goals 
  (missed_fraction : ℚ)
  (wide_right_fraction : ℚ)
  (wide_right_misses : ℕ)
  (h1 : missed_fraction = 1 / 4)
  (h2 : wide_right_fraction = 1 / 5)
  (h3 : wide_right_misses = 3) :
  ∃ (total_attempts : ℕ), 
    (↑wide_right_misses : ℚ) / wide_right_fraction / missed_fraction = total_attempts ∧ 
    total_attempts = 60 := by
sorry


end NUMINAMATH_CALUDE_wario_field_goals_l1148_114879


namespace NUMINAMATH_CALUDE_equation_represents_intersecting_lines_l1148_114896

-- Define the equation
def equation (x y : ℝ) : Prop := x^2 - y^2 = 0

-- Theorem statement
theorem equation_represents_intersecting_lines :
  ∃ (f g : ℝ → ℝ), 
    (∀ x, f x = x ∧ g x = -x) ∧
    (∀ x y, equation x y ↔ (y = f x ∨ y = g x)) :=
sorry

end NUMINAMATH_CALUDE_equation_represents_intersecting_lines_l1148_114896


namespace NUMINAMATH_CALUDE_division_result_l1148_114851

theorem division_result : (0.0204 : ℝ) / 17 = 0.0012 := by
  sorry

end NUMINAMATH_CALUDE_division_result_l1148_114851


namespace NUMINAMATH_CALUDE_sqrt_fraction_equality_l1148_114825

theorem sqrt_fraction_equality : 
  (Real.sqrt (8^2 + 15^2)) / (Real.sqrt (25 + 36)) = (17 * Real.sqrt 61) / 61 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_fraction_equality_l1148_114825


namespace NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l1148_114854

/-- Given the quadratic expression 4k^2 - 8k + 16, when rewritten in the form a(k + b)^2 + c,
    the ratio c/b equals -12 -/
theorem quadratic_rewrite_ratio : 
  ∃ (a b c : ℝ), (∀ k, 4 * k^2 - 8 * k + 16 = a * (k + b)^2 + c) ∧ c / b = -12 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_ratio_l1148_114854


namespace NUMINAMATH_CALUDE_max_third_side_of_triangle_l1148_114832

theorem max_third_side_of_triangle (a b : ℝ) (ha : a = 7) (hb : b = 11) :
  ∃ (c : ℕ), c = 17 ∧ 
  (∀ (x : ℕ), (a + b > x ∧ x > b - a ∧ x > a - b) → x ≤ c) :=
sorry

end NUMINAMATH_CALUDE_max_third_side_of_triangle_l1148_114832


namespace NUMINAMATH_CALUDE_max_notebooks_is_11_l1148_114808

/-- Represents the number of notebooks in a pack -/
inductive PackSize
  | Single
  | Pack4
  | Pack7

/-- Returns the number of notebooks for a given pack size -/
def notebooks (size : PackSize) : ℕ :=
  match size with
  | PackSize.Single => 1
  | PackSize.Pack4 => 4
  | PackSize.Pack7 => 7

/-- Returns the cost in dollars for a given pack size -/
def cost (size : PackSize) : ℕ :=
  match size with
  | PackSize.Single => 2
  | PackSize.Pack4 => 6
  | PackSize.Pack7 => 10

/-- Represents a purchase of notebook packs -/
structure Purchase where
  single : ℕ
  pack4 : ℕ
  pack7 : ℕ

/-- Calculates the total number of notebooks for a given purchase -/
def totalNotebooks (p : Purchase) : ℕ :=
  p.single * notebooks PackSize.Single +
  p.pack4 * notebooks PackSize.Pack4 +
  p.pack7 * notebooks PackSize.Pack7

/-- Calculates the total cost for a given purchase -/
def totalCost (p : Purchase) : ℕ :=
  p.single * cost PackSize.Single +
  p.pack4 * cost PackSize.Pack4 +
  p.pack7 * cost PackSize.Pack7

/-- Represents the budget constraint -/
def budget : ℕ := 17

/-- Theorem: The maximum number of notebooks that can be purchased within the budget is 11 -/
theorem max_notebooks_is_11 :
  ∀ p : Purchase, totalCost p ≤ budget → totalNotebooks p ≤ 11 ∧
  ∃ p' : Purchase, totalCost p' ≤ budget ∧ totalNotebooks p' = 11 :=
sorry

end NUMINAMATH_CALUDE_max_notebooks_is_11_l1148_114808


namespace NUMINAMATH_CALUDE_third_cube_edge_l1148_114803

theorem third_cube_edge (a b c x : ℝ) (ha : a = 3) (hb : b = 5) (hc : c = 6) :
  a^3 + b^3 + x^3 = c^3 → x = 4 := by sorry

end NUMINAMATH_CALUDE_third_cube_edge_l1148_114803


namespace NUMINAMATH_CALUDE_balance_theorem_l1148_114894

-- Define the weights of balls as real numbers
variable (R G O B : ℝ)

-- Define the balance relationships
axiom red_green : 4 * R = 8 * G
axiom orange_green : 3 * O = 6 * G
axiom green_blue : 8 * G = 6 * B

-- Theorem to prove
theorem balance_theorem : 3 * R + 2 * O + 4 * B = (46/3) * G := by
  sorry

end NUMINAMATH_CALUDE_balance_theorem_l1148_114894


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1148_114886

-- Define the repeating decimals as rational numbers
def a : ℚ := 234 / 999
def b : ℚ := 567 / 999
def c : ℚ := 891 / 999

-- Define the result of the operation
def result : ℚ := a - b + c

-- Theorem statement
theorem repeating_decimal_sum : result = 31 / 37 := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1148_114886


namespace NUMINAMATH_CALUDE_average_of_wxz_l1148_114864

variable (w x y z t : ℝ)

theorem average_of_wxz (h1 : 3/w + 3/x + 3/z = 3/(y + t))
                       (h2 : w*x*z = y + t)
                       (h3 : w*z + x*t + y*z = 3*w + 3*x + 3*z) :
  (w + x + z) / 3 = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_average_of_wxz_l1148_114864


namespace NUMINAMATH_CALUDE_partial_fraction_sum_zero_l1148_114887

theorem partial_fraction_sum_zero (x : ℝ) (A B C D E F : ℝ) :
  (1 : ℝ) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4) * (x + 5)) =
  A / x + B / (x + 1) + C / (x + 2) + D / (x + 3) + E / (x + 4) + F / (x + 5) →
  A + B + C + D + E + F = 0 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_sum_zero_l1148_114887


namespace NUMINAMATH_CALUDE_race_distance_difference_l1148_114816

theorem race_distance_difference (race_distance : ℝ) (a_time b_time : ℝ) : 
  race_distance = 80 →
  a_time = 20 →
  b_time = 25 →
  let a_speed := race_distance / a_time
  let b_speed := race_distance / b_time
  let b_distance := b_speed * a_time
  race_distance - b_distance = 16 := by sorry

end NUMINAMATH_CALUDE_race_distance_difference_l1148_114816


namespace NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1148_114876

theorem greatest_divisor_with_remainders (n : ℕ) : 
  (∃ k1 : ℕ, 1657 = n * k1 + 6) ∧ 
  (∃ k2 : ℕ, 2037 = n * k2 + 5) ∧ 
  (∀ m : ℕ, (∃ j1 : ℕ, 1657 = m * j1 + 6) ∧ (∃ j2 : ℕ, 2037 = m * j2 + 5) → m ≤ n) →
  n = 127 := by
sorry

end NUMINAMATH_CALUDE_greatest_divisor_with_remainders_l1148_114876


namespace NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l1148_114897

theorem shortest_side_of_right_triangle (a b c : ℝ) (ha : a = 5) (hb : b = 12) 
  (hright : a^2 + b^2 = c^2) : 
  min a (min b c) = 5 := by
  sorry

end NUMINAMATH_CALUDE_shortest_side_of_right_triangle_l1148_114897


namespace NUMINAMATH_CALUDE_circle_radii_formula_l1148_114880

/-- Given a triangle ABC with circumradius R and heights h_a, h_b, h_c,
    the radii t_a, t_b, t_c of circles tangent internally to the inscribed circle
    at vertices A, B, C and externally to each other satisfy the given formulas. -/
theorem circle_radii_formula (a b c R h_a h_b h_c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ R > 0 ∧ h_a > 0 ∧ h_b > 0 ∧ h_c > 0) :
  ∃ (t_a t_b t_c : ℝ),
    (t_a > 0 ∧ t_b > 0 ∧ t_c > 0) ∧
    (t_a = (R * h_a) / (a + h_a)) ∧
    (t_b = (R * h_b) / (b + h_b)) ∧
    (t_c = (R * h_c) / (c + h_c)) :=
by sorry

end NUMINAMATH_CALUDE_circle_radii_formula_l1148_114880


namespace NUMINAMATH_CALUDE_daves_rides_l1148_114830

theorem daves_rides (total_rides : ℕ) (second_day_rides : ℕ) (first_day_rides : ℕ) :
  total_rides = 7 ∧ second_day_rides = 3 ∧ total_rides = first_day_rides + second_day_rides →
  first_day_rides = 4 := by
sorry

end NUMINAMATH_CALUDE_daves_rides_l1148_114830


namespace NUMINAMATH_CALUDE_linear_system_solution_l1148_114898

/-- A system of linear equations with a parameter m -/
structure LinearSystem (m : ℝ) where
  eq1 : ∀ x y z : ℝ, x + m*y + 5*z = 0
  eq2 : ∀ x y z : ℝ, 4*x + m*y - 3*z = 0
  eq3 : ∀ x y z : ℝ, 3*x + 6*y - 4*z = 0

/-- The solution to the system exists and is nontrivial -/
def has_nontrivial_solution (m : ℝ) : Prop :=
  ∃ x y z : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧
    x + m*y + 5*z = 0 ∧
    4*x + m*y - 3*z = 0 ∧
    3*x + 6*y - 4*z = 0

theorem linear_system_solution :
  ∃ m : ℝ, has_nontrivial_solution m ∧ m = 11.5 ∧
    ∀ x y z : ℝ, x ≠ 0 → y ≠ 0 → z ≠ 0 →
      x + m*y + 5*z = 0 →
      4*x + m*y - 3*z = 0 →
      3*x + 6*y - 4*z = 0 →
      x*z / (y^2) = -108/169 := by
  sorry

end NUMINAMATH_CALUDE_linear_system_solution_l1148_114898


namespace NUMINAMATH_CALUDE_inequality_proof_l1148_114850

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b) * (b + c) * (c + a) ≥ 8 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1148_114850


namespace NUMINAMATH_CALUDE_infinite_solutions_for_continuous_function_l1148_114872

theorem infinite_solutions_for_continuous_function 
  (f : ℝ → ℝ) 
  (h_cont : Continuous f) 
  (h_dom : ∀ x, x ≥ 1 → f x > 0) 
  (h_sol : ∀ a > 0, ∃ x ≥ 1, f x = a * x) : 
  ∀ a > 0, Set.Infinite {x | x ≥ 1 ∧ f x = a * x} :=
sorry

end NUMINAMATH_CALUDE_infinite_solutions_for_continuous_function_l1148_114872


namespace NUMINAMATH_CALUDE_starters_count_l1148_114848

-- Define the total number of players
def total_players : ℕ := 15

-- Define the number of quadruplets
def num_quadruplets : ℕ := 4

-- Define the number of starters to choose
def num_starters : ℕ := 6

-- Define the number of quadruplets that must be in the starting lineup
def quadruplets_in_lineup : ℕ := 3

-- Define the function to calculate the number of ways to choose the starting lineup
def choose_starters : ℕ :=
  (Nat.choose num_quadruplets quadruplets_in_lineup) *
  (Nat.choose (total_players - num_quadruplets) (num_starters - quadruplets_in_lineup))

-- Theorem statement
theorem starters_count : choose_starters = 660 := by
  sorry

end NUMINAMATH_CALUDE_starters_count_l1148_114848


namespace NUMINAMATH_CALUDE_geometric_sequence_a2_l1148_114863

def geometric_sequence (a : ℕ → ℚ) : Prop :=
  ∃ q : ℚ, ∀ n : ℕ, a (n + 1) = a n * q

theorem geometric_sequence_a2 (a : ℕ → ℚ) :
  geometric_sequence a →
  a 1 = 1/4 →
  a 3 * a 5 = 4 * (a 4 - 1) →
  a 2 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_a2_l1148_114863


namespace NUMINAMATH_CALUDE_marble_probability_l1148_114847

theorem marble_probability (total : ℕ) (p_white p_green : ℚ) : 
  total = 120 → 
  p_white = 1/4 → 
  p_green = 1/3 → 
  ∃ (p_red_blue : ℚ), p_red_blue = 5/12 ∧ p_white + p_green + p_red_blue = 1 :=
by sorry

end NUMINAMATH_CALUDE_marble_probability_l1148_114847


namespace NUMINAMATH_CALUDE_fraction_equality_l1148_114843

theorem fraction_equality : (5 * 6 + 3) / 9 = 11 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1148_114843


namespace NUMINAMATH_CALUDE_inequality_equivalence_l1148_114812

theorem inequality_equivalence (x : ℝ) : 3 * x^2 - 5 * x > 9 ↔ x < -1 ∨ x > 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l1148_114812


namespace NUMINAMATH_CALUDE_age_of_seventh_person_l1148_114833

-- Define the ages and age differences
variable (A1 A2 A3 A4 A5 A6 A7 D1 D2 D3 D4 D5 : ℕ)

-- Define the conditions
axiom age_order : A1 < A2 ∧ A2 < A3 ∧ A3 < A4 ∧ A4 < A5 ∧ A5 < A6

axiom age_differences : 
  A2 = A1 + D1 ∧
  A3 = A2 + D2 ∧
  A4 = A3 + D3 ∧
  A5 = A4 + D4 ∧
  A6 = A5 + D5

axiom sum_of_six : A1 + A2 + A3 + A4 + A5 + A6 = 246

axiom sum_of_seven : A1 + A2 + A3 + A4 + A5 + A6 + A7 = 315

-- The theorem to prove
theorem age_of_seventh_person : A7 = 69 := by
  sorry

end NUMINAMATH_CALUDE_age_of_seventh_person_l1148_114833


namespace NUMINAMATH_CALUDE_maximize_profit_l1148_114859

/-- The production volume that maximizes profit -/
def optimal_production_volume : ℝ := 6

/-- Sales revenue as a function of production volume -/
def sales_revenue (x : ℝ) : ℝ := 17 * x^2

/-- Production cost as a function of production volume -/
def production_cost (x : ℝ) : ℝ := 2 * x^3 - x^2

/-- Profit as a function of production volume -/
def profit (x : ℝ) : ℝ := sales_revenue x - production_cost x

theorem maximize_profit (x : ℝ) (h : x > 0) :
  profit x ≤ profit optimal_production_volume := by
  sorry

end NUMINAMATH_CALUDE_maximize_profit_l1148_114859


namespace NUMINAMATH_CALUDE_product_xyz_equals_2898_l1148_114811

theorem product_xyz_equals_2898 (x y z : ℝ) 
  (eq1 : -3*x + 4*y - z = 28)
  (eq2 : 3*x - 2*y + z = 8)
  (eq3 : x + y - z = 2) :
  x * y * z = 2898 := by sorry

end NUMINAMATH_CALUDE_product_xyz_equals_2898_l1148_114811


namespace NUMINAMATH_CALUDE_fraction_ordering_l1148_114826

theorem fraction_ordering : (8 : ℚ) / 25 < 1 / 3 ∧ 1 / 3 < 10 / 31 ∧ 10 / 31 < 6 / 17 := by
  sorry

end NUMINAMATH_CALUDE_fraction_ordering_l1148_114826


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1148_114868

theorem unique_integer_solution :
  ∃! (x y z : ℤ), x^2 + y^2 + z^2 + 3 < x*y + 3*y + 2*z ∧ x = 1 ∧ y = 2 ∧ z = 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1148_114868


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l1148_114807

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * q

/-- The property that three terms form an arithmetic sequence -/
def ArithmeticSequence (x y z : ℝ) : Prop :=
  y - x = z - y

theorem geometric_sequence_ratio (a : ℕ → ℝ) :
  GeometricSequence a →
  ArithmeticSequence (3 * a 1) ((1/2) * a 3) (2 * a 2) →
  (a 2014 + a 2015) / (a 2012 + a 2013) = 9 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l1148_114807


namespace NUMINAMATH_CALUDE_project_completion_time_l1148_114819

theorem project_completion_time (m n : ℕ) (m_pos : m > 0) (n_pos : n > 0) :
  let total_people := m + n
  let days_for_total := m
  let days_for_n := m * total_people / n
  (∀ (person : ℕ), person ≤ total_people → person > 0 → 
    (1 : ℚ) / (total_people * days_for_total : ℚ) = 
    (1 : ℚ) / (person * (total_people * days_for_total / person) : ℚ)) →
  days_for_n * n = m * total_people :=
sorry

end NUMINAMATH_CALUDE_project_completion_time_l1148_114819


namespace NUMINAMATH_CALUDE_problem_solution_l1148_114849

theorem problem_solution (x y : ℝ) (h : x^2 * (y^2 + 1) = 1) :
  (xy < 1) ∧ (x^2 * y ≥ -1/2) ∧ (x^2 + x*y ≤ 5/4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1148_114849


namespace NUMINAMATH_CALUDE_multiples_of_four_median_l1148_114865

def first_seven_multiples_of_four : List ℕ := [4, 8, 12, 16, 20, 24, 28]

def a : ℚ := (first_seven_multiples_of_four.sum : ℚ) / 7

def b (n : ℕ) : ℚ := 2 * n

theorem multiples_of_four_median (n : ℕ) :
  a ^ 2 - (b n) ^ 2 = 0 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_multiples_of_four_median_l1148_114865


namespace NUMINAMATH_CALUDE_cubic_integer_values_l1148_114820

/-- A cubic polynomial function -/
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 - c * x - d

/-- Theorem: If a cubic polynomial takes integer values at -1, 0, 1, and 2, 
    then it takes integer values for all integer inputs -/
theorem cubic_integer_values 
  (a b c d : ℝ) 
  (h₁ : ∃ n₁ : ℤ, f a b c d (-1) = n₁) 
  (h₂ : ∃ n₂ : ℤ, f a b c d 0 = n₂)
  (h₃ : ∃ n₃ : ℤ, f a b c d 1 = n₃)
  (h₄ : ∃ n₄ : ℤ, f a b c d 2 = n₄) :
  ∀ x : ℤ, ∃ n : ℤ, f a b c d x = n :=
sorry

end NUMINAMATH_CALUDE_cubic_integer_values_l1148_114820


namespace NUMINAMATH_CALUDE_probability_of_two_heads_after_HTH_l1148_114853

/-- A fair coin flip sequence -/
inductive CoinFlip
| H : CoinFlip  -- Heads
| T : CoinFlip  -- Tails

/-- The stopping condition for the coin flipping process -/
def stoppingCondition (seq : List CoinFlip) : Bool :=
  match seq with
  | (CoinFlip.H :: CoinFlip.H :: _) => true
  | (CoinFlip.T :: CoinFlip.T :: _) => true
  | _ => false

/-- The probability of getting a specific sequence of coin flips -/
def probabilityOfSequence (seq : List CoinFlip) : ℚ :=
  (1 / 2) ^ seq.length

/-- The initial sequence of flips -/
def initialSequence : List CoinFlip := [CoinFlip.H, CoinFlip.T, CoinFlip.H]

/-- The theorem to be proved -/
theorem probability_of_two_heads_after_HTH :
  ∃ (p : ℚ), p = 1 / 64 ∧
  p = probabilityOfSequence initialSequence *
      (probabilityOfSequence [CoinFlip.T, CoinFlip.H, CoinFlip.H]) :=
sorry

end NUMINAMATH_CALUDE_probability_of_two_heads_after_HTH_l1148_114853


namespace NUMINAMATH_CALUDE_good_numbers_up_to_17_and_18_not_good_l1148_114867

/-- The number of positive divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- A number m is "good" if there exists a positive integer n such that m = n / d(n) -/
def is_good (m : ℕ+) : Prop :=
  ∃ n : ℕ+, (n : ℚ) / d n = m

theorem good_numbers_up_to_17_and_18_not_good :
  (∀ m : ℕ+, m ≤ 17 → is_good m) ∧ ¬ is_good 18 := by sorry

end NUMINAMATH_CALUDE_good_numbers_up_to_17_and_18_not_good_l1148_114867


namespace NUMINAMATH_CALUDE_correct_junior_teachers_in_sample_l1148_114877

/-- Represents the number of teachers in each category -/
structure TeacherPopulation where
  total : Nat
  junior : Nat

/-- Represents a stratified sample -/
structure StratifiedSample where
  populationSize : Nat
  sampleSize : Nat
  juniorInPopulation : Nat
  juniorInSample : Nat

/-- Calculates the number of junior teachers in a stratified sample -/
def calculateJuniorTeachersInSample (pop : TeacherPopulation) (sampleSize : Nat) : Nat :=
  (pop.junior * sampleSize) / pop.total

/-- Theorem stating that the calculated number of junior teachers in the sample is correct -/
theorem correct_junior_teachers_in_sample (pop : TeacherPopulation) (sample : StratifiedSample) 
    (h1 : pop.total = 200)
    (h2 : pop.junior = 80)
    (h3 : sample.populationSize = pop.total)
    (h4 : sample.sampleSize = 50)
    (h5 : sample.juniorInPopulation = pop.junior)
    (h6 : sample.juniorInSample = calculateJuniorTeachersInSample pop sample.sampleSize) :
  sample.juniorInSample = 20 := by
  sorry

#check correct_junior_teachers_in_sample

end NUMINAMATH_CALUDE_correct_junior_teachers_in_sample_l1148_114877


namespace NUMINAMATH_CALUDE_rhombus_area_l1148_114801

/-- The area of a rhombus with side length 4 cm and one angle of 45° is 8√2 square centimeters. -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π/4) :
  let area := s * s * Real.sin θ
  area = 8 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_rhombus_area_l1148_114801


namespace NUMINAMATH_CALUDE_batsman_average_increase_l1148_114835

/-- Represents a batsman's score statistics -/
structure BatsmanStats where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an inning -/
def newAverage (stats : BatsmanStats) (runsScored : ℕ) : ℚ :=
  (stats.totalRuns + runsScored : ℚ) / (stats.innings + 1)

/-- Theorem: If a batsman's average increases by 3 after scoring 84 in the 17th inning, the new average is 36 -/
theorem batsman_average_increase (stats : BatsmanStats) 
    (h1 : stats.innings = 16)
    (h2 : newAverage stats 84 = stats.average + 3) :
    newAverage stats 84 = 36 := by
  sorry


end NUMINAMATH_CALUDE_batsman_average_increase_l1148_114835


namespace NUMINAMATH_CALUDE_repeating_decimal_sum_l1148_114846

theorem repeating_decimal_sum : 
  (1 / 3 : ℚ) + (4 / 99 : ℚ) + (5 / 999 : ℚ) = (14 / 37 : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_sum_l1148_114846


namespace NUMINAMATH_CALUDE_age_difference_proof_l1148_114839

theorem age_difference_proof (patrick_age michael_age monica_age : ℕ) : 
  patrick_age * 5 = michael_age * 3 →
  michael_age * 5 = monica_age * 3 →
  patrick_age + michael_age + monica_age = 245 →
  monica_age - patrick_age = 80 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l1148_114839


namespace NUMINAMATH_CALUDE_complement_of_A_l1148_114881

-- Define the universal set U as the set of real numbers
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x : ℝ | (x - 1) * (x - 4) ≤ 0}

-- Theorem statement
theorem complement_of_A : 
  Set.compl A = {x : ℝ | x < 1 ∨ x > 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_A_l1148_114881


namespace NUMINAMATH_CALUDE_largest_integer_with_remainder_l1148_114845

theorem largest_integer_with_remainder : ∃ n : ℕ, n < 100 ∧ n % 6 = 4 ∧ ∀ m : ℕ, m < 100 ∧ m % 6 = 4 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_with_remainder_l1148_114845


namespace NUMINAMATH_CALUDE_m_plus_e_equals_22_l1148_114821

def base_value (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

theorem m_plus_e_equals_22 (m e : Nat) :
  m > 0 →
  e < 10 →
  base_value [4, 1, e] m = 346 →
  base_value [4, 1, 6] m = base_value [1, 2, e, 1] 7 →
  m + e = 22 :=
by sorry

end NUMINAMATH_CALUDE_m_plus_e_equals_22_l1148_114821


namespace NUMINAMATH_CALUDE_circles_intersect_l1148_114855

-- Define the circles
def circle_C₁ (x y : ℝ) : Prop := x^2 + y^2 + 2*x + 8*y - 8 = 0
def circle_C₂ (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 4*y - 1 = 0

-- Define the centers and radii of the circles
def center_C₁ : ℝ × ℝ := (-1, -4)
def center_C₂ : ℝ × ℝ := (2, 2)
def radius_C₁ : ℝ := 5
def radius_C₂ : ℝ := 3

-- Theorem stating that the circles are intersecting
theorem circles_intersect : 
  abs (radius_C₁ - radius_C₂) < 
  Real.sqrt (((center_C₂.1 - center_C₁.1)^2 + (center_C₂.2 - center_C₁.2)^2)) ∧
  Real.sqrt (((center_C₂.1 - center_C₁.1)^2 + (center_C₂.2 - center_C₁.2)^2)) < 
  radius_C₁ + radius_C₂ :=
sorry

end NUMINAMATH_CALUDE_circles_intersect_l1148_114855


namespace NUMINAMATH_CALUDE_subtracted_value_l1148_114893

theorem subtracted_value (n : ℝ) (v : ℝ) (h1 : n = 1) (h2 : 3 * n - v = 2 * n) : v = 1 := by
  sorry

end NUMINAMATH_CALUDE_subtracted_value_l1148_114893


namespace NUMINAMATH_CALUDE_min_value_theorem_l1148_114824

theorem min_value_theorem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (sum_constraint : a + b + c = 5) :
  (9 / a) + (16 / b) + (25 / c^2) ≥ 50 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1148_114824


namespace NUMINAMATH_CALUDE_hyperbola_equation_l1148_114827

/-- Given a hyperbola with equation x²/a² - y²/b² = 1, where a > 0 and b > 0,
    if point P(3, 5/2) lies on the hyperbola and the radius of the incircle of
    triangle PF₁F₂ (where F₁ and F₂ are the left and right foci) is 1,
    then a = 2 and b = √5. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (F₁ F₂ : ℝ × ℝ),
    -- F₁ and F₂ are the foci of the hyperbola
    (F₁.1 < 0 ∧ F₂.1 > 0) ∧
    -- P(3, 5/2) lies on the hyperbola
    3^2 / a^2 - (5/2)^2 / b^2 = 1 ∧
    -- The radius of the incircle of triangle PF₁F₂ is 1
    (∃ (r : ℝ), r = 1 ∧
      r = (dist F₁ (3, 5/2) + dist F₂ (3, 5/2) + dist F₁ F₂) /
          (dist F₁ (3, 5/2) / r + dist F₂ (3, 5/2) / r + dist F₁ F₂ / r))) →
  a = 2 ∧ b = Real.sqrt 5 := by
sorry


end NUMINAMATH_CALUDE_hyperbola_equation_l1148_114827


namespace NUMINAMATH_CALUDE_system_solution_l1148_114844

theorem system_solution (x y : ℚ) : 
  (x + y = x^2 + 2*x*y + y^2 ∧ x - y = x^2 - 2*x*y + y^2) ↔ 
  ((x = 1/2 ∧ y = -1/2) ∨ (x = 0 ∧ y = 0) ∨ (x = 1 ∧ y = 0) ∨ (x = 1/2 ∧ y = 1/2)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1148_114844


namespace NUMINAMATH_CALUDE_binomial_150_150_equals_1_l1148_114841

theorem binomial_150_150_equals_1 : Nat.choose 150 150 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_150_150_equals_1_l1148_114841


namespace NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1148_114852

theorem min_value_theorem (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y = 1) :
  x^2 + (1/4) * y^2 ≥ 1/8 := by
sorry

theorem min_value_achieved (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : 2 * x + y = 1) :
  (x^2 + (1/4) * y^2 = 1/8) ↔ (x = 1/4 ∧ y = 1/2) := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_min_value_achieved_l1148_114852


namespace NUMINAMATH_CALUDE_triangle_sine_identity_l1148_114869

theorem triangle_sine_identity (A B C : Real) (h : A + B + C = Real.pi) :
  Real.sin (2 * A) + Real.sin (2 * B) + Real.sin (2 * C) = 4 * Real.sin A * Real.sin B * Real.sin C :=
by sorry

end NUMINAMATH_CALUDE_triangle_sine_identity_l1148_114869


namespace NUMINAMATH_CALUDE_initial_amount_proof_l1148_114874

/-- Proves that if an amount increases by 1/8 of itself every year and after two years
    it becomes 40500, then the initial amount was 32000. -/
theorem initial_amount_proof (A : ℚ) : 
  (A + A/8 + (A + A/8)/8 = 40500) → A = 32000 :=
by sorry

end NUMINAMATH_CALUDE_initial_amount_proof_l1148_114874


namespace NUMINAMATH_CALUDE_rotational_symmetry_180_l1148_114804

/-- Represents a 2D shape -/
structure Shape :=
  (points : Set (ℝ × ℝ))

/-- Represents a rotation of a shape -/
def rotate (s : Shape) (angle : ℝ) : Shape :=
  sorry

/-- Defines rotational symmetry for a shape -/
def is_rotationally_symmetric (s : Shape) (angle : ℝ) : Prop :=
  rotate s angle = s

/-- The original L-like shape -/
def original_shape : Shape :=
  sorry

/-- Theorem: The shape rotated 180 degrees is rotationally symmetric to the original shape -/
theorem rotational_symmetry_180 :
  is_rotationally_symmetric (rotate original_shape π) π :=
sorry

end NUMINAMATH_CALUDE_rotational_symmetry_180_l1148_114804


namespace NUMINAMATH_CALUDE_parallel_vectors_m_equals_one_l1148_114861

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_equals_one :
  let a : ℝ × ℝ := (-1, 3)
  let b : ℝ × ℝ := (m, m - 4)
  are_parallel a b → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_equals_one_l1148_114861


namespace NUMINAMATH_CALUDE_brendas_age_l1148_114815

theorem brendas_age (addison janet brenda : ℚ) 
  (h1 : addison = 4 * brenda) 
  (h2 : janet = brenda + 8) 
  (h3 : addison = janet) : 
  brenda = 8/3 := by
sorry

end NUMINAMATH_CALUDE_brendas_age_l1148_114815


namespace NUMINAMATH_CALUDE_midpoint_trajectory_equation_l1148_114834

/-- The equation of the trajectory of the midpoint M of line segment PQ, where P is on the parabola y = x^2 + 1 and Q is (0, 1) -/
theorem midpoint_trajectory_equation :
  ∀ (x y a b : ℝ),
  y = x^2 + 1 →  -- P (x, y) is on the parabola y = x^2 + 1
  a = x / 2 →    -- M (a, b) is the midpoint of PQ, so a = x/2
  b = (y + 1) / 2 →  -- and b = (y + 1)/2
  b = 2 * a^2 + 1 :=  -- The equation of the trajectory of M is y = 2x^2 + 1
by
  sorry

end NUMINAMATH_CALUDE_midpoint_trajectory_equation_l1148_114834


namespace NUMINAMATH_CALUDE_dime_exchange_theorem_l1148_114895

/-- Represents the number of dimes each person has at each stage -/
structure DimeState :=
  (a : ℤ) (b : ℤ) (c : ℤ)

/-- Represents the transactions between A, B, and C -/
def exchange (state : DimeState) : DimeState :=
  let state1 := DimeState.mk (state.a - state.b - state.c) (2 * state.b) (2 * state.c)
  let state2 := DimeState.mk (2 * state1.a) (state1.b - state1.a - state1.c) (2 * state1.c)
  DimeState.mk (2 * state2.a) (2 * state2.b) (state2.c - state2.a - state2.b)

theorem dime_exchange_theorem (initial : DimeState) :
  exchange initial = DimeState.mk 36 36 36 → initial.a = 36 :=
by sorry

end NUMINAMATH_CALUDE_dime_exchange_theorem_l1148_114895


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_m_negative_l1148_114862

/-- If the equation 2x^2 + (m+1)x + m = 0 has one positive root and one negative root, then m < 0 -/
theorem quadratic_roots_imply_m_negative (m : ℝ) : 
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ 2 * x^2 + (m + 1) * x + m = 0 ∧ 2 * y^2 + (m + 1) * y + m = 0) →
  m < 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_m_negative_l1148_114862


namespace NUMINAMATH_CALUDE_stickers_on_last_page_l1148_114805

def total_books : Nat := 10
def pages_per_book : Nat := 30
def initial_stickers_per_page : Nat := 5
def new_stickers_per_page : Nat := 8
def full_books_after_rearrange : Nat := 6
def full_pages_in_seventh_book : Nat := 25

theorem stickers_on_last_page :
  let total_stickers := total_books * pages_per_book * initial_stickers_per_page
  let stickers_in_full_books := full_books_after_rearrange * pages_per_book * new_stickers_per_page
  let remaining_stickers := total_stickers - stickers_in_full_books
  let stickers_in_full_pages_of_seventh_book := (remaining_stickers / new_stickers_per_page) * new_stickers_per_page
  remaining_stickers - stickers_in_full_pages_of_seventh_book = 4 := by
  sorry

end NUMINAMATH_CALUDE_stickers_on_last_page_l1148_114805


namespace NUMINAMATH_CALUDE_coin_value_difference_l1148_114842

def total_coins : ℕ := 5050

def penny_value : ℕ := 1
def dime_value : ℕ := 10

def total_value (num_pennies : ℕ) : ℕ :=
  num_pennies * penny_value + (total_coins - num_pennies) * dime_value

theorem coin_value_difference :
  ∃ (max_value min_value : ℕ),
    (∀ (num_pennies : ℕ), 1 ≤ num_pennies ∧ num_pennies ≤ total_coins - 1 →
      min_value ≤ total_value num_pennies ∧ total_value num_pennies ≤ max_value) ∧
    max_value - min_value = 45432 :=
sorry

end NUMINAMATH_CALUDE_coin_value_difference_l1148_114842


namespace NUMINAMATH_CALUDE_smallest_n_congruence_l1148_114891

theorem smallest_n_congruence (n : ℕ+) : 
  (5 * n.val ≡ 2015 [MOD 26]) ↔ n = 21 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_congruence_l1148_114891


namespace NUMINAMATH_CALUDE_robin_gum_packages_l1148_114840

/-- Given that Robin has some packages of gum, with 7 pieces in each package,
    6 extra pieces, and 41 pieces in total, prove that Robin has 5 packages. -/
theorem robin_gum_packages : ∀ (p : ℕ), 
  (7 * p + 6 = 41) → p = 5 := by
  sorry

end NUMINAMATH_CALUDE_robin_gum_packages_l1148_114840


namespace NUMINAMATH_CALUDE_adjacent_complementary_angles_are_complementary_l1148_114885

/-- Two angles are complementary if their sum is 90 degrees -/
def Complementary (α β : ℝ) : Prop := α + β = 90

/-- Two angles are adjacent if they share a common vertex and a common side,
    but do not overlap -/
def Adjacent (α β : ℝ) : Prop := True  -- We simplify this for the purpose of the statement

theorem adjacent_complementary_angles_are_complementary
  (α β : ℝ) (h1 : Adjacent α β) (h2 : Complementary α β) :
  Complementary α β :=
sorry

end NUMINAMATH_CALUDE_adjacent_complementary_angles_are_complementary_l1148_114885


namespace NUMINAMATH_CALUDE_marks_increase_ratio_l1148_114883

/-- 
Given a class of 80 pupils, if one pupil's marks are increased by 40,
the ratio of the increase in average marks to the original average marks
is 1/(2A), where A is the original average marks.
-/
theorem marks_increase_ratio (T : ℝ) (A : ℝ) : 
  A = T / 80 → (T + 40) / 80 - T / 80 = 1 / 2 → 
  ((T + 40) / 80 - T / 80) / A = 1 / (2 * A) := by
  sorry

end NUMINAMATH_CALUDE_marks_increase_ratio_l1148_114883


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1148_114822

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: In an arithmetic sequence where a₂ = 4 and a₄ = 2, a₆ = 0 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a) 
  (h_a2 : a 2 = 4) 
  (h_a4 : a 4 = 2) : 
  a 6 = 0 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1148_114822


namespace NUMINAMATH_CALUDE_percentage_of_absent_students_l1148_114866

theorem percentage_of_absent_students (total : ℕ) (present : ℕ) : 
  total = 50 → present = 44 → (total - present) * 100 / total = 12 := by
sorry

end NUMINAMATH_CALUDE_percentage_of_absent_students_l1148_114866


namespace NUMINAMATH_CALUDE_joan_gave_43_seashells_l1148_114878

/-- The number of seashells Joan initially found on the beach. -/
def initial_seashells : ℕ := 70

/-- The number of seashells Joan has left after giving some to Sam. -/
def remaining_seashells : ℕ := 27

/-- The number of seashells Joan gave to Sam. -/
def seashells_given_to_sam : ℕ := initial_seashells - remaining_seashells

/-- Theorem stating that Joan gave 43 seashells to Sam. -/
theorem joan_gave_43_seashells : seashells_given_to_sam = 43 := by
  sorry

end NUMINAMATH_CALUDE_joan_gave_43_seashells_l1148_114878


namespace NUMINAMATH_CALUDE_cookies_per_pack_l1148_114806

/-- Given information about Candy's cookie distribution --/
structure CookieDistribution where
  trays : ℕ
  cookies_per_tray : ℕ
  packs : ℕ
  trays_eq : trays = 4
  cookies_per_tray_eq : cookies_per_tray = 24
  packs_eq : packs = 8

/-- Theorem: The number of cookies in each pack is 12 --/
theorem cookies_per_pack (cd : CookieDistribution) : 
  (cd.trays * cd.cookies_per_tray) / cd.packs = 12 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_pack_l1148_114806


namespace NUMINAMATH_CALUDE_students_taking_both_french_and_german_l1148_114837

theorem students_taking_both_french_and_german 
  (total : ℕ) 
  (french : ℕ) 
  (german : ℕ) 
  (neither : ℕ) 
  (h1 : total = 78) 
  (h2 : french = 41) 
  (h3 : german = 22) 
  (h4 : neither = 24) :
  french + german - (total - neither) = 9 :=
by sorry

end NUMINAMATH_CALUDE_students_taking_both_french_and_german_l1148_114837


namespace NUMINAMATH_CALUDE_ellipse_equation_l1148_114888

/-- An ellipse with parametric equations x = 5cos(α) and y = 3sin(α) has the general equation x²/25 + y²/9 = 1 -/
theorem ellipse_equation (α : ℝ) (x y : ℝ) (h1 : x = 5 * Real.cos α) (h2 : y = 3 * Real.sin α) : 
  x^2 / 25 + y^2 / 9 = 1 := by
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1148_114888


namespace NUMINAMATH_CALUDE_frog_count_l1148_114870

theorem frog_count (total_eyes : ℕ) (eyes_per_frog : ℕ) (h1 : total_eyes > 0) (h2 : eyes_per_frog > 0) :
  total_eyes / eyes_per_frog = 4 →
  total_eyes = 8 ∧ eyes_per_frog = 2 :=
by sorry

end NUMINAMATH_CALUDE_frog_count_l1148_114870


namespace NUMINAMATH_CALUDE_arrangement_count_l1148_114818

/-- Represents the number of people in the arrangement. -/
def total_people : ℕ := 6

/-- Represents the number of people who have a specific position requirement (Jia, Bing, and Yi). -/
def specific_people : ℕ := 3

/-- Calculates the number of arrangements where one person stands between two others in a line of n people. -/
def arrangements (n : ℕ) : ℕ :=
  (Nat.factorial (n - specific_people + 1)) * 2

/-- Theorem stating that the number of arrangements for 6 people with the given condition is 48. -/
theorem arrangement_count :
  arrangements total_people = 48 := by
  sorry

end NUMINAMATH_CALUDE_arrangement_count_l1148_114818


namespace NUMINAMATH_CALUDE_wanda_walking_distance_l1148_114817

/-- Represents the distance Wanda walks in miles -/
def distance_to_school : ℝ := 0.5

/-- Represents the number of round trips Wanda makes per day -/
def round_trips_per_day : ℕ := 2

/-- Represents the number of days Wanda walks to school per week -/
def school_days_per_week : ℕ := 5

/-- Represents the number of weeks we're considering -/
def weeks : ℕ := 4

/-- Theorem stating that Wanda walks 40 miles after 4 weeks -/
theorem wanda_walking_distance : 
  2 * distance_to_school * round_trips_per_day * school_days_per_week * weeks = 40 := by
  sorry


end NUMINAMATH_CALUDE_wanda_walking_distance_l1148_114817


namespace NUMINAMATH_CALUDE_restaurant_friends_l1148_114828

theorem restaurant_friends (pre_cooked wings_cooked wings_per_person : ℕ) :
  pre_cooked = 2 →
  wings_cooked = 25 →
  wings_per_person = 3 →
  (pre_cooked + wings_cooked) / wings_per_person = 9 :=
by
  sorry

end NUMINAMATH_CALUDE_restaurant_friends_l1148_114828
