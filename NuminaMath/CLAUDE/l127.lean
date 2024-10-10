import Mathlib

namespace symmetry_lines_sum_l127_12731

/-- Two parabolas intersecting at two points -/
structure IntersectingParabolas where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  h1 : -(3 - a)^2 + b = 6
  h2 : (3 - c)^2 + d = 6
  h3 : -(9 - a)^2 + b = 0
  h4 : (9 - c)^2 + d = 0

/-- The sum of x-axis symmetry lines of two intersecting parabolas equals 12 -/
theorem symmetry_lines_sum (p : IntersectingParabolas) : p.a + p.c = 12 := by
  sorry

end symmetry_lines_sum_l127_12731


namespace rent_increase_percentage_l127_12700

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.35 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 202.5 := by
  sorry

end rent_increase_percentage_l127_12700


namespace max_donated_cookies_l127_12725

def distribute_cookies (total : Nat) (employees : Nat) : Nat :=
  total - (employees * (total / employees))

theorem max_donated_cookies :
  distribute_cookies 120 7 = 1 := by
  sorry

end max_donated_cookies_l127_12725


namespace photo_gallery_total_l127_12705

theorem photo_gallery_total (initial_photos : ℕ) 
  (h1 : initial_photos = 1200) 
  (first_day : ℕ) 
  (h2 : first_day = initial_photos * 3 / 5) 
  (second_day : ℕ) 
  (h3 : second_day = first_day + 230) : 
  initial_photos + first_day + second_day = 2870 := by
  sorry

end photo_gallery_total_l127_12705


namespace problem_statement_l127_12752

theorem problem_statement (x : ℝ) (h : x = -1) : 
  2 * (-x^2 + 3*x^3) - (2*x^3 - 2*x^2) + 8 = 4 := by
  sorry

end problem_statement_l127_12752


namespace plane_relations_l127_12786

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations between planes and lines
variable (in_plane : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (perpendicular_planes : Plane → Plane → Prop)
variable (parallel_planes : Plane → Plane → Prop)

-- State the theorem
theorem plane_relations (a b : Plane) (h : a ≠ b) :
  (∀ (l : Line), in_plane l a → 
    (∀ (m : Line), in_plane m b → perpendicular l m) → 
    perpendicular_planes a b) ∧
  (∀ (l : Line), in_plane l a → 
    parallel_line_plane l b → 
    parallel_planes a b) ∧
  (parallel_planes a b → 
    ∀ (l : Line), in_plane l a → 
    parallel_line_plane l b) :=
by sorry

end plane_relations_l127_12786


namespace min_value_theorem_l127_12741

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + 2*y = 4) :
  (2/x + 1/y) ≥ 2 ∧ ∃ x y, x > 0 ∧ y > 0 ∧ x + 2*y = 4 ∧ 2/x + 1/y = 2 := by
  sorry

end min_value_theorem_l127_12741


namespace cubic_parabola_collinearity_l127_12776

/-- Represents a point on a cubic parabola -/
structure CubicPoint where
  x : ℝ
  y : ℝ

/-- Represents a cubic parabola y = x^3 + a₁x^2 + a₂x + a₃ -/
structure CubicParabola where
  a₁ : ℝ
  a₂ : ℝ
  a₃ : ℝ

/-- Check if a point lies on the cubic parabola -/
def onCubicParabola (p : CubicPoint) (c : CubicParabola) : Prop :=
  p.y = p.x^3 + c.a₁ * p.x^2 + c.a₂ * p.x + c.a₃

/-- Check if three points are collinear -/
def areCollinear (p q r : CubicPoint) : Prop :=
  (q.y - p.y) * (r.x - p.x) = (r.y - p.y) * (q.x - p.x)

/-- Main theorem: Given a cubic parabola and three points on it with x-coordinates summing to -a₁, the points are collinear -/
theorem cubic_parabola_collinearity (c : CubicParabola) (p q r : CubicPoint)
    (h_p : onCubicParabola p c)
    (h_q : onCubicParabola q c)
    (h_r : onCubicParabola r c)
    (h_sum : p.x + q.x + r.x = -c.a₁) :
    areCollinear p q r := by
  sorry

end cubic_parabola_collinearity_l127_12776


namespace sufficient_but_not_necessary_l127_12749

theorem sufficient_but_not_necessary : 
  (∃ x : ℝ, x < 2 ∧ ¬(1 < x ∧ x < 2)) ∧ 
  (∀ x : ℝ, 1 < x ∧ x < 2 → x < 2) :=
by sorry

end sufficient_but_not_necessary_l127_12749


namespace leftover_value_l127_12772

/-- Represents the number of coins in a roll --/
structure RollSize where
  quarters : Nat
  dimes : Nat

/-- Represents a person's coin collection --/
structure CoinCollection where
  quarters : Nat
  dimes : Nat

/-- Calculates the dollar value of a given number of quarters and dimes --/
def dollarValue (quarters dimes : Nat) : ℚ :=
  (quarters : ℚ) * (1 / 4) + (dimes : ℚ) * (1 / 10)

/-- Theorem stating the dollar value of leftover coins --/
theorem leftover_value (roll_size : RollSize) (ana_coins ben_coins : CoinCollection) :
  roll_size.quarters = 30 →
  roll_size.dimes = 40 →
  ana_coins.quarters = 95 →
  ana_coins.dimes = 183 →
  ben_coins.quarters = 104 →
  ben_coins.dimes = 219 →
  dollarValue 
    ((ana_coins.quarters + ben_coins.quarters) % roll_size.quarters)
    ((ana_coins.dimes + ben_coins.dimes) % roll_size.dimes) = 695 / 100 := by
  sorry

#eval dollarValue 19 22

end leftover_value_l127_12772


namespace repeating_decimal_equals_fraction_l127_12758

/-- The repeating decimal 0.8̄23 as a rational number -/
def repeating_decimal : ℚ := 0.8 + 23 / 99

/-- The expected fraction representation of 0.8̄23 -/
def expected_fraction : ℚ := 511 / 495

/-- Theorem stating that the repeating decimal 0.8̄23 is equal to 511/495 -/
theorem repeating_decimal_equals_fraction : repeating_decimal = expected_fraction := by
  sorry

end repeating_decimal_equals_fraction_l127_12758


namespace arithmetic_sequence_1010th_term_l127_12717

/-- An arithmetic sequence with the given first four terms -/
def arithmetic_sequence (p r : ℚ) : ℕ → ℚ
| 0 => p / 2
| 1 => 18
| 2 => 2 * p - r
| 3 => 2 * p + r
| n + 4 => arithmetic_sequence p r 3 + (n + 1) * (arithmetic_sequence p r 3 - arithmetic_sequence p r 2)

/-- The 1010th term of the sequence is 72774/11 -/
theorem arithmetic_sequence_1010th_term (p r : ℚ) :
  arithmetic_sequence p r 1009 = 72774 / 11 :=
by sorry

end arithmetic_sequence_1010th_term_l127_12717


namespace rational_identity_product_l127_12783

theorem rational_identity_product (M₁ M₂ : ℚ) : 
  (∀ x : ℚ, x ≠ 2 ∧ x ≠ 3 → (45 * x - 55) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) →
  M₁ * M₂ = 200 := by
sorry

end rational_identity_product_l127_12783


namespace crayon_difference_l127_12755

theorem crayon_difference (willy_crayons lucy_crayons : ℕ) 
  (hw : willy_crayons = 5092) 
  (hl : lucy_crayons = 3971) : 
  willy_crayons - lucy_crayons = 1121 := by
  sorry

end crayon_difference_l127_12755


namespace chord_intersection_tangent_circle_l127_12757

/-- Given a point A and a circle S with center O and radius R, 
    prove that the line through A intersecting S in a chord PQ of length d 
    is tangent to a circle with center O and radius sqrt(R^2 - d^2/4) -/
theorem chord_intersection_tangent_circle 
  (A O : ℝ × ℝ) (R d : ℝ) (S : Set (ℝ × ℝ)) :
  let circle_S := {p : ℝ × ℝ | dist p O = R}
  let chord_length (l : Set (ℝ × ℝ)) := ∃ P Q : ℝ × ℝ, P ∈ l ∩ S ∧ Q ∈ l ∩ S ∧ dist P Q = d
  let tangent_circle := {p : ℝ × ℝ | dist p O = Real.sqrt (R^2 - d^2/4)}
  ∀ l : Set (ℝ × ℝ), A ∈ l → S = circle_S → chord_length l → 
    ∃ p : ℝ × ℝ, p ∈ l ∩ tangent_circle :=
by sorry

end chord_intersection_tangent_circle_l127_12757


namespace sports_meet_participation_l127_12754

/-- The number of students participating in both track and field and ball games -/
def students_in_track_and_ball (total : ℕ) (swimming : ℕ) (track : ℕ) (ball : ℕ)
  (swimming_and_track : ℕ) (swimming_and_ball : ℕ) : ℕ :=
  swimming + track + ball - swimming_and_track - swimming_and_ball - total

theorem sports_meet_participation :
  students_in_track_and_ball 28 15 8 14 3 3 = 6 := by
  sorry

end sports_meet_participation_l127_12754


namespace valid_parameterization_l127_12796

/-- Represents a 2D vector -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- Represents a vector parameterization of a line -/
structure VectorParam where
  point : Vector2D
  direction : Vector2D

def isOnLine (v : Vector2D) : Prop :=
  v.y = 3 * v.x + 5

def isParallel (v : Vector2D) : Prop :=
  ∃ k : ℝ, v.x = k * 1 ∧ v.y = k * 3

theorem valid_parameterization (param : VectorParam) :
  (isOnLine param.point ∧ isParallel param.direction) ↔
  ∀ t : ℝ, isOnLine (Vector2D.mk
    (param.point.x + t * param.direction.x)
    (param.point.y + t * param.direction.y)) :=
sorry

end valid_parameterization_l127_12796


namespace arithmetic_progression_problem_l127_12746

theorem arithmetic_progression_problem (a d : ℝ) : 
  2 * (a - d) * a * (a + d + 7) = 1000 ∧ 
  a^2 = 2 * (a - d) * (a + d + 7) →
  d = 8 ∨ d = -8 :=
sorry

end arithmetic_progression_problem_l127_12746


namespace sum_interior_angles_pentagon_l127_12727

/-- The sum of interior angles of a polygon with n sides is (n - 2) * 180 degrees. -/
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

/-- A pentagon is a polygon with 5 sides. -/
def pentagon_sides : ℕ := 5

/-- The sum of the interior angles of a pentagon is 540 degrees. -/
theorem sum_interior_angles_pentagon : 
  sum_interior_angles pentagon_sides = 540 := by
  sorry

end sum_interior_angles_pentagon_l127_12727


namespace cat_whiskers_problem_l127_12747

/-- Represents the number of whiskers for each cat -/
structure CatWhiskers where
  juniper : ℕ
  puffy : ℕ
  scruffy : ℕ
  buffy : ℕ
  whisper : ℕ
  bella : ℕ
  max : ℕ
  felix : ℕ

/-- Theorem stating the conditions and the result to be proved -/
theorem cat_whiskers_problem (c : CatWhiskers) : 
  c.juniper = 12 ∧
  c.puffy = 3 * c.juniper ∧
  c.scruffy = 2 * c.puffy ∧
  c.buffy = (c.puffy + c.scruffy + c.juniper) / 3 ∧
  c.whisper = 2 * c.puffy ∧
  c.whisper = c.scruffy / 3 ∧
  c.bella = c.juniper + c.puffy - 4 ∧
  c.max = c.scruffy + c.buffy ∧
  c.felix = min c.juniper (min c.puffy (min c.scruffy (min c.buffy (min c.whisper (min c.bella c.max)))))
  →
  c.max = 112 := by
sorry

end cat_whiskers_problem_l127_12747


namespace exactly_21_numbers_reach_one_in_8_steps_l127_12787

def operation (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else n + 1

def reachesOneIn (steps : ℕ) (n : ℕ) : Prop :=
  ∃ (sequence : Fin (steps + 1) → ℕ),
    sequence 0 = n ∧
    sequence steps = 1 ∧
    ∀ i : Fin steps, sequence (i + 1) = operation (sequence i)

theorem exactly_21_numbers_reach_one_in_8_steps :
  ∃! (s : Finset ℕ), s.card = 21 ∧ ∀ n, n ∈ s ↔ reachesOneIn 8 n :=
sorry

end exactly_21_numbers_reach_one_in_8_steps_l127_12787


namespace central_angle_values_l127_12759

/-- A circular sector with given perimeter and area -/
structure CircularSector where
  perimeter : ℝ
  area : ℝ

/-- The central angle of a circular sector in radians -/
def central_angle (s : CircularSector) : Set ℝ :=
  {θ : ℝ | ∃ r : ℝ, 
    s.area = 1/2 * r^2 * θ ∧ 
    s.perimeter = 2 * r + r * θ}

/-- Theorem: For a circular sector with perimeter 3 cm and area 1/2 cm², 
    the central angle is either 1 or 4 radians -/
theorem central_angle_values (s : CircularSector) 
  (h_perimeter : s.perimeter = 3)
  (h_area : s.area = 1/2) : 
  central_angle s = {1, 4} := by
  sorry

end central_angle_values_l127_12759


namespace f_is_even_l127_12790

def f (x : ℝ) : ℝ := -3 * x^4

theorem f_is_even : ∀ x : ℝ, f (-x) = f x := by sorry

end f_is_even_l127_12790


namespace third_shift_participation_rate_l127_12773

-- Define the total number of employees in each shift
def first_shift : ℕ := 60
def second_shift : ℕ := 50
def third_shift : ℕ := 40

-- Define the participation rates for the first two shifts
def first_shift_rate : ℚ := 1/5
def second_shift_rate : ℚ := 2/5

-- Define the total participation rate
def total_participation_rate : ℚ := 6/25

-- Theorem statement
theorem third_shift_participation_rate :
  let total_employees := first_shift + second_shift + third_shift
  let total_participants := total_employees * total_participation_rate
  let first_shift_participants := first_shift * first_shift_rate
  let second_shift_participants := second_shift * second_shift_rate
  let third_shift_participants := total_participants - first_shift_participants - second_shift_participants
  third_shift_participants / third_shift = 1/10 := by
sorry

end third_shift_participation_rate_l127_12773


namespace sum_of_three_numbers_l127_12740

theorem sum_of_three_numbers (a b c : ℝ) 
  (h1 : a^2 + b^2 + c^2 = 156) 
  (h2 : a*b + b*c + c*a = 50) : 
  a + b + c = 16 := by
sorry

end sum_of_three_numbers_l127_12740


namespace sum_of_squares_of_roots_l127_12782

theorem sum_of_squares_of_roots (p q : ℝ) (a b : ℝ) 
  (h : ∀ x, 3 * x^2 - 2 * p * x + q = 0 ↔ x = a ∨ x = b) :
  a^2 + b^2 = 4 * p^2 - 6 * q := by
  sorry

end sum_of_squares_of_roots_l127_12782


namespace polygon_diagonals_theorem_l127_12751

theorem polygon_diagonals_theorem (n : ℕ) :
  n ≥ 3 →
  (n - 2 = 6) →
  n = 8 := by
sorry

end polygon_diagonals_theorem_l127_12751


namespace percent_cat_owners_l127_12722

def total_students : ℕ := 500
def cat_owners : ℕ := 90

theorem percent_cat_owners : (cat_owners : ℚ) / total_students * 100 = 18 := by
  sorry

end percent_cat_owners_l127_12722


namespace definite_integral_equals_twenty_minus_six_pi_l127_12750

theorem definite_integral_equals_twenty_minus_six_pi :
  let f : ℝ → ℝ := λ x => x^4 / ((16 - x^2) * Real.sqrt (16 - x^2))
  let a : ℝ := 0
  let b : ℝ := 2 * Real.sqrt 2
  ∫ x in a..b, f x = 20 - 6 * Real.pi := by sorry

end definite_integral_equals_twenty_minus_six_pi_l127_12750


namespace more_difficult_than_easy_l127_12788

/-- Represents the number of problems solved by a specific number of people -/
structure ProblemCounts where
  total : ℕ
  solvedByOne : ℕ
  solvedByTwo : ℕ
  solvedByThree : ℕ

/-- The total number of problems solved by each person -/
def problemsPerPerson : ℕ := 60

theorem more_difficult_than_easy (p : ProblemCounts) :
  p.total = 100 →
  p.solvedByOne + p.solvedByTwo + p.solvedByThree = p.total →
  p.solvedByOne + 3 * p.solvedByThree + 2 * p.solvedByTwo = 3 * problemsPerPerson →
  p.solvedByOne = p.solvedByThree + 20 :=
by
  sorry

#check more_difficult_than_easy

end more_difficult_than_easy_l127_12788


namespace ratio_sum_theorem_l127_12793

theorem ratio_sum_theorem (a b : ℕ) (h1 : a * 4 = b * 3) (h2 : a = 180) : a + b = 420 := by
  sorry

end ratio_sum_theorem_l127_12793


namespace sum_squares_and_products_ge_ten_l127_12781

theorem sum_squares_and_products_ge_ten (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (prod_eq_one : a * b * c * d = 1) :
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end sum_squares_and_products_ge_ten_l127_12781


namespace triangle_formation_l127_12762

def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_formation :
  ¬(can_form_triangle 3 6 3) ∧
  (can_form_triangle 3 4 5) ∧
  (can_form_triangle (Real.sqrt 3) 1 2) ∧
  (can_form_triangle 1.5 2.5 3) :=
by sorry

end triangle_formation_l127_12762


namespace strictly_increasing_function_l127_12710

theorem strictly_increasing_function
  (a b c d : ℝ)
  (h1 : a > c)
  (h2 : c > d)
  (h3 : d > b)
  (h4 : b > 1)
  (h5 : a * b > c * d) :
  let f : ℝ → ℝ := λ x ↦ a^x + b^x - c^x - d^x
  ∀ x ≥ 0, (deriv f) x > 0 :=
by sorry

end strictly_increasing_function_l127_12710


namespace evaluate_expression_l127_12704

theorem evaluate_expression : ((4^4 - 4*(4-2)^4)^4) = 136048896 := by sorry

end evaluate_expression_l127_12704


namespace larger_box_capacity_l127_12738

/-- Represents the capacity of a box in terms of volume and paperclips -/
structure Box where
  volume : ℝ
  paperclipCapacity : ℕ

/-- The maximum number of paperclips a box can hold -/
def maxPaperclips (b : Box) : ℕ := b.paperclipCapacity

theorem larger_box_capacity (smallBox largeBox : Box)
  (h1 : smallBox.volume = 20)
  (h2 : smallBox.paperclipCapacity = 80)
  (h3 : largeBox.volume = 100)
  (h4 : largeBox.paperclipCapacity = 380) :
  maxPaperclips largeBox = 380 := by
  sorry

end larger_box_capacity_l127_12738


namespace least_N_for_P_condition_l127_12733

/-- The probability that at least 3/5 of N green balls are on the same side of a red ball
    when arranged randomly in a line -/
def P (N : ℕ) : ℚ :=
  (⌊(2 * N : ℚ) / 5⌋ + 1 + (N - ⌈(3 * N : ℚ) / 5⌉ + 1)) / (N + 1)

theorem least_N_for_P_condition :
  ∀ N : ℕ, N % 5 = 0 → N > 0 →
    (∀ k : ℕ, k % 5 = 0 → k > 0 → k < N → P k ≥ 321/400) ∧
    P N < 321/400 →
    N = 480 :=
sorry

end least_N_for_P_condition_l127_12733


namespace luke_fish_catching_l127_12760

theorem luke_fish_catching (days : ℕ) (fillets_per_fish : ℕ) (total_fillets : ℕ) :
  days = 30 →
  fillets_per_fish = 2 →
  total_fillets = 120 →
  total_fillets / (days * fillets_per_fish) = 2 :=
by sorry

end luke_fish_catching_l127_12760


namespace triangular_pyramid_base_balls_l127_12797

/-- The number of balls in a triangular pyramid with n rows -/
def triangular_pyramid_balls (n : ℕ) : ℕ := n * (n + 1) * (n + 2) / 6

/-- The number of balls in the base of a triangular pyramid with n rows -/
def base_balls (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Theorem: In a regular triangular pyramid with 165 tightly packed identical balls,
    the number of balls in the base is 45 -/
theorem triangular_pyramid_base_balls :
  ∃ n : ℕ, triangular_pyramid_balls n = 165 ∧ base_balls n = 45 :=
by
  sorry

end triangular_pyramid_base_balls_l127_12797


namespace g_inv_composition_l127_12780

-- Define the function g
def g : Fin 5 → Fin 5
| 1 => 4
| 2 => 3
| 3 => 1
| 4 => 5
| 5 => 2

-- Define the inverse function g⁻¹
def g_inv : Fin 5 → Fin 5
| 1 => 3
| 2 => 5
| 3 => 2
| 4 => 1
| 5 => 4

-- State the theorem
theorem g_inv_composition :
  g_inv (g_inv (g_inv 3)) = 4 := by sorry

end g_inv_composition_l127_12780


namespace ella_work_days_l127_12726

/-- Represents the number of days Ella worked at each age --/
structure WorkDays where
  age10 : ℕ
  age11 : ℕ
  age12 : ℕ

/-- Calculates the total pay for the given work days --/
def totalPay (w : WorkDays) : ℕ :=
  4 * (10 * w.age10 + 11 * w.age11 + 12 * w.age12)

theorem ella_work_days :
  ∃ (w : WorkDays),
    w.age10 + w.age11 + w.age12 = 180 ∧
    totalPay w = 7920 ∧
    w.age11 = 60 := by
  sorry

end ella_work_days_l127_12726


namespace smallest_y_value_l127_12798

theorem smallest_y_value (x y : ℝ) 
  (h1 : 2 < x ∧ x < y)
  (h2 : 2 + x ≤ y)
  (h3 : 1 / x + 1 / y ≤ 1) :
  y ≥ 2 + Real.sqrt 2 ∧ ∀ z, (2 < z ∧ 2 + z ≤ y ∧ 1 / z + 1 / y ≤ 1) → y ≤ z :=
by sorry

end smallest_y_value_l127_12798


namespace jean_price_satisfies_conditions_l127_12716

/-- The price of a jean that satisfies the given conditions -/
def jean_price : ℝ := 11

/-- The price of a tee -/
def tee_price : ℝ := 8

/-- The number of tees sold -/
def tees_sold : ℕ := 7

/-- The number of jeans sold -/
def jeans_sold : ℕ := 4

/-- The total revenue -/
def total_revenue : ℝ := 100

/-- Theorem stating that the jean price satisfies the given conditions -/
theorem jean_price_satisfies_conditions :
  tee_price * tees_sold + jean_price * jeans_sold = total_revenue := by
  sorry

#check jean_price_satisfies_conditions

end jean_price_satisfies_conditions_l127_12716


namespace mushroom_collection_l127_12711

theorem mushroom_collection : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧  -- three-digit number
    (n / 100 + (n / 10) % 10 + n % 10 = 14) ∧  -- sum of digits is 14
    n % 50 = 0 ∧  -- divisible by 50
    n = 950 := by
  sorry

end mushroom_collection_l127_12711


namespace mikes_video_game_earnings_l127_12769

theorem mikes_video_game_earnings :
  let working_game_prices : List ℕ := [5, 7, 12, 9, 6, 15, 11, 10]
  List.sum working_game_prices = 75 := by
sorry

end mikes_video_game_earnings_l127_12769


namespace number_problem_l127_12767

theorem number_problem (x : ℝ) : 0.5 * x = 0.25 * x + 2 → x = 8 := by
  sorry

end number_problem_l127_12767


namespace largest_integer_l127_12795

theorem largest_integer (a b c d : ℤ) 
  (sum1 : a + b + c = 210)
  (sum2 : a + b + d = 230)
  (sum3 : a + c + d = 245)
  (sum4 : b + c + d = 260) :
  max a (max b (max c d)) = 105 := by
sorry

end largest_integer_l127_12795


namespace point_on_graph_and_coordinate_sum_l127_12777

theorem point_on_graph_and_coordinate_sum 
  (f : ℝ → ℝ) 
  (h : f 6 = 10) : 
  ∃ (x y : ℝ), 
    x = 2 ∧ 
    y = 28.5 ∧ 
    2 * y = 5 * f (3 * x) + 7 ∧ 
    x + y = 30.5 := by
  sorry

end point_on_graph_and_coordinate_sum_l127_12777


namespace line_erased_length_l127_12744

theorem line_erased_length (initial_length : ℝ) (final_length : ℝ) (erased_length : ℝ) : 
  initial_length = 1 →
  final_length = 0.67 →
  erased_length = initial_length * 100 - final_length * 100 →
  erased_length = 33 := by
sorry

end line_erased_length_l127_12744


namespace smallest_dual_base_palindrome_l127_12713

/-- Checks if a number is a palindrome in a given base -/
def isPalindrome (n : ℕ) (base : ℕ) : Prop := sorry

/-- Converts a number from one base to another -/
def baseConvert (n : ℕ) (fromBase toBase : ℕ) : ℕ := sorry

/-- Number of digits in a number in a given base -/
def numDigits (n : ℕ) (base : ℕ) : ℕ := sorry

theorem smallest_dual_base_palindrome :
  let n := 10001 -- In base 3
  ∀ m : ℕ,
    (numDigits n 3 = 5) →
    (isPalindrome n 3) →
    (∃ b : ℕ, b > 3 ∧ isPalindrome (baseConvert n 3 b) b ∧ numDigits (baseConvert n 3 b) b = 4) →
    (numDigits m 3 = 5) →
    (isPalindrome m 3) →
    (∃ b : ℕ, b > 3 ∧ isPalindrome (baseConvert m 3 b) b ∧ numDigits (baseConvert m 3 b) b = 4) →
    m ≥ n :=
by sorry

end smallest_dual_base_palindrome_l127_12713


namespace mrs_hilt_remaining_money_l127_12736

/-- Calculates the remaining money after purchases -/
def remaining_money (initial : ℚ) (pencil notebook pens : ℚ) : ℚ :=
  initial - (pencil + notebook + pens)

/-- Proves that Mrs. Hilt's remaining money is $3.00 -/
theorem mrs_hilt_remaining_money :
  remaining_money 12.5 1.25 3.45 4.8 = 3 := by
  sorry

end mrs_hilt_remaining_money_l127_12736


namespace seating_arrangements_3_8_l127_12766

/-- The number of distinct seating arrangements for 3 people in a row of 8 seats,
    with empty seats on both sides of each person. -/
def seating_arrangements (total_seats : ℕ) (people : ℕ) : ℕ :=
  sorry

/-- Theorem stating that the number of seating arrangements
    for 3 people in 8 seats is 24. -/
theorem seating_arrangements_3_8 :
  seating_arrangements 8 3 = 24 := by
  sorry

end seating_arrangements_3_8_l127_12766


namespace cos_alpha_plus_pi_third_l127_12718

theorem cos_alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/6) = 1/3) :
  Real.cos (α + π/3) = -1/3 := by
sorry

end cos_alpha_plus_pi_third_l127_12718


namespace transformed_dataset_properties_l127_12701

/-- Represents a dataset with its average and variance -/
structure Dataset where
  average : ℝ
  variance : ℝ

/-- Represents a linear transformation of a dataset -/
structure LinearTransform where
  scale : ℝ
  shift : ℝ

/-- Theorem stating the properties of a transformed dataset -/
theorem transformed_dataset_properties (original : Dataset) (transform : LinearTransform) :
  original.average = 3 ∧ 
  original.variance = 4 ∧ 
  transform.scale = 3 ∧ 
  transform.shift = -1 →
  ∃ (transformed : Dataset),
    transformed.average = 8 ∧
    transformed.variance = 36 := by
  sorry

end transformed_dataset_properties_l127_12701


namespace quadratic_equation_roots_l127_12771

theorem quadratic_equation_roots (a : ℝ) :
  let f : ℝ → ℝ := λ x => 4 * x^2 - 4 * (a + 2) * x + a^2 + 11
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ - x₂ = 3 → a = 4 := by
  sorry

end quadratic_equation_roots_l127_12771


namespace polynomial_divisibility_implies_perfect_powers_l127_12778

/-- Given a polynomial ax³ + 3bx² + 3cx + d that is divisible by ax² + 2bx + c,
    prove that it's a perfect cube and the divisor is a perfect square. -/
theorem polynomial_divisibility_implies_perfect_powers
  (a b c d : ℝ) (h : a ≠ 0) :
  (∃ (q : ℝ → ℝ), ∀ x, a * x^3 + 3*b * x^2 + 3*c * x + d = (a * x^2 + 2*b * x + c) * q x) →
  (∃ y, a * x^3 + 3*b * x^2 + 3*c * x + d = (a * x + y)^3) ∧
  (∃ z, a * x^2 + 2*b * x + c = (a * x + z)^2) ∧
  c = 2 * b^2 / a ∧
  d = 2 * b^3 / a^2 :=
by sorry

end polynomial_divisibility_implies_perfect_powers_l127_12778


namespace postman_pete_miles_l127_12775

/-- Represents a pedometer with a maximum step count before resetting --/
structure Pedometer where
  max_steps : ℕ
  resets : ℕ
  final_reading : ℕ

/-- Calculates the total steps recorded by a pedometer --/
def total_steps (p : Pedometer) : ℕ :=
  p.max_steps * (p.resets + 1) + p.final_reading

/-- Converts steps to miles, rounded to the nearest mile --/
def steps_to_miles (steps : ℕ) (steps_per_mile : ℕ) : ℕ :=
  (steps + steps_per_mile / 2) / steps_per_mile

/-- Theorem stating the total miles walked by Postman Pete --/
theorem postman_pete_miles :
  let p : Pedometer := { max_steps := 100000, resets := 48, final_reading := 25000 }
  let steps_per_mile : ℕ := 1600
  steps_to_miles (total_steps p) steps_per_mile = 3016 := by
  sorry

end postman_pete_miles_l127_12775


namespace expression_simplification_l127_12714

theorem expression_simplification (x : ℝ) (hx : x > 0) :
  (x - 1) / (x^(3/4) + x^(1/2)) * (x^(1/2) + x^(1/4)) / (x^(1/2) + 1) * x^(1/4) + 1 = x^(1/2) := by
  sorry

end expression_simplification_l127_12714


namespace expression_result_l127_12765

theorem expression_result : (3.241 * 14) / 100 = 0.45374 := by
  sorry

end expression_result_l127_12765


namespace largest_divisor_of_consecutive_odds_l127_12785

theorem largest_divisor_of_consecutive_odds (n : ℕ) (h : Even n) (h_pos : 0 < n) :
  ∃ (k : ℕ), k = 105 ∧ 
  (∀ (d : ℕ), d ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) → d ≤ k) ∧
  k ∣ ((n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
sorry

end largest_divisor_of_consecutive_odds_l127_12785


namespace f_shifted_l127_12794

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_shifted (x : ℝ) : f (x + 1) = x^2 + 2*x → f (x - 1) = x^2 - 2*x := by
  sorry

end f_shifted_l127_12794


namespace reflection_across_y_axis_l127_12707

/-- A point in a 2D coordinate system. -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis. -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The theorem stating that reflecting P(3, -4) across the y-axis results in P'(-3, -4). -/
theorem reflection_across_y_axis :
  let P : Point := { x := 3, y := -4 }
  let P' : Point := reflectAcrossYAxis P
  P'.x = -3 ∧ P'.y = -4 := by sorry

end reflection_across_y_axis_l127_12707


namespace parabola_intersects_x_axis_twice_and_integer_intersection_l127_12712

/-- Represents a quadratic function y = mx^2 - (m+2)x + 2 --/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - (m + 2) * x + 2

theorem parabola_intersects_x_axis_twice_and_integer_intersection (m : ℝ) 
  (hm_nonzero : m ≠ 0) (hm_not_two : m ≠ 2) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ quadratic_function m x1 = 0 ∧ quadratic_function m x2 = 0) ∧
  (∃! m : ℕ+, m ≠ 2 ∧ ∃ x1 x2 : ℤ, quadratic_function ↑m ↑x1 = 0 ∧ quadratic_function ↑m ↑x2 = 0 ∧ x1 ≠ x2) :=
by sorry

end parabola_intersects_x_axis_twice_and_integer_intersection_l127_12712


namespace sum_of_coefficients_is_five_l127_12719

def sequence_u (n : ℕ) : ℚ :=
  sorry

theorem sum_of_coefficients_is_five :
  ∃ (a b c : ℚ),
    (∀ n : ℕ, sequence_u n = a * n^2 + b * n + c) ∧
    (sequence_u 1 = 5) ∧
    (∀ n : ℕ, sequence_u (n + 1) - sequence_u n = 3 + 4 * (n - 1)) ∧
    (a + b + c = 5) :=
  sorry

end sum_of_coefficients_is_five_l127_12719


namespace equality_from_sum_squares_l127_12748

theorem equality_from_sum_squares (x y z : ℝ) :
  x^2 + y^2 + z^2 = x*y + y*z + z*x → x = y ∧ y = z := by
  sorry

end equality_from_sum_squares_l127_12748


namespace amy_money_calculation_l127_12703

theorem amy_money_calculation (initial : ℕ) (chores : ℕ) (birthday : ℕ) : 
  initial = 2 → chores = 13 → birthday = 3 → initial + chores + birthday = 18 := by
  sorry

end amy_money_calculation_l127_12703


namespace consecutive_odd_squares_difference_l127_12745

theorem consecutive_odd_squares_difference (x : ℤ) : 
  Odd x → Odd (x + 2) → (x + 2)^2 - x^2 = 2000 → (x = 499 ∨ x = -501) :=
by sorry

end consecutive_odd_squares_difference_l127_12745


namespace scientific_notation_125000_l127_12791

theorem scientific_notation_125000 : 
  125000 = 1.25 * (10 ^ 5) := by
  sorry

end scientific_notation_125000_l127_12791


namespace tangent_line_minimum_two_roots_inequality_l127_12737

noncomputable section

variables (m : ℝ) (x x₁ x₂ : ℝ) (a b n : ℝ)

def f (x : ℝ) : ℝ := Real.log x - m * x

theorem tangent_line_minimum (h : f e x = a * x + b) :
  ∃ (x₀ : ℝ), a + 2 * b = 1 / x₀ + 2 * Real.log x₀ - e - 2 ∧ 
  ∀ (x : ℝ), 1 / x + 2 * Real.log x - e - 2 ≥ 1 / x₀ + 2 * Real.log x₀ - e - 2 :=
sorry

theorem two_roots_inequality (h1 : f m x₁ = (2 - m) * x₁ + n) 
                             (h2 : f m x₂ = (2 - m) * x₂ + n) 
                             (h3 : x₁ < x₂) :
  2 * x₁ + x₂ > e / 2 :=
sorry

end tangent_line_minimum_two_roots_inequality_l127_12737


namespace ellipse_foci_l127_12730

theorem ellipse_foci (x y : ℝ) :
  (x^2 / 25 + y^2 / 169 = 1) →
  (∃ f₁ f₂ : ℝ × ℝ, 
    (f₁ = (0, 12) ∧ f₂ = (0, -12)) ∧
    (∀ p : ℝ × ℝ, p.1^2 / 25 + p.2^2 / 169 = 1 →
      (Real.sqrt ((p.1 - f₁.1)^2 + (p.2 - f₁.2)^2) +
       Real.sqrt ((p.1 - f₂.1)^2 + (p.2 - f₂.2)^2) =
       2 * Real.sqrt 169))) :=
by sorry

end ellipse_foci_l127_12730


namespace small_cylinder_radius_l127_12764

/-- Proves that the radius of smaller cylinders is √(24/5) meters given the specified conditions -/
theorem small_cylinder_radius 
  (large_diameter : ℝ) 
  (large_height : ℝ) 
  (small_height : ℝ) 
  (num_small_cylinders : ℕ) 
  (h_large_diameter : large_diameter = 6)
  (h_large_height : large_height = 8)
  (h_small_height : small_height = 5)
  (h_num_small_cylinders : num_small_cylinders = 3)
  : ∃ (small_radius : ℝ), small_radius = Real.sqrt (24 / 5) := by
  sorry

#check small_cylinder_radius

end small_cylinder_radius_l127_12764


namespace least_possible_third_side_length_l127_12728

theorem least_possible_third_side_length (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a = 5 → b = 12 →
  (a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2 ∨ a^2 + b^2 = c^2) →
  c ≥ Real.sqrt 119 := by
  sorry

end least_possible_third_side_length_l127_12728


namespace distance_after_7km_l127_12753

/-- Regular hexagon with side length 3 km -/
structure RegularHexagon where
  side_length : ℝ
  is_regular : side_length = 3

/-- Point on the perimeter of the hexagon -/
structure PerimeterPoint (h : RegularHexagon) where
  distance_from_start : ℝ
  on_perimeter : distance_from_start ≥ 0 ∧ distance_from_start ≤ 6 * h.side_length

/-- The distance from the starting point to a point on the perimeter -/
def distance_to_start (h : RegularHexagon) (p : PerimeterPoint h) : ℝ :=
  sorry

theorem distance_after_7km (h : RegularHexagon) (p : PerimeterPoint h) 
  (h_distance : p.distance_from_start = 7) :
  distance_to_start h p = 2 :=
sorry

end distance_after_7km_l127_12753


namespace expression_multiple_of_six_l127_12789

theorem expression_multiple_of_six (n : ℕ) (h : n ≥ 10) :
  ∃ k : ℤ, ((n + 3).factorial - (n + 1).factorial) / n.factorial = 6 * k := by
  sorry

end expression_multiple_of_six_l127_12789


namespace square_commutes_with_multiplication_l127_12706

theorem square_commutes_with_multiplication (m n : ℝ) : m^2 * n - n * m^2 = 0 := by
  sorry

end square_commutes_with_multiplication_l127_12706


namespace cubic_root_sum_cube_l127_12779

theorem cubic_root_sum_cube (a b c : ℂ) : 
  (a^3 - 2*a^2 + 3*a - 4 = 0) → 
  (b^3 - 2*b^2 + 3*b - 4 = 0) → 
  (c^3 - 2*c^2 + 3*c - 4 = 0) → 
  a^3 + b^3 + c^3 = 2 := by
sorry

end cubic_root_sum_cube_l127_12779


namespace rectangle_length_l127_12799

/-- Represents a rectangle with perimeter P, width W, length L, and area A. -/
structure Rectangle where
  P : ℝ  -- Perimeter
  W : ℝ  -- Width
  L : ℝ  -- Length
  A : ℝ  -- Area
  h1 : P = 2 * (L + W)  -- Perimeter formula
  h2 : A = L * W        -- Area formula
  h3 : P / W = 5        -- Given ratio
  h4 : A = 150          -- Given area

/-- Proves that a rectangle with the given properties has a length of 15. -/
theorem rectangle_length (rect : Rectangle) : rect.L = 15 := by
  sorry

#check rectangle_length

end rectangle_length_l127_12799


namespace tan_alpha_plus_pi_third_l127_12756

theorem tan_alpha_plus_pi_third (α β : Real) 
  (h1 : Real.tan (α + β) = 3/5)
  (h2 : Real.tan (β - Real.pi/3) = 1/4) : 
  Real.tan (α + Real.pi/3) = 7/23 := by sorry

end tan_alpha_plus_pi_third_l127_12756


namespace prob_fewer_heads_12_coins_l127_12774

/-- The number of coins Lucy flips -/
def n : ℕ := 12

/-- The probability of getting fewer heads than tails when flipping n coins -/
def prob_fewer_heads (n : ℕ) : ℚ :=
  793 / 2048

theorem prob_fewer_heads_12_coins : 
  prob_fewer_heads n = 793 / 2048 := by sorry

end prob_fewer_heads_12_coins_l127_12774


namespace wait_time_difference_l127_12742

/-- Proves that the difference in wait times between swings and slide is 270 seconds -/
theorem wait_time_difference : 
  let kids_on_swings : ℕ := 3
  let kids_on_slide : ℕ := 2 * kids_on_swings
  let swing_wait_time : ℕ := 2 * 60  -- 2 minutes in seconds
  let slide_wait_time : ℕ := 15      -- 15 seconds
  let total_swing_wait : ℕ := kids_on_swings * swing_wait_time
  let total_slide_wait : ℕ := kids_on_slide * slide_wait_time
  total_swing_wait - total_slide_wait = 270 := by
sorry


end wait_time_difference_l127_12742


namespace max_candy_leftover_l127_12732

theorem max_candy_leftover (y : ℕ) : ∃ (q r : ℕ), y = 7 * q + r ∧ r < 7 ∧ r ≤ 6 :=
by sorry

end max_candy_leftover_l127_12732


namespace gcd_nine_digit_repeats_l127_12723

/-- The set of all nine-digit integers formed by repeating a three-digit integer three times -/
def NineDigitRepeats : Set ℕ :=
  {n | ∃ k : ℕ, 100 ≤ k ∧ k ≤ 999 ∧ n = 1001001 * k}

/-- The greatest common divisor of all numbers in NineDigitRepeats is 1001001 -/
theorem gcd_nine_digit_repeats :
  ∃ d : ℕ, d > 0 ∧ (∀ n ∈ NineDigitRepeats, d ∣ n) ∧
  (∀ m : ℕ, m > 0 → (∀ n ∈ NineDigitRepeats, m ∣ n) → m ≤ d) ∧
  d = 1001001 := by
  sorry


end gcd_nine_digit_repeats_l127_12723


namespace line_intersects_circle_l127_12743

/-- The line y = x + 1 intersects the circle x^2 + y^2 = 1 -/
theorem line_intersects_circle :
  let line : ℝ → ℝ := λ x ↦ x + 1
  let circle : ℝ × ℝ → Prop := λ p ↦ p.1^2 + p.2^2 = 1
  let center : ℝ × ℝ := (0, 0)
  let radius : ℝ := 1
  let distance_to_line : ℝ := |1| / Real.sqrt 2
  distance_to_line < radius →
  ∃ p : ℝ × ℝ, line p.1 = p.2 ∧ circle p :=
by sorry

end line_intersects_circle_l127_12743


namespace angelina_speed_l127_12724

/-- Proves that Angelina's speed from the grocery to the gym is 3 meters per second --/
theorem angelina_speed (home_to_grocery : ℝ) (grocery_to_gym : ℝ) (v : ℝ) :
  home_to_grocery = 180 →
  grocery_to_gym = 240 →
  (home_to_grocery / v) - (grocery_to_gym / (2 * v)) = 40 →
  2 * v = 3 := by
  sorry

end angelina_speed_l127_12724


namespace sachin_age_l127_12735

theorem sachin_age (sachin_age rahul_age : ℕ) 
  (h1 : rahul_age = sachin_age + 7)
  (h2 : sachin_age * 12 = rahul_age * 5) : 
  sachin_age = 5 := by
  sorry

end sachin_age_l127_12735


namespace book_writing_time_difference_l127_12761

/-- The time difference in months between Ivanka's and Woody's book writing time -/
def time_difference (ivanka_time woody_time : ℕ) : ℕ :=
  ivanka_time - woody_time

/-- Proof that the time difference is 3 months given the conditions -/
theorem book_writing_time_difference :
  ∀ (ivanka_time woody_time : ℕ),
    woody_time = 18 →
    ivanka_time + woody_time = 39 →
    time_difference ivanka_time woody_time = 3 := by
  sorry

end book_writing_time_difference_l127_12761


namespace water_used_l127_12721

theorem water_used (total_liquid oil : ℝ) (h1 : total_liquid = 1.33) (h2 : oil = 0.17) :
  total_liquid - oil = 1.16 := by
  sorry

end water_used_l127_12721


namespace seven_layer_tower_lights_l127_12709

/-- Represents a tower with lights -/
structure LightTower where
  layers : ℕ
  top_lights : ℕ
  total_lights : ℕ

/-- The sum of a geometric sequence -/
def geometricSum (a₁ : ℕ) (r : ℕ) (n : ℕ) : ℕ :=
  a₁ * (r^n - 1) / (r - 1)

/-- The theorem statement -/
theorem seven_layer_tower_lights (tower : LightTower) :
  tower.layers = 7 ∧
  tower.total_lights = 381 ∧
  (∀ i : ℕ, i < 7 → geometricSum tower.top_lights 2 (i + 1) ≤ tower.total_lights) →
  tower.top_lights = 3 := by
  sorry

end seven_layer_tower_lights_l127_12709


namespace bridge_length_calculation_l127_12792

/-- Calculates the length of a bridge given a person's walking speed and time to cross -/
theorem bridge_length_calculation (speed : ℝ) (time_minutes : ℝ) : 
  speed = 6 → time_minutes = 15 → speed * (time_minutes / 60) = 1.5 := by
  sorry

#check bridge_length_calculation

end bridge_length_calculation_l127_12792


namespace triangle_radii_relation_l127_12770

/-- Given a triangle ABC with sides a, b, c, inradius r, circumradius R, and excircle radii rA, rB, rC,
    prove the following equation. -/
theorem triangle_radii_relation (a b c r R rA rB rC : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ r > 0 ∧ R > 0 ∧ rA > 0 ∧ rB > 0 ∧ rC > 0) :
  a^2 * (2/rA - r/(rB*rC)) + b^2 * (2/rB - r/(rA*rC)) + c^2 * (2/rC - r/(rA*rB)) = 4*(R + 3*r) := by
  sorry

end triangle_radii_relation_l127_12770


namespace smallest_a_in_special_progression_l127_12734

theorem smallest_a_in_special_progression (a b c : ℤ) 
  (h1 : a < b) (h2 : b < c)
  (h3 : 2 * b = a + c)  -- arithmetic progression
  (h4 : a * a = c * b)  -- geometric progression
  : a ≥ 1 ∧ ∃ (a₀ b₀ c₀ : ℤ), a₀ = 1 ∧ 
    a₀ < b₀ ∧ b₀ < c₀ ∧ 
    2 * b₀ = a₀ + c₀ ∧
    a₀ * a₀ = c₀ * b₀ := by
  sorry

end smallest_a_in_special_progression_l127_12734


namespace yellow_then_not_yellow_probability_l127_12784

/-- A deck of cards with 5 suits and 13 ranks. -/
structure Deck :=
  (cards : Finset (Fin 5 × Fin 13))
  (card_count : cards.card = 65)
  (suit_rank_unique : ∀ (s : Fin 5) (r : Fin 13), (s, r) ∈ cards)

/-- The probability of drawing a yellow card followed by a non-yellow card from a shuffled deck. -/
def yellow_then_not_yellow_prob (d : Deck) : ℚ :=
  169 / 1040

/-- Theorem stating the probability of drawing a yellow card followed by a non-yellow card. -/
theorem yellow_then_not_yellow_probability (d : Deck) :
  yellow_then_not_yellow_prob d = 169 / 1040 := by
  sorry

end yellow_then_not_yellow_probability_l127_12784


namespace ratio_of_numbers_l127_12715

theorem ratio_of_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b) (h4 : a + b = 7 * (a - b)) : a / b = 4 / 3 := by
  sorry

end ratio_of_numbers_l127_12715


namespace tony_squat_weight_l127_12739

/-- Represents Tony's weight lifting capabilities -/
structure WeightLifter where
  curl_weight : ℕ
  military_press_multiplier : ℕ
  squat_multiplier : ℕ

/-- Calculates the weight Tony can lift in the squat exercise -/
def squat_weight (lifter : WeightLifter) : ℕ :=
  lifter.curl_weight * lifter.military_press_multiplier * lifter.squat_multiplier

/-- Theorem: Tony can lift 900 pounds in the squat exercise -/
theorem tony_squat_weight :
  ∃ (tony : WeightLifter),
    tony.curl_weight = 90 ∧
    tony.military_press_multiplier = 2 ∧
    tony.squat_multiplier = 5 ∧
    squat_weight tony = 900 :=
by
  sorry

end tony_squat_weight_l127_12739


namespace square_difference_emily_calculation_l127_12708

theorem square_difference (n : ℕ) : (n - 1)^2 = n^2 - (2*n - 1) := by sorry

theorem emily_calculation : 39^2 = 40^2 - 79 := by sorry

end square_difference_emily_calculation_l127_12708


namespace kangaroo_population_change_l127_12768

theorem kangaroo_population_change 
  (G : ℝ) -- Initial number of grey kangaroos
  (R : ℝ) -- Initial number of red kangaroos
  (h1 : G > 0) -- Assumption: initial grey kangaroo population is positive
  (h2 : R > 0) -- Assumption: initial red kangaroo population is positive
  (h3 : 1.28 * G / (0.72 * R) = R / G) -- Ratio reversal condition
  : (2.24 * G) / ((7/3) * G) = 0.96 := by
  sorry

end kangaroo_population_change_l127_12768


namespace magnitude_v_l127_12763

theorem magnitude_v (u v : ℂ) (h1 : u * v = 24 - 10 * I) (h2 : Complex.abs u = 5) :
  Complex.abs v = 26 / 5 := by
  sorry

end magnitude_v_l127_12763


namespace train_departure_time_difference_l127_12720

/-- Proves that Train A leaves 40 minutes before Train B, given their speeds and overtake time --/
theorem train_departure_time_difference 
  (speed_A : ℝ) 
  (speed_B : ℝ) 
  (overtake_time : ℝ) 
  (h1 : speed_A = 60) 
  (h2 : speed_B = 80) 
  (h3 : overtake_time = 120) :
  ∃ (time_diff : ℝ), 
    time_diff = 40 ∧ 
    speed_A * (time_diff / 60 + overtake_time / 60) = speed_B * (overtake_time / 60) := by
  sorry


end train_departure_time_difference_l127_12720


namespace trigonometric_problem_l127_12729

theorem trigonometric_problem (x : ℝ) (h : 3 * Real.sin (x/2) - Real.cos (x/2) = 0) :
  Real.tan x = 3/4 ∧ (Real.cos (2*x)) / (Real.sqrt 2 * Real.cos (π/4 + x) * Real.sin x) = 7/3 := by
  sorry

end trigonometric_problem_l127_12729


namespace absolute_value_inequality_l127_12702

theorem absolute_value_inequality (x : ℝ) : |2*x + 1| < 3 ↔ -2 < x ∧ x < 1 := by sorry

end absolute_value_inequality_l127_12702
