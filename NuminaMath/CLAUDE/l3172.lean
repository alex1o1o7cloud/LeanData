import Mathlib

namespace rectangular_field_area_l3172_317255

/-- Calculates the area of a rectangular field given specific fencing conditions -/
theorem rectangular_field_area 
  (uncovered_side : ℝ) 
  (total_fencing : ℝ) 
  (h1 : uncovered_side = 20) 
  (h2 : total_fencing = 88) : 
  uncovered_side * ((total_fencing - uncovered_side) / 2) = 680 :=
by
  sorry

#check rectangular_field_area

end rectangular_field_area_l3172_317255


namespace last_ball_black_prob_specific_box_l3172_317221

/-- A box containing black and white balls -/
structure Box where
  black : ℕ
  white : ℕ

/-- The probability of the last ball being black in a drawing process -/
def last_ball_black_prob (b : Box) : ℚ :=
  b.white / (b.black + b.white)

/-- The theorem stating the probability of the last ball being black for a specific box -/
theorem last_ball_black_prob_specific_box :
  let b : Box := { black := 3, white := 4 }
  last_ball_black_prob b = 4/7 := by
  sorry

#check last_ball_black_prob_specific_box

end last_ball_black_prob_specific_box_l3172_317221


namespace bret_reading_time_l3172_317217

/-- The time Bret spends reading a book during a train ride -/
def time_reading_book (total_time dinner_time movie_time nap_time : ℕ) : ℕ :=
  total_time - (dinner_time + movie_time + nap_time)

/-- Theorem: Bret spends 2 hours reading a book during his train ride -/
theorem bret_reading_time :
  time_reading_book 9 1 3 3 = 2 := by
  sorry

end bret_reading_time_l3172_317217


namespace weekly_distance_is_1760_l3172_317237

/-- Calculates the total distance traveled by a driver in a week -/
def weekly_distance : ℕ :=
  let weekday_speed1 : ℕ := 30
  let weekday_time1 : ℕ := 3
  let weekday_speed2 : ℕ := 25
  let weekday_time2 : ℕ := 4
  let weekday_speed3 : ℕ := 40
  let weekday_time3 : ℕ := 2
  let weekday_days : ℕ := 6
  let sunday_speed : ℕ := 35
  let sunday_time : ℕ := 5
  let sunday_breaks : ℕ := 2
  let break_duration : ℕ := 30

  let weekday_distance := (weekday_speed1 * weekday_time1 + 
                           weekday_speed2 * weekday_time2 + 
                           weekday_speed3 * weekday_time3) * weekday_days
  let sunday_distance := sunday_speed * (sunday_time - sunday_breaks * break_duration / 60)
  
  weekday_distance + sunday_distance

theorem weekly_distance_is_1760 : weekly_distance = 1760 := by
  sorry

end weekly_distance_is_1760_l3172_317237


namespace tan_alpha_value_l3172_317263

theorem tan_alpha_value (α : Real) :
  (∃ x y : Real, x = -1 ∧ y = Real.sqrt 3 ∧ 
   (Real.cos α * x - Real.sin α * y = 0)) →
  Real.tan α = -Real.sqrt 3 := by
sorry

end tan_alpha_value_l3172_317263


namespace min_comparisons_for_max_l3172_317224

/-- Represents a list of n pairwise distinct numbers -/
def DistinctNumbers (n : ℕ) := { l : List ℝ // l.length = n ∧ l.Pairwise (· ≠ ·) }

/-- Represents a comparison between two numbers -/
def Comparison := ℝ × ℝ

/-- A function that finds the maximum number in a list using pairwise comparisons -/
def FindMax (n : ℕ) (numbers : DistinctNumbers n) : 
  { comparisons : List Comparison // comparisons.length = n - 1 ∧ 
    ∃ max, max ∈ numbers.val ∧ ∀ x ∈ numbers.val, x ≤ max } :=
sorry

theorem min_comparisons_for_max (n : ℕ) (numbers : DistinctNumbers n) :
  (∀ comparisons : List Comparison, 
    (∃ max, max ∈ numbers.val ∧ ∀ x ∈ numbers.val, x ≤ max) → 
    comparisons.length ≥ n - 1) ∧
  (∃ comparisons : List Comparison, 
    comparisons.length = n - 1 ∧ 
    ∃ max, max ∈ numbers.val ∧ ∀ x ∈ numbers.val, x ≤ max) :=
sorry

end min_comparisons_for_max_l3172_317224


namespace general_term_formula_l3172_317248

/-- Given a sequence {a_n} where S_n is the sum of its first n terms -/
def S (n : ℕ) : ℝ := 3^n - 1

/-- The general term of the sequence -/
def a (n : ℕ) : ℝ := 2 * 3^(n - 1)

/-- Theorem stating that the given general term formula is correct -/
theorem general_term_formula (n : ℕ) (h : n ≥ 1) : 
  a n = S n - S (n - 1) := by sorry

end general_term_formula_l3172_317248


namespace apple_cost_for_two_weeks_l3172_317223

/-- Represents the cost of apples for Irene and her dog for 2 weeks -/
def appleCost (daysPerWeek : ℕ) (weeks : ℕ) (appleWeight : ℚ) (pricePerPound : ℚ) : ℚ :=
  let totalDays : ℕ := daysPerWeek * weeks
  let totalApples : ℕ := totalDays
  let totalWeight : ℚ := appleWeight * totalApples
  totalWeight * pricePerPound

/-- Theorem stating that the cost of apples for 2 weeks is $7.00 -/
theorem apple_cost_for_two_weeks :
  appleCost 7 2 (1/4) 2 = 7 :=
sorry

end apple_cost_for_two_weeks_l3172_317223


namespace expression_equality_l3172_317251

theorem expression_equality : 5 * 399 + 4 * 399 + 3 * 399 + 397 = 5185 := by
  sorry

end expression_equality_l3172_317251


namespace inequality_system_solutions_l3172_317265

theorem inequality_system_solutions : 
  ∃! (s : Finset ℕ), 
    (∀ x ∈ s, (3 - 2*x > 0 ∧ 2*x - 7 ≤ 4*x + 7)) ∧ 
    (∀ x : ℕ, (3 - 2*x > 0 ∧ 2*x - 7 ≤ 4*x + 7) → x ∈ s) ∧
    Finset.card s = 2 :=
by sorry

end inequality_system_solutions_l3172_317265


namespace number_equation_l3172_317209

theorem number_equation (x : ℝ) : 0.833 * x = -60 → x = -72 := by
  sorry

end number_equation_l3172_317209


namespace cos_pi_minus_2alpha_l3172_317222

theorem cos_pi_minus_2alpha (α : Real) (h : Real.sin α = 2/3) : 
  Real.cos (Real.pi - 2*α) = -1/9 := by sorry

end cos_pi_minus_2alpha_l3172_317222


namespace cube_edge_length_l3172_317260

/-- Given a cube with surface area 18 dm², prove that the length of its edge is √3 dm. -/
theorem cube_edge_length (S : ℝ) (edge : ℝ) (h1 : S = 18) (h2 : S = 6 * edge ^ 2) : 
  edge = Real.sqrt 3 := by
sorry

end cube_edge_length_l3172_317260


namespace zeros_order_l3172_317212

noncomputable def f (x : ℝ) := Real.exp x + x
noncomputable def g (x : ℝ) := Real.log x + x
noncomputable def h (x : ℝ) := Real.log x - 1

theorem zeros_order (a b c : ℝ) 
  (ha : f a = 0) 
  (hb : g b = 0) 
  (hc : h c = 0) : 
  a < b ∧ b < c := by sorry

end zeros_order_l3172_317212


namespace weekly_earnings_correct_l3172_317235

/-- Represents the weekly earnings of Jake, Jacob, and Jim --/
structure WeeklyEarnings where
  jacob : ℕ
  jake : ℕ
  jim : ℕ

/-- Calculates the weekly earnings based on the given conditions --/
def calculateWeeklyEarnings : WeeklyEarnings :=
  let jacobWeekdayRate := 6
  let jacobWeekendRate := 8
  let weekdayHours := 8
  let weekendHours := 5
  let weekdays := 5
  let weekendDays := 2

  let jacobWeekdayEarnings := jacobWeekdayRate * weekdayHours * weekdays
  let jacobWeekendEarnings := jacobWeekendRate * weekendHours * weekendDays
  let jacobTotal := jacobWeekdayEarnings + jacobWeekendEarnings

  let jakeWeekdayRate := 3 * jacobWeekdayRate
  let jakeWeekdayEarnings := jakeWeekdayRate * weekdayHours * weekdays
  let jakeWeekendEarnings := jacobWeekendEarnings
  let jakeTotal := jakeWeekdayEarnings + jakeWeekendEarnings

  let jimWeekdayRate := 2 * jakeWeekdayRate
  let jimWeekdayEarnings := jimWeekdayRate * weekdayHours * weekdays
  let jimWeekendEarnings := jacobWeekendEarnings
  let jimTotal := jimWeekdayEarnings + jimWeekendEarnings

  { jacob := jacobTotal, jake := jakeTotal, jim := jimTotal }

/-- Theorem stating that the calculated weekly earnings match the expected values --/
theorem weekly_earnings_correct : 
  let earnings := calculateWeeklyEarnings
  earnings.jacob = 320 ∧ earnings.jake = 800 ∧ earnings.jim = 1520 := by
  sorry

end weekly_earnings_correct_l3172_317235


namespace complement_union_theorem_l3172_317294

def U : Set ℕ := {0, 1, 2, 3, 4}
def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {2, 3, 4}

theorem complement_union_theorem :
  (U \ A) ∪ (U \ B) = {0, 1, 4} := by sorry

end complement_union_theorem_l3172_317294


namespace max_projection_equals_face_area_l3172_317266

/-- A tetrahedron with two adjacent isosceles right triangle faces -/
structure Tetrahedron where
  /-- Length of the hypotenuse of the isosceles right triangle faces -/
  hypotenuse_length : ℝ
  /-- Dihedral angle between the two adjacent isosceles right triangle faces -/
  dihedral_angle : ℝ
  /-- Assumption that the hypotenuse length is 2 -/
  hypotenuse_is_two : hypotenuse_length = 2
  /-- Assumption that the dihedral angle is 60 degrees (π/3 radians) -/
  angle_is_sixty_degrees : dihedral_angle = π / 3

/-- The area of one isosceles right triangle face of the tetrahedron -/
def face_area (t : Tetrahedron) : ℝ := 1

/-- The maximum area of the projection of the rotating tetrahedron -/
def max_projection_area (t : Tetrahedron) : ℝ := 1

/-- Theorem stating that the maximum projection area equals the face area -/
theorem max_projection_equals_face_area (t : Tetrahedron) :
  max_projection_area t = face_area t := by sorry

end max_projection_equals_face_area_l3172_317266


namespace points_per_treasure_l3172_317273

theorem points_per_treasure (total_treasures : ℕ) (total_score : ℕ) (points_per_treasure : ℕ) : 
  total_treasures = 7 → total_score = 63 → points_per_treasure * total_treasures = total_score → points_per_treasure = 9 := by
  sorry

end points_per_treasure_l3172_317273


namespace granger_age_multiple_l3172_317225

/-- The multiple of Mr. Granger's son's age last year that Mr. Granger's age last year was 4 years less than -/
def multiple_last_year (grangers_age : ℕ) (sons_age : ℕ) : ℚ :=
  (grangers_age - 1) / (sons_age - 1)

/-- Mr. Granger's current age -/
def grangers_age : ℕ := 42

/-- Mr. Granger's son's current age -/
def sons_age : ℕ := 16

theorem granger_age_multiple : multiple_last_year grangers_age sons_age = 3 := by
  sorry

end granger_age_multiple_l3172_317225


namespace dinner_seating_arrangements_l3172_317245

theorem dinner_seating_arrangements (n : ℕ) (k : ℕ) (h1 : n = 8) (h2 : k = 6) :
  (Nat.choose n k) * (Nat.factorial (k - 1)) = 3360 := by
  sorry

end dinner_seating_arrangements_l3172_317245


namespace largest_constant_divisor_l3172_317281

theorem largest_constant_divisor (n : ℤ) : 
  let x : ℤ := 4 * n - 1
  ∃ (k : ℤ), (12 * x + 2) * (8 * x + 6) * (6 * x + 3) = 60 * k ∧ 
  ∀ (m : ℤ), m > 60 → 
    ∃ (l : ℤ), (12 * x + 2) * (8 * x + 6) * (6 * x + 3) ≠ m * l :=
by sorry

end largest_constant_divisor_l3172_317281


namespace perfect_square_from_fraction_pairs_l3172_317277

theorem perfect_square_from_fraction_pairs (N : ℕ+) 
  (h : ∃! (pairs : Finset (ℕ+ × ℕ+)), pairs.card = 2005 ∧ 
    ∀ (x y : ℕ+), (x, y) ∈ pairs ↔ (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / N) : 
  ∃ (k : ℕ+), N = k ^ 2 := by
  sorry

end perfect_square_from_fraction_pairs_l3172_317277


namespace total_muffins_for_sale_l3172_317200

theorem total_muffins_for_sale : 
  let num_boys : ℕ := 3
  let num_girls : ℕ := 2
  let muffins_per_boy : ℕ := 12
  let muffins_per_girl : ℕ := 20
  let total_muffins : ℕ := num_boys * muffins_per_boy + num_girls * muffins_per_girl
  total_muffins = 76 :=
by
  sorry

end total_muffins_for_sale_l3172_317200


namespace circle_passes_through_point_l3172_317215

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (x : ℝ) : Prop := x + 2 = 0

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : parabola x y

-- Define a circle
structure Circle where
  center : PointOnParabola
  radius : ℝ
  tangent_to_line : radius = center.x + 2

-- Theorem to prove
theorem circle_passes_through_point :
  ∀ (c : Circle), (c.center.x - 2)^2 + c.center.y^2 = c.radius^2 := by
  sorry

end circle_passes_through_point_l3172_317215


namespace jerky_order_theorem_l3172_317242

/-- Calculates the total number of jerky bags for a customer order -/
def customer_order_bags (production_rate : ℕ) (initial_inventory : ℕ) (production_days : ℕ) : ℕ :=
  production_rate * production_days + initial_inventory

/-- Theorem stating that given the specific conditions, the customer order is 60 bags -/
theorem jerky_order_theorem :
  let production_rate := 10
  let initial_inventory := 20
  let production_days := 4
  customer_order_bags production_rate initial_inventory production_days = 60 := by
  sorry

#eval customer_order_bags 10 20 4  -- Should output 60

end jerky_order_theorem_l3172_317242


namespace infinitely_many_unreachable_integers_l3172_317282

/-- Sum of digits in base b -/
def sum_of_digits (b : ℕ) (n : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem infinitely_many_unreachable_integers (b : ℕ) (h : b ≥ 2) :
  ∀ M : ℕ, ∃ S : Finset ℕ, (Finset.card S = M) ∧ 
  (∀ k ∈ S, ∀ n : ℕ, n + sum_of_digits b n ≠ k) :=
sorry

end infinitely_many_unreachable_integers_l3172_317282


namespace triangle_max_area_l3172_317296

/-- Given a triangle ABC with AB = 10 and BC:AC = 35:36, its maximum area is 1260 -/
theorem triangle_max_area (A B C : ℝ × ℝ) :
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AC := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let s := (AB + BC + AC) / 2
  let area := Real.sqrt (s * (s - AB) * (s - BC) * (s - AC))
  AB = 10 ∧ BC / AC = 35 / 36 → area ≤ 1260 := by
  sorry

#check triangle_max_area

end triangle_max_area_l3172_317296


namespace paving_cost_calculation_l3172_317232

/-- The cost of paving a rectangular floor -/
def paving_cost (length width rate : ℝ) : ℝ :=
  length * width * rate

/-- Theorem: The cost of paving a rectangular floor with length 5.5 m and width 3.75 m at a rate of Rs. 400 per square metre is Rs. 8250 -/
theorem paving_cost_calculation :
  paving_cost 5.5 3.75 400 = 8250 := by
  sorry

end paving_cost_calculation_l3172_317232


namespace angle_bisector_sum_l3172_317264

/-- A triangle with vertices P, Q, and R -/
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

/-- The angle bisector equation of the form ax + 2y + c = 0 -/
structure AngleBisectorEq where
  a : ℝ
  c : ℝ

/-- Given a triangle PQR, returns the angle bisector equation of ∠P -/
def angleBisectorP (t : Triangle) : AngleBisectorEq := sorry

theorem angle_bisector_sum (t : Triangle) 
  (h : t.P = (-7, 4) ∧ t.Q = (-14, -20) ∧ t.R = (2, -8)) : 
  let eq := angleBisectorP t
  eq.a + eq.c = 40 := by sorry

end angle_bisector_sum_l3172_317264


namespace ski_trips_theorem_l3172_317216

/-- Represents the ski lift problem -/
structure SkiLiftProblem where
  lift_time : ℕ  -- Time to ride the lift up (in minutes)
  ski_time : ℕ   -- Time to ski down (in minutes)
  known_trips : ℕ  -- Known number of trips in 2 hours
  known_hours : ℕ  -- Known number of hours for known_trips

/-- Calculates the number of ski trips possible in a given number of hours -/
def ski_trips (problem : SkiLiftProblem) (hours : ℕ) : ℕ :=
  3 * hours

/-- Theorem stating the relationship between hours and number of ski trips -/
theorem ski_trips_theorem (problem : SkiLiftProblem) (hours : ℕ) :
  problem.lift_time = 15 →
  problem.ski_time = 5 →
  problem.known_trips = 6 →
  problem.known_hours = 2 →
  ski_trips problem hours = 3 * hours :=
by
  sorry

#check ski_trips_theorem

end ski_trips_theorem_l3172_317216


namespace rooks_diagonal_move_l3172_317272

/-- Represents a position on an 8x8 chessboard -/
structure Position :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents a configuration of 8 rooks on an 8x8 chessboard -/
structure RookConfiguration :=
  (positions : Fin 8 → Position)
  (no_attacks : ∀ i j, i ≠ j → 
    (positions i).row ≠ (positions j).row ∧ 
    (positions i).col ≠ (positions j).col)

/-- Checks if a position is adjacent diagonally to another position -/
def is_adjacent_diagonal (p1 p2 : Position) : Prop :=
  (p1.row.val + 1 = p2.row.val ∧ p1.col.val + 1 = p2.col.val) ∨
  (p1.row.val + 1 = p2.row.val ∧ p1.col.val = p2.col.val + 1) ∨
  (p1.row.val = p2.row.val + 1 ∧ p1.col.val + 1 = p2.col.val) ∨
  (p1.row.val = p2.row.val + 1 ∧ p1.col.val = p2.col.val + 1)

/-- The main theorem to be proved -/
theorem rooks_diagonal_move (initial : RookConfiguration) :
  ∃ (final : RookConfiguration),
    ∀ i, is_adjacent_diagonal (initial.positions i) (final.positions i) :=
sorry

end rooks_diagonal_move_l3172_317272


namespace complex_sum_representation_l3172_317219

theorem complex_sum_representation : ∃ (r θ : ℝ), 
  15 * Complex.exp (Complex.I * (π / 7)) + 15 * Complex.exp (Complex.I * (9 * π / 14)) = r * Complex.exp (Complex.I * θ) ∧ 
  r = 15 * Real.sqrt 2 ∧ 
  θ = 11 * π / 28 := by
  sorry

end complex_sum_representation_l3172_317219


namespace max_value_cube_ratio_l3172_317240

theorem max_value_cube_ratio (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x + y)^3 / (x^3 + y^3) ≤ 4 := by
  sorry

end max_value_cube_ratio_l3172_317240


namespace product_357_sum_28_l3172_317259

theorem product_357_sum_28 (a b c d : ℕ+) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d → 
  a * b * c * d = 357 → 
  a + b + c + d = 28 := by
sorry

end product_357_sum_28_l3172_317259


namespace range_of_m_l3172_317293

theorem range_of_m (x : ℝ) (m : ℝ) : 
  (∃ x ∈ Set.Ioo (π/2) π, 2 * Real.sin x ^ 2 - Real.sqrt 3 * Real.sin (2 * x) + m - 1 = 0) →
  m ∈ Set.Ioo (-2) (-1) :=
sorry

end range_of_m_l3172_317293


namespace six_legs_is_insect_l3172_317203

/-- Represents an animal with a certain number of legs -/
structure Animal where
  legs : ℕ

/-- Definition of an insect based on number of legs -/
def is_insect (a : Animal) : Prop := a.legs = 6

/-- Theorem stating that an animal with 6 legs satisfies the definition of an insect -/
theorem six_legs_is_insect (a : Animal) (h : a.legs = 6) : is_insect a := by
  sorry

end six_legs_is_insect_l3172_317203


namespace gcf_lcm_60_72_l3172_317207

theorem gcf_lcm_60_72 : 
  (Nat.gcd 60 72 = 12) ∧ (Nat.lcm 60 72 = 360) := by
  sorry

end gcf_lcm_60_72_l3172_317207


namespace journey_distance_l3172_317291

theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ)
  (h1 : total_time = 40)
  (h2 : speed1 = 20)
  (h3 : speed2 = 30)
  (h4 : total_time = (distance / 2) / speed1 + (distance / 2) / speed2) :
  distance = 960 :=
by
  sorry

end journey_distance_l3172_317291


namespace football_team_selection_l3172_317295

theorem football_team_selection (n : ℕ) (k : ℕ) :
  let total_students : ℕ := 31
  let team_size : ℕ := 11
  let remaining_students : ℕ := total_students - 2
  (Nat.choose total_students team_size) - (Nat.choose remaining_students team_size) =
    2 * (Nat.choose remaining_students (team_size - 1)) + (Nat.choose remaining_students (team_size - 2)) :=
by sorry

end football_team_selection_l3172_317295


namespace inez_remaining_money_l3172_317278

def initial_amount : ℕ := 150
def pad_cost : ℕ := 50

theorem inez_remaining_money :
  let skate_cost : ℕ := initial_amount / 2
  let after_skates : ℕ := initial_amount - skate_cost
  let remaining : ℕ := after_skates - pad_cost
  remaining = 25 := by sorry

end inez_remaining_money_l3172_317278


namespace abs_sum_min_value_l3172_317247

theorem abs_sum_min_value :
  (∀ x : ℝ, |x + 1| + |2 - x| ≥ 3) ∧
  (∃ x : ℝ, |x + 1| + |2 - x| = 3) :=
by sorry

end abs_sum_min_value_l3172_317247


namespace joe_fruit_probability_l3172_317268

/-- The number of meals Joe has in a day. -/
def num_meals : ℕ := 3

/-- The number of fruit options Joe has for each meal. -/
def num_fruits : ℕ := 4

/-- The probability of choosing a specific fruit for a meal. -/
def prob_single_fruit : ℚ := 1 / num_fruits

/-- The probability of eating the same fruit for all meals. -/
def prob_same_fruit : ℚ := prob_single_fruit ^ num_meals

/-- The probability of eating at least two different kinds of fruit in a day. -/
def prob_different_fruits : ℚ := 1 - (num_fruits * prob_same_fruit)

theorem joe_fruit_probability : prob_different_fruits = 15 / 16 := by
  sorry

end joe_fruit_probability_l3172_317268


namespace divisors_of_2018_or_2019_l3172_317274

theorem divisors_of_2018_or_2019 (h1 : Nat.Prime 673) (h2 : Nat.Prime 1009) :
  (Finset.filter (fun n => n ∣ 2018 ∨ n ∣ 2019) (Finset.range 2020)).card = 7 := by
  sorry

end divisors_of_2018_or_2019_l3172_317274


namespace machine_production_l3172_317202

/-- Given that 4 machines produce x units in 6 days at a constant rate,
    prove that 16 machines will produce 2x units in 3 days at the same rate. -/
theorem machine_production (x : ℝ) : 
  (∃ (rate : ℝ), rate > 0 ∧ 4 * rate * 6 = x) →
  (∃ (output : ℝ), 16 * (x / (4 * 6)) * 3 = output ∧ output = 2 * x) :=
by sorry

end machine_production_l3172_317202


namespace sin_sum_identity_l3172_317228

theorem sin_sum_identity : 
  Real.sin (13 * π / 180) * Real.sin (58 * π / 180) + 
  Real.sin (77 * π / 180) * Real.sin (32 * π / 180) = 
  Real.sqrt 2 / 2 := by
  sorry

end sin_sum_identity_l3172_317228


namespace otimes_four_two_l3172_317214

def otimes (a b : ℝ) : ℝ := 2 * a + 5 * b

theorem otimes_four_two : otimes 4 2 = 18 := by
  sorry

end otimes_four_two_l3172_317214


namespace isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l3172_317270

/-- An isosceles triangle with perimeter 3.74 and leg length 1.5 has a base length of 0.74 -/
theorem isosceles_triangle_base_length : ℝ → Prop :=
  fun base =>
    let perimeter : ℝ := 3.74
    let leg : ℝ := 1.5
    (2 * leg + base = perimeter) → (base = 0.74)

/-- Proof of the theorem -/
theorem isosceles_triangle_base_length_proof : isosceles_triangle_base_length 0.74 := by
  sorry

end isosceles_triangle_base_length_isosceles_triangle_base_length_proof_l3172_317270


namespace hyperbola_asymptote_a_value_l3172_317236

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop := x^2 / 4 - y^2 / a = 1

-- Define the asymptotes of the hyperbola
def asymptote (a : ℝ) (x y : ℝ) : Prop := y = (Real.sqrt a / 2) * x ∨ y = -(Real.sqrt a / 2) * x

-- Theorem statement
theorem hyperbola_asymptote_a_value :
  ∀ a : ℝ, a > 1 →
  asymptote a 2 (Real.sqrt 3) →
  hyperbola a 2 (Real.sqrt 3) →
  a = 3 := by
  sorry

end hyperbola_asymptote_a_value_l3172_317236


namespace colored_paper_count_l3172_317244

theorem colored_paper_count (people : ℕ) (pieces_per_person : ℕ) (leftover : ℕ) : 
  people = 6 → pieces_per_person = 7 → leftover = 3 → 
  people * pieces_per_person + leftover = 45 := by
  sorry

end colored_paper_count_l3172_317244


namespace union_condition_intersection_empty_l3172_317275

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -3 ≤ x ∧ x ≤ 7}
def B (t : ℝ) : Set ℝ := {x : ℝ | t + 1 ≤ x ∧ x ≤ 2*t - 2}

-- Statement 1: A ∪ B = A if and only if t ∈ (-∞, 9/2]
theorem union_condition (t : ℝ) : A ∪ B t = A ↔ t ≤ 9/2 := by sorry

-- Statement 2: A ∩ B = ∅ if and only if t ∈ (-∞, 3) ∪ (6, +∞)
theorem intersection_empty (t : ℝ) : A ∩ B t = ∅ ↔ t < 3 ∨ t > 6 := by sorry

end union_condition_intersection_empty_l3172_317275


namespace line_slope_equals_k_l3172_317229

/-- Given a line passing through points (-1, -4) and (5, k), 
    if the slope of the line is equal to k, then k = 4/5 -/
theorem line_slope_equals_k (k : ℚ) : 
  (k - (-4)) / (5 - (-1)) = k → k = 4/5 := by
  sorry

end line_slope_equals_k_l3172_317229


namespace book_club_groups_l3172_317234

theorem book_club_groups (n m : ℕ) (hn : n = 7) (hm : m = 4) :
  Nat.choose n m = 35 := by
  sorry

end book_club_groups_l3172_317234


namespace average_lunchmeat_price_l3172_317252

def joan_bologna_weight : ℝ := 3
def joan_bologna_price : ℝ := 2.80
def grant_pastrami_weight : ℝ := 2
def grant_pastrami_price : ℝ := 1.80

theorem average_lunchmeat_price :
  let total_weight := joan_bologna_weight + grant_pastrami_weight
  let total_cost := joan_bologna_weight * joan_bologna_price + grant_pastrami_weight * grant_pastrami_price
  total_cost / total_weight = 2.40 := by
sorry

end average_lunchmeat_price_l3172_317252


namespace sqrt_x6_plus_x4_l3172_317288

theorem sqrt_x6_plus_x4 (x : ℝ) : Real.sqrt (x^6 + x^4) = |x|^2 * Real.sqrt (x^2 + 1) := by sorry

end sqrt_x6_plus_x4_l3172_317288


namespace complex_equation_difference_l3172_317262

theorem complex_equation_difference (m n : ℝ) (i : ℂ) (h : i * i = -1) :
  (m + 2 * i) / i = n + i → n - m = 3 := by
  sorry

end complex_equation_difference_l3172_317262


namespace optimal_route_unchanged_for_given_network_l3172_317218

/-- Represents the transportation network of a country -/
structure TransportNetwork where
  num_cities : Nat
  capital_travel_time : Real
  city_connection_time : Real
  initial_transfer_time : Real
  reduced_transfer_time : Real

/-- Calculates the travel time via the capital -/
def time_via_capital (network : TransportNetwork) (transfer_time : Real) : Real :=
  2 * network.capital_travel_time + transfer_time

/-- Calculates the maximum travel time via cyclic connections -/
def time_via_cycle (network : TransportNetwork) (transfer_time : Real) : Real :=
  5 * network.city_connection_time + 4 * transfer_time

/-- Determines if the optimal route remains unchanged after reducing transfer time -/
def optimal_route_unchanged (network : TransportNetwork) : Prop :=
  let initial_time_via_capital := time_via_capital network network.initial_transfer_time
  let initial_time_via_cycle := time_via_cycle network network.initial_transfer_time
  let reduced_time_via_capital := time_via_capital network network.reduced_transfer_time
  let reduced_time_via_cycle := time_via_cycle network network.reduced_transfer_time
  (initial_time_via_capital ≤ initial_time_via_cycle) ∧
  (reduced_time_via_capital ≤ reduced_time_via_cycle)

theorem optimal_route_unchanged_for_given_network :
  optimal_route_unchanged
    { num_cities := 11
    , capital_travel_time := 7
    , city_connection_time := 3
    , initial_transfer_time := 2
    , reduced_transfer_time := 1.5 } := by
  sorry

end optimal_route_unchanged_for_given_network_l3172_317218


namespace condition_iff_in_solution_set_l3172_317246

/-- A pair of positive integers (x, y) satisfies the given condition -/
def satisfies_condition (x y : ℕ+) : Prop :=
  ∃ k : ℕ, x^2 * y + x = k * (x * y^2 + 7)

/-- The set of all pairs (x, y) that satisfy the condition -/
def solution_set : Set (ℕ+ × ℕ+) :=
  {p | p = (7, 1) ∨ p = (14, 1) ∨ p = (35, 1) ∨ p = (7, 2) ∨
       ∃ k : ℕ+, p = (7 * k, 7)}

/-- The main theorem stating the equivalence between the condition and the solution set -/
theorem condition_iff_in_solution_set (x y : ℕ+) :
  satisfies_condition x y ↔ (x, y) ∈ solution_set := by
  sorry

end condition_iff_in_solution_set_l3172_317246


namespace exam_probability_l3172_317210

/-- The probability of passing the exam -/
def prob_pass : ℚ := 4/7

/-- The probability of not passing the exam -/
def prob_not_pass : ℚ := 1 - prob_pass

theorem exam_probability : prob_not_pass = 3/7 := by
  sorry

end exam_probability_l3172_317210


namespace sqrt_c_value_l3172_317287

theorem sqrt_c_value (a b c : ℝ) :
  (a^2 + 2020 * a + c = 0) →
  (b^2 + 2020 * b + c = 0) →
  (a / b + b / a = 98) →
  Real.sqrt c = 202 := by
sorry

end sqrt_c_value_l3172_317287


namespace intersection_points_correct_l3172_317211

/-- Parallelogram with given dimensions divided into three equal areas -/
structure EqualAreaParallelogram where
  AB : ℝ
  AD : ℝ
  BE : ℝ
  h_AB : AB = 153
  h_AD : AD = 180
  h_BE : BE = 135

/-- The points where perpendicular lines intersect AD -/
def intersection_points (p : EqualAreaParallelogram) : ℝ × ℝ :=
  (96, 156)

/-- Theorem stating that the intersection points are correct -/
theorem intersection_points_correct (p : EqualAreaParallelogram) :
  intersection_points p = (96, 156) :=
sorry

end intersection_points_correct_l3172_317211


namespace regular_polygon_140_degrees_has_9_sides_l3172_317206

/-- A regular polygon with interior angles of 140 degrees has 9 sides -/
theorem regular_polygon_140_degrees_has_9_sides :
  ∀ n : ℕ,
  n > 2 →
  (∀ angle : ℝ, angle = 140 →
    (180 * (n - 2) : ℝ) = n * angle) →
  n = 9 := by
sorry

end regular_polygon_140_degrees_has_9_sides_l3172_317206


namespace triangle_equilateral_condition_l3172_317298

/-- If in a triangle ABC, a/cos(A) = b/cos(B) = c/cos(C), then the triangle is equilateral -/
theorem triangle_equilateral_condition (A B C : ℝ) (a b c : ℝ) 
  (h_triangle : A + B + C = π)
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_condition : a / Real.cos A = b / Real.cos B ∧ b / Real.cos B = c / Real.cos C) :
  A = B ∧ B = C := by
  sorry

end triangle_equilateral_condition_l3172_317298


namespace line_property_l3172_317213

/-- Given a line passing through points (2, -1) and (-1, 6), prove that 3m - 2b = -19 where m is the slope and b is the y-intercept -/
theorem line_property (m b : ℚ) : 
  (∀ (x y : ℚ), (x = 2 ∧ y = -1) ∨ (x = -1 ∧ y = 6) → y = m * x + b) →
  3 * m - 2 * b = -19 := by
  sorry

end line_property_l3172_317213


namespace fraction_to_seventh_power_l3172_317226

theorem fraction_to_seventh_power : (2 / 5 : ℚ) ^ 7 = 128 / 78125 := by
  sorry

end fraction_to_seventh_power_l3172_317226


namespace composition_of_even_is_even_l3172_317230

-- Define an even function
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

-- State the theorem
theorem composition_of_even_is_even (f : ℝ → ℝ) (h : IsEven f) :
  IsEven (fun x ↦ f (f x)) := by
  sorry

end composition_of_even_is_even_l3172_317230


namespace sixth_root_equation_solution_l3172_317254

theorem sixth_root_equation_solution (x : ℝ) :
  (x^2 * (x^4)^(1/3))^(1/6) = 4 ↔ x = 4^(18/5) := by sorry

end sixth_root_equation_solution_l3172_317254


namespace square_side_length_l3172_317257

theorem square_side_length (area : Real) (side : Real) : 
  area = 25 → side * side = area → side = 5 := by
  sorry

end square_side_length_l3172_317257


namespace range_of_a_theorem_l3172_317220

-- Define propositions p and q
def p (a : ℝ) : Prop := ∀ x : ℝ, 0 < a * x^2 - x + 1/16 * a

def q (a : ℝ) : Prop := ∀ x y : ℝ, x < y → (a - 3/2)^y < (a - 3/2)^x

-- Define the range of a
def range_of_a (a : ℝ) : Prop := (3/2 < a ∧ a ≤ 2) ∨ a ≥ 5/2

-- State the theorem
theorem range_of_a_theorem (a : ℝ) :
  ((p a ∨ q a) ∧ ¬(p a ∧ q a)) → range_of_a a :=
by sorry

end range_of_a_theorem_l3172_317220


namespace asafa_arrives_5_min_after_florence_l3172_317231

/-- Represents a point in the route -/
inductive Point | P | Q | R | S

/-- Represents a runner -/
inductive Runner | Asafa | Florence

/-- The speed of a runner in km/h -/
def speed (r : Runner) : ℝ :=
  match r with
  | Runner.Asafa => 21
  | Runner.Florence => 16.8  -- This is derived, not given directly

/-- The distance between two points in km -/
def distance (p1 p2 : Point) : ℝ :=
  match p1, p2 with
  | Point.P, Point.Q => 8
  | Point.Q, Point.R => 15
  | Point.R, Point.S => 7
  | Point.P, Point.R => 17  -- This is derived, not given directly
  | _, _ => 0  -- For all other combinations

/-- The time difference in minutes between Florence and Asafa arriving at point R -/
def time_difference_at_R : ℝ := 5

/-- The theorem to be proved -/
theorem asafa_arrives_5_min_after_florence :
  let total_distance_asafa := distance Point.P Point.Q + distance Point.Q Point.R + distance Point.R Point.S
  let total_distance_florence := distance Point.P Point.R + distance Point.R Point.S
  let total_time := total_distance_asafa / speed Runner.Asafa
  let time_asafa_RS := distance Point.R Point.S / speed Runner.Asafa
  let time_florence_RS := distance Point.R Point.S / speed Runner.Florence
  time_florence_RS - time_asafa_RS = time_difference_at_R / 60 := by
  sorry

end asafa_arrives_5_min_after_florence_l3172_317231


namespace expansion_temperature_difference_l3172_317258

-- Define the initial conditions and coefficients
def initial_length : ℝ := 2
def initial_temp : ℝ := 80
def alpha_iron : ℝ := 0.0000118
def alpha_zinc : ℝ := 0.000031
def length_difference : ℝ := 0.0015

-- Define the function for the length of a rod at temperature x
def rod_length (alpha : ℝ) (x : ℝ) : ℝ :=
  initial_length * (1 + alpha * (x - initial_temp))

-- Define the theorem to prove
theorem expansion_temperature_difference :
  ∃ (x₁ x₂ : ℝ),
    x₁ ≠ x₂ ∧
    (rod_length alpha_zinc x₁ - rod_length alpha_iron x₁ = length_difference ∨
     rod_length alpha_iron x₁ - rod_length alpha_zinc x₁ = length_difference) ∧
    (rod_length alpha_zinc x₂ - rod_length alpha_iron x₂ = length_difference ∨
     rod_length alpha_iron x₂ - rod_length alpha_zinc x₂ = length_difference) ∧
    ((x₁ = 119 ∧ x₂ = 41) ∨ (x₁ = 41 ∧ x₂ = 119)) :=
sorry

end expansion_temperature_difference_l3172_317258


namespace march_first_is_wednesday_l3172_317283

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a date in March -/
structure MarchDate where
  day : Nat
  dayOfWeek : DayOfWeek

/-- Function to get the day of the week for a given number of days before a reference day -/
def daysBefore (referenceDay : DayOfWeek) (days : Nat) : DayOfWeek :=
  sorry

theorem march_first_is_wednesday (march13 : MarchDate) 
  (h : march13.day = 13 ∧ march13.dayOfWeek = DayOfWeek.Monday) :
  ∃ (march1 : MarchDate), march1.day = 1 ∧ march1.dayOfWeek = DayOfWeek.Wednesday :=
  sorry

end march_first_is_wednesday_l3172_317283


namespace marlon_gift_card_balance_l3172_317279

def gift_card_balance (initial_balance : ℝ) (monday_fraction : ℝ) (tuesday_fraction : ℝ) : ℝ :=
  let remaining_after_monday := initial_balance * (1 - monday_fraction)
  remaining_after_monday * (1 - tuesday_fraction)

theorem marlon_gift_card_balance :
  gift_card_balance 200 (1/2) (1/4) = 75 := by
  sorry

end marlon_gift_card_balance_l3172_317279


namespace exponent_division_l3172_317204

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^6 / a^2 = a^4 := by
  sorry

end exponent_division_l3172_317204


namespace ellipse_line_intersection_l3172_317233

/-- Given an ellipse C and a line l, if l intersects C at two points with a specific distance, then the y-intercept of l is 0. -/
theorem ellipse_line_intersection (x y m : ℝ) : 
  (4 * x^2 + y^2 = 1) →  -- Ellipse equation
  (y = x + m) →          -- Line equation
  (∃ A B : ℝ × ℝ, A ≠ B ∧ 
    (4 * A.1^2 + A.2^2 = 1) ∧ 
    (4 * B.1^2 + B.2^2 = 1) ∧ 
    (A.2 = A.1 + m) ∧ 
    (B.2 = B.1 + m) ∧ 
    ((A.1 - B.1)^2 + (A.2 - B.2)^2 = (2 * Real.sqrt 10 / 5)^2)) →
  m = 0 := by
sorry

end ellipse_line_intersection_l3172_317233


namespace average_mark_first_class_l3172_317284

theorem average_mark_first_class 
  (n1 : ℕ) (n2 : ℕ) (avg2 : ℝ) (avg_total : ℝ)
  (h1 : n1 = 30)
  (h2 : n2 = 50)
  (h3 : avg2 = 80)
  (h4 : avg_total = 65) :
  (n1 + n2) * avg_total = n1 * ((n1 + n2) * avg_total - n2 * avg2) / n1 + n2 * avg2 :=
by sorry

end average_mark_first_class_l3172_317284


namespace x_x_minus_3_is_quadratic_l3172_317285

/-- Definition of a quadratic equation in one variable -/
def is_quadratic_equation_in_one_variable (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The equation x(x-3) = 0 -/
def f (x : ℝ) : ℝ := x * (x - 3)

/-- Theorem: x(x-3) = 0 is a quadratic equation in one variable -/
theorem x_x_minus_3_is_quadratic : is_quadratic_equation_in_one_variable f := by
  sorry


end x_x_minus_3_is_quadratic_l3172_317285


namespace expression_equals_twenty_times_ten_to_1234_l3172_317205

theorem expression_equals_twenty_times_ten_to_1234 :
  (2^1234 + 5^1235)^2 - (2^1234 - 5^1235)^2 = 20 * 10^1234 := by
sorry

end expression_equals_twenty_times_ten_to_1234_l3172_317205


namespace BF_length_is_four_l3172_317286

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D E F : ℝ × ℝ)

-- Define the properties of the quadrilateral
def is_right_angled_at_A_and_C (q : Quadrilateral) : Prop := sorry
def E_and_F_on_AC (q : Quadrilateral) : Prop := sorry
def DE_perpendicular_to_AC (q : Quadrilateral) : Prop := sorry
def BF_perpendicular_to_AC (q : Quadrilateral) : Prop := sorry

-- Define the given lengths
def AE_length (q : Quadrilateral) : ℝ := 4
def DE_length (q : Quadrilateral) : ℝ := 6
def CE_length (q : Quadrilateral) : ℝ := 6

-- Define the length of BF
def BF_length (q : Quadrilateral) : ℝ := sorry

-- Theorem statement
theorem BF_length_is_four (q : Quadrilateral) 
  (h1 : is_right_angled_at_A_and_C q)
  (h2 : E_and_F_on_AC q)
  (h3 : DE_perpendicular_to_AC q)
  (h4 : BF_perpendicular_to_AC q) :
  BF_length q = 4 := by sorry

end BF_length_is_four_l3172_317286


namespace sqrt_fraction_simplification_l3172_317290

theorem sqrt_fraction_simplification (x : ℝ) (h : x < -1) :
  Real.sqrt ((x + 1) / (2 - (x + 2) / x)) = Real.sqrt (|x^2 + x| / |x - 2|) := by
  sorry

end sqrt_fraction_simplification_l3172_317290


namespace james_and_louise_ages_l3172_317299

/-- Proves that given the conditions about James and Louise's ages, the sum of their current ages is 25. -/
theorem james_and_louise_ages (J L : ℕ) : 
  J = L + 5 ∧ 
  J + 6 = 3 * (L - 3) →
  J + L = 25 := by
  sorry

end james_and_louise_ages_l3172_317299


namespace second_item_cost_price_l3172_317289

/-- Given two items sold together for 432 yuan, where one item is sold at a 20% loss
    and the combined sale results in a 20% profit, prove that the cost price of the second item is 90 yuan. -/
theorem second_item_cost_price (total_selling_price : ℝ) (loss_percentage : ℝ) (profit_percentage : ℝ) 
  (h1 : total_selling_price = 432)
  (h2 : loss_percentage = 0.20)
  (h3 : profit_percentage = 0.20) :
  ∃ (cost_price_1 cost_price_2 : ℝ),
    cost_price_1 * (1 - loss_percentage) = total_selling_price / 2 ∧
    total_selling_price = (cost_price_1 + cost_price_2) * (1 + profit_percentage) ∧
    cost_price_2 = 90 := by
  sorry

end second_item_cost_price_l3172_317289


namespace gcd_seven_digit_special_l3172_317201

def seven_digit_special (n : ℕ) : ℕ := 1001000 * n + n / 100

theorem gcd_seven_digit_special :
  ∃ (k : ℕ), k > 0 ∧ ∀ (n : ℕ), 100 ≤ n ∧ n < 1000 → 
    (k ∣ seven_digit_special n) ∧
    ∀ (m : ℕ), m > k ∧ (∀ (p : ℕ), 100 ≤ p ∧ p < 1000 → m ∣ seven_digit_special p) → False :=
by sorry

end gcd_seven_digit_special_l3172_317201


namespace slope_equation_l3172_317280

theorem slope_equation (m : ℝ) (h1 : m > 0) 
  (h2 : (m - 3) / (1 - m) = m) : m = Real.sqrt 3 := by
  sorry

end slope_equation_l3172_317280


namespace dhoni_doll_expenditure_l3172_317256

theorem dhoni_doll_expenditure :
  ∀ (total_spent : ℕ) (large_price small_price : ℕ),
    large_price = 6 →
    small_price = large_price - 2 →
    (total_spent / small_price) - (total_spent / large_price) = 25 →
    total_spent = 300 := by
  sorry

end dhoni_doll_expenditure_l3172_317256


namespace plant_growth_mean_l3172_317238

theorem plant_growth_mean (measurements : List ℝ) 
  (h1 : measurements.length = 15)
  (h2 : (measurements.filter (λ x => 10 ≤ x ∧ x < 20)).length = 3)
  (h3 : (measurements.filter (λ x => 20 ≤ x ∧ x < 30)).length = 7)
  (h4 : (measurements.filter (λ x => 30 ≤ x ∧ x < 40)).length = 5)
  (h5 : measurements.sum = 401) :
  measurements.sum / measurements.length = 401 / 15 := by
sorry

end plant_growth_mean_l3172_317238


namespace system_solution_iff_b_in_range_l3172_317250

/-- The system of equations has a solution for any real a if and only if 0 ≤ b ≤ 2 -/
theorem system_solution_iff_b_in_range (b : ℝ) :
  (∀ a : ℝ, ∃ x y : ℝ, x^2 - 2*x + y^2 = 0 ∧ a*x + y = a*b) ↔ 0 ≤ b ∧ b ≤ 2 := by
  sorry

end system_solution_iff_b_in_range_l3172_317250


namespace parallelogram_diagonals_l3172_317261

/-- Represents a parallelogram with given side length and diagonal lengths -/
structure Parallelogram where
  side : ℝ
  diagonal1 : ℝ
  diagonal2 : ℝ

/-- Check if three lengths can form a triangle -/
def canFormTriangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- The main theorem to prove -/
theorem parallelogram_diagonals (p : Parallelogram) : 
  p.side = 10 →
  (p.diagonal1 = 20 ∧ p.diagonal2 = 30) ↔
  (canFormTriangle (p.side) (p.diagonal1 / 2) (p.diagonal2 / 2) ∧
   ¬(canFormTriangle p.side 2 3) ∧
   ¬(canFormTriangle p.side 3 4) ∧
   ¬(canFormTriangle p.side 4 6)) :=
by sorry

end parallelogram_diagonals_l3172_317261


namespace cubic_root_identity_l3172_317208

theorem cubic_root_identity (a b : ℝ) (h1 : a ≠ b) (h2 : (Real.rpow a (1/3) + Real.rpow b (1/3))^3 = a^2 * b^2) : 
  (3*a + 1)*(3*b + 1) - 3*a^2*b^2 = 1 := by sorry

end cubic_root_identity_l3172_317208


namespace amount_from_cars_and_buses_is_309_l3172_317227

/-- Calculates the amount raised from cars and buses given the total amount raised and the amounts from other vehicle types. -/
def amount_from_cars_and_buses (total_raised : ℕ) (suv_charge truck_charge motorcycle_charge : ℕ) (num_suvs num_trucks num_motorcycles : ℕ) : ℕ :=
  total_raised - (suv_charge * num_suvs + truck_charge * num_trucks + motorcycle_charge * num_motorcycles)

/-- Theorem stating that the amount raised from cars and buses is $309. -/
theorem amount_from_cars_and_buses_is_309 :
  amount_from_cars_and_buses 500 12 10 15 3 8 5 = 309 := by
  sorry

end amount_from_cars_and_buses_is_309_l3172_317227


namespace s_value_l3172_317241

theorem s_value (n : ℝ) (s : ℝ) (h1 : n ≠ 0) 
  (h2 : s = (20 / (2^(2*n+4) + 2^(2*n+2)))^(1/n)) : s = 1/4 := by
sorry

end s_value_l3172_317241


namespace line_segment_representation_l3172_317239

/-- Represents the scale factor of the drawing -/
def scale_factor : ℝ := 800

/-- Represents the length of the line segment in the drawing (in inches) -/
def line_segment_length : ℝ := 4.75

/-- Calculates the actual length in feet represented by a given length in the drawing -/
def actual_length (drawing_length : ℝ) : ℝ := drawing_length * scale_factor

/-- Theorem stating that a 4.75-inch line segment on the scale drawing represents 3800 feet -/
theorem line_segment_representation : 
  actual_length line_segment_length = 3800 := by sorry

end line_segment_representation_l3172_317239


namespace geometric_sequence_sum_l3172_317271

theorem geometric_sequence_sum (n : ℕ) (a : ℕ → ℝ) (S : ℕ → ℝ) (r : ℝ) :
  (∀ k, k ≥ 1 → S k = 3^k + r) →
  (∀ k, k ≥ 2 → a k = S k - S (k-1)) →
  (∀ k, k ≥ 2 → a k = 2 * 3^(k-1)) →
  a 1 = S 1 →
  r = -1 :=
by sorry

end geometric_sequence_sum_l3172_317271


namespace circle_trajectory_and_tangent_line_l3172_317269

-- Define the circles M and N
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := (x - 1)^2 + y^2 = 9

-- Define the moving circle P
def circle_P (x y r : ℝ) : Prop := (x - 2)^2 + y^2 = r^2

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 3 = 1

-- Define the tangent line l
def line_l (x y k : ℝ) : Prop := y = k * (x + 4)

-- Theorem statement
theorem circle_trajectory_and_tangent_line :
  ∀ (x y r k : ℝ),
  (∃ (x₁ y₁ : ℝ), circle_M x₁ y₁ ∧ circle_P (x₁ - 1) y₁ r) →
  (∃ (x₂ y₂ : ℝ), circle_N x₂ y₂ ∧ circle_P (x₂ + 1) y₂ (3 - r)) →
  curve_C x y →
  line_l x y k →
  (∃ (x₃ y₃ : ℝ), circle_M x₃ y₃ ∧ line_l x₃ y₃ k) →
  (∃ (x₄ y₄ : ℝ), circle_P x₄ y₄ 2 ∧ line_l x₄ y₄ k) →
  (∀ (x₅ y₅ : ℝ), curve_C x₅ y₅ → line_l x₅ y₅ k → 
    (x₅ - x)^2 + (y₅ - y)^2 ≤ (18/7)^2) :=
by
  sorry

end circle_trajectory_and_tangent_line_l3172_317269


namespace sufficient_not_necessary_l3172_317253

theorem sufficient_not_necessary (x y : ℝ) :
  (x < y ∧ y < 0 → x^2 > y^2) ∧
  ∃ x y, x^2 > y^2 ∧ ¬(x < y ∧ y < 0) := by
sorry

end sufficient_not_necessary_l3172_317253


namespace proportion_third_number_l3172_317292

theorem proportion_third_number (y : ℝ) : 
  (0.75 : ℝ) / 1.05 = y / 7 → y = 5 := by
  sorry

end proportion_third_number_l3172_317292


namespace concert_ticket_price_l3172_317249

/-- The price of each ticket in dollars -/
def ticket_price : ℚ := 4

/-- The total number of tickets bought -/
def total_tickets : ℕ := 8

/-- The total amount spent in dollars -/
def total_spent : ℚ := 32

theorem concert_ticket_price : 
  ticket_price * total_tickets = total_spent :=
sorry

end concert_ticket_price_l3172_317249


namespace sector_circumradius_l3172_317267

theorem sector_circumradius (r : ℝ) (θ : ℝ) (h1 : r = 8) (h2 : θ = 2 * π / 3) :
  let R := r / (2 * Real.sin (θ / 2))
  R = 8 * Real.sqrt 3 / 3 := by
sorry

end sector_circumradius_l3172_317267


namespace a_range_for_increasing_f_l3172_317276

/-- A cubic function f(x) that is increasing on the entire real line. -/
noncomputable def f (a : ℝ) : ℝ → ℝ := λ x ↦ x^3 + a*x^2 + 7*a*x

/-- The property that f is increasing on the entire real line. -/
def is_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x < f y

/-- The theorem stating the range of a for which f is increasing. -/
theorem a_range_for_increasing_f :
  ∀ a : ℝ, is_increasing (f a) ↔ 0 ≤ a ∧ a ≤ 21 :=
sorry

end a_range_for_increasing_f_l3172_317276


namespace difference_of_squares_l3172_317243

theorem difference_of_squares (n : ℝ) : n^2 - 9 = (n + 3) * (n - 3) := by
  sorry

end difference_of_squares_l3172_317243


namespace b_payment_is_360_l3172_317297

/-- Represents the payment for a group of horses in a pasture -/
structure Payment where
  horses : ℕ
  months : ℕ
  amount : ℚ

/-- Calculates the total horse-months for a payment -/
def horse_months (p : Payment) : ℕ := p.horses * p.months

/-- Theorem: Given the conditions of the pasture rental, B's payment is Rs. 360 -/
theorem b_payment_is_360 
  (total_rent : ℚ)
  (a_payment : Payment)
  (b_payment : Payment)
  (c_payment : Payment)
  (h1 : total_rent = 870)
  (h2 : a_payment.horses = 12 ∧ a_payment.months = 8)
  (h3 : b_payment.horses = 16 ∧ b_payment.months = 9)
  (h4 : c_payment.horses = 18 ∧ c_payment.months = 6) :
  b_payment.amount = 360 := by
  sorry

end b_payment_is_360_l3172_317297
