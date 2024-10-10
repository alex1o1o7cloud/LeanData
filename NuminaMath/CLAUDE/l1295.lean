import Mathlib

namespace units_digit_of_18_power_l1295_129509

theorem units_digit_of_18_power : ∃ n : ℕ, (18^(18*(7^7))) % 10 = 4 := by
  sorry

end units_digit_of_18_power_l1295_129509


namespace negative_reciprocal_equality_l1295_129503

theorem negative_reciprocal_equality (a : ℝ) (ha : a ≠ 0) :
  -(1 / a) = (-1) / a := by
  sorry

end negative_reciprocal_equality_l1295_129503


namespace perpendicular_vectors_k_value_l1295_129590

/-- Given vectors a and b in ℝ², prove that if k*a + b is perpendicular to a - 3*b, then k = 19 -/
theorem perpendicular_vectors_k_value (a b : ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 2))
    (h2 : b = (-3, 2))
    (h3 : (k * a.1 + b.1, k * a.2 + b.2) • (a.1 - 3 * b.1, a.2 - 3 * b.2) = 0) :
  k = 19 := by
  sorry

end perpendicular_vectors_k_value_l1295_129590


namespace f_of_one_equals_one_l1295_129588

theorem f_of_one_equals_one (f : ℝ → ℝ) (h : ∀ x, f (x + 1) = Real.cos x) : f 1 = 1 := by
  sorry

end f_of_one_equals_one_l1295_129588


namespace liangliang_speed_l1295_129563

theorem liangliang_speed (initial_distance : ℝ) (remaining_distance : ℝ) (time : ℝ) (mingming_speed : ℝ) :
  initial_distance = 3000 →
  remaining_distance = 2900 →
  time = 20 →
  mingming_speed = 80 →
  ∃ (liangliang_speed : ℝ), (liangliang_speed = 75 ∨ liangliang_speed = 85) ∧
    (initial_distance - remaining_distance = (mingming_speed - liangliang_speed) * time) :=
by sorry

end liangliang_speed_l1295_129563


namespace probability_second_yellow_ball_l1295_129577

def initial_white_balls : ℕ := 5
def initial_yellow_balls : ℕ := 3

def remaining_white_balls : ℕ := initial_white_balls
def remaining_yellow_balls : ℕ := initial_yellow_balls - 1

def total_remaining_balls : ℕ := remaining_white_balls + remaining_yellow_balls

theorem probability_second_yellow_ball :
  (remaining_yellow_balls : ℚ) / total_remaining_balls = 2 / 7 := by
  sorry

end probability_second_yellow_ball_l1295_129577


namespace males_band_not_orchestra_l1295_129518

/-- Represents the number of students in various categories of the school's music program -/
structure MusicProgram where
  female_band : ℕ
  male_band : ℕ
  female_orchestra : ℕ
  male_orchestra : ℕ
  female_both : ℕ
  left_band : ℕ
  total_either : ℕ

/-- Theorem stating the number of males in the band who are not in the orchestra -/
theorem males_band_not_orchestra (mp : MusicProgram) : 
  mp.female_band = 120 →
  mp.male_band = 90 →
  mp.female_orchestra = 70 →
  mp.male_orchestra = 110 →
  mp.female_both = 55 →
  mp.left_band = 10 →
  mp.total_either = 250 →
  mp.male_band - (mp.male_band + mp.male_orchestra - (mp.total_either - ((mp.female_band + mp.female_orchestra - mp.female_both) + mp.left_band))) = 15 := by
  sorry


end males_band_not_orchestra_l1295_129518


namespace monthly_income_A_l1295_129528

/-- Given the average monthly incomes of pairs of individuals, 
    prove that the monthly income of A is 4000. -/
theorem monthly_income_A (a b c : ℝ) 
  (avg_ab : (a + b) / 2 = 5050)
  (avg_bc : (b + c) / 2 = 6250)
  (avg_ac : (a + c) / 2 = 5200) :
  a = 4000 := by
  sorry

end monthly_income_A_l1295_129528


namespace cone_lateral_surface_area_l1295_129557

/-- A cone with an isosceles right triangle cross-section and volume 8π/3 has lateral surface area 4√2π -/
theorem cone_lateral_surface_area (V : ℝ) (r h l : ℝ) : 
  V = (8 / 3) * Real.pi →  -- Volume condition
  V = (1 / 3) * Real.pi * r^2 * h →  -- Volume formula
  r = (Real.sqrt 2 * l) / 2 →  -- Relationship between radius and slant height
  h = r →  -- Height equals radius in isosceles right triangle
  (Real.pi * r * l) = 4 * Real.sqrt 2 * Real.pi := by
sorry

end cone_lateral_surface_area_l1295_129557


namespace arithmetic_sequence_formula_l1295_129504

def f (x : ℝ) : ℝ := x^2 - 3*x + 1

theorem arithmetic_sequence_formula (a : ℝ) (a_n : ℕ → ℝ) :
  (∀ n, a_n (n + 2) - a_n (n + 1) = a_n (n + 1) - a_n n) →  -- arithmetic sequence
  a_n 1 = f (a + 1) →
  a_n 2 = 0 →
  a_n 3 = f (a - 1) →
  ((a = 1 ∧ ∀ n, a_n n = n - 2) ∨ (a = 2 ∧ ∀ n, a_n n = 2 - n)) :=
by sorry

end arithmetic_sequence_formula_l1295_129504


namespace min_distance_line_circle_l1295_129537

/-- The minimum distance between a point on the given line and a point on the given circle is √5/5 -/
theorem min_distance_line_circle :
  let line := {p : ℝ × ℝ | ∃ t : ℝ, p.1 = t ∧ p.2 = 6 - 2*t}
  let circle := {q : ℝ × ℝ | (q.1 - 1)^2 + (q.2 + 2)^2 = 5}
  ∃ d : ℝ, d = Real.sqrt 5 / 5 ∧
    ∀ p ∈ line, ∀ q ∈ circle,
      Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) ≥ d ∧
      ∃ p' ∈ line, ∃ q' ∈ circle,
        Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) = d :=
by sorry

end min_distance_line_circle_l1295_129537


namespace sum_of_even_integers_between_0_and_18_l1295_129568

theorem sum_of_even_integers_between_0_and_18 : 
  (Finset.filter (fun n => n % 2 = 0) (Finset.range 18)).sum id = 72 := by
  sorry

end sum_of_even_integers_between_0_and_18_l1295_129568


namespace chess_tournament_participants_l1295_129536

theorem chess_tournament_participants (n : ℕ) : 
  (∃ (y : ℚ), 2 * y + n * y = (n + 2) * (n + 1) / 2) → 
  (n = 7 ∨ n = 14) := by
sorry

end chess_tournament_participants_l1295_129536


namespace intersection_of_B_and_complement_of_A_l1295_129598

def U : Set Int := {-1, 0, 1, 2, 3, 4, 5}
def A : Set Int := {1, 2, 5}
def B : Set Int := {0, 1, 2, 3}

theorem intersection_of_B_and_complement_of_A : B ∩ (U \ A) = {0, 3} := by
  sorry

end intersection_of_B_and_complement_of_A_l1295_129598


namespace lines_parallel_or_skew_l1295_129525

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (plane_parallel : Plane → Plane → Prop)

-- Define the subset relation for lines and planes
variable (line_in_plane : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (line_parallel : Line → Line → Prop)

-- Define the skew relation for lines
variable (line_skew : Line → Line → Prop)

-- Theorem statement
theorem lines_parallel_or_skew
  (α β : Plane) (a b : Line)
  (h_parallel : plane_parallel α β)
  (h_a_in_α : line_in_plane a α)
  (h_b_in_β : line_in_plane b β) :
  line_parallel a b ∨ line_skew a b :=
sorry

end lines_parallel_or_skew_l1295_129525


namespace no_solutions_to_radical_equation_l1295_129576

theorem no_solutions_to_radical_equation :
  ∀ x : ℝ, x ≥ 2 →
    ¬ (Real.sqrt (x + 7 - 6 * Real.sqrt (x - 2)) + Real.sqrt (x + 12 - 8 * Real.sqrt (x - 2)) = 2) :=
by sorry

end no_solutions_to_radical_equation_l1295_129576


namespace min_odd_integers_l1295_129508

theorem min_odd_integers (a b c d e f : ℤ) 
  (sum1 : a + b = 28)
  (sum2 : a + b + c + d = 46)
  (sum3 : a + b + c + d + e + f = 65) :
  ∃ (x : Finset ℤ), x ⊆ {a, b, c, d, e, f} ∧ x.card = 1 ∧ ∀ i ∈ x, Odd i ∧
  ∀ (y : Finset ℤ), y ⊆ {a, b, c, d, e, f} ∧ (∀ i ∈ y, Odd i) → y.card ≥ 1 :=
by sorry

end min_odd_integers_l1295_129508


namespace bread_cost_l1295_129545

/-- The cost of a loaf of bread, given the costs of ham and cake, and that the combined cost of ham and bread equals the cost of cake. -/
theorem bread_cost (ham_cost cake_cost : ℕ) (h1 : ham_cost = 150) (h2 : cake_cost = 200)
  (h3 : ∃ (bread_cost : ℕ), bread_cost + ham_cost = cake_cost) : 
  ∃ (bread_cost : ℕ), bread_cost = 50 := by
  sorry

end bread_cost_l1295_129545


namespace cyclist_speed_problem_l1295_129560

/-- Proves that given two cyclists on a 45-mile course, one traveling at 16 mph,
    meeting after 1.5 hours, the speed of the other cyclist must be 14 mph. -/
theorem cyclist_speed_problem (course_length : ℝ) (second_cyclist_speed : ℝ) (meeting_time : ℝ)
  (h1 : course_length = 45)
  (h2 : second_cyclist_speed = 16)
  (h3 : meeting_time = 1.5) :
  ∃ (first_cyclist_speed : ℝ),
    first_cyclist_speed * meeting_time + second_cyclist_speed * meeting_time = course_length ∧
    first_cyclist_speed = 14 :=
by sorry

end cyclist_speed_problem_l1295_129560


namespace contrapositive_example_l1295_129516

theorem contrapositive_example (a b : ℝ) :
  (¬(a = 0 → a * b = 0) ↔ (a * b ≠ 0 → a ≠ 0)) := by sorry

end contrapositive_example_l1295_129516


namespace unique_common_point_modulo25_l1295_129586

/-- Given two congruences on modulo 25 graph paper, prove there's exactly one common point with x-coordinate 1 --/
theorem unique_common_point_modulo25 : ∃! p : ℕ × ℕ, 
  p.1 < 25 ∧ 
  p.2 < 25 ∧
  p.2 ≡ 10 * p.1 + 3 [ZMOD 25] ∧ 
  p.2 ≡ p.1^2 + 15 * p.1 + 20 [ZMOD 25] ∧
  p.1 = 1 := by
  sorry

end unique_common_point_modulo25_l1295_129586


namespace calculation_proof_l1295_129583

theorem calculation_proof :
  (- 2^2 * (1/4) + 4 / (4/9) + (-1)^2023 = 7) ∧
  (- 1^4 + |2 - (-3)^2| + (1/2) / (-(3/2)) = 17/3) := by
sorry

end calculation_proof_l1295_129583


namespace eggs_leftover_l1295_129538

def david_eggs : ℕ := 45
def ella_eggs : ℕ := 58
def fiona_eggs : ℕ := 29
def carton_size : ℕ := 10

theorem eggs_leftover :
  (david_eggs + ella_eggs + fiona_eggs) % carton_size = 2 := by
  sorry

end eggs_leftover_l1295_129538


namespace correct_ways_to_spend_l1295_129564

/-- Represents the number of magazines costing 2 yuan -/
def magazines_2yuan : ℕ := 8

/-- Represents the number of magazines costing 1 yuan -/
def magazines_1yuan : ℕ := 3

/-- Represents the total budget in yuan -/
def budget : ℕ := 10

/-- Calculates the number of ways to select magazines to spend exactly the budget -/
def ways_to_spend_budget : ℕ := sorry

theorem correct_ways_to_spend : ways_to_spend_budget = 266 := by sorry

end correct_ways_to_spend_l1295_129564


namespace quadratic_factorization_l1295_129524

theorem quadratic_factorization (x : ℝ) : 16 * x^2 - 40 * x + 25 = (4 * x - 5)^2 := by
  sorry

end quadratic_factorization_l1295_129524


namespace saturday_balls_count_l1295_129515

/-- The number of golf balls Corey wants to find every weekend -/
def weekend_goal : ℕ := 48

/-- The number of golf balls Corey found on Sunday -/
def sunday_balls : ℕ := 18

/-- The number of additional golf balls Corey needs to reach his goal -/
def additional_balls_needed : ℕ := 14

/-- The number of golf balls Corey found on Saturday -/
def saturday_balls : ℕ := weekend_goal - sunday_balls - additional_balls_needed

theorem saturday_balls_count : saturday_balls = 16 := by
  sorry

end saturday_balls_count_l1295_129515


namespace max_inradius_value_l1295_129599

/-- A parabola with equation y^2 = 4x and focus at (1,0) -/
structure Parabola where
  equation : ℝ → ℝ → Prop := fun x y ↦ y^2 = 4*x
  focus : ℝ × ℝ := (1, 0)

/-- The inradius of a triangle formed by a point on the parabola, the focus, and the origin -/
def inradius (p : Parabola) (P : ℝ × ℝ) : ℝ :=
  sorry

/-- The maximum inradius of triangle OPF -/
def max_inradius (p : Parabola) : ℝ :=
  sorry

theorem max_inradius_value (p : Parabola) :
  max_inradius p = 2 * Real.sqrt 3 / 9 :=
sorry

end max_inradius_value_l1295_129599


namespace largest_prime_divisor_l1295_129567

theorem largest_prime_divisor (crayons paper : ℕ) 
  (h1 : crayons = 385) (h2 : paper = 95) : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ crayons ∧ p ∣ paper ∧ 
  ∀ (q : ℕ), Nat.Prime q → q ∣ crayons → q ∣ paper → q ≤ p := by
  sorry

end largest_prime_divisor_l1295_129567


namespace distance_between_circle_centers_l1295_129521

-- Define the rectangle
def rectangle_width : ℝ := 11
def rectangle_height : ℝ := 7

-- Define the circles
def circle_diameter : ℝ := rectangle_height

-- Theorem statement
theorem distance_between_circle_centers : 
  let circle_radius : ℝ := circle_diameter / 2
  let distance : ℝ := rectangle_width - 2 * circle_radius
  distance = 4 := by sorry

end distance_between_circle_centers_l1295_129521


namespace scientific_notation_equality_l1295_129541

/-- Proves that -0.000008691 is equal to -8.691×10^(-6) in scientific notation -/
theorem scientific_notation_equality : ∃ (a : ℝ) (n : ℤ), 
  -0.000008691 = a * 10^n ∧ 
  1 ≤ |a| ∧ 
  |a| < 10 ∧ 
  a = -8.691 ∧ 
  n = -6 := by
  sorry

end scientific_notation_equality_l1295_129541


namespace inequality_implies_a_geq_two_l1295_129534

theorem inequality_implies_a_geq_two (a : ℝ) :
  (∀ x y : ℝ, x^2 + 2*x + a ≥ -y^2 - 2*y) → a ≥ 2 := by
  sorry

end inequality_implies_a_geq_two_l1295_129534


namespace abs_6y_minus_8_not_positive_l1295_129593

theorem abs_6y_minus_8_not_positive (y : ℚ) : ¬(0 < |6 * y - 8|) ↔ y = 4/3 := by
  sorry

end abs_6y_minus_8_not_positive_l1295_129593


namespace gelato_sundae_combinations_l1295_129514

theorem gelato_sundae_combinations :
  (Finset.univ.filter (fun s : Finset (Fin 8) => s.card = 3)).card = 56 := by
  sorry

end gelato_sundae_combinations_l1295_129514


namespace divisibility_by_nine_highest_power_of_three_in_M_l1295_129544

/-- The integer formed by concatenating 2-digit integers from 15 to 95 -/
def M : ℕ := sorry

/-- The sum of digits of M -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A number is divisible by 9 if and only if the sum of its digits is divisible by 9 -/
theorem divisibility_by_nine (n : ℕ) : n % 9 = 0 ↔ sum_of_digits n % 9 = 0 := sorry

/-- The highest power of 3 that divides M is 3^2 -/
theorem highest_power_of_three_in_M : 
  ∃ (k : ℕ), M % (3^3) ≠ 0 ∧ M % (3^2) = 0 := sorry

end divisibility_by_nine_highest_power_of_three_in_M_l1295_129544


namespace sufficient_not_necessary_condition_l1295_129546

theorem sufficient_not_necessary_condition (a : ℝ) :
  (∀ a, a > 1 → 1 / a < 1) ∧ 
  (∃ a, 1 / a < 1 ∧ ¬(a > 1)) :=
by sorry

end sufficient_not_necessary_condition_l1295_129546


namespace garden_transformation_cost_and_area_increase_l1295_129596

/-- Represents a rectangular garden with its dimensions and fence cost -/
structure RectGarden where
  length : ℝ
  width : ℝ
  fence_cost : ℝ

/-- Represents a square garden with its side length and fence cost -/
structure SquareGarden where
  side : ℝ
  fence_cost : ℝ

/-- Calculates the perimeter of a rectangular garden -/
def rect_perimeter (g : RectGarden) : ℝ :=
  2 * (g.length + g.width)

/-- Calculates the area of a rectangular garden -/
def rect_area (g : RectGarden) : ℝ :=
  g.length * g.width

/-- Calculates the total fencing cost of a rectangular garden -/
def rect_fence_cost (g : RectGarden) : ℝ :=
  rect_perimeter g * g.fence_cost

/-- Calculates the area of a square garden -/
def square_area (g : SquareGarden) : ℝ :=
  g.side * g.side

/-- Calculates the total fencing cost of a square garden -/
def square_fence_cost (g : SquareGarden) : ℝ :=
  4 * g.side * g.fence_cost

/-- The main theorem to prove -/
theorem garden_transformation_cost_and_area_increase :
  let rect := RectGarden.mk 60 20 15
  let square := SquareGarden.mk (rect_perimeter rect / 4) 20
  square_fence_cost square - rect_fence_cost rect = 800 ∧
  square_area square - rect_area rect = 400 := by
  sorry


end garden_transformation_cost_and_area_increase_l1295_129596


namespace contrapositive_prop2_true_l1295_129587

-- Proposition 1
axiom prop1 : ∀ a b : ℝ, a > b → (1 / a) < (1 / b)

-- Proposition 2
axiom prop2 : ∀ x : ℝ, -2 ≤ x ∧ x ≤ 0 → (x + 2) * (x - 3) ≤ 0

-- Theorem: The contrapositive of Proposition 2 is true
theorem contrapositive_prop2_true :
  ∀ x : ℝ, (x + 2) * (x - 3) > 0 → (x < -2 ∨ x > 0) := by
  sorry

end contrapositive_prop2_true_l1295_129587


namespace count_valid_integers_l1295_129512

def is_valid_integer (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999 ∧ n % 100 = 45 ∧ n % 90 = 0

theorem count_valid_integers :
  ∃! (count : ℕ), ∃ (S : Finset ℕ),
    S.card = count ∧
    (∀ n, n ∈ S ↔ is_valid_integer n) ∧
    count = 9 := by sorry

end count_valid_integers_l1295_129512


namespace inequality_equivalence_l1295_129526

theorem inequality_equivalence (x : ℝ) : 
  (x + 3) / 2 - (5 * x - 1) / 5 ≥ 0 ↔ x ≤ 17 / 5 := by
sorry

end inequality_equivalence_l1295_129526


namespace quadratic_root_cube_l1295_129584

theorem quadratic_root_cube (A B C : ℝ) (r s : ℝ) (h1 : A ≠ 0) :
  (A * r^2 + B * r + C = 0) →
  (A * s^2 + B * s + C = 0) →
  (r + s = -B / A) →
  (r * s = C / A) →
  let p := (B^3 - 3*A*B*C) / A^3
  ∃ q, (r^3)^2 + p*(r^3) + q = 0 ∧ (s^3)^2 + p*(s^3) + q = 0 :=
by sorry

end quadratic_root_cube_l1295_129584


namespace equation_positive_root_implies_m_eq_neg_one_l1295_129559

-- Define the equation
def equation (x m : ℝ) : Prop :=
  x / (x - 1) - m / (1 - x) = 2

-- Define the theorem
theorem equation_positive_root_implies_m_eq_neg_one :
  (∃ x : ℝ, x > 0 ∧ equation x m) → m = -1 := by
  sorry

end equation_positive_root_implies_m_eq_neg_one_l1295_129559


namespace eliminate_denominators_l1295_129566

theorem eliminate_denominators (x : ℝ) : 
  (3 * x + (2 * x - 1) / 3 = 3 - (x + 1) / 2) ↔ 
  (18 * x + 2 * (2 * x - 1) = 18 - 3 * (x + 1)) := by
  sorry

end eliminate_denominators_l1295_129566


namespace correct_hours_calculation_l1295_129571

/-- Calculates the number of hours worked given hourly rates and total payment -/
def hours_worked (bricklayer_rate electrician_rate total_payment : ℚ) : ℚ :=
  total_payment / (bricklayer_rate + electrician_rate)

/-- Theorem stating that the calculated hours worked is correct -/
theorem correct_hours_calculation 
  (bricklayer_rate electrician_rate total_payment : ℚ) 
  (h1 : bricklayer_rate = 12)
  (h2 : electrician_rate = 16)
  (h3 : total_payment = 1350) :
  hours_worked bricklayer_rate electrician_rate total_payment = 1350 / 28 :=
by sorry

end correct_hours_calculation_l1295_129571


namespace three_true_propositions_l1295_129539

/-- A line in 3D space -/
structure Line3D where
  -- Define properties of a line

/-- A plane in 3D space -/
structure Plane3D where
  -- Define properties of a plane

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

theorem three_true_propositions
  (a : Line3D) (α β : Plane3D) (h_diff : α ≠ β) :
  (perpendicular a α ∧ perpendicular a β → parallel α β) ∧
  (perpendicular a α ∧ parallel α β → perpendicular a β) ∧
  (perpendicular a β ∧ parallel α β → perpendicular a α) :=
by sorry

end three_true_propositions_l1295_129539


namespace bicycle_not_in_motion_time_l1295_129533

-- Define the constants
def total_distance : ℝ := 22.5
def bert_ride_speed : ℝ := 8
def bert_walk_speed : ℝ := 5
def al_walk_speed : ℝ := 4
def al_ride_speed : ℝ := 10

-- Define the theorem
theorem bicycle_not_in_motion_time :
  ∃ (x : ℝ),
    (x / bert_ride_speed + (total_distance - x) / bert_walk_speed =
     x / al_walk_speed + (total_distance - x) / al_ride_speed) ∧
    ((x / al_walk_speed - x / bert_ride_speed) * 60 = 75) :=
by sorry

end bicycle_not_in_motion_time_l1295_129533


namespace integral_sqrt_plus_xcosx_equals_pi_half_l1295_129549

open Real MeasureTheory Interval

theorem integral_sqrt_plus_xcosx_equals_pi_half :
  ∫ x in (-1)..1, (Real.sqrt (1 - x^2) + x * Real.cos x) = π / 2 := by
  sorry

end integral_sqrt_plus_xcosx_equals_pi_half_l1295_129549


namespace ball_drawing_game_l1295_129531

def total_balls : ℕ := 10
def red_balls : ℕ := 2
def black_balls : ℕ := 4
def white_balls : ℕ := 4
def win_reward : ℚ := 10
def loss_fine : ℚ := 2
def num_draws : ℕ := 10

def prob_win : ℚ := 1 / 15

theorem ball_drawing_game :
  -- Probability of winning in a single draw
  prob_win = (Nat.choose black_balls 3 + Nat.choose white_balls 3) / Nat.choose total_balls 3 ∧
  -- Probability of more than one win in 10 draws
  1 - (1 - prob_win) ^ num_draws - num_draws * prob_win * (1 - prob_win) ^ (num_draws - 1) = 1 / 6 ∧
  -- Expected total amount won (or lost) by 10 people
  (prob_win * win_reward - (1 - prob_win) * loss_fine) * num_draws = -12
  := by sorry

end ball_drawing_game_l1295_129531


namespace amy_bob_games_l1295_129555

theorem amy_bob_games (n : ℕ) (h : n = 9) :
  let total_combinations := Nat.choose n 3
  let games_per_player := total_combinations / n
  let games_together := games_per_player / 4
  games_together = 7 := by
  sorry

end amy_bob_games_l1295_129555


namespace correct_assignment_is_correct_l1295_129551

-- Define the color type
inductive Color
| Red
| Blue
| Green

-- Define the assignment type
structure Assignment where
  one : Color
  two : Color
  three : Color

-- Define the correct assignment
def correct_assignment : Assignment :=
  { one := Color.Green
  , two := Color.Blue
  , three := Color.Red }

-- Theorem stating that the correct_assignment is indeed correct
theorem correct_assignment_is_correct : 
  correct_assignment.one = Color.Green ∧ 
  correct_assignment.two = Color.Blue ∧ 
  correct_assignment.three = Color.Red :=
by sorry

end correct_assignment_is_correct_l1295_129551


namespace multiply_by_0_064_l1295_129530

theorem multiply_by_0_064 (x : ℝ) (h : 13.26 * x = 132.6) : 0.064 * x = 0.64 := by
  sorry

end multiply_by_0_064_l1295_129530


namespace car_speed_second_hour_l1295_129520

/-- Given a car traveling for two hours with an average speed of 82.5 km/h
    and a speed of 90 km/h in the first hour, the speed in the second hour is 75 km/h. -/
theorem car_speed_second_hour
  (average_speed : ℝ)
  (first_hour_speed : ℝ)
  (h_average : average_speed = 82.5)
  (h_first : first_hour_speed = 90)
  : ∃ (second_hour_speed : ℝ),
    second_hour_speed = 75 ∧
    average_speed = (first_hour_speed + second_hour_speed) / 2 :=
by sorry

end car_speed_second_hour_l1295_129520


namespace y_sum_equals_4360_l1295_129595

/-- Given real numbers y₁ to y₈ satisfying four equations, 
    prove that a specific linear combination of these numbers equals 4360 -/
theorem y_sum_equals_4360 
  (y₁ y₂ y₃ y₄ y₅ y₆ y₇ y₈ : ℝ) 
  (eq1 : y₁ + 4*y₂ + 9*y₃ + 16*y₄ + 25*y₅ + 36*y₆ + 49*y₇ + 64*y₈ = 2)
  (eq2 : 4*y₁ + 9*y₂ + 16*y₃ + 25*y₄ + 36*y₅ + 49*y₆ + 64*y₇ + 81*y₈ = 15)
  (eq3 : 9*y₁ + 16*y₂ + 25*y₃ + 36*y₄ + 49*y₅ + 64*y₆ + 81*y₇ + 100*y₈ = 156)
  (eq4 : 16*y₁ + 25*y₂ + 36*y₃ + 49*y₄ + 64*y₅ + 81*y₆ + 100*y₇ + 121*y₈ = 1305) :
  25*y₁ + 36*y₂ + 49*y₃ + 64*y₄ + 81*y₅ + 100*y₆ + 121*y₇ + 144*y₈ = 4360 := by
  sorry


end y_sum_equals_4360_l1295_129595


namespace coefficient_x2y3z2_is_120_l1295_129558

/-- The coefficient of x^2 * y^3 * z^2 in the expansion of (x-y)(x+2y+z)^6 -/
def coefficient_x2y3z2 (x y z : ℤ) : ℤ :=
  let expansion := (x - y) * (x + 2*y + z)^6
  -- The actual computation of the coefficient would go here
  120

/-- Theorem stating that the coefficient of x^2 * y^3 * z^2 in the expansion of (x-y)(x+2y+z)^6 is 120 -/
theorem coefficient_x2y3z2_is_120 (x y z : ℤ) :
  coefficient_x2y3z2 x y z = 120 := by
  sorry

end coefficient_x2y3z2_is_120_l1295_129558


namespace mom_bought_14_packages_l1295_129553

/-- The number of packages Mom bought -/
def num_packages (total_shirts : ℕ) (shirts_per_package : ℕ) : ℕ :=
  total_shirts / shirts_per_package

/-- Proof that Mom bought 14 packages of white t-shirts -/
theorem mom_bought_14_packages :
  num_packages 70 5 = 14 := by
  sorry

end mom_bought_14_packages_l1295_129553


namespace smaug_hoard_value_l1295_129510

/-- Calculates the total value of Smaug's hoard in copper coins -/
def smaugsHoardValue (goldCoins silverCoins copperCoins : ℕ) 
  (silverToCopperRatio goldToSilverRatio : ℕ) : ℕ :=
  goldCoins * goldToSilverRatio * silverToCopperRatio + 
  silverCoins * silverToCopperRatio + 
  copperCoins

/-- Proves that Smaug's hoard has a total value of 2913 copper coins -/
theorem smaug_hoard_value : 
  smaugsHoardValue 100 60 33 8 3 = 2913 := by
  sorry

end smaug_hoard_value_l1295_129510


namespace expression_simplification_l1295_129594

theorem expression_simplification (y : ℝ) : 
  (3 - 4*y) * (3 + 4*y) + (3 + 4*y)^2 = 18 + 24*y := by
  sorry

end expression_simplification_l1295_129594


namespace divisible_by_three_l1295_129597

theorem divisible_by_three (n : ℕ) : 3 ∣ (5^n - 2^n) := by
  sorry

end divisible_by_three_l1295_129597


namespace half_three_abs_diff_squares_l1295_129540

theorem half_three_abs_diff_squares : (1/2 : ℝ) * 3 * |20^2 - 15^2| = 262.5 := by
  sorry

end half_three_abs_diff_squares_l1295_129540


namespace smallest_multiple_l1295_129517

theorem smallest_multiple (x : ℕ) : x = 54 ↔ 
  (x > 0 ∧ 
   250 * x % 1080 = 0 ∧ 
   ∀ y : ℕ, y > 0 → y < x → 250 * y % 1080 ≠ 0) :=
sorry

end smallest_multiple_l1295_129517


namespace first_group_size_l1295_129554

/-- Represents the work rate of a group of people -/
structure WorkRate where
  people : ℕ
  work : ℕ
  days : ℕ

/-- The work rate of the first group -/
def first_group : WorkRate :=
  { people := 0,  -- We don't know this value yet
    work := 3,
    days := 3 }

/-- The work rate of the second group -/
def second_group : WorkRate :=
  { people := 9,
    work := 9,
    days := 3 }

/-- Calculates the daily work rate -/
def daily_rate (wr : WorkRate) : ℚ :=
  wr.work / wr.days

theorem first_group_size :
  first_group.people = 3 :=
by
  sorry

end first_group_size_l1295_129554


namespace volume_ratio_in_cycle_l1295_129532

/-- Represents the state of an ideal gas -/
structure GasState where
  volume : ℝ
  pressure : ℝ
  temperature : ℝ

/-- Represents a cycle of an ideal gas -/
structure GasCycle where
  state1 : GasState
  state2 : GasState
  state3 : GasState

/-- Conditions for the gas cycle -/
def cycleConditions (cycle : GasCycle) : Prop :=
  -- 1-2 is isobaric and volume increases by 4 times
  cycle.state1.pressure = cycle.state2.pressure ∧
  cycle.state2.volume = 4 * cycle.state1.volume ∧
  -- 2-3 is isothermal
  cycle.state2.temperature = cycle.state3.temperature ∧
  cycle.state3.pressure > cycle.state2.pressure ∧
  -- 3-1 follows T = γV²
  ∃ γ : ℝ, cycle.state3.temperature = γ * cycle.state1.volume^2

theorem volume_ratio_in_cycle (cycle : GasCycle) 
  (h : cycleConditions cycle) : 
  cycle.state3.volume = 2 * cycle.state1.volume :=
sorry

end volume_ratio_in_cycle_l1295_129532


namespace race_finish_times_l1295_129506

/-- Race parameters and results -/
structure RaceData where
  malcolm_speed : ℝ  -- Malcolm's speed in minutes per mile
  joshua_speed : ℝ   -- Joshua's speed in minutes per mile
  ellie_speed : ℝ    -- Ellie's speed in minutes per mile
  race_distance : ℝ  -- Race distance in miles

def finish_time (speed : ℝ) (distance : ℝ) : ℝ := speed * distance

/-- Theorem stating the time differences for Joshua and Ellie compared to Malcolm -/
theorem race_finish_times (data : RaceData) 
  (h_malcolm : data.malcolm_speed = 5)
  (h_joshua : data.joshua_speed = 7)
  (h_ellie : data.ellie_speed = 6)
  (h_distance : data.race_distance = 15) :
  let malcolm_time := finish_time data.malcolm_speed data.race_distance
  let joshua_time := finish_time data.joshua_speed data.race_distance
  let ellie_time := finish_time data.ellie_speed data.race_distance
  (joshua_time - malcolm_time = 30 ∧ ellie_time - malcolm_time = 15) := by
  sorry


end race_finish_times_l1295_129506


namespace area_of_DBCE_l1295_129579

/-- Represents a triangle in the diagram -/
structure Triangle where
  area : ℝ

/-- Represents the trapezoid DBCE in the diagram -/
structure Trapezoid where
  area : ℝ

/-- The isosceles triangle ABC -/
def ABC : Triangle := { area := 96 }

/-- One of the smallest triangles in the diagram -/
def smallTriangle : Triangle := { area := 2 }

/-- The number of smallest triangles in the diagram -/
def numSmallTriangles : ℕ := 12

/-- The triangle ADF formed by 8 smallest triangles -/
def ADF : Triangle := { area := 8 * smallTriangle.area }

/-- The trapezoid DBCE -/
def DBCE : Trapezoid := { area := ABC.area - ADF.area }

theorem area_of_DBCE : DBCE.area = 80 := by
  sorry

end area_of_DBCE_l1295_129579


namespace range_of_f_l1295_129542

def f (x : ℤ) : ℤ := (x - 1)^2 - 1

def domain : Set ℤ := {-1, 0, 1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {-1, 0, 3} :=
by sorry

end range_of_f_l1295_129542


namespace smallest_b_is_correct_l1295_129523

/-- A function that checks if a number is a perfect cube -/
def is_perfect_cube (n : ℕ) : Prop :=
  ∃ m : ℕ, m^3 = n

/-- The smallest integer b > 5 for which 43_b is a perfect cube -/
def smallest_b : ℕ := 6

theorem smallest_b_is_correct :
  (smallest_b > 5) ∧ 
  (is_perfect_cube (4 * smallest_b + 3)) ∧ 
  (∀ b : ℕ, b > 5 ∧ b < smallest_b → ¬(is_perfect_cube (4 * b + 3))) :=
by sorry

end smallest_b_is_correct_l1295_129523


namespace quadratic_inequality_l1295_129502

theorem quadratic_inequality (a b : ℝ) : ∃ x₀ ∈ Set.Icc (-1 : ℝ) 1, |x₀^2 + a*x₀ + b| + a ≥ 0 := by
  sorry

end quadratic_inequality_l1295_129502


namespace palindrome_count_l1295_129592

/-- A multiset representing the available digits -/
def availableDigits : Multiset ℕ := {1, 1, 2, 2, 2, 4, 4, 5, 5}

/-- The length of the palindrome -/
def palindromeLength : ℕ := 9

/-- Function to count valid 9-digit palindromes -/
def countPalindromes (digits : Multiset ℕ) (length : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of valid palindromes -/
theorem palindrome_count :
  countPalindromes availableDigits palindromeLength = 36 :=
sorry

end palindrome_count_l1295_129592


namespace total_height_difference_l1295_129589

def heightProblem (anne cathy bella daisy ellie : ℝ) : Prop :=
  anne = 80 ∧
  cathy = anne / 2 ∧
  bella = 3 * anne ∧
  daisy = (cathy + anne) / 2 ∧
  ellie = Real.sqrt (bella * cathy) ∧
  |bella - cathy| + |bella - daisy| + |bella - ellie| + 
  |cathy - daisy| + |cathy - ellie| + |daisy - ellie| = 638

theorem total_height_difference :
  ∃ (anne cathy bella daisy ellie : ℝ),
    heightProblem anne cathy bella daisy ellie :=
by
  sorry

end total_height_difference_l1295_129589


namespace circle_equation_l1295_129529

/-- A circle with center (h, k) and radius r -/
structure Circle where
  h : ℝ
  k : ℝ
  r : ℝ

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Theorem: For a circle with center (4, -6) and radius 3,
    any point (x, y) on the circle satisfies (x - 4)^2 + (y + 6)^2 = 9 -/
theorem circle_equation (c : Circle) (p : Point) :
  c.h = 4 ∧ c.k = -6 ∧ c.r = 3 →
  (p.x - c.h)^2 + (p.y - c.k)^2 = c.r^2 →
  (p.x - 4)^2 + (p.y + 6)^2 = 9 := by
  sorry

end circle_equation_l1295_129529


namespace fraction_equality_l1295_129552

theorem fraction_equality (x y : ℝ) (h : x / y = 3 / 2) : (x - y) / (x + y) = 1 / 5 := by
  sorry

end fraction_equality_l1295_129552


namespace irrational_plus_five_less_than_five_necessary_for_less_than_three_l1295_129570

-- Define the property of being irrational
def IsIrrational (x : ℝ) : Prop := ∀ (p q : ℤ), q ≠ 0 → x ≠ p / q

-- Proposition ②
theorem irrational_plus_five (a : ℝ) : IsIrrational (a + 5) ↔ IsIrrational a := by sorry

-- Proposition ④
theorem less_than_five_necessary_for_less_than_three (a : ℝ) : a < 3 → a < 5 := by sorry

end irrational_plus_five_less_than_five_necessary_for_less_than_three_l1295_129570


namespace glen_village_count_l1295_129578

theorem glen_village_count (p h s c d : ℕ) : 
  p = 2 * h →  -- 2 people for each horse
  s = 5 * c →  -- 5 sheep for each cow
  d = 4 * p →  -- 4 ducks for each person
  p + h + s + c + d ≠ 47 :=
by sorry

end glen_village_count_l1295_129578


namespace sum_first_150_remainder_l1295_129572

theorem sum_first_150_remainder (n : Nat) (divisor : Nat) : n = 150 → divisor = 11200 → 
  (n * (n + 1) / 2) % divisor = 125 := by
  sorry

end sum_first_150_remainder_l1295_129572


namespace average_marks_l1295_129575

theorem average_marks (num_subjects : ℕ) (avg_five : ℝ) (sixth_mark : ℝ) :
  num_subjects = 6 →
  avg_five = 74 →
  sixth_mark = 98 →
  ((avg_five * 5 + sixth_mark) / num_subjects : ℝ) = 78 := by
  sorry

end average_marks_l1295_129575


namespace polynomial_roots_imply_a_ge_5_l1295_129573

theorem polynomial_roots_imply_a_ge_5 (a b c : ℤ) (ha : a > 0) 
  (h_roots : ∃ x y : ℝ, x ≠ y ∧ 0 < x ∧ x < 1 ∧ 0 < y ∧ y < 1 ∧ 
    a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) : 
  a ≥ 5 := by
sorry

end polynomial_roots_imply_a_ge_5_l1295_129573


namespace median_in_70_79_interval_l1295_129550

/-- Represents a score interval with its lower bound and frequency -/
structure ScoreInterval :=
  (lower_bound : ℕ)
  (frequency : ℕ)

/-- The list of score intervals representing the histogram -/
def histogram : List ScoreInterval :=
  [⟨90, 18⟩, ⟨80, 20⟩, ⟨70, 19⟩, ⟨60, 17⟩, ⟨50, 26⟩]

/-- The total number of students -/
def total_students : ℕ := 100

/-- Function to find the interval containing the median score -/
def median_interval (hist : List ScoreInterval) (total : ℕ) : Option ScoreInterval :=
  sorry

/-- Theorem stating that the median score is in the 70-79 interval -/
theorem median_in_70_79_interval :
  median_interval histogram total_students = some ⟨70, 19⟩ := by sorry

end median_in_70_79_interval_l1295_129550


namespace fox_max_berries_l1295_129582

/-- The number of bear cubs --/
def num_cubs : ℕ := 100

/-- The initial number of berries for the n-th bear cub --/
def initial_berries (n : ℕ) : ℕ := 2^(n-1)

/-- The total number of berries initially --/
def total_berries : ℕ := 2^num_cubs - 1

/-- The maximum number of berries the fox can eat --/
def max_fox_berries : ℕ := 2^num_cubs - (num_cubs + 1)

theorem fox_max_berries :
  ∀ (redistribution : ℕ → ℕ → ℕ),
  (∀ (a b : ℕ), redistribution a b ≤ a + b) →
  (∀ (a b : ℕ), redistribution a b = redistribution b a) →
  (∃ (final_berries : ℕ), ∀ (i : ℕ), i ≤ num_cubs → redistribution (initial_berries i) final_berries = final_berries) →
  (total_berries - num_cubs * final_berries) ≤ max_fox_berries :=
sorry

end fox_max_berries_l1295_129582


namespace amy_school_year_hours_l1295_129522

/-- Amy's summer work and earnings information -/
structure SummerWork where
  hours_per_week : ℕ
  weeks : ℕ
  total_earnings : ℕ

/-- Amy's school year work plan -/
structure SchoolYearPlan where
  weeks : ℕ
  target_earnings : ℕ

/-- Calculate required weekly hours for school year -/
def required_weekly_hours (summer : SummerWork) (school : SchoolYearPlan) : ℕ :=
  15

/-- Theorem: Amy must work 15 hours per week during the school year -/
theorem amy_school_year_hours 
  (summer : SummerWork) 
  (school : SchoolYearPlan) 
  (h1 : summer.hours_per_week = 45)
  (h2 : summer.weeks = 8)
  (h3 : summer.total_earnings = 3600)
  (h4 : school.weeks = 24)
  (h5 : school.target_earnings = 3600) :
  required_weekly_hours summer school = 15 := by
  sorry

#check amy_school_year_hours

end amy_school_year_hours_l1295_129522


namespace greatest_plants_per_row_l1295_129519

theorem greatest_plants_per_row (sunflowers corn tomatoes : ℕ) 
  (h1 : sunflowers = 45)
  (h2 : corn = 81)
  (h3 : tomatoes = 63) :
  Nat.gcd sunflowers (Nat.gcd corn tomatoes) = 9 :=
by sorry

end greatest_plants_per_row_l1295_129519


namespace units_digit_of_m_squared_plus_two_to_m_l1295_129548

def m : ℕ := 2017^2 + 2^2017

theorem units_digit_of_m_squared_plus_two_to_m (m : ℕ := m) : (m^2 + 2^m) % 10 = 3 := by
  sorry

end units_digit_of_m_squared_plus_two_to_m_l1295_129548


namespace bear_weight_gain_ratio_l1295_129569

theorem bear_weight_gain_ratio :
  let total_weight : ℝ := 1000
  let berry_weight : ℝ := total_weight / 5
  let small_animal_weight : ℝ := 200
  let salmon_weight : ℝ := (total_weight - berry_weight - small_animal_weight) / 2
  let acorn_weight : ℝ := total_weight - berry_weight - small_animal_weight - salmon_weight
  acorn_weight / berry_weight = 3 / 2 := by
  sorry

end bear_weight_gain_ratio_l1295_129569


namespace no_p_q_for_all_x_divisible_by_3_l1295_129505

theorem no_p_q_for_all_x_divisible_by_3 : 
  ¬ ∃ (p q : ℤ), ∀ (x : ℤ), (3 : ℤ) ∣ (x^2 + p*x + q) := by
  sorry

end no_p_q_for_all_x_divisible_by_3_l1295_129505


namespace volleyball_team_theorem_l1295_129501

def volleyball_team_selection (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) : ℕ :=
  quadruplets * (Nat.choose (total_players - quadruplets) (starters - 1))

theorem volleyball_team_theorem :
  volleyball_team_selection 16 4 6 = 3168 := by
  sorry

end volleyball_team_theorem_l1295_129501


namespace binomial_7_2_l1295_129543

theorem binomial_7_2 : Nat.choose 7 2 = 21 := by
  sorry

end binomial_7_2_l1295_129543


namespace complex_on_line_l1295_129581

theorem complex_on_line (z : ℂ) (a : ℝ) :
  z = (2 + a * Complex.I) / (1 + Complex.I) →
  (z.re = -z.im) →
  a = 0 := by sorry

end complex_on_line_l1295_129581


namespace geometric_arithmetic_sequence_l1295_129591

theorem geometric_arithmetic_sequence :
  ∀ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 →
  (y^2 = x*z) →                   -- geometric progression
  (2*y = x + z - 16) →            -- arithmetic progression after subtracting 16 from z
  ((y-2)^2 = x*(z-16)) →          -- geometric progression after subtracting 2 from y
  ((x = 1 ∧ y = 5 ∧ z = 25) ∨ (x = 1/9 ∧ y = 13/9 ∧ z = 169/9)) :=
by sorry

end geometric_arithmetic_sequence_l1295_129591


namespace donuts_per_box_l1295_129535

/-- Proves that the number of donuts per box is 10 given the conditions of Jeff's donut-making and eating scenario. -/
theorem donuts_per_box :
  let total_donuts := 10 * 12
  let jeff_eaten := 1 * 12
  let chris_eaten := 8
  let boxes := 10
  let remaining_donuts := total_donuts - jeff_eaten - chris_eaten
  remaining_donuts / boxes = 10 := by
  sorry

end donuts_per_box_l1295_129535


namespace min_seats_for_adjacent_seating_l1295_129511

/-- Represents a seating arrangement on a train -/
def SeatingArrangement (total_seats : ℕ) (occupied_seats : ℕ) : Prop :=
  occupied_seats ≤ total_seats

/-- Checks if the next person must sit next to someone already seated -/
def ForceAdjacentSeat (total_seats : ℕ) (occupied_seats : ℕ) : Prop :=
  ∀ (empty_seat : ℕ), empty_seat ≤ total_seats - occupied_seats →
    ∃ (adjacent_seat : ℕ), adjacent_seat ≤ total_seats ∧
      (adjacent_seat = empty_seat + 1 ∨ adjacent_seat = empty_seat - 1) ∧
      (adjacent_seat ≤ occupied_seats)

/-- The main theorem to prove -/
theorem min_seats_for_adjacent_seating :
  ∃ (min_occupied : ℕ),
    SeatingArrangement 150 min_occupied ∧
    ForceAdjacentSeat 150 min_occupied ∧
    (∀ (n : ℕ), n < min_occupied → ¬ForceAdjacentSeat 150 n) ∧
    min_occupied = 37 := by
  sorry

end min_seats_for_adjacent_seating_l1295_129511


namespace square_difference_formula_l1295_129574

theorem square_difference_formula (a b : ℚ) 
  (sum_eq : a + b = 3/4)
  (diff_eq : a - b = 1/8) : 
  a^2 - b^2 = 3/32 := by
sorry

end square_difference_formula_l1295_129574


namespace arithmetic_sequence_count_l1295_129562

theorem arithmetic_sequence_count : 
  let a₁ : ℝ := 2.6
  let aₙ : ℝ := 52.1
  let d : ℝ := 4.5
  let n := (aₙ - a₁) / d + 1
  n = 12 := by sorry

end arithmetic_sequence_count_l1295_129562


namespace travis_apple_sales_l1295_129500

/-- Calculates the total money Travis takes home from selling apples -/
def total_money (total_apples : ℕ) (apples_per_box : ℕ) (price_per_box : ℕ) : ℕ :=
  (total_apples / apples_per_box) * price_per_box

/-- Proves that Travis will take home $7000 -/
theorem travis_apple_sales : total_money 10000 50 35 = 7000 := by
  sorry

end travis_apple_sales_l1295_129500


namespace arccos_one_over_sqrt_two_l1295_129561

theorem arccos_one_over_sqrt_two (π : Real) : 
  Real.arccos (1 / Real.sqrt 2) = π / 4 := by
  sorry

end arccos_one_over_sqrt_two_l1295_129561


namespace divisibility_by_19_l1295_129556

theorem divisibility_by_19 (n : ℕ) : ∃ k : ℤ, 
  120 * 10^(n+2) + 3 * ((10^(n+1) - 1) / 9) * 100 + 8 = 19 * k := by
  sorry

end divisibility_by_19_l1295_129556


namespace line_equation_proof_l1295_129513

/-- Given a point (2, 1) and a slope of -2, prove that the equation 2x + y - 5 = 0 represents the line passing through this point with the given slope. -/
theorem line_equation_proof (x y : ℝ) :
  let point : ℝ × ℝ := (2, 1)
  let slope : ℝ := -2
  (2 * x + y - 5 = 0) ↔ (y - point.2 = slope * (x - point.1)) :=
by sorry

end line_equation_proof_l1295_129513


namespace sin_2theta_value_l1295_129580

theorem sin_2theta_value (θ : Real) 
  (h1 : Real.cos (π/4 - θ) * Real.cos (π/4 + θ) = Real.sqrt 2 / 6)
  (h2 : 0 < θ) (h3 : θ < π/2) : 
  Real.sin (2*θ) = Real.sqrt 7 / 3 := by
  sorry

end sin_2theta_value_l1295_129580


namespace odd_function_property_l1295_129585

def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem odd_function_property (f : ℝ → ℝ) 
  (h1 : is_odd_function f)
  (h2 : is_even_function (fun x ↦ f (x + 2)))
  (h3 : f (-1) = -1) :
  f 2017 + f 2016 = 1 := by
  sorry

end odd_function_property_l1295_129585


namespace sin_cos_tan_product_l1295_129527

theorem sin_cos_tan_product : 
  Real.sin (4/3 * Real.pi) * Real.cos (5/6 * Real.pi) * Real.tan (-4/3 * Real.pi) = -3 * Real.sqrt 3 / 4 := by
  sorry

end sin_cos_tan_product_l1295_129527


namespace common_root_quadratic_equations_l1295_129565

theorem common_root_quadratic_equations (a : ℝ) :
  (∃ x : ℝ, x^2 + x + a = 0 ∧ x^2 + a*x + 1 = 0) → a = -2 := by
  sorry

end common_root_quadratic_equations_l1295_129565


namespace min_value_of_expression_l1295_129547

theorem min_value_of_expression (a : ℝ) (h : a > 0) :
  a + 4 / a ≥ 4 ∧ (a + 4 / a = 4 ↔ a = 2) := by
  sorry

end min_value_of_expression_l1295_129547


namespace ice_cream_truck_expenses_l1295_129507

/-- Proves that for an ice cream truck business where each cone costs $5, 
    if 200 cones are sold and a $200 profit is made, 
    then the expenses are 80% of the total sales. -/
theorem ice_cream_truck_expenses (cone_price : ℝ) (cones_sold : ℕ) (profit : ℝ) :
  cone_price = 5 →
  cones_sold = 200 →
  profit = 200 →
  let total_sales := cone_price * cones_sold
  let expenses := total_sales - profit
  expenses / total_sales = 0.8 := by sorry

end ice_cream_truck_expenses_l1295_129507
