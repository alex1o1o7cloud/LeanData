import Mathlib

namespace unique_quadratic_pair_l90_9014

theorem unique_quadratic_pair : 
  ∃! (b c : ℕ+), 
    (∀ x : ℝ, (x^2 + 2*b*x + c ≤ 0 → x^2 + 2*b*x + c = 0)) ∧ 
    (∀ x : ℝ, (x^2 + 2*c*x + b ≤ 0 → x^2 + 2*c*x + b = 0)) :=
by sorry

end unique_quadratic_pair_l90_9014


namespace simplified_expression_ratio_l90_9000

theorem simplified_expression_ratio (m : ℤ) : 
  let simplified := (6 * m + 12) / 3
  ∃ (c d : ℤ), simplified = c * m + d ∧ c / d = 1 / 2 := by
  sorry

end simplified_expression_ratio_l90_9000


namespace positive_rationals_decomposition_l90_9052

-- Define the set of positive integers
def PositiveIntegers : Set ℚ := {x : ℚ | x > 0 ∧ x.den = 1}

-- Define the set of positive fractions
def PositiveFractions : Set ℚ := {x : ℚ | x > 0 ∧ x.den ≠ 1}

-- Define the set of positive rational numbers
def PositiveRationals : Set ℚ := {x : ℚ | x > 0}

-- Theorem statement
theorem positive_rationals_decomposition :
  PositiveRationals = PositiveIntegers ∪ PositiveFractions :=
by sorry

end positive_rationals_decomposition_l90_9052


namespace square_root_expression_simplification_l90_9050

theorem square_root_expression_simplification :
  (2 + Real.sqrt 3)^2 - Real.sqrt 18 * Real.sqrt (2/3) = 7 + 2 * Real.sqrt 3 := by
  sorry

end square_root_expression_simplification_l90_9050


namespace carlo_thursday_practice_l90_9070

/-- Represents the practice schedule for Carlo's music recital --/
structure PracticeSchedule where
  thursday : ℕ  -- Minutes practiced on Thursday
  wednesday : ℕ := thursday + 5  -- Minutes practiced on Wednesday
  tuesday : ℕ := wednesday - 10  -- Minutes practiced on Tuesday
  monday : ℕ := 2 * tuesday  -- Minutes practiced on Monday
  friday : ℕ := 60  -- Minutes practiced on Friday

/-- Calculates the total practice time for the week --/
def totalPracticeTime (schedule : PracticeSchedule) : ℕ :=
  schedule.monday + schedule.tuesday + schedule.wednesday + schedule.thursday + schedule.friday

/-- Theorem stating that Carlo practiced for 50 minutes on Thursday --/
theorem carlo_thursday_practice :
  ∃ (schedule : PracticeSchedule), totalPracticeTime schedule = 300 ∧ schedule.thursday = 50 := by
  sorry

end carlo_thursday_practice_l90_9070


namespace inner_triangle_perimeter_l90_9078

/-- A right triangle with sides 9, 12, and 15 units -/
structure RightTriangle where
  side_a : ℝ
  side_b : ℝ
  side_c : ℝ
  is_right_triangle : side_a^2 + side_b^2 = side_c^2
  side_a_eq : side_a = 9
  side_b_eq : side_b = 12
  side_c_eq : side_c = 15

/-- A circle with radius 2 units -/
def circle_radius : ℝ := 2

/-- The inner triangle formed by the path of the circle's center -/
def inner_triangle (t : RightTriangle) : ℝ × ℝ × ℝ :=
  (t.side_a - 2 * circle_radius, t.side_b - 2 * circle_radius, t.side_c - 2 * circle_radius)

/-- Theorem: The perimeter of the inner triangle is 24 units -/
theorem inner_triangle_perimeter (t : RightTriangle) :
  let (a, b, c) := inner_triangle t
  a + b + c = 24 := by
  sorry

end inner_triangle_perimeter_l90_9078


namespace prob_four_blue_exact_l90_9095

-- Define the number of blue pens, red pens, and total draws
def blue_pens : ℕ := 5
def red_pens : ℕ := 4
def total_draws : ℕ := 7

-- Define the probability of picking a blue pen in a single draw
def prob_blue : ℚ := blue_pens / (blue_pens + red_pens)

-- Define the probability of picking a red pen in a single draw
def prob_red : ℚ := red_pens / (blue_pens + red_pens)

-- Define the number of ways to choose 4 blue pens out of 7 draws
def ways_to_choose : ℕ := Nat.choose total_draws 4

-- Define the probability of picking exactly 4 blue pens in 7 draws
def prob_four_blue : ℚ := ways_to_choose * (prob_blue ^ 4 * prob_red ^ 3)

-- Theorem statement
theorem prob_four_blue_exact :
  prob_four_blue = 35 * 40000 / 4782969 := by sorry

end prob_four_blue_exact_l90_9095


namespace parallel_vectors_l90_9004

/-- Given two 2D vectors a and b, find the value of k that makes (k*a + b) parallel to (a - 3*b) -/
theorem parallel_vectors (a b : ℝ × ℝ) (h1 : a = (1, 2)) (h2 : b = (-3, 2)) :
  ∃ k : ℝ, k * a.1 + b.1 = (a.1 - 3 * b.1) * ((k * a.2 + b.2) / (a.2 - 3 * b.2)) ∧ k = -1/3 := by
  sorry

end parallel_vectors_l90_9004


namespace fraction_calculation_l90_9075

theorem fraction_calculation : (1/4 + 1/5) / (3/7 - 1/8) = 126/85 := by
  sorry

end fraction_calculation_l90_9075


namespace student_marks_l90_9013

theorem student_marks (total_marks : ℕ) (passing_percentage : ℚ) (failed_by : ℕ) (obtained_marks : ℕ) : 
  total_marks = 400 →
  passing_percentage = 33 / 100 →
  failed_by = 40 →
  obtained_marks = (total_marks * passing_percentage).floor - failed_by →
  obtained_marks = 92 := by
sorry

end student_marks_l90_9013


namespace coincident_centers_of_inscribed_ngons_l90_9082

/-- A regular n-gon in a 2D plane. -/
structure RegularNGon where
  n : ℕ
  center : ℝ × ℝ
  radius : ℝ
  rotation : ℝ  -- Rotation angle of the first vertex

/-- The vertices of a regular n-gon. -/
def vertices (ngon : RegularNGon) : Finset (ℝ × ℝ) :=
  sorry

/-- Predicate to check if a point lies on the perimeter of an n-gon. -/
def on_perimeter (point : ℝ × ℝ) (ngon : RegularNGon) : Prop :=
  sorry

/-- Theorem: If the vertices of one regular n-gon lie on the perimeter of another,
    their centers coincide (for n ≥ 4). -/
theorem coincident_centers_of_inscribed_ngons
  (n : ℕ)
  (h_n : n ≥ 4)
  (ngon1 ngon2 : RegularNGon)
  (h_same_n : ngon1.n = n ∧ ngon2.n = n)
  (h_inscribed : ∀ v ∈ vertices ngon1, on_perimeter v ngon2) :
  ngon1.center = ngon2.center :=
sorry

end coincident_centers_of_inscribed_ngons_l90_9082


namespace total_edges_after_ten_cuts_l90_9033

/-- Represents the number of edges after a given number of cuts -/
def num_edges : ℕ → ℕ
| 0 => 4  -- Initial square has 4 edges
| n + 1 => num_edges n + 3  -- Each cut adds 3 edges

/-- The theorem stating that after 10 cuts, there are 34 edges in total -/
theorem total_edges_after_ten_cuts :
  num_edges 10 = 34 := by
  sorry

end total_edges_after_ten_cuts_l90_9033


namespace jim_age_l90_9092

theorem jim_age (jim fred sam : ℕ) 
  (h1 : jim = 2 * fred)
  (h2 : fred = sam + 9)
  (h3 : jim - 6 = 5 * (sam - 6)) :
  jim = 46 := by sorry

end jim_age_l90_9092


namespace sum_segment_lengths_equals_78_l90_9068

/-- Triangle with vertices A, B, C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Sum of lengths of segments cut by horizontal integer lines -/
def sumSegmentLengths (t : Triangle) : ℝ :=
  sorry

/-- The specific triangle in the problem -/
def problemTriangle : Triangle :=
  { A := (1, 3.5),
    B := (13.5, 3.5),
    C := (11, 16) }

theorem sum_segment_lengths_equals_78 :
  sumSegmentLengths problemTriangle = 78 :=
sorry

end sum_segment_lengths_equals_78_l90_9068


namespace hospital_staff_count_l90_9087

theorem hospital_staff_count (total : ℕ) (doctor_ratio nurse_ratio : ℕ) (h1 : total = 456) (h2 : doctor_ratio = 8) (h3 : nurse_ratio = 11) : 
  (nurse_ratio * total) / (doctor_ratio + nurse_ratio) = 264 := by
  sorry

end hospital_staff_count_l90_9087


namespace prime_triplets_equation_l90_9038

theorem prime_triplets_equation :
  ∀ p q r : ℕ,
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r →
    (p : ℚ) / q = 8 / (r - 1) + 1 →
    ((p = 3 ∧ q = 2 ∧ r = 17) ∨
     (p = 7 ∧ q = 3 ∧ r = 7) ∨
     (p = 5 ∧ q = 3 ∧ r = 13)) :=
by sorry

end prime_triplets_equation_l90_9038


namespace initial_population_proof_l90_9010

def population_change (initial : ℕ) : ℕ := 
  let after_first_year := initial * 125 / 100
  (after_first_year * 70) / 100

theorem initial_population_proof : 
  ∃ (P : ℕ), population_change P = 363650 ∧ P = 415600 := by
  sorry

end initial_population_proof_l90_9010


namespace arccos_one_half_l90_9045

theorem arccos_one_half : Real.arccos (1/2) = π/3 := by
  sorry

end arccos_one_half_l90_9045


namespace rod_speed_l90_9091

/-- 
Given a rod moving freely between a horizontal floor and a slanted wall:
- v: speed of the end in contact with the floor
- θ: angle between the rod and the horizontal floor
- α: angle such that (α - θ) is the angle between the rod and the slanted wall

This theorem states that the speed of the end in contact with the wall 
is v * cos(θ) / cos(α - θ)
-/
theorem rod_speed (v θ α : ℝ) : ℝ := by
  sorry

end rod_speed_l90_9091


namespace smallest_n_perfect_square_and_fifth_power_l90_9043

theorem smallest_n_perfect_square_and_fifth_power : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (a : ℕ), 4 * n = a^2) ∧
  (∃ (b : ℕ), 5 * n = b^5) ∧
  (∀ (m : ℕ), m > 0 → 
    (∃ (x : ℕ), 4 * m = x^2) → 
    (∃ (y : ℕ), 5 * m = y^5) → 
    m ≥ n) ∧
  n = 3125 := by
sorry

end smallest_n_perfect_square_and_fifth_power_l90_9043


namespace jellybean_problem_l90_9085

theorem jellybean_problem :
  ∃ n : ℕ, n = 164 ∧ 
  (∀ m : ℕ, m ≥ 150 ∧ m % 15 = 14 → m ≥ n) ∧
  n ≥ 150 ∧ 
  n % 15 = 14 :=
sorry

end jellybean_problem_l90_9085


namespace billys_restaurant_bill_l90_9030

/-- Calculates the total bill for three families at Billy's Restaurant -/
theorem billys_restaurant_bill (adult_meal_cost child_meal_cost drink_cost : ℕ) 
  (family1_adults family1_children : ℕ)
  (family2_adults family2_children : ℕ)
  (family3_adults family3_children : ℕ) :
  adult_meal_cost = 8 →
  child_meal_cost = 5 →
  drink_cost = 2 →
  family1_adults = 2 →
  family1_children = 3 →
  family2_adults = 4 →
  family2_children = 2 →
  family3_adults = 3 →
  family3_children = 4 →
  (family1_adults * adult_meal_cost + family1_children * child_meal_cost + 
   (family1_adults + family1_children) * drink_cost) +
  (family2_adults * adult_meal_cost + family2_children * child_meal_cost + 
   (family2_adults + family2_children) * drink_cost) +
  (family3_adults * adult_meal_cost + family3_children * child_meal_cost + 
   (family3_adults + family3_children) * drink_cost) = 153 :=
by
  sorry

end billys_restaurant_bill_l90_9030


namespace alpha_para_beta_sufficient_not_necessary_l90_9002

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Plane → Prop)
variable (para : Plane → Plane → Prop)
variable (perpLine : Line → Line → Prop)
variable (paraLine : Line → Plane → Prop)

-- State the theorem
theorem alpha_para_beta_sufficient_not_necessary 
  (l m : Line) (α β : Plane) 
  (h1 : perp l α) 
  (h2 : paraLine m β) : 
  (∃ (config : Type), 
    (∀ (α β : Plane), para α β → perpLine l m) ∧ 
    (∃ (α β : Plane), perpLine l m ∧ ¬ para α β)) :=
sorry

end alpha_para_beta_sufficient_not_necessary_l90_9002


namespace quadratic_root_theorem_l90_9081

theorem quadratic_root_theorem (c : ℝ) : 
  (∀ x : ℝ, 2*x^2 + 8*x + c = 0 ↔ x = -2 + Real.sqrt 3 ∨ x = -2 - Real.sqrt 3) →
  c = 13/2 := by
sorry

end quadratic_root_theorem_l90_9081


namespace smallest_five_digit_divisible_by_2_5_11_l90_9067

theorem smallest_five_digit_divisible_by_2_5_11 :
  ∀ n : ℕ, 10000 ≤ n ∧ n < 100000 ∧ 2 ∣ n ∧ 5 ∣ n ∧ 11 ∣ n → 10010 ≤ n :=
by
  sorry

end smallest_five_digit_divisible_by_2_5_11_l90_9067


namespace recurrence_relation_initial_conditions_sequence_satisfies_conditions_l90_9076

/-- Sequence definition -/
def a (n : ℕ) : ℚ := 3 * 2^n + 3^n + (1/2) * n + 11/4

/-- Recurrence relation -/
theorem recurrence_relation (n : ℕ) (h : n ≥ 2) :
  a n = 5 * a (n-1) - 6 * a (n-2) + n + 2 := by sorry

/-- Initial conditions -/
theorem initial_conditions :
  a 0 = 27/4 ∧ a 1 = 49/4 := by sorry

/-- Main theorem: The sequence satisfies the recurrence relation and initial conditions -/
theorem sequence_satisfies_conditions :
  (∀ n : ℕ, n ≥ 2 → a n = 5 * a (n-1) - 6 * a (n-2) + n + 2) ∧
  a 0 = 27/4 ∧ a 1 = 49/4 := by sorry

end recurrence_relation_initial_conditions_sequence_satisfies_conditions_l90_9076


namespace marble_probability_l90_9084

/-- Represents a box of marbles -/
structure MarbleBox where
  total : ℕ
  black : ℕ
  white : ℕ
  hSum : total = black + white

/-- The probability of drawing a specific color from a box -/
def drawProbability (box : MarbleBox) (color : ℕ) : ℚ :=
  color / box.total

theorem marble_probability (box1 box2 : MarbleBox)
  (hTotal : box1.total + box2.total = 30)
  (hBlackProb : drawProbability box1 box1.black * drawProbability box2 box2.black = 1/2) :
  drawProbability box1 box1.white * drawProbability box2 box2.white = 0 := by
  sorry

end marble_probability_l90_9084


namespace tournament_theorem_l90_9079

/-- Represents a team in the tournament -/
structure Team :=
  (city : Fin 16)
  (is_team_a : Bool)

/-- The number of matches played by a team -/
def matches_played (t : Team) : Fin 32 := sorry

/-- The statement that all teams except one have unique match counts -/
def all_but_one_unique (exception : Team) : Prop :=
  ∀ t1 t2 : Team, t1 ≠ exception → t2 ≠ exception → t1 ≠ t2 → matches_played t1 ≠ matches_played t2

theorem tournament_theorem :
  ∃ (exception : Team),
    (all_but_one_unique exception) →
    (matches_played exception = 15) :=
  sorry

end tournament_theorem_l90_9079


namespace team_formation_with_girls_l90_9061

-- Define the total number of people
def total_people : Nat := 10

-- Define the number of boys
def num_boys : Nat := 5

-- Define the number of girls
def num_girls : Nat := 5

-- Define the team size
def team_size : Nat := 3

-- Theorem statement
theorem team_formation_with_girls (total_people num_boys num_girls team_size : Nat) 
  (h1 : total_people = num_boys + num_girls)
  (h2 : num_boys = 5)
  (h3 : num_girls = 5)
  (h4 : team_size = 3) :
  (Nat.choose total_people team_size) - (Nat.choose num_boys team_size) = 110 := by
  sorry

end team_formation_with_girls_l90_9061


namespace complex_power_difference_l90_9032

theorem complex_power_difference (i : ℂ) : i * i = -1 → (1 + i)^20 - (1 - i)^20 = 0 := by
  sorry

end complex_power_difference_l90_9032


namespace parallel_lines_perpendicular_lines_l90_9028

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := (3 - m) * x + 2 * m * y + 1 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := 2 * m * x + 2 * y + m = 0

-- Define parallel and perpendicular conditions
def parallel (m : ℝ) : Prop := ∀ x y, l₁ m x y ↔ l₂ m x y
def perpendicular (m : ℝ) : Prop := ∀ x₁ y₁ x₂ y₂, l₁ m x₁ y₁ → l₂ m x₂ y₂ → 
  ((3 - m) * (2 * m) + (2 * m) * 2 = 0)

-- Theorem for parallel lines
theorem parallel_lines : ∀ m : ℝ, parallel m ↔ m = -3/2 :=
sorry

-- Theorem for perpendicular lines
theorem perpendicular_lines : ∀ m : ℝ, perpendicular m ↔ (m = 0 ∨ m = 5) :=
sorry

end parallel_lines_perpendicular_lines_l90_9028


namespace de_moivre_formula_l90_9046

theorem de_moivre_formula (x : ℝ) (n : ℕ) (h : x ∈ Set.Ioo 0 (π / 2)) :
  (Complex.exp (Complex.I * x)) ^ n = Complex.exp (Complex.I * (n : ℝ) * x) := by
  sorry

#check de_moivre_formula

end de_moivre_formula_l90_9046


namespace citizenship_test_study_time_l90_9040

/-- Calculates the total study time in hours for a citizenship test -/
theorem citizenship_test_study_time :
  let total_questions : ℕ := 60
  let multiple_choice_questions : ℕ := 30
  let fill_in_blank_questions : ℕ := 30
  let multiple_choice_time : ℕ := 15  -- minutes per question
  let fill_in_blank_time : ℕ := 25    -- minutes per question
  
  total_questions = multiple_choice_questions + fill_in_blank_questions →
  (multiple_choice_questions * multiple_choice_time + fill_in_blank_questions * fill_in_blank_time) / 60 = 20 :=
by
  sorry


end citizenship_test_study_time_l90_9040


namespace tangent_line_determines_function_l90_9064

noncomputable def f (a b x : ℝ) : ℝ := a * x / (x^2 + b)

theorem tangent_line_determines_function (a b : ℝ) :
  (∃ x, f a b x = 2 ∧ (deriv (f a b)) x = 0) ∧ f a b 1 = 2 →
  ∀ x, f a b x = 4 * x / (x^2 + 1) :=
by sorry

end tangent_line_determines_function_l90_9064


namespace charity_event_total_is_1080_l90_9041

/-- Represents the total money raised from a charity event with raffle ticket sales and donations -/
def charity_event_total (a_price b_price c_price : ℚ) 
                        (a_sold b_sold c_sold : ℕ) 
                        (donations : List ℚ) : ℚ :=
  a_price * a_sold + b_price * b_sold + c_price * c_sold + donations.sum

/-- Theorem stating the total money raised from the charity event -/
theorem charity_event_total_is_1080 : 
  charity_event_total 3 5.5 10 100 50 25 [30, 30, 50, 45, 100] = 1080 := by
  sorry

end charity_event_total_is_1080_l90_9041


namespace shortest_distance_parabola_to_line_l90_9019

/-- The shortest distance between a point on the parabola y = x^2 - 4x and a point on the line y = 2x - 3 is 6√5/5 -/
theorem shortest_distance_parabola_to_line :
  let parabola := {p : ℝ × ℝ | p.2 = p.1^2 - 4*p.1}
  let line := {p : ℝ × ℝ | p.2 = 2*p.1 - 3}
  ∀ A ∈ parabola, ∀ B ∈ line,
  ∃ C ∈ parabola, ∃ D ∈ line,
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) ≤ Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ∧
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 6 * Real.sqrt 5 / 5 := by
sorry

end shortest_distance_parabola_to_line_l90_9019


namespace candies_added_l90_9083

theorem candies_added (initial_candies final_candies : ℕ) (h1 : initial_candies = 6) (h2 : final_candies = 10) :
  final_candies - initial_candies = 4 := by
  sorry

end candies_added_l90_9083


namespace bowl_game_points_l90_9089

/-- The total points scored by Noa and Phillip in a bowl game. -/
def total_points (noa_points phillip_points : ℕ) : ℕ := noa_points + phillip_points

/-- Theorem stating that given Noa's score and Phillip scoring twice as much,
    the total points scored by Noa and Phillip is 90. -/
theorem bowl_game_points :
  let noa_points : ℕ := 30
  let phillip_points : ℕ := 2 * noa_points
  total_points noa_points phillip_points = 90 := by
  sorry

end bowl_game_points_l90_9089


namespace equation_represents_three_lines_lines_do_not_all_intersect_l90_9005

-- Define the equation
def equation (x y : ℝ) : Prop :=
  x^2 * (x - y - 2) = y^2 * (x - y - 2)

-- Define the three lines
def line1 (x y : ℝ) : Prop := y = x
def line2 (x y : ℝ) : Prop := y = -x
def line3 (x y : ℝ) : Prop := y = x - 2

-- Theorem statement
theorem equation_represents_three_lines :
  ∀ x y : ℝ, equation x y ↔ (line1 x y ∨ line2 x y ∨ line3 x y) :=
by sorry

-- Theorem stating that the three lines do not all intersect at a common point
theorem lines_do_not_all_intersect :
  ¬∃ x y : ℝ, line1 x y ∧ line2 x y ∧ line3 x y :=
by sorry

end equation_represents_three_lines_lines_do_not_all_intersect_l90_9005


namespace polygon_exterior_angle_pairs_l90_9086

def is_valid_pair (m n : ℕ) : Prop :=
  m ≥ 3 ∧ n ≥ 3 ∧ 360 / m = n ∧ 360 / n = m

theorem polygon_exterior_angle_pairs :
  ∃! (S : Finset (ℕ × ℕ)), S.card = 20 ∧ ∀ p : ℕ × ℕ, p ∈ S ↔ is_valid_pair p.1 p.2 := by
  sorry

end polygon_exterior_angle_pairs_l90_9086


namespace josh_candy_purchase_l90_9059

/-- Given an initial amount of money and the cost of a purchase, 
    calculate the remaining change. -/
def calculate_change (initial_amount cost : ℚ) : ℚ :=
  initial_amount - cost

/-- Prove that given an initial amount of $1.80 and a purchase of $0.45, 
    the remaining change is $1.35. -/
theorem josh_candy_purchase : 
  calculate_change (180/100) (45/100) = 135/100 := by
  sorry

end josh_candy_purchase_l90_9059


namespace particle_position_1989_l90_9063

/-- Represents the position of a particle -/
structure Position :=
  (x : ℕ) (y : ℕ)

/-- Calculates the position of the particle after a given number of minutes -/
def particlePosition (minutes : ℕ) : Position :=
  sorry

/-- The theorem stating the particle's position after 1989 minutes -/
theorem particle_position_1989 : particlePosition 1989 = Position.mk 44 35 := by
  sorry

end particle_position_1989_l90_9063


namespace sam_candy_bars_l90_9096

/-- Represents the number of candy bars Sam bought -/
def candy_bars : ℕ := sorry

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of dimes Sam initially had -/
def initial_dimes : ℕ := 19

/-- The number of quarters Sam initially had -/
def initial_quarters : ℕ := 6

/-- The cost of a candy bar in dimes -/
def candy_bar_cost_dimes : ℕ := 3

/-- The cost of a lollipop in quarters -/
def lollipop_cost_quarters : ℕ := 1

/-- The amount of money Sam has left after purchases, in cents -/
def remaining_cents : ℕ := 195

theorem sam_candy_bars : 
  candy_bars = 4 ∧
  initial_dimes * dime_value + initial_quarters * quarter_value = 
  remaining_cents + candy_bars * (candy_bar_cost_dimes * dime_value) + 
  lollipop_cost_quarters * quarter_value :=
sorry

end sam_candy_bars_l90_9096


namespace least_positive_integer_with_remainders_l90_9073

theorem least_positive_integer_with_remainders : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a % 2 = 0) ∧ 
  (a % 5 = 1) ∧ 
  (a % 4 = 2) ∧ 
  (∀ (b : ℕ), b > 0 ∧ b % 2 = 0 ∧ b % 5 = 1 ∧ b % 4 = 2 → a ≤ b) ∧
  (a = 6) := by
sorry

end least_positive_integer_with_remainders_l90_9073


namespace triangle_properties_l90_9065

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

def isArithmeticSequence (t : Triangle) : Prop :=
  t.a + t.c = 2 * t.b

def aEquals2c (t : Triangle) : Prop :=
  t.a = 2 * t.c

def areaIs3Sqrt15Over4 (t : Triangle) : Prop :=
  1/2 * t.b * t.c * Real.sin t.A = 3 * Real.sqrt 15 / 4

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h1 : isValidTriangle t)
  (h2 : isArithmeticSequence t)
  (h3 : aEquals2c t)
  (h4 : areaIs3Sqrt15Over4 t) :
  Real.cos t.A = -1/4 ∧ t.b = 3 := by
  sorry


end triangle_properties_l90_9065


namespace FGH_supermarket_count_l90_9037

def FGH_supermarkets : Type := Unit

def location : FGH_supermarkets → Bool
  | _ => sorry

def in_US (s : FGH_supermarkets) : Prop := location s = true
def in_Canada (s : FGH_supermarkets) : Prop := location s = false

axiom all_in_US_or_Canada : ∀ s : FGH_supermarkets, in_US s ∨ in_Canada s

def count_US : Nat := 42
def count_Canada : Nat := count_US - 14

def total_count : Nat := count_US + count_Canada

theorem FGH_supermarket_count : total_count = 70 := by sorry

end FGH_supermarket_count_l90_9037


namespace puppies_adoption_l90_9001

theorem puppies_adoption (first_week : ℕ) : 
  first_week + (2/5 : ℚ) * first_week + 2 * ((2/5 : ℚ) * first_week) + (first_week + 10) = 74 → 
  first_week = 20 := by
sorry

end puppies_adoption_l90_9001


namespace circumradius_is_five_l90_9057

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2/4 - y^2/12 = 1

-- Define the foci
def F₁ : ℝ × ℝ := (-4, 0)
def F₂ : ℝ × ℝ := (4, 0)

-- Define a point P on the hyperbola
def P : ℝ × ℝ := sorry

-- Assert that P is on the hyperbola
axiom P_on_hyperbola : hyperbola P.1 P.2

-- Define the centroid G of triangle F₁PF₂
def G : ℝ × ℝ := sorry

-- Define the incenter I of triangle F₁PF₂
def I : ℝ × ℝ := sorry

-- Assert that G and I are parallel to the x-axis
axiom G_I_parallel_x : G.2 = I.2

-- Define the circumradius of triangle F₁PF₂
def circumradius (F₁ F₂ P : ℝ × ℝ) : ℝ := sorry

-- Theorem: The circumradius of triangle F₁PF₂ is 5
theorem circumradius_is_five : circumradius F₁ F₂ P = 5 := by
  sorry

end circumradius_is_five_l90_9057


namespace joan_balloons_l90_9036

theorem joan_balloons (total : ℕ) (melanie : ℕ) (joan : ℕ) : 
  total = 81 → melanie = 41 → total = joan + melanie → joan = 40 := by
  sorry

end joan_balloons_l90_9036


namespace maintenance_check_increase_l90_9044

theorem maintenance_check_increase (original_days : ℝ) (new_days : ℝ) 
  (h1 : original_days = 30) 
  (h2 : new_days = 45) : 
  ((new_days - original_days) / original_days) * 100 = 50 := by
  sorry

end maintenance_check_increase_l90_9044


namespace arrangement_count_is_24_l90_9034

/-- The number of ways to arrange 8 balls in a row, with 5 red balls and 3 white balls,
    such that exactly three red balls are consecutive. -/
def arrangement_count : ℕ := 24

/-- The total number of balls -/
def total_balls : ℕ := 8

/-- The number of red balls -/
def red_balls : ℕ := 5

/-- The number of white balls -/
def white_balls : ℕ := 3

/-- The number of consecutive red balls required -/
def consecutive_red : ℕ := 3

theorem arrangement_count_is_24 :
  arrangement_count = 24 ∧
  total_balls = 8 ∧
  red_balls = 5 ∧
  white_balls = 3 ∧
  consecutive_red = 3 :=
by sorry

end arrangement_count_is_24_l90_9034


namespace merill_marble_count_l90_9055

/-- The number of marbles each person has -/
structure MarbleCount where
  merill : ℕ
  elliot : ℕ
  selma : ℕ

/-- The conditions of the marble problem -/
def marbleProblemConditions (m : MarbleCount) : Prop :=
  m.merill = 2 * m.elliot ∧
  m.merill + m.elliot = m.selma - 5 ∧
  m.selma = 50

/-- Theorem stating that under the given conditions, Merill has 30 marbles -/
theorem merill_marble_count (m : MarbleCount) 
  (h : marbleProblemConditions m) : m.merill = 30 := by
  sorry


end merill_marble_count_l90_9055


namespace arithmetic_sequence_general_term_l90_9062

/-- An arithmetic sequence with first term 6 and the sum of the 3rd and 5th terms equal to 0 -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  a 1 = 6 ∧ 
  a 3 + a 5 = 0 ∧ 
  ∀ n : ℕ, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- The general term formula for the arithmetic sequence -/
def GeneralTermFormula (a : ℕ → ℤ) : Prop :=
  ∀ n : ℕ, a n = 8 - 2 * n

theorem arithmetic_sequence_general_term 
  (a : ℕ → ℤ) (h : ArithmeticSequence a) : GeneralTermFormula a :=
sorry

end arithmetic_sequence_general_term_l90_9062


namespace initial_pups_proof_l90_9022

/-- The number of initial mice -/
def initial_mice : ℕ := 8

/-- The number of additional pups each mouse has in the second round -/
def second_round_pups : ℕ := 6

/-- The number of pups eaten by each adult mouse -/
def eaten_pups : ℕ := 2

/-- The total number of mice at the end -/
def total_mice : ℕ := 280

/-- The initial number of pups per mouse -/
def initial_pups_per_mouse : ℕ := 6

theorem initial_pups_proof :
  initial_mice +
  initial_mice * initial_pups_per_mouse +
  (initial_mice + initial_mice * initial_pups_per_mouse) * second_round_pups -
  (initial_mice + initial_mice * initial_pups_per_mouse) * eaten_pups =
  total_mice :=
by sorry

end initial_pups_proof_l90_9022


namespace quadratic_roots_sum_l90_9024

theorem quadratic_roots_sum (x : ℝ) (h : x^2 - 9*x + 20 = 0) : 
  ∃ (y : ℝ), y ≠ x ∧ y^2 - 9*y + 20 = 0 ∧ x + y = 9 := by
sorry

end quadratic_roots_sum_l90_9024


namespace postcards_cost_l90_9060

/-- Represents a country --/
inductive Country
| Italy
| Germany
| Canada
| Japan

/-- Represents a decade --/
inductive Decade
| Fifties
| Sixties
| Seventies
| Eighties
| Nineties

/-- Price of a postcard in cents for a given country --/
def price (c : Country) : ℕ :=
  match c with
  | Country.Italy => 8
  | Country.Germany => 8
  | Country.Canada => 5
  | Country.Japan => 7

/-- Number of postcards for a given country and decade --/
def quantity (c : Country) (d : Decade) : ℕ :=
  match c, d with
  | Country.Italy, Decade.Fifties => 5
  | Country.Italy, Decade.Sixties => 12
  | Country.Italy, Decade.Seventies => 11
  | Country.Italy, Decade.Eighties => 10
  | Country.Italy, Decade.Nineties => 6
  | Country.Germany, Decade.Fifties => 9
  | Country.Germany, Decade.Sixties => 5
  | Country.Germany, Decade.Seventies => 13
  | Country.Germany, Decade.Eighties => 15
  | Country.Germany, Decade.Nineties => 7
  | Country.Canada, Decade.Fifties => 3
  | Country.Canada, Decade.Sixties => 7
  | Country.Canada, Decade.Seventies => 6
  | Country.Canada, Decade.Eighties => 10
  | Country.Canada, Decade.Nineties => 11
  | Country.Japan, Decade.Fifties => 6
  | Country.Japan, Decade.Sixties => 8
  | Country.Japan, Decade.Seventies => 9
  | Country.Japan, Decade.Eighties => 5
  | Country.Japan, Decade.Nineties => 9

/-- Total cost of postcards for a given country and set of decades --/
def totalCost (c : Country) (decades : List Decade) : ℕ :=
  (decades.map (quantity c)).sum * price c

/-- Theorem: The total cost of postcards from Canada and Japan issued in the '60s, '70s, and '80s is 269 cents --/
theorem postcards_cost :
  totalCost Country.Canada [Decade.Sixties, Decade.Seventies, Decade.Eighties] +
  totalCost Country.Japan [Decade.Sixties, Decade.Seventies, Decade.Eighties] = 269 := by
  sorry

end postcards_cost_l90_9060


namespace arithmetic_square_root_of_four_l90_9017

theorem arithmetic_square_root_of_four : ∃ x : ℝ, x > 0 ∧ x^2 = 4 ∧ ∀ y : ℝ, y > 0 ∧ y^2 = 4 → y = x :=
sorry

end arithmetic_square_root_of_four_l90_9017


namespace club_members_after_five_years_l90_9026

/-- Represents the number of people in the club after k years -/
def club_members (k : ℕ) : ℕ :=
  if k = 0 then 18
  else 3 * club_members (k - 1) - 10

/-- The number of people in the club after 5 years is 3164 -/
theorem club_members_after_five_years :
  club_members 5 = 3164 := by
  sorry

end club_members_after_five_years_l90_9026


namespace inequality_proof_l90_9039

theorem inequality_proof (a b c x y z : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (hsum : a + b + c = 1)
  (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) : 
  (x^2 + y^2 + z^2) * (a^3 / (x^2 + 2*y^2) + b^3 / (y^2 + 2*z^2) + c^3 / (z^2 + 2*x^2)) ≥ 1/9 := by
  sorry

end inequality_proof_l90_9039


namespace problem_solution_l90_9098

theorem problem_solution (a b c d : ℝ) : 
  8 = (4 / 100) * a →
  4 = (d / 100) * a →
  8 = (d / 100) * b →
  c = b / a →
  c = 2 := by
sorry

end problem_solution_l90_9098


namespace cab_driver_average_income_l90_9012

def daily_incomes : List ℝ := [250, 400, 750, 400, 500]

theorem cab_driver_average_income :
  (daily_incomes.sum / daily_incomes.length : ℝ) = 460 := by
  sorry

end cab_driver_average_income_l90_9012


namespace equality_of_positive_reals_l90_9023

theorem equality_of_positive_reals (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (a^2 - b*d) / (b + 2*c + d) + (b^2 - c*a) / (c + 2*d + a) + 
  (c^2 - d*b) / (d + 2*a + b) + (d^2 - a*c) / (a + 2*b + c) = 0 →
  a = b ∧ b = c ∧ c = d := by
sorry

end equality_of_positive_reals_l90_9023


namespace problem_statement_l90_9027

theorem problem_statement (a b : ℝ) (h : |a + 2| + (b - 1)^2 = 0) : 
  (a + b)^2023 = -1 := by
sorry

end problem_statement_l90_9027


namespace remainder_252_power_252_mod_13_l90_9054

theorem remainder_252_power_252_mod_13 : 252^252 ≡ 1 [ZMOD 13] := by
  sorry

end remainder_252_power_252_mod_13_l90_9054


namespace gcd_50420_35313_l90_9008

theorem gcd_50420_35313 : Nat.gcd 50420 35313 = 19 := by
  sorry

end gcd_50420_35313_l90_9008


namespace fraction_irreducible_l90_9053

theorem fraction_irreducible (n : ℕ) : Nat.gcd (21 * n + 4) (14 * n + 3) = 1 := by
  sorry

end fraction_irreducible_l90_9053


namespace remaining_document_arrangements_l90_9009

/-- Represents the number of documents --/
def total_documents : ℕ := 12

/-- Represents the number of the processed document --/
def processed_document : ℕ := 10

/-- Calculates the number of possible arrangements for the remaining documents --/
def possible_arrangements : ℕ :=
  2 * (Nat.factorial 9 + 2 * Nat.factorial 10 + Nat.factorial 11)

/-- Theorem stating the number of possible ways to handle the remaining documents --/
theorem remaining_document_arrangements :
  possible_arrangements = 95116960 := by sorry

end remaining_document_arrangements_l90_9009


namespace amount_transferred_l90_9074

def initial_balance : ℕ := 27004
def remaining_balance : ℕ := 26935

theorem amount_transferred : initial_balance - remaining_balance = 69 := by
  sorry

end amount_transferred_l90_9074


namespace division_result_l90_9035

theorem division_result : (5 / 2) / 7 = 5 / 14 := by sorry

end division_result_l90_9035


namespace xy_reciprocal_l90_9049

theorem xy_reciprocal (x y : ℝ) 
  (h1 : x * y > 0) 
  (h2 : 1 / x + 1 / y = 15) 
  (h3 : (x + y) / 5 = 0.6) : 
  1 / (x * y) = 5 := by
  sorry

end xy_reciprocal_l90_9049


namespace room_breadth_calculation_l90_9051

theorem room_breadth_calculation (room_length : ℝ) (carpet_width_cm : ℝ) (cost_per_meter : ℝ) (total_cost : ℝ) :
  room_length = 18 →
  carpet_width_cm = 75 →
  cost_per_meter = 4.50 →
  total_cost = 810 →
  (total_cost / cost_per_meter) / room_length * (carpet_width_cm / 100) = 7.5 := by
  sorry

end room_breadth_calculation_l90_9051


namespace license_plate_count_is_9750000_l90_9011

/-- The number of possible distinct license plates -/
def license_plate_count : ℕ :=
  (Nat.choose 6 2) * 26 * 25 * (10^4)

/-- Theorem stating the number of distinct license plates -/
theorem license_plate_count_is_9750000 :
  license_plate_count = 9750000 := by
  sorry

end license_plate_count_is_9750000_l90_9011


namespace monotonic_functional_equation_implies_f_zero_eq_one_l90_9058

/-- A function f: ℝ → ℝ is monotonic if for all x, y ∈ ℝ, x ≤ y implies f(x) ≤ f(y) -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

/-- A function f: ℝ → ℝ satisfies the functional equation f(x+y) = f(x)f(y) for all x, y ∈ ℝ -/
def SatisfiesFunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y, f (x + y) = f x * f y

theorem monotonic_functional_equation_implies_f_zero_eq_one
  (f : ℝ → ℝ) (h_mono : Monotonic f) (h_eq : SatisfiesFunctionalEquation f) :
  f 0 = 1 :=
sorry

end monotonic_functional_equation_implies_f_zero_eq_one_l90_9058


namespace logical_equivalences_l90_9048

theorem logical_equivalences :
  (∀ A B C : Prop,
    (A ∨ (B ∧ C) ↔ (A ∨ B) ∧ (A ∨ C)) ∧
    (¬((A ∨ (¬B)) ∨ (C ∧ (A ∨ (¬B)))) ↔ (¬A) ∧ B)) := by
  sorry

end logical_equivalences_l90_9048


namespace triangle_theorem_l90_9031

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : Real.sin t.A ^ 2 + Real.sin t.A * Real.sin t.B - 6 * Real.sin t.B ^ 2 = 0) :
  (t.a / t.b = 2) ∧ 
  (Real.cos t.C = 3/4 → Real.sin t.B = Real.sqrt 14 / 8) := by
  sorry

end triangle_theorem_l90_9031


namespace gerbils_sold_l90_9020

theorem gerbils_sold (initial : ℕ) (remaining : ℕ) (sold : ℕ) : 
  initial = 85 → remaining = 16 → sold = initial - remaining → sold = 69 := by
  sorry

end gerbils_sold_l90_9020


namespace james_max_lift_l90_9018

def farmers_walk_20m : ℝ := 300

def increase_20m : ℝ := 50

def short_distance_increase_percent : ℝ := 0.3

def strap_increase_percent : ℝ := 0.2

def calculate_max_weight (base_weight : ℝ) (short_distance_increase : ℝ) (strap_increase : ℝ) : ℝ :=
  base_weight * (1 + short_distance_increase) * (1 + strap_increase)

theorem james_max_lift :
  calculate_max_weight (farmers_walk_20m + increase_20m) short_distance_increase_percent strap_increase_percent = 546 := by
  sorry

end james_max_lift_l90_9018


namespace l_shaped_paper_area_l90_9007

/-- The area of an "L" shaped paper formed by cutting rectangles from a larger rectangle --/
theorem l_shaped_paper_area (original_length original_width cut1_length cut1_width cut2_length cut2_width : ℕ) 
  (h1 : original_length = 10)
  (h2 : original_width = 7)
  (h3 : cut1_length = 3)
  (h4 : cut1_width = 2)
  (h5 : cut2_length = 2)
  (h6 : cut2_width = 4) :
  original_length * original_width - cut1_length * cut1_width - cut2_length * cut2_width = 56 := by
  sorry

end l_shaped_paper_area_l90_9007


namespace min_value_M_min_value_expression_equality_condition_l90_9015

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1|

-- Theorem 1: Minimum value of M
theorem min_value_M : 
  (∃ (M : ℝ), ∀ (m : ℝ), (∃ (x₀ : ℝ), f x₀ ≤ m) → M ≤ m) ∧ 
  (∃ (x₀ : ℝ), f x₀ ≤ 2) := by sorry

-- Theorem 2: Minimum value of 1/(2a) + 1/(a+b)
theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3*a + b = 2) :
  1/(2*a) + 1/(a+b) ≥ 2 := by sorry

-- Theorem 3: Equality condition
theorem equality_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : 3*a + b = 2) :
  1/(2*a) + 1/(a+b) = 2 ↔ a = 1/2 ∧ b = 1/2 := by sorry

end min_value_M_min_value_expression_equality_condition_l90_9015


namespace zed_wye_value_l90_9080

-- Define the types of coins
structure Coin where
  value : ℚ

-- Define the coins
def Ex : Coin := ⟨1⟩
def Wye : Coin := ⟨1⟩
def Zed : Coin := ⟨1⟩

-- Define the given conditions
axiom ex_wye_relation : 2 * Ex.value = 29 * Wye.value
axiom zed_ex_relation : Zed.value = 16 * Ex.value

theorem zed_wye_value : Zed.value = 232 * Wye.value :=
by sorry

end zed_wye_value_l90_9080


namespace gcd_problems_l90_9003

theorem gcd_problems :
  (Nat.gcd 72 168 = 24) ∧ (Nat.gcd 98 280 = 14) := by
  sorry

end gcd_problems_l90_9003


namespace tournament_prize_orders_l90_9021

/-- Represents the number of players in the tournament -/
def num_players : ℕ := 6

/-- Represents the number of elimination rounds in the tournament -/
def num_rounds : ℕ := 5

/-- Represents the number of possible outcomes for each match -/
def outcomes_per_match : ℕ := 2

/-- Theorem stating the number of possible prize orders in the tournament -/
theorem tournament_prize_orders :
  (outcomes_per_match ^ num_rounds : ℕ) = 32 := by sorry

end tournament_prize_orders_l90_9021


namespace pictures_hung_vertically_l90_9025

/-- Given a total of 30 pictures, with half hung horizontally and 5 hung haphazardly,
    prove that 10 pictures are hung vertically. -/
theorem pictures_hung_vertically (total : ℕ) (horizontal : ℕ) (haphazard : ℕ) :
  total = 30 →
  horizontal = total / 2 →
  haphazard = 5 →
  total - horizontal - haphazard = 10 := by
sorry

end pictures_hung_vertically_l90_9025


namespace fathers_age_l90_9077

theorem fathers_age (n m f : ℕ) (h1 : n * m = f / 7) (h2 : (n + 3) * (m + 3) = f + 3) : f = 21 := by
  sorry

end fathers_age_l90_9077


namespace problem_solution_l90_9093

theorem problem_solution (a b : ℝ) (h1 : a > 1) (h2 : a * b = a + b + 8) :
  (b > 1) ∧ 
  (∀ x y : ℝ, x > 1 ∧ x * y = x + y + 8 → a + b ≤ x + y) ∧
  (a * b = 16 ∧ ∀ x y : ℝ, x > 1 ∧ x * y = x + y + 8 → 16 ≤ x * y) := by
  sorry

end problem_solution_l90_9093


namespace rationalize_denominator_l90_9029

theorem rationalize_denominator : 7 / Real.sqrt 63 = Real.sqrt 7 / 3 := by sorry

end rationalize_denominator_l90_9029


namespace quadratic_inequality_range_l90_9047

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → x^2 - 2*x + a ≤ 0) → a ≤ -3 := by
  sorry

end quadratic_inequality_range_l90_9047


namespace tangent_point_coordinates_l90_9006

theorem tangent_point_coordinates :
  ∀ x y : ℝ,
  y = x^2 →
  (2 : ℝ) = 2*x →
  x = 1 ∧ y = 1 :=
by sorry

end tangent_point_coordinates_l90_9006


namespace vasyas_numbers_l90_9097

theorem vasyas_numbers (x y : ℝ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1/2 ∧ y = -1 := by
sorry

end vasyas_numbers_l90_9097


namespace probability_theorem_l90_9069

/-- The probability that the straight-line distance between two randomly chosen points 
    on the sides of a square with side length 2 is at least 1 -/
def probability_distance_at_least_one (S : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- A square with side length 2 -/
def square_side_two : Set (ℝ × ℝ) :=
  sorry

theorem probability_theorem :
  let S := square_side_two
  probability_distance_at_least_one S = (22 - π) / 32 := by
  sorry

end probability_theorem_l90_9069


namespace sin_difference_equals_four_l90_9016

theorem sin_difference_equals_four : 
  1 / Real.sin (10 * π / 180) - Real.sqrt 3 / Real.sin (80 * π / 180) = 4 := by
  sorry

end sin_difference_equals_four_l90_9016


namespace imaginary_part_of_fraction_l90_9088

theorem imaginary_part_of_fraction (i : ℂ) : i * i = -1 → Complex.im (5 * i / (2 - i)) = 2 := by
  sorry

end imaginary_part_of_fraction_l90_9088


namespace inequality_system_solution_set_l90_9066

theorem inequality_system_solution_set :
  ∀ x : ℝ, (3 * x - 1 ≥ x + 1 ∧ x + 4 > 4 * x - 2) ↔ (1 ≤ x ∧ x < 2) := by
  sorry

end inequality_system_solution_set_l90_9066


namespace chord_length_l90_9056

/-- The length of the chord cut off by a circle on a line --/
theorem chord_length (x y : ℝ) : 
  let line := {(x, y) | x - y - 3 = 0}
  let circle := {(x, y) | (x - 2)^2 + y^2 = 4}
  let chord := line ∩ circle
  ∃ p q : ℝ × ℝ, p ∈ chord ∧ q ∈ chord ∧ p ≠ q ∧ 
    Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) = Real.sqrt 14 :=
by sorry

end chord_length_l90_9056


namespace complementary_events_l90_9071

-- Define the sample space
def SampleSpace := Fin 4 × Fin 4

-- Define the events
def AtLeastOneBlack (outcome : SampleSpace) : Prop :=
  outcome.1 < 2 ∨ outcome.2 < 2

def BothRed (outcome : SampleSpace) : Prop :=
  outcome.1 ≥ 2 ∧ outcome.2 ≥ 2

-- Theorem statement
theorem complementary_events :
  ∀ (outcome : SampleSpace), AtLeastOneBlack outcome ↔ ¬BothRed outcome := by
  sorry

end complementary_events_l90_9071


namespace polynomial_evaluation_l90_9099

theorem polynomial_evaluation : (3 : ℝ)^3 + (3 : ℝ)^2 + (3 : ℝ) + 1 = 40 := by
  sorry

end polynomial_evaluation_l90_9099


namespace tylenol_interval_l90_9072

/-- Represents the duration of Jeremy's Tylenol regimen in weeks -/
def duration : ℕ := 2

/-- Represents the total number of pills Jeremy takes -/
def total_pills : ℕ := 112

/-- Represents the amount of Tylenol in each pill in milligrams -/
def mg_per_pill : ℕ := 500

/-- Represents the amount of Tylenol Jeremy takes per dose in milligrams -/
def mg_per_dose : ℕ := 1000

/-- Theorem stating that the time interval between doses is 6 hours -/
theorem tylenol_interval : 
  (duration * 7 * 24) / ((total_pills * mg_per_pill) / mg_per_dose) = 6 := by
  sorry


end tylenol_interval_l90_9072


namespace revenue_growth_equation_l90_9090

def january_revenue : ℝ := 250
def quarter_target : ℝ := 900

theorem revenue_growth_equation (x : ℝ) :
  january_revenue + january_revenue * (1 + x) + january_revenue * (1 + x)^2 = quarter_target :=
by sorry

end revenue_growth_equation_l90_9090


namespace sqrt_fraction_simplification_l90_9042

theorem sqrt_fraction_simplification : 
  Real.sqrt ((25 : ℝ) / 36 - 4 / 9) = (1 : ℝ) / 2 := by sorry

end sqrt_fraction_simplification_l90_9042


namespace pond_to_field_ratio_l90_9094

theorem pond_to_field_ratio : 
  let field_length : ℝ := 48
  let field_width : ℝ := 24
  let pond_side : ℝ := 8
  let field_area : ℝ := field_length * field_width
  let pond_area : ℝ := pond_side * pond_side
  pond_area / field_area = 1 / 18 := by
  sorry

end pond_to_field_ratio_l90_9094
