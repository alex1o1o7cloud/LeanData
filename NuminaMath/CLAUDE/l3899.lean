import Mathlib

namespace inequality_preservation_l3899_389945

theorem inequality_preservation (a b : ℝ) (h : a > b) : a - 3 > b - 3 := by
  sorry

end inequality_preservation_l3899_389945


namespace debate_club_committee_compositions_l3899_389903

def total_candidates : ℕ := 20
def past_members : ℕ := 10
def committee_size : ℕ := 5
def min_past_members : ℕ := 3

theorem debate_club_committee_compositions :
  (Nat.choose past_members min_past_members * Nat.choose (total_candidates - past_members) (committee_size - min_past_members)) +
  (Nat.choose past_members (min_past_members + 1) * Nat.choose (total_candidates - past_members) (committee_size - (min_past_members + 1))) +
  (Nat.choose past_members committee_size) = 7752 := by
  sorry

end debate_club_committee_compositions_l3899_389903


namespace function_translation_l3899_389992

-- Define a function type for f
def FunctionType := ℝ → ℝ

-- Define the translation vector
def TranslationVector := ℝ × ℝ

-- State the theorem
theorem function_translation
  (f : FunctionType)
  (a : TranslationVector) :
  (∀ x y : ℝ, y = f (2*x - 1) + 1 ↔ 
              y + a.2 = f (2*(x + a.1) - 1) + 1) →
  (∀ x y : ℝ, y = f (2*x + 1) - 1 ↔ 
              y = f (2*(x + a.1) + 1) - 1) →
  a = (1, -2) :=
sorry

end function_translation_l3899_389992


namespace movie_trip_cost_l3899_389965

/-- The total cost of a movie trip for a group of adults and children -/
def total_cost (num_adults num_children : ℕ) (adult_ticket_price child_ticket_price concession_cost : ℚ) : ℚ :=
  num_adults * adult_ticket_price + num_children * child_ticket_price + concession_cost

/-- Theorem stating that the total cost for the given group is $76 -/
theorem movie_trip_cost : 
  total_cost 5 2 10 7 12 = 76 := by
  sorry

end movie_trip_cost_l3899_389965


namespace division_problem_l3899_389987

theorem division_problem (a b : ℕ+) (q r : ℤ) 
  (h1 : (a : ℤ) * (b : ℤ) = q * ((a : ℤ) + (b : ℤ)) + r)
  (h2 : 0 ≤ r ∧ r < (a : ℤ) + (b : ℤ))
  (h3 : q^2 + r = 2011) :
  ∃ t : ℕ, 1 ≤ t ∧ t ≤ 45 ∧ 
  ((a : ℤ) = t ∧ (b : ℤ) = t + 2012 ∨ (a : ℤ) = t + 2012 ∧ (b : ℤ) = t) :=
by sorry

end division_problem_l3899_389987


namespace fraction_over_65_l3899_389962

theorem fraction_over_65 (total : ℕ) (under_21 : ℕ) (over_65 : ℕ) : 
  (3 : ℚ) / 7 * total = under_21 →
  50 < total →
  total < 100 →
  under_21 = 33 →
  (over_65 : ℚ) / total = over_65 / 77 :=
by sorry

end fraction_over_65_l3899_389962


namespace binomial_cube_special_case_l3899_389913

theorem binomial_cube_special_case : 8^3 + 3*(8^2) + 3*8 + 1 = 729 := by
  sorry

end binomial_cube_special_case_l3899_389913


namespace quadratic_expression_value_l3899_389976

theorem quadratic_expression_value (x : ℝ) : 
  x = -2 → x^2 + 6*x - 8 = -16 := by
  sorry

end quadratic_expression_value_l3899_389976


namespace operation_result_l3899_389916

theorem operation_result (c : ℚ) : 
  2 * ((3 * c + 6 - 5 * c) / 3) = -4/3 * c + 4 := by
  sorry

end operation_result_l3899_389916


namespace max_value_of_a_l3899_389920

theorem max_value_of_a (a b c : ℝ) (sum_zero : a + b + c = 0) (sum_squares_six : a^2 + b^2 + c^2 = 6) :
  ∃ (max_a : ℝ), max_a = 2 ∧ ∀ x, (∃ y z, x + y + z = 0 ∧ x^2 + y^2 + z^2 = 6) → x ≤ max_a :=
sorry

end max_value_of_a_l3899_389920


namespace lowest_cost_plan_l3899_389995

/-- Represents a gardening style arrangement plan -/
structure ArrangementPlan where
  style_a : ℕ
  style_b : ℕ

/-- Represents the gardening problem setup -/
structure GardeningProblem where
  total_sets : ℕ
  type_a_flowers : ℕ
  type_b_flowers : ℕ
  style_a_type_a : ℕ
  style_a_type_b : ℕ
  style_b_type_a : ℕ
  style_b_type_b : ℕ
  style_a_cost : ℕ
  style_b_cost : ℕ

/-- Checks if an arrangement plan is feasible -/
def is_feasible (problem : GardeningProblem) (plan : ArrangementPlan) : Prop :=
  plan.style_a + plan.style_b = problem.total_sets ∧
  plan.style_a * problem.style_a_type_a + plan.style_b * problem.style_b_type_a ≤ problem.type_a_flowers ∧
  plan.style_a * problem.style_a_type_b + plan.style_b * problem.style_b_type_b ≤ problem.type_b_flowers

/-- Calculates the cost of an arrangement plan -/
def cost (problem : GardeningProblem) (plan : ArrangementPlan) : ℕ :=
  plan.style_a * problem.style_a_cost + plan.style_b * problem.style_b_cost

/-- The main theorem to be proved -/
theorem lowest_cost_plan (problem : GardeningProblem) 
  (h_problem : problem = { 
    total_sets := 50,
    type_a_flowers := 2660,
    type_b_flowers := 3000,
    style_a_type_a := 70,
    style_a_type_b := 30,
    style_b_type_a := 40,
    style_b_type_b := 80,
    style_a_cost := 800,
    style_b_cost := 960
  }) :
  ∃ (optimal_plan : ArrangementPlan),
    is_feasible problem optimal_plan ∧
    cost problem optimal_plan = 44480 ∧
    ∀ (other_plan : ArrangementPlan), 
      is_feasible problem other_plan → 
      cost problem other_plan ≥ cost problem optimal_plan :=
sorry

end lowest_cost_plan_l3899_389995


namespace least_three_digit_multiple_of_3_4_5_l3899_389980

theorem least_three_digit_multiple_of_3_4_5 : 
  (∀ n : ℕ, n ≥ 100 ∧ n < 120 → ¬(3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n)) ∧ 
  (120 ≥ 100 ∧ 3 ∣ 120 ∧ 4 ∣ 120 ∧ 5 ∣ 120) := by
  sorry

end least_three_digit_multiple_of_3_4_5_l3899_389980


namespace min_fence_length_l3899_389969

theorem min_fence_length (w : ℝ) (l : ℝ) (area : ℝ) (perimeter : ℝ) : 
  w > 0 →
  l = 2 * w →
  area = l * w →
  area ≥ 500 →
  perimeter = 2 * (l + w) →
  perimeter ≥ 96 :=
by sorry

end min_fence_length_l3899_389969


namespace pencils_remaining_l3899_389946

theorem pencils_remaining (initial_pencils : ℕ) (pencils_removed : ℕ) : 
  initial_pencils = 9 → pencils_removed = 4 → initial_pencils - pencils_removed = 5 := by
  sorry

end pencils_remaining_l3899_389946


namespace max_sum_of_digits_24hour_clock_l3899_389935

/-- Represents a time in 24-hour format -/
structure Time24 where
  hours : Nat
  minutes : Nat
  hour_valid : hours < 24
  minute_valid : minutes < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  let digits := n.repr.toList.map (fun c => c.toString.toNat!)
  digits.sum

/-- Calculates the sum of digits for a Time24 -/
def timeSumOfDigits (t : Time24) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes

/-- The largest possible sum of digits in a 24-hour digital clock display is 24 -/
theorem max_sum_of_digits_24hour_clock :
  (∀ t : Time24, timeSumOfDigits t ≤ 24) ∧
  (∃ t : Time24, timeSumOfDigits t = 24) :=
sorry

end max_sum_of_digits_24hour_clock_l3899_389935


namespace prime_power_form_l3899_389912

theorem prime_power_form (n : ℕ) (h : Nat.Prime (4^n + 2^n + 1)) :
  ∃ k : ℕ, n = 3^k :=
by sorry

end prime_power_form_l3899_389912


namespace farmer_pomelo_shipment_l3899_389900

/-- Calculates the total number of dozens of pomelos shipped given the number of boxes and pomelos from last week and the number of boxes shipped this week. -/
def totalDozensShipped (lastWeekBoxes : ℕ) (lastWeekPomelos : ℕ) (thisWeekBoxes : ℕ) : ℕ :=
  let pomelosPerBox := lastWeekPomelos / lastWeekBoxes
  let totalPomelos := lastWeekPomelos + thisWeekBoxes * pomelosPerBox
  totalPomelos / 12

/-- Proves that given 10 boxes containing 240 pomelos in total last week, and 20 boxes shipped this week, the total number of dozens of pomelos shipped is 40. -/
theorem farmer_pomelo_shipment :
  totalDozensShipped 10 240 20 = 40 := by
  sorry

end farmer_pomelo_shipment_l3899_389900


namespace toys_needed_l3899_389991

theorem toys_needed (available : ℕ) (people : ℕ) (per_person : ℕ) : 
  available = 68 → people = 14 → per_person = 5 → 
  (people * per_person - available : ℕ) = 2 := by
  sorry

end toys_needed_l3899_389991


namespace total_spent_on_car_parts_l3899_389941

def speakers : ℚ := 235.87
def newTires : ℚ := 281.45
def steeringWheelCover : ℚ := 179.99
def seatCovers : ℚ := 122.31
def headlights : ℚ := 98.63

theorem total_spent_on_car_parts : 
  speakers + newTires + steeringWheelCover + seatCovers + headlights = 918.25 := by
  sorry

end total_spent_on_car_parts_l3899_389941


namespace vector_b_exists_l3899_389922

def a : ℝ × ℝ := (1, -2)

theorem vector_b_exists : ∃ (b : ℝ × ℝ), 
  (∃ (k : ℝ), b = k • a) ∧ 
  (‖a + b‖ < ‖a‖) ∧
  b = (-1, 2) := by
  sorry

end vector_b_exists_l3899_389922


namespace siena_bookmarks_theorem_l3899_389939

/-- The number of pages Siena bookmarks every day -/
def pages_per_day : ℕ := 30

/-- The number of pages Siena has at the start of March -/
def initial_pages : ℕ := 400

/-- The number of pages Siena will have at the end of March -/
def final_pages : ℕ := 1330

/-- The number of days in March -/
def days_in_march : ℕ := 31

theorem siena_bookmarks_theorem :
  initial_pages + pages_per_day * days_in_march = final_pages :=
by sorry

end siena_bookmarks_theorem_l3899_389939


namespace tournament_battles_one_team_remains_l3899_389944

/-- The number of battles needed to determine a champion in a tournament --/
def battles_to_champion (initial_teams : ℕ) : ℕ :=
  if initial_teams ≤ 1 then 0
  else if initial_teams = 2 then 1
  else (initial_teams - 1) / 2

/-- Theorem: In a tournament with 2017 teams, 1008 battles are needed to determine a champion --/
theorem tournament_battles :
  battles_to_champion 2017 = 1008 := by
  sorry

/-- Lemma: The number of teams remaining after n battles --/
lemma teams_remaining (initial_teams n : ℕ) : ℕ :=
  if n ≥ (initial_teams - 1) / 2 then 1
  else initial_teams - 2 * n

/-- Theorem: After 1008 battles, only one team remains in a tournament of 2017 teams --/
theorem one_team_remains :
  teams_remaining 2017 1008 = 1 := by
  sorry

end tournament_battles_one_team_remains_l3899_389944


namespace jenny_egg_distribution_l3899_389921

theorem jenny_egg_distribution (n : ℕ) : 
  n ∣ 18 ∧ n ∣ 24 ∧ n ≥ 4 → n = 6 :=
by sorry

end jenny_egg_distribution_l3899_389921


namespace triangular_prism_ratio_l3899_389934

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a triangular prism -/
structure TriangularPrism where
  A : Point3D
  B : Point3D
  C : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D

/-- Checks if two planes are perpendicular -/
def arePlanesPerp (p1 p2 p3 q1 q2 q3 : Point3D) : Prop := sorry

/-- Checks if two vectors are perpendicular -/
def areVectorsPerp (v1 v2 : Point3D) : Prop := sorry

/-- Calculates the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Checks if a point lies on a line segment -/
def isOnLineSegment (p1 p2 p : Point3D) : Prop := sorry

/-- Main theorem -/
theorem triangular_prism_ratio 
  (prism : TriangularPrism)
  (D : Point3D)
  (h1 : distance prism.A prism.A₁ = 4)
  (h2 : distance prism.A prism.C = 4)
  (h3 : distance prism.A₁ prism.C₁ = 4)
  (h4 : distance prism.C prism.C₁ = 4)
  (h5 : arePlanesPerp prism.A prism.B prism.C prism.A prism.A₁ prism.C₁)
  (h6 : distance prism.A prism.B = 3)
  (h7 : distance prism.B prism.C = 5)
  (h8 : isOnLineSegment prism.B prism.C₁ D)
  (h9 : areVectorsPerp (Point3D.mk (D.x - prism.A.x) (D.y - prism.A.y) (D.z - prism.A.z))
                       (Point3D.mk (prism.B.x - prism.A₁.x) (prism.B.y - prism.A₁.y) (prism.B.z - prism.A₁.z))) :
  distance prism.B D / distance prism.B prism.C₁ = 9 / 25 := by
  sorry

end triangular_prism_ratio_l3899_389934


namespace translation_theorem_l3899_389972

/-- A translation in the complex plane that moves 1 - 3i to 5 + 2i also moves 3 - 4i to 7 + i -/
theorem translation_theorem (t : ℂ → ℂ) :
  (t (1 - 3*I) = 5 + 2*I) →
  (∃ w : ℂ, ∀ z : ℂ, t z = z + w) →
  t (3 - 4*I) = 7 + I :=
by sorry

end translation_theorem_l3899_389972


namespace compare_powers_l3899_389902

theorem compare_powers (x m n : ℝ) (hx_pos : x > 0) (hx_neq_one : x ≠ 1) (hm_gt_n : m > n) (hn_pos : n > 0) :
  x^m + 1/x^m > x^n + 1/x^n := by
  sorry

end compare_powers_l3899_389902


namespace necessary_not_sufficient_condition_l3899_389918

theorem necessary_not_sufficient_condition (a : ℝ) :
  (((a - 1) * (a - 2) = 0) → (a = 2)) ∧
  ¬(∀ a : ℝ, ((a - 1) * (a - 2) = 0) ↔ (a = 2)) :=
by sorry

end necessary_not_sufficient_condition_l3899_389918


namespace tile_border_ratio_l3899_389975

theorem tile_border_ratio (p b : ℝ) (h_positive : p > 0 ∧ b > 0) : 
  (225 * p^2) / ((15 * p + 30 * b)^2) = 49/100 → b/p = 4/7 := by
sorry

end tile_border_ratio_l3899_389975


namespace sum_of_repeating_decimals_l3899_389988

/-- Represents a repeating decimal with a given numerator and denominator. -/
def repeating_decimal (numerator denominator : ℕ) : ℚ :=
  numerator / denominator

/-- The sum of the given repeating decimals is equal to 2224/9999. -/
theorem sum_of_repeating_decimals :
  repeating_decimal 2 9 + repeating_decimal 2 99 + repeating_decimal 2 9999 = 2224 / 9999 := by
  sorry

end sum_of_repeating_decimals_l3899_389988


namespace scenario1_probability_scenario2_probability_l3899_389949

-- Define the probabilities
def prob_A_hit : ℚ := 2/3
def prob_B_hit : ℚ := 3/4

-- Define the number of shots for each scenario
def shots_scenario1 : ℕ := 3
def shots_scenario2 : ℕ := 2

-- Theorem for scenario 1
theorem scenario1_probability : 
  (1 - prob_A_hit ^ shots_scenario1) = 19/27 := by sorry

-- Theorem for scenario 2
theorem scenario2_probability : 
  (Nat.choose shots_scenario2 shots_scenario2 * prob_A_hit ^ shots_scenario2) *
  (Nat.choose shots_scenario2 1 * prob_B_hit ^ 1 * (1 - prob_B_hit) ^ (shots_scenario2 - 1)) = 1/6 := by sorry

end scenario1_probability_scenario2_probability_l3899_389949


namespace parking_lot_perimeter_l3899_389910

theorem parking_lot_perimeter 
  (d : ℝ) (A : ℝ) (x y : ℝ) (P : ℝ) 
  (h1 : d = 20) 
  (h2 : A = 120) 
  (h3 : x = (2/3) * y) 
  (h4 : x^2 + y^2 = d^2) 
  (h5 : x * y = A) 
  (h6 : P = 2 * (x + y)) : 
  P = 20 * Real.sqrt 5 := by
  sorry

end parking_lot_perimeter_l3899_389910


namespace final_box_weight_l3899_389998

/-- The weight of the box after each step of adding ingredients --/
def box_weight (initial : ℝ) (triple : ℝ → ℝ) (add_two : ℝ → ℝ) (double : ℝ → ℝ) : ℝ :=
  double (add_two (triple initial))

/-- The theorem stating the final weight of the box --/
theorem final_box_weight :
  box_weight 2 (fun x => 3 * x) (fun x => x + 2) (fun x => 2 * x) = 16 := by
  sorry

#check final_box_weight

end final_box_weight_l3899_389998


namespace unique_five_digit_number_l3899_389954

def is_valid_digit (d : ℕ) : Prop := d ≥ 1 ∧ d ≤ 5

def digits_to_number (a b c : ℕ) : ℕ := 100 * a + 10 * b + c

theorem unique_five_digit_number 
  (P Q R S T : ℕ) 
  (h_valid : ∀ d ∈ [P, Q, R, S, T], is_valid_digit d)
  (h_distinct : P ≠ Q ∧ P ≠ R ∧ P ≠ S ∧ P ≠ T ∧ Q ≠ R ∧ Q ≠ S ∧ Q ≠ T ∧ R ≠ S ∧ R ≠ T ∧ S ≠ T)
  (h_div_4 : digits_to_number P Q R % 4 = 0)
  (h_div_5 : digits_to_number Q R S % 5 = 0)
  (h_div_3 : digits_to_number R S T % 3 = 0) :
  P = 1 := by
  sorry

end unique_five_digit_number_l3899_389954


namespace inequality_solution_set_l3899_389953

theorem inequality_solution_set (x : ℝ) : 
  (x - 2) * (x + 2) < 5 ↔ -3 < x ∧ x < 3 := by sorry

end inequality_solution_set_l3899_389953


namespace solution_equation1_solution_equation2_l3899_389923

-- Define the equations
def equation1 (x : ℝ) : Prop := 3 * (x - 1)^3 = 24
def equation2 (x : ℝ) : Prop := (x - 3)^2 = 64

-- Theorem for the first equation
theorem solution_equation1 : ∃ x : ℝ, equation1 x ∧ x = 3 := by sorry

-- Theorem for the second equation
theorem solution_equation2 : ∃ x₁ x₂ : ℝ, equation2 x₁ ∧ equation2 x₂ ∧ x₁ = 11 ∧ x₂ = -5 := by sorry

end solution_equation1_solution_equation2_l3899_389923


namespace sqrt_combinable_with_sqrt_6_l3899_389957

theorem sqrt_combinable_with_sqrt_6 :
  ∀ x : ℝ, x > 0 →
  (x = 12 ∨ x = 15 ∨ x = 18 ∨ x = 24) →
  (∃ q : ℚ, Real.sqrt x = q * Real.sqrt 6) ↔ x = 24 := by
sorry

end sqrt_combinable_with_sqrt_6_l3899_389957


namespace dalton_needs_sixteen_more_l3899_389979

theorem dalton_needs_sixteen_more : ∀ (jump_rope board_game ball puzzle saved uncle_gift : ℕ),
  jump_rope = 9 →
  board_game = 15 →
  ball = 5 →
  puzzle = 8 →
  saved = 7 →
  uncle_gift = 14 →
  jump_rope + board_game + ball + puzzle - (saved + uncle_gift) = 16 := by
  sorry

end dalton_needs_sixteen_more_l3899_389979


namespace right_pyramid_height_l3899_389955

/-- The height of a right pyramid with a square base -/
theorem right_pyramid_height (perimeter base_side diagonal_half : ℝ) 
  (apex_to_vertex : ℝ) (h_perimeter : perimeter = 40) 
  (h_base_side : base_side = perimeter / 4)
  (h_diagonal_half : diagonal_half = base_side * Real.sqrt 2 / 2)
  (h_apex_to_vertex : apex_to_vertex = 15) : 
  Real.sqrt (apex_to_vertex ^ 2 - diagonal_half ^ 2) = 5 * Real.sqrt 7 := by
  sorry

#check right_pyramid_height

end right_pyramid_height_l3899_389955


namespace sphere_properties_l3899_389958

/-- Given a sphere with volume 288π cubic inches, prove its surface area is 144π square inches and its diameter is 12 inches -/
theorem sphere_properties (r : ℝ) (h : (4/3) * Real.pi * r^3 = 288 * Real.pi) :
  (4 * Real.pi * r^2 = 144 * Real.pi) ∧ (2 * r = 12) := by
  sorry

end sphere_properties_l3899_389958


namespace parabola_properties_l3899_389919

/-- Parabola C with vertex at origin and focus on y-axis -/
structure Parabola where
  focus : ℝ
  equation : ℝ → ℝ → Prop

/-- Point on the parabola -/
structure PointOnParabola (C : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : C.equation x y

/-- Line segment on the parabola -/
structure LineSegmentOnParabola (C : Parabola) where
  A : PointOnParabola C
  B : PointOnParabola C

/-- Triangle on the parabola -/
structure TriangleOnParabola (C : Parabola) where
  A : PointOnParabola C
  B : PointOnParabola C
  D : PointOnParabola C

theorem parabola_properties (C : Parabola) (Q : PointOnParabola C) 
    (AB : LineSegmentOnParabola C) (M : ℝ) (ABD : TriangleOnParabola C) :
  Q.x = Real.sqrt 8 ∧ Q.y = 2 ∧ (Q.x - C.focus)^2 + Q.y^2 = 9 →
  (∃ (m : ℝ), m > 0 ∧ 
    (∃ (k : ℝ), AB.A.y = k * AB.A.x + m ∧ AB.B.y = k * AB.B.x + m) ∧
    AB.A.x * AB.B.x + AB.A.y * AB.B.y = 0) →
  ABD.D.x < ABD.A.x ∧ ABD.A.x < ABD.B.x ∧
  (ABD.B.x - ABD.A.x)^2 + (ABD.B.y - ABD.A.y)^2 = 
    (ABD.D.x - ABD.A.x)^2 + (ABD.D.y - ABD.A.y)^2 ∧
  (ABD.B.x - ABD.A.x) * (ABD.D.x - ABD.A.x) + 
    (ABD.B.y - ABD.A.y) * (ABD.D.y - ABD.A.y) = 0 →
  C.equation = (fun x y => x^2 = 4*y) ∧ 
  M = 4 ∧
  (∀ (ABD' : TriangleOnParabola C), 
    ABD'.D.x < ABD'.A.x ∧ ABD'.A.x < ABD'.B.x ∧
    (ABD'.B.x - ABD'.A.x)^2 + (ABD'.B.y - ABD'.A.y)^2 = 
      (ABD'.D.x - ABD'.A.x)^2 + (ABD'.D.y - ABD'.A.y)^2 ∧
    (ABD'.B.x - ABD'.A.x) * (ABD'.D.x - ABD'.A.x) + 
      (ABD'.B.y - ABD'.A.y) * (ABD'.D.y - ABD'.A.y) = 0 →
    (ABD'.B.x - ABD'.A.x) * (ABD'.D.y - ABD'.A.y) -
      (ABD'.B.y - ABD'.A.y) * (ABD'.D.x - ABD'.A.x) ≥ 8) :=
by sorry

end parabola_properties_l3899_389919


namespace corn_plants_per_row_l3899_389989

/-- Calculates the number of corn plants in each row given the water pumping conditions. -/
theorem corn_plants_per_row (
  pump_rate : ℝ)
  (pump_time : ℝ)
  (num_rows : ℕ)
  (water_per_plant : ℝ)
  (num_pigs : ℕ)
  (water_per_pig : ℝ)
  (num_ducks : ℕ)
  (water_per_duck : ℝ)
  (h_pump_rate : pump_rate = 3)
  (h_pump_time : pump_time = 25)
  (h_num_rows : num_rows = 4)
  (h_water_per_plant : water_per_plant = 0.5)
  (h_num_pigs : num_pigs = 10)
  (h_water_per_pig : water_per_pig = 4)
  (h_num_ducks : num_ducks = 20)
  (h_water_per_duck : water_per_duck = 0.25) :
  (pump_rate * pump_time - (num_pigs * water_per_pig + num_ducks * water_per_duck)) / (num_rows * water_per_plant) = 15 := by
sorry

end corn_plants_per_row_l3899_389989


namespace jellybean_theorem_l3899_389911

def jellybean_problem (initial : ℕ) (samantha_took : ℕ) (shelby_ate : ℕ) : ℕ :=
  let remaining_after_samantha := initial - samantha_took
  let remaining_after_shelby := remaining_after_samantha - shelby_ate
  let total_removed := samantha_took + shelby_ate
  let shannon_added := total_removed / 2
  remaining_after_shelby + shannon_added

theorem jellybean_theorem :
  jellybean_problem 90 24 12 = 72 := by
  sorry

#eval jellybean_problem 90 24 12

end jellybean_theorem_l3899_389911


namespace triangle_properties_l3899_389971

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  0 < a ∧ 0 < b ∧ 0 < c ∧
  A + B + C = Real.pi ∧
  (Real.cos A - 2 * Real.cos C) / Real.cos B = (2 * c - a) / b ∧
  Real.cos B = 1/4 ∧
  1/2 * a * c * Real.sin B = Real.sqrt 15 / 4 →
  Real.sin C / Real.sin A = 2 ∧
  a + b + c = 5 := by sorry

end triangle_properties_l3899_389971


namespace fraction_equality_l3899_389964

theorem fraction_equality : (1992^2 - 1985^2) / (2001^2 - 1976^2) = 7 / 25 := by
  sorry

end fraction_equality_l3899_389964


namespace ben_bonus_allocation_l3899_389966

theorem ben_bonus_allocation (bonus : ℚ) (holiday_fraction : ℚ) (gift_fraction : ℚ) (remaining : ℚ) 
  (h1 : bonus = 1496)
  (h2 : holiday_fraction = 1/4)
  (h3 : gift_fraction = 1/8)
  (h4 : remaining = 867) :
  let kitchen_fraction := (bonus - remaining - holiday_fraction * bonus - gift_fraction * bonus) / bonus
  kitchen_fraction = 221/748 := by sorry

end ben_bonus_allocation_l3899_389966


namespace sqrt_inequality_solution_set_l3899_389985

theorem sqrt_inequality_solution_set (x : ℝ) :
  (x + 3 ≥ 0) → (∃ y, y > 1 ∧ x = y) ↔ (Real.sqrt (x + 3) > 3 - x) :=
by sorry

end sqrt_inequality_solution_set_l3899_389985


namespace circles_intersect_l3899_389963

theorem circles_intersect (r R d : ℝ) (hr : r = 4) (hR : R = 5) (hd : d = 6) :
  let sum := r + R
  let diff := R - r
  d > diff ∧ d < sum := by sorry

end circles_intersect_l3899_389963


namespace bargaining_range_l3899_389961

def marked_price : ℝ := 100

def min_markup_percent : ℝ := 50
def max_markup_percent : ℝ := 100

def min_profit_percent : ℝ := 20

def lower_bound : ℝ := 60
def upper_bound : ℝ := 80

theorem bargaining_range :
  ∀ (cost_price : ℝ),
    (cost_price * (1 + min_markup_percent / 100) ≤ marked_price) →
    (cost_price * (1 + max_markup_percent / 100) ≥ marked_price) →
    (lower_bound ≥ cost_price * (1 + min_profit_percent / 100)) ∧
    (upper_bound ≤ marked_price) ∧
    (lower_bound ≤ upper_bound) :=
by sorry

end bargaining_range_l3899_389961


namespace repeating_digit_divisible_by_101_l3899_389977

/-- A 9-digit integer where the first three digits are the same as the middle three and last three digits -/
def RepeatingDigitInteger (x y z : ℕ) : ℕ :=
  100100100 * x + 10010010 * y + 1001001 * z

/-- Theorem stating that 101 is a factor of any RepeatingDigitInteger -/
theorem repeating_digit_divisible_by_101 (x y z : ℕ) (h : 0 < x ∧ x < 10 ∧ y < 10 ∧ z < 10) :
  101 ∣ RepeatingDigitInteger x y z := by
  sorry

#check repeating_digit_divisible_by_101

end repeating_digit_divisible_by_101_l3899_389977


namespace min_correct_answers_to_advance_l3899_389933

/-- Given a math competition with the following conditions:
  * There are 25 questions in total
  * Each correct answer is worth 4 points
  * Each incorrect or unanswered question results in -1 point
  * A minimum of 60 points is required to advance
  This theorem proves that the minimum number of correctly answered questions
  to advance is 17. -/
theorem min_correct_answers_to_advance (total_questions : ℕ) (correct_points : ℤ) 
  (incorrect_points : ℤ) (min_points_to_advance : ℤ) :
  total_questions = 25 →
  correct_points = 4 →
  incorrect_points = -1 →
  min_points_to_advance = 60 →
  ∃ (min_correct : ℕ), 
    min_correct = 17 ∧ 
    (min_correct : ℤ) * correct_points + (total_questions - min_correct) * incorrect_points ≥ min_points_to_advance ∧
    ∀ (x : ℕ), x < min_correct → 
      (x : ℤ) * correct_points + (total_questions - x) * incorrect_points < min_points_to_advance :=
by sorry

end min_correct_answers_to_advance_l3899_389933


namespace star_properties_l3899_389952

-- Define the * operation
def star (x y : ℝ) : ℝ := (x + 1) * (y + 1) - 1

-- State the theorem
theorem star_properties :
  ∀ x y : ℝ,
  (star x y = star y x) ∧
  (star (x + 1) (x - 1) = x * x - 1) := by
  sorry

end star_properties_l3899_389952


namespace smallest_possible_area_l3899_389956

theorem smallest_possible_area (S : ℕ) (A : ℕ) : 
  S * S = 2019 + A  -- Total area equation
  → A ≠ 1  -- Area of 2020th square is not 1
  → A ≥ 9  -- Smallest possible area is at least 9
  → ∃ (S' : ℕ), S' * S' = 2019 + 9  -- There exists a solution with area 9
  → A = 9  -- The smallest area is indeed 9
  := by sorry

end smallest_possible_area_l3899_389956


namespace paint_coverage_l3899_389959

/-- Proves that a quart of paint covers 60 square feet given the specified conditions -/
theorem paint_coverage (cube_edge : Real) (paint_cost_per_quart : Real) (total_paint_cost : Real)
  (h1 : cube_edge = 10)
  (h2 : paint_cost_per_quart = 3.2)
  (h3 : total_paint_cost = 32) :
  (6 * cube_edge^2) / (total_paint_cost / paint_cost_per_quart) = 60 := by
  sorry

end paint_coverage_l3899_389959


namespace lead_in_mixture_l3899_389908

theorem lead_in_mixture (total : ℝ) (copper_weight : ℝ) (lead_percent : ℝ) (copper_percent : ℝ)
  (h1 : copper_weight = 12)
  (h2 : copper_percent = 0.60)
  (h3 : lead_percent = 0.25)
  (h4 : copper_weight = copper_percent * total) :
  lead_percent * total = 5 := by
sorry

end lead_in_mixture_l3899_389908


namespace bank_document_error_l3899_389994

def ends_with (n : ℕ) (d : ℕ) : Prop := n % 10 = d

theorem bank_document_error (S D N R : ℕ) : 
  ends_with S 7 →
  ends_with N 3 →
  ends_with D 5 →
  ends_with R 1 →
  S = D * N + R →
  False :=
by sorry

end bank_document_error_l3899_389994


namespace power_function_monotone_iff_m_eq_three_l3899_389997

/-- A power function f(x) = (m^2 - 2m - 2) * x^(m-2) is monotonically increasing on (0, +∞) if and only if m = 3 -/
theorem power_function_monotone_iff_m_eq_three (m : ℝ) :
  (∀ x > 0, Monotone (fun x => (m^2 - 2*m - 2) * x^(m-2))) ↔ m = 3 := by
  sorry

end power_function_monotone_iff_m_eq_three_l3899_389997


namespace least_number_with_remainder_l3899_389940

theorem least_number_with_remainder (n : ℕ) : n = 256 ↔ 
  (∀ m, m < n → ¬(m % 7 = 4 ∧ m % 9 = 4 ∧ m % 12 = 4 ∧ m % 18 = 4)) ∧
  n % 7 = 4 ∧ n % 9 = 4 ∧ n % 12 = 4 ∧ n % 18 = 4 :=
by sorry

end least_number_with_remainder_l3899_389940


namespace shortest_path_across_river_l3899_389999

/-- Given two points A and B on opposite sides of a straight line (river),
    with A being 5 km north and 1 km west of B,
    prove that the shortest path from A to B crossing the line perpendicularly
    is 6 km long. -/
theorem shortest_path_across_river (A B : ℝ × ℝ) : 
  A.1 = B.1 - 1 →  -- A is 1 km west of B
  A.2 = B.2 + 5 →  -- A is 5 km north of B
  ∃ (C : ℝ × ℝ), 
    (C.1 - A.1) * (B.2 - A.2) = (C.2 - A.2) * (B.1 - A.1) ∧  -- C is on the line AB
    (C.2 = A.2 ∨ C.2 = B.2) ∧  -- C is on the same level as A or B (representing the river)
    Real.sqrt ((C.1 - A.1)^2 + (C.2 - A.2)^2) + 
    Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 6 :=
by sorry

end shortest_path_across_river_l3899_389999


namespace remainder_equality_l3899_389914

theorem remainder_equality (A B D S S' : ℕ) (hA : A > B) :
  A % D = S →
  B % D = S' →
  (A + B) % D = (S + S') % D :=
by sorry

end remainder_equality_l3899_389914


namespace crayons_given_to_friends_l3899_389973

theorem crayons_given_to_friends (initial : ℕ) (lost : ℕ) (remaining : ℕ) 
  (h1 : initial = 440)
  (h2 : lost = 106)
  (h3 : remaining = 223) :
  initial - lost - remaining = 111 := by
  sorry

end crayons_given_to_friends_l3899_389973


namespace min_value_theorem_l3899_389970

theorem min_value_theorem (a b x : ℝ) (ha : a > 0) (hb : b > 0) (hx : 0 < x ∧ x < 1) :
  a^2 / x + b^2 / (1 - x) ≥ (a + b)^2 ∧ 
  ∃ y, 0 < y ∧ y < 1 ∧ a^2 / y + b^2 / (1 - y) = (a + b)^2 :=
by sorry

end min_value_theorem_l3899_389970


namespace taxi_fare_calculation_l3899_389948

/-- Proves that the charge for each additional 1/5 mile is $0.40 --/
theorem taxi_fare_calculation (initial_charge : ℚ) (total_distance : ℚ) (total_charge : ℚ) :
  initial_charge = 280/100 →
  total_distance = 8 →
  total_charge = 1840/100 →
  let additional_distance : ℚ := total_distance - 1/5
  let additional_increments : ℚ := additional_distance / (1/5)
  let charge_per_increment : ℚ := (total_charge - initial_charge) / additional_increments
  charge_per_increment = 40/100 := by
  sorry

end taxi_fare_calculation_l3899_389948


namespace eleventh_flip_probability_l3899_389932

def is_fair_coin (coin : Type) : Prop := sorry

def probability_of_tails (coin : Type) : ℚ := sorry

def previous_flips_heads (coin : Type) (n : ℕ) : Prop := sorry

theorem eleventh_flip_probability (coin : Type) 
  (h_fair : is_fair_coin coin)
  (h_previous : previous_flips_heads coin 10) :
  probability_of_tails coin = 1/2 := by sorry

end eleventh_flip_probability_l3899_389932


namespace go_pieces_theorem_l3899_389926

/-- Represents the set of Go pieces -/
structure GoPieces where
  white : ℕ
  black : ℕ

/-- Calculates the probability of drawing two pieces of the same color in the first two draws -/
def prob_same_color (pieces : GoPieces) : ℚ :=
  sorry

/-- Calculates the expected value of the number of white Go pieces drawn in the first four draws -/
def expected_white_pieces (pieces : GoPieces) : ℚ :=
  sorry

theorem go_pieces_theorem (pieces : GoPieces) 
  (h1 : pieces.white = 4) 
  (h2 : pieces.black = 3) : 
  prob_same_color pieces = 3/7 ∧ expected_white_pieces pieces = 16/7 := by
  sorry

end go_pieces_theorem_l3899_389926


namespace inequality_solution_range_l3899_389993

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) := by
  sorry

end inequality_solution_range_l3899_389993


namespace dynamic_load_calculation_l3899_389942

/-- Given an architectural formula for dynamic load on cylindrical columns -/
theorem dynamic_load_calculation (T H : ℝ) (hT : T = 3) (hH : H = 6) :
  (50 * T^3) / H^3 = 6.25 := by
  sorry

end dynamic_load_calculation_l3899_389942


namespace quadratic_roots_condition_l3899_389974

theorem quadratic_roots_condition (a : ℝ) (h1 : a ≠ 0) (h2 : a < -1) :
  ∃ (x y : ℝ), x > 0 ∧ y < 0 ∧ 
  (a * x^2 + 2 * x + 1 = 0) ∧ 
  (a * y^2 + 2 * y + 1 = 0) := by
sorry

end quadratic_roots_condition_l3899_389974


namespace equation_solution_l3899_389927

theorem equation_solution : ∃ x : ℚ, (4 / 7) * (1 / 5) * x + 2 = 8 ∧ x = 105 / 2 := by
  sorry

end equation_solution_l3899_389927


namespace solution_values_l3899_389984

theorem solution_values (a : ℝ) : (a - 2) ^ (a + 1) = 1 → a = -1 ∨ a = 3 ∨ a = 1 := by
  sorry

end solution_values_l3899_389984


namespace right_triangle_7_24_25_l3899_389904

theorem right_triangle_7_24_25 (a b c : ℝ) :
  a = 7 ∧ b = 24 ∧ c = 25 → a^2 + b^2 = c^2 :=
by sorry

end right_triangle_7_24_25_l3899_389904


namespace preimages_of_one_l3899_389983

def f (x : ℝ) : ℝ := x^3 - x + 1

theorem preimages_of_one (x : ℝ) : 
  f x = 1 ↔ x = -1 ∨ x = 0 ∨ x = 1 := by sorry

end preimages_of_one_l3899_389983


namespace triangle_area_product_l3899_389996

theorem triangle_area_product (p q : ℝ) : 
  p > 0 → q > 0 → 
  (∃ (x y : ℝ), x ≥ 0 ∧ y ≥ 0 ∧ p * x + q * y = 12) →
  (1/2 * (12/p) * (12/q) = 12) →
  p * q = 6 := by
sorry

end triangle_area_product_l3899_389996


namespace max_sum_constrained_max_sum_constrained_attained_l3899_389943

theorem max_sum_constrained (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h : 16 * x * y * z = (x + y)^2 * (x + z)^2) :
  x + y + z ≤ 4 := by
sorry

theorem max_sum_constrained_attained :
  ∃ (x y z : ℝ), x > 0 ∧ y > 0 ∧ z > 0 ∧
  16 * x * y * z = (x + y)^2 * (x + z)^2 ∧
  x + y + z = 4 := by
sorry

end max_sum_constrained_max_sum_constrained_attained_l3899_389943


namespace fraction_relation_l3899_389968

theorem fraction_relation (x y z : ℚ) 
  (h1 : x / y = 3)
  (h2 : y / z = 5 / 2) :
  z / x = 2 / 15 := by
  sorry

end fraction_relation_l3899_389968


namespace seat_difference_is_three_l3899_389917

/-- Represents a bus with seats on left and right sides, and a back seat. -/
structure Bus where
  leftSeats : Nat
  rightSeats : Nat
  backSeatCapacity : Nat
  seatCapacity : Nat
  totalCapacity : Nat

/-- The number of fewer seats on the right side compared to the left side. -/
def seatDifference (bus : Bus) : Nat :=
  bus.leftSeats - bus.rightSeats

/-- Theorem stating the difference in seats between left and right sides. -/
theorem seat_difference_is_three :
  ∃ (bus : Bus),
    bus.leftSeats = 15 ∧
    bus.seatCapacity = 3 ∧
    bus.backSeatCapacity = 10 ∧
    bus.totalCapacity = 91 ∧
    seatDifference bus = 3 := by
  sorry

#check seat_difference_is_three

end seat_difference_is_three_l3899_389917


namespace expand_and_simplify_l3899_389936

theorem expand_and_simplify (x : ℝ) : (x - 3) * (x + 7) + x = x^2 + 5*x - 21 := by
  sorry

end expand_and_simplify_l3899_389936


namespace bus_problem_l3899_389930

/-- The number of children on a bus after a stop, given the initial number,
    the number who got off, and the difference between those who got off and on. -/
def children_after_stop (initial : ℕ) (got_off : ℕ) (diff : ℕ) : ℤ :=
  initial - got_off + (got_off - diff)

/-- Theorem stating that given the initial conditions, 
    the number of children on the bus after the stop is 12. -/
theorem bus_problem : children_after_stop 36 68 24 = 12 := by
  sorry

end bus_problem_l3899_389930


namespace weight_replacement_l3899_389931

theorem weight_replacement (n : ℕ) (old_weight new_weight avg_increase : ℝ) :
  n = 8 ∧
  new_weight = 65 ∧
  avg_increase = 2.5 →
  old_weight = new_weight - n * avg_increase :=
by sorry

end weight_replacement_l3899_389931


namespace remainder_problem_l3899_389924

theorem remainder_problem (n : ℕ) : 
  (n / 44 = 432 ∧ n % 44 = 0) → n % 38 = 8 := by
  sorry

end remainder_problem_l3899_389924


namespace expand_product_l3899_389925

theorem expand_product (x : ℝ) : 3 * (x - 2) * (x^2 + x + 1) = 3*x^3 - 3*x^2 - 3*x - 6 := by
  sorry

end expand_product_l3899_389925


namespace cannot_make_65_cents_l3899_389981

def coin_value (coin : Nat) : Nat :=
  match coin with
  | 0 => 5  -- nickel
  | 1 => 10 -- dime
  | 2 => 25 -- quarter
  | 3 => 50 -- half-dollar
  | _ => 0  -- invalid coin

def is_valid_coin (c : Nat) : Prop := c ≤ 3

theorem cannot_make_65_cents :
  ¬ ∃ (a b c d e : Nat),
    is_valid_coin a ∧ is_valid_coin b ∧ is_valid_coin c ∧ is_valid_coin d ∧ is_valid_coin e ∧
    coin_value a + coin_value b + coin_value c + coin_value d + coin_value e = 65 :=
by sorry

end cannot_make_65_cents_l3899_389981


namespace collinear_points_sum_l3899_389906

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Check if three points are collinear -/
def collinear (p1 p2 p3 : Point3D) : Prop := sorry

/-- The main theorem -/
theorem collinear_points_sum (a b : ℝ) : 
  collinear (Point3D.mk 1 a b) (Point3D.mk a 2 3) (Point3D.mk a b 3) → a + b = 4 := by
  sorry

end collinear_points_sum_l3899_389906


namespace valid_arrangements_count_l3899_389915

/-- Represents the colors of the pegs -/
inductive Color
| Red
| Blue
| Green

/-- Represents a position on the triangular board -/
structure Position :=
  (row : Nat)
  (col : Nat)

/-- Represents the triangular board -/
def Board := List Position

/-- Defines a valid triangular board with 3 rows -/
def validBoard : Board :=
  [(Position.mk 1 1),
   (Position.mk 2 1), (Position.mk 2 2),
   (Position.mk 3 1), (Position.mk 3 2), (Position.mk 3 3)]

/-- Represents a peg placement on the board -/
structure Placement :=
  (pos : Position)
  (color : Color)

/-- Checks if a list of placements is valid according to the color restriction rule -/
def isValidPlacement (placements : List Placement) : Bool :=
  sorry

/-- Counts the number of valid arrangements of pegs on the board -/
def countValidArrangements (board : Board) (redPegs bluePegs greenPegs : Nat) : Nat :=
  sorry

/-- The main theorem to be proved -/
theorem valid_arrangements_count :
  countValidArrangements validBoard 4 3 2 = 6 :=
sorry

end valid_arrangements_count_l3899_389915


namespace hallway_tiles_l3899_389937

/-- Calculates the total number of tiles used in a rectangular hallway with specific tiling patterns. -/
def total_tiles (length width : ℕ) : ℕ :=
  let outer_border := 2 * (length - 2) + 2 * (width - 2) + 4
  let second_border := 2 * ((length - 4) / 2) + 2 * ((width - 4) / 2)
  let inner_area := ((length - 6) * (width - 6)) / 9
  outer_border + second_border + inner_area

/-- Theorem stating that the total number of tiles used in a 20x30 foot rectangular hallway
    with specific tiling patterns is 175. -/
theorem hallway_tiles : total_tiles 30 20 = 175 := by
  sorry

end hallway_tiles_l3899_389937


namespace logarithm_difference_equals_three_l3899_389990

theorem logarithm_difference_equals_three :
  (Real.log 320 / Real.log 4) / (Real.log 80 / Real.log 4) -
  (Real.log 640 / Real.log 4) / (Real.log 40 / Real.log 4) = 3 := by
  sorry

end logarithm_difference_equals_three_l3899_389990


namespace selection_methods_count_l3899_389950

def num_type_a : ℕ := 3
def num_type_b : ℕ := 4
def total_selected : ℕ := 3

theorem selection_methods_count :
  (Finset.sum (Finset.range (total_selected + 1)) (λ k =>
    if k ≥ 1 ∧ (total_selected - k) ≥ 1 then
      (Nat.choose num_type_a k) * (Nat.choose num_type_b (total_selected - k))
    else
      0
  )) = 30 := by
  sorry

end selection_methods_count_l3899_389950


namespace feb_29_is_sunday_l3899_389907

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a date in February of a leap year -/
structure FebruaryDate :=
  (day : Nat)
  (isLeapYear : Bool)

/-- Returns the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

/-- Advances the day of the week by n days -/
def advanceDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => advanceDays (nextDay d) n

/-- Main theorem: If February 11th is a Wednesday in a leap year, then February 29th is a Sunday -/
theorem feb_29_is_sunday (d : FebruaryDate) (dow : DayOfWeek) :
  d.day = 11 → d.isLeapYear = true → dow = DayOfWeek.Wednesday →
  advanceDays dow 18 = DayOfWeek.Sunday :=
by
  sorry


end feb_29_is_sunday_l3899_389907


namespace perimeter_of_square_figure_l3899_389967

/-- A figure composed of four identical squares -/
structure SquareFigure where
  -- Side length of each square
  side_length : ℝ
  -- Total area of the figure
  total_area : ℝ
  -- Number of vertical segments
  vertical_segments : ℕ
  -- Number of horizontal segments
  horizontal_segments : ℕ
  -- Condition: Total area is the area of four squares
  area_condition : total_area = 4 * side_length ^ 2

/-- The perimeter of the square figure -/
def perimeter (f : SquareFigure) : ℝ :=
  (f.vertical_segments + f.horizontal_segments) * f.side_length

/-- Theorem: If the total area is 144 cm² and the figure has 4 vertical and 6 horizontal segments,
    then the perimeter is 60 cm -/
theorem perimeter_of_square_figure (f : SquareFigure) 
    (h_area : f.total_area = 144) 
    (h_vertical : f.vertical_segments = 4) 
    (h_horizontal : f.horizontal_segments = 6) : 
    perimeter f = 60 := by
  sorry


end perimeter_of_square_figure_l3899_389967


namespace game_strategies_l3899_389947

/-- The game state -/
structure GameState where
  board : ℝ
  turn : ℕ

/-- The game rules -/
def valid_move (x y : ℝ) : Prop :=
  0 < y - x ∧ y - x < 1

/-- The winning condition for the first variant -/
def winning_condition_1 (s : GameState) : Prop :=
  s.board ≥ 2010

/-- The winning condition for the second variant -/
def winning_condition_2 (s : GameState) : Prop :=
  s.board ≥ 2010 ∧ s.turn ≥ 2011

/-- The losing condition for the second variant -/
def losing_condition_2 (s : GameState) : Prop :=
  s.board ≥ 2010 ∧ s.turn ≤ 2010

/-- The theorem statement -/
theorem game_strategies :
  (∃ (strategy : ℕ → ℝ → ℝ),
    (∀ (n : ℕ) (x : ℝ), valid_move x (strategy n x)) ∧
    (∀ (play : ℕ → ℝ),
      (∀ (n : ℕ), valid_move (play n) (play (n+1))) →
      ∃ (k : ℕ), winning_condition_1 ⟨play k, k⟩ ∧
        k % 2 = 0)) ∧
  (∃ (strategy : ℕ → ℝ → ℝ),
    (∀ (n : ℕ) (x : ℝ), valid_move x (strategy n x)) ∧
    (∀ (play : ℕ → ℝ),
      (∀ (n : ℕ), valid_move (play n) (play (n+1))) →
      (∃ (k : ℕ), winning_condition_2 ⟨play k, k⟩ ∧
        k % 2 = 1) ∧
      (∀ (k : ℕ), k ≤ 2010 → ¬losing_condition_2 ⟨play k, k⟩))) :=
by sorry

end game_strategies_l3899_389947


namespace smallest_n_for_inequality_l3899_389938

theorem smallest_n_for_inequality : ∃ (n : ℕ), n = 2 ∧ 
  (∀ (x y : ℝ), (x^2 + y^2)^2 ≤ n * (x^4 + y^4)) ∧ 
  (∀ (m : ℕ), m < n → ∃ (x y : ℝ), (x^2 + y^2)^2 > m * (x^4 + y^4)) := by
  sorry

end smallest_n_for_inequality_l3899_389938


namespace two_digit_property_three_digit_property_l3899_389986

/-- Two-digit positive integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

/-- Three-digit positive integer -/
def ThreeDigitInt (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

/-- Converts a two-digit number to its digits -/
def toDigits2 (n : ℕ) : ℕ × ℕ := (n / 10, n % 10)

/-- Converts a three-digit number to its digits -/
def toDigits3 (n : ℕ) : ℕ × ℕ × ℕ := (n / 100, (n / 10) % 10, n % 10)

theorem two_digit_property (n : ℕ) (h : TwoDigitInt n) :
  let (a, b) := toDigits2 n
  (a + 1) * (b + 1) = n + 1 ↔ b = 9 := by sorry

theorem three_digit_property (n : ℕ) (h : ThreeDigitInt n) :
  let (a, b, c) := toDigits3 n
  (a + 1) * (b + 1) * (c + 1) = n + 1 ↔ b = 9 ∧ c = 9 := by sorry

end two_digit_property_three_digit_property_l3899_389986


namespace inequality_solution_implies_n_range_l3899_389951

theorem inequality_solution_implies_n_range (n : ℝ) : 
  (∀ x : ℝ, ((n - 3) * x > 2) ↔ (x < 2 / (n - 3))) → n < 3 := by
  sorry

end inequality_solution_implies_n_range_l3899_389951


namespace min_value_sum_l3899_389929

theorem min_value_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : 2 * a + b = 1) :
  ∀ x y : ℝ, 0 < x ∧ 0 < y ∧ 2 * x + y = 1 → a + b ≤ x + y ∧ a + b = 4 :=
sorry

end min_value_sum_l3899_389929


namespace parabola_sum_coefficients_l3899_389982

/-- Represents a parabola of the form x = ay^2 + by + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The x-coordinate of a point on the parabola given its y-coordinate -/
def Parabola.x_coord (p : Parabola) (y : ℝ) : ℝ :=
  p.a * y^2 + p.b * y + p.c

theorem parabola_sum_coefficients (p : Parabola) :
  p.x_coord 2 = -3 →  -- vertex at (-3, 2)
  p.x_coord (-1) = 1 →  -- passes through (1, -1)
  p.a < 0 →  -- opens to the left
  p.a + p.b + p.c = -23/9 := by
  sorry

end parabola_sum_coefficients_l3899_389982


namespace license_plate_count_l3899_389928

/-- The number of consonants in the alphabet. -/
def num_consonants : ℕ := 20

/-- The number of vowels in the alphabet (including Y). -/
def num_vowels : ℕ := 6

/-- The number of digits (0-9). -/
def num_digits : ℕ := 10

/-- The total number of letters in the alphabet. -/
def num_letters : ℕ := 26

/-- The number of unique five-character license plates with the sequence:
    consonant, vowel, consonant, digit, any letter. -/
def num_license_plates : ℕ := num_consonants * num_vowels * num_consonants * num_digits * num_letters

theorem license_plate_count :
  num_license_plates = 624000 :=
sorry

end license_plate_count_l3899_389928


namespace linda_classmates_l3899_389909

/-- The number of cookies each student receives -/
def cookies_per_student : ℕ := 10

/-- The number of cookies in one dozen -/
def cookies_per_dozen : ℕ := 12

/-- The number of dozens of cookies in each batch -/
def dozens_per_batch : ℕ := 4

/-- The number of batches of chocolate chip cookies Linda made -/
def chocolate_chip_batches : ℕ := 2

/-- The number of batches of oatmeal raisin cookies Linda made -/
def oatmeal_raisin_batches : ℕ := 1

/-- The number of additional batches Linda needs to bake -/
def additional_batches : ℕ := 2

/-- The total number of cookies Linda will have after baking all batches -/
def total_cookies : ℕ := 
  (chocolate_chip_batches + oatmeal_raisin_batches + additional_batches) * 
  dozens_per_batch * cookies_per_dozen

/-- The number of Linda's classmates -/
def number_of_classmates : ℕ := total_cookies / cookies_per_student

theorem linda_classmates : number_of_classmates = 24 := by
  sorry

end linda_classmates_l3899_389909


namespace wendy_furniture_assembly_time_l3899_389978

/-- Calculates the total time spent assembling furniture --/
def total_assembly_time (chair_count : ℕ) (table_count : ℕ) (bookshelf_count : ℕ)
                        (chair_time : ℕ) (table_time : ℕ) (bookshelf_time : ℕ) : ℕ :=
  chair_count * chair_time + table_count * table_time + bookshelf_count * bookshelf_time

/-- Theorem stating that the total assembly time for Wendy's furniture is 84 minutes --/
theorem wendy_furniture_assembly_time :
  total_assembly_time 4 3 2 6 10 15 = 84 := by
  sorry

#eval total_assembly_time 4 3 2 6 10 15

end wendy_furniture_assembly_time_l3899_389978


namespace rice_purchase_l3899_389905

theorem rice_purchase (rice_price lentil_price total_weight total_cost : ℚ)
  (h1 : rice_price = 105/100)
  (h2 : lentil_price = 33/100)
  (h3 : total_weight = 30)
  (h4 : total_cost = 2340/100) :
  ∃ (rice_weight : ℚ),
    rice_weight + (total_weight - rice_weight) = total_weight ∧
    rice_price * rice_weight + lentil_price * (total_weight - rice_weight) = total_cost ∧
    rice_weight = 75/4 := by
  sorry

end rice_purchase_l3899_389905


namespace pitcher_distribution_l3899_389960

theorem pitcher_distribution (C : ℝ) (h : C > 0) : 
  let juice_amount : ℝ := (2/3) * C
  let cups : ℕ := 6
  let juice_per_cup : ℝ := juice_amount / cups
  juice_per_cup / C = 1/9 := by sorry

end pitcher_distribution_l3899_389960


namespace triangle_properties_l3899_389901

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem states properties of a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : t.b * Real.cos t.C = (2 * t.a + t.c) * Real.cos (π - t.B))
  (h2 : t.b = Real.sqrt 13)
  (h3 : (1 / 2) * t.a * t.c * Real.sin t.B = (3 * Real.sqrt 3) / 4) :
  t.B = 2 * π / 3 ∧ t.a + t.c = 4 := by
  sorry


end triangle_properties_l3899_389901
