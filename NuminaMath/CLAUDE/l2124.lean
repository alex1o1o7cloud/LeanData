import Mathlib

namespace specific_prism_volume_l2124_212452

/-- Represents a triangular prism -/
structure TriangularPrism :=
  (lateral_face_area : ℝ)
  (distance_to_face : ℝ)

/-- The volume of a triangular prism -/
def volume (prism : TriangularPrism) : ℝ :=
  sorry

/-- Theorem stating the volume of the specific triangular prism -/
theorem specific_prism_volume :
  ∀ (prism : TriangularPrism),
    prism.lateral_face_area = 4 →
    prism.distance_to_face = 2 →
    volume prism = 4 :=
by
  sorry

end specific_prism_volume_l2124_212452


namespace justin_tim_games_l2124_212464

theorem justin_tim_games (n : ℕ) (k : ℕ) (total_players : ℕ) (justin tim : Fin total_players) :
  n = 12 →
  k = 6 →
  total_players = n →
  justin ≠ tim →
  (Nat.choose n k : ℚ) * k / n = 210 :=
sorry

end justin_tim_games_l2124_212464


namespace milk_production_per_cow_l2124_212419

theorem milk_production_per_cow 
  (num_cows : ℕ) 
  (milk_price : ℚ) 
  (butter_ratio : ℕ) 
  (butter_price : ℚ) 
  (num_customers : ℕ) 
  (milk_per_customer : ℕ) 
  (total_earnings : ℚ) 
  (h1 : num_cows = 12)
  (h2 : milk_price = 3)
  (h3 : butter_ratio = 2)
  (h4 : butter_price = 3/2)
  (h5 : num_customers = 6)
  (h6 : milk_per_customer = 6)
  (h7 : total_earnings = 144) :
  (total_earnings / num_cows : ℚ) / milk_price = 4 := by
sorry

end milk_production_per_cow_l2124_212419


namespace intersection_A_B_l2124_212433

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ Set.Icc 0 2, y = 2^x}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ico 1 3 := by
  sorry

end intersection_A_B_l2124_212433


namespace sum_of_integers_l2124_212478

theorem sum_of_integers (x y : ℕ) (h1 : x > y) (h2 : x - y = 8) (h3 : x * y = 240) : x + y = 32 := by
  sorry

end sum_of_integers_l2124_212478


namespace expression_evaluation_l2124_212480

/-- Given a = 3, b = 2, and c = 1, prove that (a^3 + b^2 + c)^2 - (a^3 + b^2 - c)^2 = 124 -/
theorem expression_evaluation (a b c : ℕ) (ha : a = 3) (hb : b = 2) (hc : c = 1) :
  (a^3 + b^2 + c)^2 - (a^3 + b^2 - c)^2 = 124 := by
  sorry

end expression_evaluation_l2124_212480


namespace largest_c_for_negative_two_in_range_l2124_212492

/-- The function f(x) with parameter c -/
def f (c : ℝ) (x : ℝ) : ℝ := x^2 + 4*x + c

/-- The largest value of c such that -2 is in the range of f(x) -/
theorem largest_c_for_negative_two_in_range :
  ∃ (c_max : ℝ), 
    (∃ (x : ℝ), f c_max x = -2) ∧ 
    (∀ (c : ℝ), c > c_max → ¬∃ (x : ℝ), f c x = -2) ∧
    c_max = 2 := by
  sorry

end largest_c_for_negative_two_in_range_l2124_212492


namespace quadratic_real_roots_condition_l2124_212491

theorem quadratic_real_roots_condition (k : ℝ) : 
  (∃ x : ℝ, (k - 1) * x^2 + 2 * x + 1 = 0) ↔ (k ≤ 2 ∧ k ≠ 1) :=
sorry

end quadratic_real_roots_condition_l2124_212491


namespace bowtie_equation_solution_l2124_212440

-- Define the operation ⋈
noncomputable def bowtie (a b : ℝ) : ℝ :=
  a + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2 + Real.sqrt (b^2))))

-- State the theorem
theorem bowtie_equation_solution :
  ∀ x : ℝ, bowtie 3 x = 15 → x = 2 * Real.sqrt 33 ∨ x = -2 * Real.sqrt 33 := by
  sorry

end bowtie_equation_solution_l2124_212440


namespace shorter_train_length_l2124_212417

/-- Calculates the length of the shorter train given the speeds of two trains,
    the time they take to cross each other, and the length of the longer train. -/
theorem shorter_train_length
  (speed1 : ℝ) (speed2 : ℝ) (crossing_time : ℝ) (longer_train_length : ℝ)
  (h1 : speed1 = 60) -- km/hr
  (h2 : speed2 = 40) -- km/hr
  (h3 : crossing_time = 10.799136069114471) -- seconds
  (h4 : longer_train_length = 160) -- meters
  : ∃ (shorter_train_length : ℝ),
    shorter_train_length = 140 ∧ 
    shorter_train_length = 
      (speed1 + speed2) * (5 / 18) * crossing_time - longer_train_length :=
by
  sorry

end shorter_train_length_l2124_212417


namespace outfit_count_l2124_212447

/-- The number of distinct outfits that can be made -/
def number_of_outfits (red_shirts green_shirts pants blue_hats red_hats : ℕ) : ℕ :=
  (red_shirts * pants * blue_hats) + (green_shirts * pants * red_hats)

/-- Theorem stating the number of outfits under the given conditions -/
theorem outfit_count :
  number_of_outfits 6 7 9 10 10 = 1170 :=
by sorry

end outfit_count_l2124_212447


namespace elena_garden_lilies_l2124_212421

/-- The number of lilies in Elena's garden -/
def num_lilies : ℕ := 8

/-- The number of tulips in Elena's garden -/
def num_tulips : ℕ := 5

/-- The number of petals each lily has -/
def petals_per_lily : ℕ := 6

/-- The number of petals each tulip has -/
def petals_per_tulip : ℕ := 3

/-- The total number of petals in Elena's garden -/
def total_petals : ℕ := 63

theorem elena_garden_lilies :
  num_lilies * petals_per_lily + num_tulips * petals_per_tulip = total_petals :=
by sorry

end elena_garden_lilies_l2124_212421


namespace swimming_running_speed_ratio_l2124_212446

/-- Proves that given the specified conditions, the ratio of running speed to swimming speed is 4 -/
theorem swimming_running_speed_ratio :
  ∀ (swimming_speed swimming_time running_time total_distance : ℝ),
  swimming_speed = 2 →
  swimming_time = 2 →
  running_time = swimming_time / 2 →
  total_distance = 12 →
  total_distance = swimming_speed * swimming_time + running_time * (total_distance - swimming_speed * swimming_time) / running_time →
  (total_distance - swimming_speed * swimming_time) / running_time / swimming_speed = 4 := by
  sorry

end swimming_running_speed_ratio_l2124_212446


namespace complex_modulus_equality_l2124_212490

theorem complex_modulus_equality (x : ℝ) (h : x > 0) :
  Complex.abs (10 + Complex.I * x) = 15 ↔ x = 5 * Real.sqrt 5 := by
  sorry

end complex_modulus_equality_l2124_212490


namespace train_bridge_crossing_time_l2124_212432

/-- Represents the time for a train to cross a bridge given its parameters -/
theorem train_bridge_crossing_time 
  (L : ℝ) -- Length of the train
  (u : ℝ) -- Initial speed of the train
  (a : ℝ) -- Constant acceleration of the train
  (t : ℝ) -- Time to cross the signal post
  (B : ℝ) -- Length of the bridge
  (h1 : L > 0) -- Train has positive length
  (h2 : u ≥ 0) -- Initial speed is non-negative
  (h3 : a > 0) -- Acceleration is positive
  (h4 : t > 0) -- Time to cross signal post is positive
  (h5 : B > 0) -- Bridge has positive length
  (h6 : L = u * t + (1/2) * a * t^2) -- Equation for crossing signal post
  : ∃ T, T > 0 ∧ B + L = u * T + (1/2) * a * T^2 :=
sorry


end train_bridge_crossing_time_l2124_212432


namespace no_two_right_angles_l2124_212441

-- Define a triangle
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

-- Define properties of a triangle
def Triangle.sumOfAngles (t : Triangle) : ℝ := t.angle1 + t.angle2 + t.angle3
def Triangle.isRight (t : Triangle) : Prop := t.angle1 = 90 ∨ t.angle2 = 90 ∨ t.angle3 = 90

-- Theorem: A triangle cannot have two right angles
theorem no_two_right_angles (t : Triangle) : 
  t.sumOfAngles = 180 → ¬(t.angle1 = 90 ∧ t.angle2 = 90) :=
by
  sorry


end no_two_right_angles_l2124_212441


namespace complement_M_in_U_l2124_212465

-- Define the universal set U
def U : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define the set M
def M : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}

-- Theorem statement
theorem complement_M_in_U : 
  (U \ M) = {x | 1 < x ∧ x ≤ 3} :=
sorry

end complement_M_in_U_l2124_212465


namespace fish_population_estimate_l2124_212450

/-- The number of fish initially tagged and returned to the pond -/
def tagged_fish : ℕ := 50

/-- The number of fish caught in the second catch -/
def second_catch : ℕ := 50

/-- The number of tagged fish found in the second catch -/
def tagged_in_second_catch : ℕ := 2

/-- The total number of fish in the pond -/
def total_fish : ℕ := 1250

/-- Theorem stating that the given conditions lead to the correct total number of fish -/
theorem fish_population_estimate :
  (tagged_in_second_catch : ℚ) / second_catch = tagged_fish / total_fish :=
by sorry

end fish_population_estimate_l2124_212450


namespace inflation_time_is_20_l2124_212425

/-- The time it takes to inflate one soccer ball -/
def inflation_time : ℕ := sorry

/-- The number of balls Alexia inflates -/
def alexia_balls : ℕ := 20

/-- The number of balls Ermias inflates -/
def ermias_balls : ℕ := alexia_balls + 5

/-- The total time taken to inflate all balls -/
def total_time : ℕ := 900

/-- Theorem stating that the inflation time for one ball is 20 minutes -/
theorem inflation_time_is_20 : inflation_time = 20 := by
  sorry

end inflation_time_is_20_l2124_212425


namespace focus_of_specific_parabola_l2124_212487

/-- The focus of a parabola defined by y = ax^2 + bx + c -/
def parabola_focus (a b c : ℝ) : ℝ × ℝ :=
  sorry

theorem focus_of_specific_parabola :
  parabola_focus 9 6 (-4) = (-1/3, -59/12) := by
  sorry

end focus_of_specific_parabola_l2124_212487


namespace quadratic_function_proof_l2124_212454

/-- Given a quadratic function g(x) = x^2 + cx + d, prove that if g(g(x) + x) / g(x) = x^2 + 44x + 50, then g(x) = x^2 + 44x + 50 -/
theorem quadratic_function_proof (c d : ℝ) (g : ℝ → ℝ) 
  (h1 : ∀ x, g x = x^2 + c*x + d)
  (h2 : ∀ x, g (g x + x) / g x = x^2 + 44*x + 50) :
  ∀ x, g x = x^2 + 44*x + 50 := by
  sorry

end quadratic_function_proof_l2124_212454


namespace circle_c_equation_l2124_212493

/-- A circle C with the following properties:
    1. Its center is on the line x - 3y = 0
    2. It is tangent to the negative half-axis of the y-axis
    3. The chord cut by C on the x-axis is 4√2 in length -/
structure CircleC where
  center : ℝ × ℝ
  radius : ℝ
  center_on_line : center.1 - 3 * center.2 = 0
  tangent_to_negative_y : center.2 < 0 ∧ radius = -center.2
  chord_length : 4 * Real.sqrt 2 = 2 * Real.sqrt (2 * radius * center.1)

/-- The equation of circle C is (x + 3)² + (y + 1)² = 9 -/
theorem circle_c_equation (c : CircleC) : 
  ∀ x y : ℝ, (x + 3)^2 + (y + 1)^2 = 9 ↔ 
    (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2 := by
  sorry

end circle_c_equation_l2124_212493


namespace polar_to_cartesian_line_l2124_212485

/-- The curve defined by the polar equation r = 1 / (2sin(θ) - cos(θ)) is a line. -/
theorem polar_to_cartesian_line :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  ∀ (θ : ℝ), ∀ (r : ℝ), r > 0 →
  r = 1 / (2 * Real.sin θ - Real.cos θ) →
  a * (r * Real.cos θ) + b * (r * Real.sin θ) + c = 0 :=
sorry

end polar_to_cartesian_line_l2124_212485


namespace science_quiz_passing_requirement_l2124_212431

theorem science_quiz_passing_requirement (total_questions physics_questions chemistry_questions biology_questions : ℕ)
  (physics_correct_percent chemistry_correct_percent biology_correct_percent passing_percent : ℚ) :
  total_questions = 100 →
  physics_questions = 20 →
  chemistry_questions = 40 →
  biology_questions = 40 →
  physics_correct_percent = 80 / 100 →
  chemistry_correct_percent = 50 / 100 →
  biology_correct_percent = 70 / 100 →
  passing_percent = 65 / 100 →
  (passing_percent * total_questions).ceil -
    (physics_correct_percent * physics_questions +
     chemistry_correct_percent * chemistry_questions +
     biology_correct_percent * biology_questions) = 1 := by
  sorry

end science_quiz_passing_requirement_l2124_212431


namespace complete_square_sum_l2124_212408

theorem complete_square_sum (b c : ℤ) : 
  (∀ x : ℝ, x^2 - 8*x + 8 = 0 ↔ (x + b)^2 = c) → b + c = 4 := by
  sorry

end complete_square_sum_l2124_212408


namespace triangle_side_equations_l2124_212449

/-- Given a triangle ABC with point A at (1,3) and medians from A satisfying specific equations,
    prove that the sides of the triangle have the given equations. -/
theorem triangle_side_equations (B C : ℝ × ℝ) : 
  let A : ℝ × ℝ := (1, 3)
  let median_to_BC (x y : ℝ) := x - 2*y + 1 = 0
  let median_to_AC (x y : ℝ) := y = 1
  (∃ t : ℝ, median_to_BC (C.1 + t*(B.1 - C.1)) (C.2 + t*(B.2 - C.2))) ∧ 
  (∃ s : ℝ, median_to_AC ((1 + C.1)/2) ((3 + C.2)/2)) →
  (B.2 - 3 = (3 - 1)/(1 - B.1) * (B.1 - 1)) ∧ 
  (C.2 - B.2 = (C.2 - B.2)/(C.1 - B.1) * (C.1 - B.1)) ∧ 
  (C.2 - 3 = (C.2 - 3)/(C.1 - 1) * (C.1 - 1)) := by
sorry

end triangle_side_equations_l2124_212449


namespace largest_integer_inequality_l2124_212497

theorem largest_integer_inequality : 
  ∀ y : ℤ, y ≤ 0 ↔ (y : ℚ) / 4 + 3 / 7 < 2 / 3 :=
by sorry

end largest_integer_inequality_l2124_212497


namespace no_quadratic_with_discriminant_2019_l2124_212414

theorem no_quadratic_with_discriminant_2019 : ¬ ∃ (a b c : ℤ), b^2 - 4*a*c = 2019 := by
  sorry

end no_quadratic_with_discriminant_2019_l2124_212414


namespace parabola_intersection_l2124_212472

theorem parabola_intersection :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 2
  let g (x : ℝ) := 9 * x^2 + 6 * x + 2
  ∀ x y : ℝ, f x = y ∧ g x = y ↔ (x = 0 ∧ y = 2) ∨ (x = -5/3 ∧ y = 17) :=
by sorry

end parabola_intersection_l2124_212472


namespace angle_A_is_60_degrees_perimeter_range_l2124_212469

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given equation
def given_equation (t : Triangle) : Prop :=
  t.a * Real.cos t.C + Real.sqrt 3 * t.a * Real.sin t.C - t.b - t.c = 0

-- Theorem 1: Prove that A = 60°
theorem angle_A_is_60_degrees (t : Triangle) (h : given_equation t) : t.A = Real.pi / 3 := by
  sorry

-- Theorem 2: Prove the range of perimeters when a = 7
theorem perimeter_range (t : Triangle) (h1 : given_equation t) (h2 : t.a = 7) :
  14 < t.a + t.b + t.c ∧ t.a + t.b + t.c ≤ 21 := by
  sorry

end angle_A_is_60_degrees_perimeter_range_l2124_212469


namespace susans_chairs_l2124_212444

theorem susans_chairs (red : ℕ) (yellow : ℕ) (blue : ℕ) 
  (h1 : yellow = 4 * red)
  (h2 : blue = yellow - 2)
  (h3 : red + yellow + blue = 43) :
  red = 5 := by
  sorry

end susans_chairs_l2124_212444


namespace polynomial_division_remainder_l2124_212406

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  (3 * X^5 + 15 * X^4 - 42 * X^3 - 60 * X^2 + 48 * X - 47) = 
  (X^3 + 7 * X^2 + 5 * X - 5) * q + (3 * X - 47) := by
  sorry

end polynomial_division_remainder_l2124_212406


namespace functional_equation_defined_everywhere_l2124_212411

noncomputable def f (c : ℝ) : ℝ → ℝ :=
  λ x => if x = 0 then c
         else if x = 1 then 3 - 2*c
         else (-x^3 + 3*x^2 + 2) / (3*x*(1-x))

theorem functional_equation (c : ℝ) :
  ∀ x : ℝ, x ≠ 0 → f c x + 2 * f c ((x - 1) / x) = 3 * x :=
by sorry

theorem defined_everywhere (c : ℝ) :
  ∀ x : ℝ, ∃ y : ℝ, f c x = y :=
by sorry

end functional_equation_defined_everywhere_l2124_212411


namespace geometric_sequence_condition_l2124_212471

/-- A sequence (a, b, c) is geometric if there exists a non-zero real number r such that b = a * r and c = b * r. -/
def IsGeometricSequence (a b c : ℝ) : Prop :=
  ∃ (r : ℝ), r ≠ 0 ∧ b = a * r ∧ c = b * r

/-- The condition ac = b^2 is necessary but not sufficient for a, b, c to form a geometric sequence. -/
theorem geometric_sequence_condition (a b c : ℝ) :
  (IsGeometricSequence a b c → a * c = b ^ 2) ∧
  ¬(a * c = b ^ 2 → IsGeometricSequence a b c) := by
  sorry

end geometric_sequence_condition_l2124_212471


namespace fraction_of_books_sold_l2124_212410

/-- Prove that the fraction of books sold is 2/3, given the conditions -/
theorem fraction_of_books_sold (price : ℝ) (unsold : ℕ) (total_received : ℝ) :
  price = 4.25 →
  unsold = 30 →
  total_received = 255 →
  (total_received / price) / ((total_received / price) + unsold) = 2 / 3 := by
  sorry

end fraction_of_books_sold_l2124_212410


namespace absolute_value_equation_solutions_l2124_212424

theorem absolute_value_equation_solutions :
  ∃! (s : Finset ℝ), s.card = 2 ∧ 
    (∀ x : ℝ, x ∈ s ↔ |x - 1| = |x - 2| + |x - 3| + |x - 4|) ∧
    (2 ∈ s ∧ 4 ∈ s) := by
  sorry

end absolute_value_equation_solutions_l2124_212424


namespace vasilyev_car_loan_payment_l2124_212477

/-- Calculates the maximum monthly car loan payment for the Vasilyev family --/
def max_car_loan_payment (total_income : ℝ) (total_expenses : ℝ) (emergency_fund_rate : ℝ) : ℝ :=
  let remaining_income := total_income - total_expenses
  let emergency_fund := emergency_fund_rate * remaining_income
  total_income - total_expenses - emergency_fund

/-- Theorem stating the maximum monthly car loan payment for the Vasilyev family --/
theorem vasilyev_car_loan_payment :
  max_car_loan_payment 84600 49800 0.1 = 31320 := by
  sorry

end vasilyev_car_loan_payment_l2124_212477


namespace express_y_in_terms_of_x_l2124_212481

theorem express_y_in_terms_of_x (x y : ℝ) (h : 3 * x + y = 2) : y = 2 - 3 * x := by
  sorry

end express_y_in_terms_of_x_l2124_212481


namespace next_sunday_rest_l2124_212405

/-- Represents the number of days in a week -/
def daysInWeek : ℕ := 7

/-- Represents the length of the work-rest cycle in days -/
def cycleDays : ℕ := 10

/-- Represents the number of consecutive work days -/
def workDays : ℕ := 8

/-- Represents the number of consecutive rest days -/
def restDays : ℕ := 2

/-- Theorem stating that the next Sunday rest day occurs after 7 weeks -/
theorem next_sunday_rest (n : ℕ) : 
  cycleDays * n + restDays - 1 = daysInWeek * 7 := by sorry

end next_sunday_rest_l2124_212405


namespace max_regions_theorem_l2124_212448

/-- Maximum number of regions formed by n lines in R^2 -/
def max_regions_lines (n : ℕ) : ℕ := n * (n + 1) / 2 + 1

/-- Maximum number of regions formed by n planes in R^3 -/
def max_regions_planes (n : ℕ) : ℕ := (n^3 + 5*n) / 6 + 1

/-- Maximum number of regions formed by n circles in R^2 -/
def max_regions_circles (n : ℕ) : ℕ := (n - 1) * n + 2

/-- Maximum number of regions formed by n spheres in R^3 -/
def max_regions_spheres (n : ℕ) : ℕ := n * (n^2 - 3*n + 8) / 3

theorem max_regions_theorem :
  ∀ n : ℕ,
  (max_regions_lines n = n * (n + 1) / 2 + 1) ∧
  (max_regions_planes n = (n^3 + 5*n) / 6 + 1) ∧
  (max_regions_circles n = (n - 1) * n + 2) ∧
  (max_regions_spheres n = n * (n^2 - 3*n + 8) / 3) := by
  sorry

end max_regions_theorem_l2124_212448


namespace johns_original_earnings_l2124_212402

/-- Given that John's weekly earnings increased by 20% to $72, prove that his original weekly earnings were $60. -/
theorem johns_original_earnings (current_earnings : ℝ) (increase_rate : ℝ) : 
  current_earnings = 72 ∧ increase_rate = 0.20 → 
  current_earnings / (1 + increase_rate) = 60 := by
  sorry

end johns_original_earnings_l2124_212402


namespace constant_expression_theorem_l2124_212445

theorem constant_expression_theorem (x y m n : ℝ) :
  (∀ y, (x + y) * (x - 2*y) - m*y*(n*x - y) = 25) →
  m = 2 ∧ n = -1/2 ∧ (x = 5 ∨ x = -5) := by sorry

end constant_expression_theorem_l2124_212445


namespace garden_problem_l2124_212475

theorem garden_problem (a : ℝ) : 
  a > 0 → 
  (a + 3)^2 = 2 * a^2 + 9 → 
  a = 6 := by
sorry

end garden_problem_l2124_212475


namespace quadratic_equation_roots_l2124_212426

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ m * x^2 - 2 * x + 5 = 0 ∧ m * y^2 - 2 * y + 5 = 0) ↔ 
  (m < (1 : ℝ) / 5 ∧ m ≠ 0) := by
sorry

end quadratic_equation_roots_l2124_212426


namespace fish_ratio_l2124_212479

def fish_problem (O B R : ℕ) : Prop :=
  O = B + 25 ∧
  B = 75 ∧
  (O + B + R) / 3 = 75

theorem fish_ratio : ∀ O B R : ℕ, fish_problem O B R → R * 2 = O :=
sorry

end fish_ratio_l2124_212479


namespace fibonacci_sum_convergence_l2124_212466

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The sum of F_n / 2^n from n = 0 to infinity -/
noncomputable def fibSum : ℝ := ∑' n, (fib n : ℝ) / (2 ^ n)

theorem fibonacci_sum_convergence : fibSum = 4 / 5 := by
  sorry

end fibonacci_sum_convergence_l2124_212466


namespace min_students_in_class_l2124_212423

theorem min_students_in_class (n b g : ℕ) : 
  n ≡ 2 [MOD 5] →
  (3 * g : ℕ) = (2 * b : ℕ) →
  n = b + g →
  n ≥ 57 ∧ (∀ m : ℕ, m < 57 → ¬(m ≡ 2 [MOD 5] ∧ ∃ b' g' : ℕ, (3 * g' : ℕ) = (2 * b' : ℕ) ∧ m = b' + g')) :=
by sorry

end min_students_in_class_l2124_212423


namespace loom_weaving_rate_l2124_212451

/-- The rate at which an industrial loom weaves cloth, given the amount of cloth woven and the time taken. -/
theorem loom_weaving_rate (cloth_woven : Real) (time_taken : Real) (h : cloth_woven = 25 ∧ time_taken = 195.3125) :
  cloth_woven / time_taken = 0.128 := by
  sorry

end loom_weaving_rate_l2124_212451


namespace bird_population_theorem_l2124_212496

theorem bird_population_theorem (total : ℝ) (total_pos : total > 0) : 
  let hawks := 0.3 * total
  let non_hawks := total - hawks
  let paddyfield_warblers := 0.4 * non_hawks
  let kingfishers := 0.25 * paddyfield_warblers
  let other_birds := total - (hawks + paddyfield_warblers + kingfishers)
  (other_birds / total) * 100 = 35 := by
sorry

end bird_population_theorem_l2124_212496


namespace first_player_seeds_l2124_212455

/-- Given a sunflower seed eating contest with three players, where:
  * The second player eats 53 seeds
  * The third player eats 30 more seeds than the second
  * The total number of seeds eaten is 214
  This theorem proves that the first player eats 78 seeds. -/
theorem first_player_seeds (second_player : ℕ) (third_player : ℕ) (total_seeds : ℕ) :
  second_player = 53 →
  third_player = second_player + 30 →
  total_seeds = 214 →
  total_seeds = second_player + third_player + 78 :=
by sorry

end first_player_seeds_l2124_212455


namespace sqrt_expression_equality_l2124_212407

theorem sqrt_expression_equality : 
  Real.sqrt 8 - 2 * Real.sqrt 18 + Real.sqrt 24 = -4 * Real.sqrt 2 + 2 * Real.sqrt 6 := by
  sorry

end sqrt_expression_equality_l2124_212407


namespace complementary_angles_ratio_l2124_212459

theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- The angles are complementary (sum to 90°)
  a = 4 * b →   -- The ratio of the angles is 4:1
  a = 72 :=     -- The larger angle is 72°
by sorry

end complementary_angles_ratio_l2124_212459


namespace f_properties_l2124_212483

/-- A function f from positive integers to positive integers with a parameter k -/
def f (k : ℕ+) : ℕ+ → ℕ+ :=
  fun n => if n > k then n - k else sorry

/-- The number of different functions f when k = 5 and 1 ≤ f(n) ≤ 2 for n ≤ 5 -/
def count_functions : ℕ := sorry

theorem f_properties :
  (∃ (a : ℕ+), f 1 1 = a) ∧
  count_functions = 32 := by sorry

end f_properties_l2124_212483


namespace gregs_gold_is_20_l2124_212498

/-- Represents the amount of gold Greg has -/
def gregs_gold : ℝ := sorry

/-- Represents the amount of gold Katie has -/
def katies_gold : ℝ := sorry

/-- The total amount of gold is 100 -/
axiom total_gold : gregs_gold + katies_gold = 100

/-- Greg has four times less gold than Katie -/
axiom gold_ratio : gregs_gold = katies_gold / 4

/-- Theorem stating that Greg's gold amount is 20 -/
theorem gregs_gold_is_20 : gregs_gold = 20 := by sorry

end gregs_gold_is_20_l2124_212498


namespace river_flow_speed_l2124_212463

/-- Proves that the speed of river flow is 2 km/hr given the conditions of the boat journey --/
theorem river_flow_speed (boat_speed : ℝ) (distance : ℝ) (total_time : ℝ) 
  (h1 : boat_speed = 6)
  (h2 : distance = 64)
  (h3 : total_time = 24) :
  ∃ (v : ℝ), v = 2 ∧ 
  (distance / (boat_speed - v) + distance / (boat_speed + v) = total_time) :=
by
  sorry

#check river_flow_speed

end river_flow_speed_l2124_212463


namespace bruce_shopping_theorem_l2124_212439

/-- Calculates the remaining money after Bruce's shopping trip. -/
def remaining_money (initial_amount : ℕ) (shirt_price : ℕ) (num_shirts : ℕ) (pants_price : ℕ) : ℕ :=
  initial_amount - (shirt_price * num_shirts + pants_price)

/-- Theorem stating that Bruce has $20 left after his shopping trip. -/
theorem bruce_shopping_theorem :
  remaining_money 71 5 5 26 = 20 := by
  sorry

#eval remaining_money 71 5 5 26

end bruce_shopping_theorem_l2124_212439


namespace sum_of_first_cards_theorem_l2124_212484

/-- The sum of points of the first cards in card piles -/
def sum_of_first_cards (a b c d : ℕ) : ℕ :=
  b * (c + 1) + d - a

/-- Theorem stating the sum of points of the first cards in card piles -/
theorem sum_of_first_cards_theorem (a b c d : ℕ) :
  ∃ x : ℕ, x = sum_of_first_cards a b c d :=
by
  sorry

#check sum_of_first_cards_theorem

end sum_of_first_cards_theorem_l2124_212484


namespace inequality_proof_l2124_212437

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a + b / 3 + c ≥ Real.sqrt ((a + b) * b * (c + a)) ∧
  Real.sqrt ((a + b) * b * (c + a)) ≥ Real.sqrt (a * b) + (Real.sqrt (b * c) + Real.sqrt (c * a)) / 3 := by
  sorry

end inequality_proof_l2124_212437


namespace triangle_angle_relation_l2124_212467

theorem triangle_angle_relation (a b c α β γ : ℝ) : 
  b = (a + c) / Real.sqrt 2 →
  β = (α + γ) / 2 →
  c > a →
  γ = α + 90 :=
sorry

end triangle_angle_relation_l2124_212467


namespace fixed_point_on_line_l2124_212482

theorem fixed_point_on_line (k : ℝ) : 1 = k * (-2) + 2 * k + 1 := by
  sorry

end fixed_point_on_line_l2124_212482


namespace tetrahedron_volume_and_height_l2124_212461

/-- Tetrahedron with given vertices -/
structure Tetrahedron where
  A₁ : ℝ × ℝ × ℝ := (-1, 2, -3)
  A₂ : ℝ × ℝ × ℝ := (4, -1, 0)
  A₃ : ℝ × ℝ × ℝ := (2, 1, -2)
  A₄ : ℝ × ℝ × ℝ := (3, 4, 5)

/-- Calculate the volume of the tetrahedron -/
def tetrahedronVolume (t : Tetrahedron) : ℝ := by sorry

/-- Calculate the height from A₄ to face A₁A₂A₃ -/
def tetrahedronHeight (t : Tetrahedron) : ℝ := by sorry

/-- Main theorem: Volume and height of the tetrahedron -/
theorem tetrahedron_volume_and_height (t : Tetrahedron) :
  tetrahedronVolume t = 20 / 3 ∧ tetrahedronHeight t = 5 * Real.sqrt 2 := by sorry

end tetrahedron_volume_and_height_l2124_212461


namespace green_ball_packs_l2124_212415

/-- Given the number of packs of red and yellow balls, the number of balls per pack,
    and the total number of balls, calculate the number of packs of green balls. -/
theorem green_ball_packs (red_packs yellow_packs : ℕ) (balls_per_pack : ℕ) (total_balls : ℕ) : 
  red_packs = 3 → 
  yellow_packs = 10 → 
  balls_per_pack = 19 → 
  total_balls = 399 → 
  (total_balls - (red_packs + yellow_packs) * balls_per_pack) / balls_per_pack = 8 :=
by sorry

end green_ball_packs_l2124_212415


namespace boat_length_boat_length_is_four_l2124_212494

/-- The length of a boat given specific conditions --/
theorem boat_length (breadth : ℝ) (sink_depth : ℝ) (man_mass : ℝ) 
                    (water_density : ℝ) (gravity : ℝ) : ℝ :=
  let boat_length := 4
  let volume_displaced := man_mass * gravity / (water_density * gravity)
  let calculated_length := volume_displaced / (breadth * sink_depth)
  
  -- Assumptions
  have h1 : breadth = 3 := by sorry
  have h2 : sink_depth = 0.01 := by sorry
  have h3 : man_mass = 120 := by sorry
  have h4 : water_density = 1000 := by sorry
  have h5 : gravity = 9.8 := by sorry
  
  -- Proof that the calculated length equals the given boat length
  have h6 : calculated_length = boat_length := by sorry
  
  boat_length

/-- Main theorem stating the boat length is 4 meters --/
theorem boat_length_is_four : 
  boat_length 3 0.01 120 1000 9.8 = 4 := by sorry

end boat_length_boat_length_is_four_l2124_212494


namespace concert_attendance_l2124_212460

-- Define the total number of people at the concert
variable (P : ℕ)

-- Define the conditions
def second_band_audience : ℚ := 2/3
def first_band_audience : ℚ := 1/3
def under_30_second_band : ℚ := 1/2
def women_under_30_second_band : ℚ := 3/5
def men_under_30_second_band : ℕ := 20

-- Theorem statement
theorem concert_attendance : 
  second_band_audience + first_band_audience = 1 →
  (second_band_audience * under_30_second_band * (1 - women_under_30_second_band)) * P = men_under_30_second_band →
  P = 150 :=
by sorry

end concert_attendance_l2124_212460


namespace height_difference_l2124_212489

-- Define variables for heights
variable (h_A h_B h_D h_E h_F h_G : ℝ)

-- Define the conditions
def condition1 : Prop := h_A - h_D = 4.5
def condition2 : Prop := h_E - h_D = -1.7
def condition3 : Prop := h_F - h_E = -0.8
def condition4 : Prop := h_G - h_F = 1.9
def condition5 : Prop := h_B - h_G = 3.6

-- Theorem statement
theorem height_difference 
  (c1 : condition1 h_A h_D)
  (c2 : condition2 h_E h_D)
  (c3 : condition3 h_F h_E)
  (c4 : condition4 h_G h_F)
  (c5 : condition5 h_B h_G) :
  h_A > h_B :=
by sorry

end height_difference_l2124_212489


namespace store_holiday_customers_l2124_212453

/-- The number of customers a store sees during holiday season -/
def holiday_customers (regular_rate : ℕ) (hours : ℕ) : ℕ :=
  2 * regular_rate * hours

/-- Theorem: Given the regular customer rate and time period, 
    the store will see 2800 customers during the holiday season -/
theorem store_holiday_customers :
  holiday_customers 175 8 = 2800 := by
  sorry

end store_holiday_customers_l2124_212453


namespace difference_of_squares_l2124_212442

theorem difference_of_squares (x : ℝ) : x^2 - 36 = (x + 6) * (x - 6) := by
  sorry

end difference_of_squares_l2124_212442


namespace unique_plane_through_and_parallel_l2124_212495

-- Define the concept of skew lines
def SkewLines (l₁ l₂ : Set (Point)) : Prop := sorry

-- Define a plane passing through a line and parallel to another line
def PlaneThroughAndParallel (π : Set (Point)) (l₁ l₂ : Set (Point)) : Prop := sorry

theorem unique_plane_through_and_parallel 
  (l₁ l₂ : Set (Point)) 
  (h : SkewLines l₁ l₂) : 
  ∃! π, PlaneThroughAndParallel π l₁ l₂ := by sorry

end unique_plane_through_and_parallel_l2124_212495


namespace initial_withdrawal_l2124_212427

theorem initial_withdrawal (initial_balance : ℚ) : 
  let remaining_balance := initial_balance - (2/5) * initial_balance
  let final_balance := remaining_balance + (1/4) * remaining_balance
  final_balance = 750 →
  (2/5) * initial_balance = 400 := by
sorry

end initial_withdrawal_l2124_212427


namespace perp_line_plane_from_conditions_l2124_212474

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between lines and planes
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_line_line : Line → Line → Prop)

-- Axiom: If a line is perpendicular to two planes, those planes are parallel
axiom perp_two_planes_parallel (n : Line) (α β : Plane) :
  perp_line_plane n α → perp_line_plane n β → α = β

-- Axiom: If a line is perpendicular to one of two parallel planes, it's perpendicular to the other
axiom perp_parallel_planes (m : Line) (α β : Plane) :
  α = β → perp_line_plane m α → perp_line_plane m β

-- Theorem to prove
theorem perp_line_plane_from_conditions (n m : Line) (α β : Plane) :
  perp_line_plane n α →
  perp_line_plane n β →
  perp_line_plane m α →
  perp_line_plane m β :=
by sorry

end perp_line_plane_from_conditions_l2124_212474


namespace base_three_20121_equals_178_l2124_212435

def base_three_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (3 ^ (digits.length - 1 - i))) 0

theorem base_three_20121_equals_178 :
  base_three_to_ten [2, 0, 1, 2, 1] = 178 := by sorry

end base_three_20121_equals_178_l2124_212435


namespace connie_markers_total_l2124_212428

/-- The total number of markers Connie has is 3343, given that she has 2315 red markers and 1028 blue markers. -/
theorem connie_markers_total : 
  let red_markers : ℕ := 2315
  let blue_markers : ℕ := 1028
  red_markers + blue_markers = 3343 := by sorry

end connie_markers_total_l2124_212428


namespace three_true_inequalities_l2124_212434

theorem three_true_inequalities
  (x y a b : ℝ)
  (hx_pos : x > 0)
  (hy_pos : y > 0)
  (ha_pos : a > 0)
  (hb_pos : b > 0)
  (hx : x^2 < a^2)
  (hy : y^2 < b^2) :
  (x^2 + y^2 < a^2 + b^2) ∧
  (x^2 * y^2 < a^2 * b^2) ∧
  (x^2 / y^2 < a^2 / b^2) ∧
  ¬(∀ x y a b, x > 0 → y > 0 → a > 0 → b > 0 → x^2 < a^2 → y^2 < b^2 → x^2 - y^2 < a^2 - b^2) :=
by sorry

end three_true_inequalities_l2124_212434


namespace ticket_price_is_three_l2124_212470

/-- Represents an amusement park's weekly operations and revenue -/
structure AmusementPark where
  regularDays : Nat
  regularVisitors : Nat
  specialDay1Visitors : Nat
  specialDay2Visitors : Nat
  weeklyRevenue : Nat

/-- Calculates the ticket price given the park's weekly data -/
def calculateTicketPrice (park : AmusementPark) : Rat :=
  park.weeklyRevenue / (park.regularDays * park.regularVisitors + park.specialDay1Visitors + park.specialDay2Visitors)

/-- Theorem stating that the ticket price is $3 given the specific conditions -/
theorem ticket_price_is_three (park : AmusementPark) 
  (h1 : park.regularDays = 5)
  (h2 : park.regularVisitors = 100)
  (h3 : park.specialDay1Visitors = 200)
  (h4 : park.specialDay2Visitors = 300)
  (h5 : park.weeklyRevenue = 3000) :
  calculateTicketPrice park = 3 := by
  sorry

#eval calculateTicketPrice { 
  regularDays := 5, 
  regularVisitors := 100, 
  specialDay1Visitors := 200, 
  specialDay2Visitors := 300, 
  weeklyRevenue := 3000 
}

end ticket_price_is_three_l2124_212470


namespace color_film_fraction_l2124_212443

theorem color_film_fraction (x y : ℝ) (h1 : x ≠ 0) : 
  let total_bw := 20 * x
  let total_color := 6 * y
  let selected_bw := (y / x) * (1 / 100) * total_bw
  let selected_color := total_color
  let total_selected := selected_bw + selected_color
  (selected_color / total_selected : ℝ) = 6 / 31 := by
  sorry

end color_film_fraction_l2124_212443


namespace f_value_at_2_l2124_212401

/-- A function f(x) = x^5 + ax^3 + bx - 8 where f(-2) = 10 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

/-- Theorem: If f(-2) = 10, then f(2) = -26 -/
theorem f_value_at_2 (a b : ℝ) (h : f a b (-2) = 10) : f a b 2 = -26 := by
  sorry

end f_value_at_2_l2124_212401


namespace bargain_bin_books_l2124_212457

def total_books (x y : ℕ) (z : ℚ) : ℚ :=
  (x : ℚ) - (y : ℚ) + z / 100 * (x : ℚ)

theorem bargain_bin_books (x y : ℕ) (z : ℚ) :
  total_books x y z = (x : ℚ) - (y : ℚ) + z / 100 * (x : ℚ) :=
by sorry

end bargain_bin_books_l2124_212457


namespace mothers_age_l2124_212422

/-- Proves that the mother's age is 42 given the conditions of the problem -/
theorem mothers_age (daughter_age : ℕ) (future_years : ℕ) (mother_age : ℕ) : 
  daughter_age = 8 →
  future_years = 9 →
  mother_age + future_years = 3 * (daughter_age + future_years) →
  mother_age = 42 := by
  sorry

end mothers_age_l2124_212422


namespace robins_gum_problem_l2124_212400

/-- Given that Robin initially had 18 pieces of gum and now has 44 pieces in total,
    prove that Robin's brother gave her 26 pieces of gum. -/
theorem robins_gum_problem (initial : ℕ) (total : ℕ) (h1 : initial = 18) (h2 : total = 44) :
  total - initial = 26 := by
  sorry

end robins_gum_problem_l2124_212400


namespace absolute_value_not_positive_l2124_212473

theorem absolute_value_not_positive (x : ℚ) : |4*x + 6| ≤ 0 ↔ x = -3/2 := by
  sorry

end absolute_value_not_positive_l2124_212473


namespace probability_both_red_correct_l2124_212416

def total_balls : ℕ := 10
def red_balls : ℕ := 4
def blue_balls : ℕ := 4
def green_balls : ℕ := 2
def balls_picked : ℕ := 2

def probability_both_red : ℚ := 2 / 15

theorem probability_both_red_correct :
  (Nat.choose red_balls balls_picked : ℚ) / (Nat.choose total_balls balls_picked : ℚ) = probability_both_red :=
sorry

end probability_both_red_correct_l2124_212416


namespace extended_equilateral_area_ratio_l2124_212488

/-- Represents a triangle with vertices A, B, and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Calculates the area of a triangle -/
def area (t : Triangle) : ℝ := sorry

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Extends a triangle by a factor along each side -/
def extendTriangle (t : Triangle) (factor : ℝ) : Triangle := sorry

/-- Main theorem: The area of an extended equilateral triangle is 9 times the original -/
theorem extended_equilateral_area_ratio (t : Triangle) :
  isEquilateral t →
  area (extendTriangle t 3) = 9 * area t := by sorry

end extended_equilateral_area_ratio_l2124_212488


namespace sequence_formula_smallest_m_bound_l2124_212456

def S (n : ℕ) : ℚ := 3/2 * n^2 - 1/2 * n

def a (n : ℕ+) : ℚ := 3 * n - 2

def T (n : ℕ+) : ℚ := 1 - 1 / (3 * n + 1)

theorem sequence_formula (n : ℕ+) : a n = 3 * n - 2 :=
sorry

theorem smallest_m_bound : 
  ∃ m : ℕ, (∀ n : ℕ+, T n < m / 20) ∧ (∀ k : ℕ, k < m → ∃ n : ℕ+, T n ≥ k / 20) :=
sorry

end sequence_formula_smallest_m_bound_l2124_212456


namespace rectangular_yard_area_l2124_212413

theorem rectangular_yard_area (L W : ℝ) : 
  L = 40 →  -- One full side (length) is 40 feet
  2 * W + L = 52 →  -- Total fencing for three sides is 52 feet
  L * W = 240 :=  -- Area of the yard is 240 square feet
by
  sorry

end rectangular_yard_area_l2124_212413


namespace quiz_goal_achievement_l2124_212420

theorem quiz_goal_achievement (total_quizzes : ℕ) (goal_percentage : ℚ) 
  (completed_quizzes : ℕ) (current_as : ℕ) : 
  total_quizzes = 60 →
  goal_percentage = 3/4 →
  completed_quizzes = 40 →
  current_as = 27 →
  (total_quizzes - completed_quizzes) - 
    (↑(total_quizzes) * goal_percentage - current_as).ceil = 2 := by
  sorry

end quiz_goal_achievement_l2124_212420


namespace trigonometric_inequalities_l2124_212404

theorem trigonometric_inequalities (x : Real) (h : 0 ≤ x ∧ x ≤ Real.pi / 2) :
  (1 - Real.cos x ≤ x^2 / 2) ∧
  (x * Real.cos x ≤ Real.sin x ∧ Real.sin x ≤ x * Real.cos (x / 2)) := by
  sorry

end trigonometric_inequalities_l2124_212404


namespace tablet_count_l2124_212438

theorem tablet_count : 
  ∀ (n : ℕ) (x y : ℕ),
  -- Lenovo (x), Samsung (x+6), and Huawei (y) make up less than a third of the total
  (2*x + y + 6 < n/3) →
  -- Apple iPads are three times as many as Huawei tablets
  (n - 2*x - y - 6 = 3*y) →
  -- If Lenovo tablets were tripled, there would be 59 Apple iPads
  (n - 3*x - (x+6) - y = 59) →
  (n = 94) := by
sorry

end tablet_count_l2124_212438


namespace seminar_duration_is_428_l2124_212409

/-- Represents the duration of a seminar session in minutes -/
def seminar_duration (first_part_hours : ℕ) (first_part_minutes : ℕ) (second_part_minutes : ℕ) (closing_event_seconds : ℕ) : ℕ :=
  (first_part_hours * 60 + first_part_minutes) + second_part_minutes + (closing_event_seconds / 60)

/-- Theorem stating that the seminar duration is 428 minutes given the specified conditions -/
theorem seminar_duration_is_428 :
  seminar_duration 4 45 135 500 = 428 := by
  sorry

end seminar_duration_is_428_l2124_212409


namespace unique_cubic_function_l2124_212436

def f (a b x : ℝ) : ℝ := x^3 + a*x^2 + b

theorem unique_cubic_function (a b : ℝ) (h1 : a > 0) 
  (h2 : ∃ (x : ℝ), f a b x = 5 ∧ ∀ (y : ℝ), f a b y ≤ 5)
  (h3 : ∃ (x : ℝ), f a b x = 1 ∧ ∀ (y : ℝ), f a b y ≥ 1) :
  ∀ (x : ℝ), f a b x = x^3 + 3*x^2 + 1 := by
sorry

end unique_cubic_function_l2124_212436


namespace symmetric_points_m_value_l2124_212430

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry about the origin -/
def symmetricAboutOrigin (p q : Point) : Prop :=
  q.x = -p.x ∧ q.y = -p.y

theorem symmetric_points_m_value :
  let p : Point := ⟨2, -1⟩
  let q : Point := ⟨-2, m⟩
  symmetricAboutOrigin p q → m = 1 := by
  sorry

end symmetric_points_m_value_l2124_212430


namespace pentagon_side_length_l2124_212499

/-- The side length of a regular pentagon with perimeter equal to that of an equilateral triangle with side length 20/9 cm is 4/3 cm. -/
theorem pentagon_side_length (s : ℝ) : 
  (5 * s = 3 * (20 / 9)) → s = 4 / 3 := by
  sorry

end pentagon_side_length_l2124_212499


namespace largest_zero_S_l2124_212458

def S : ℕ → ℤ
  | 0 => 0
  | n + 1 => S n + (n + 1) * (if S n < n + 1 then 1 else -1)

theorem largest_zero_S : ∃ k : ℕ, k ≤ 2010 ∧ S k = 0 ∧ ∀ m : ℕ, m ≤ 2010 ∧ m > k → S m ≠ 0 :=
by
  sorry

end largest_zero_S_l2124_212458


namespace blue_balls_removed_l2124_212468

theorem blue_balls_removed (initial_total : ℕ) (initial_blue : ℕ) (final_probability : ℚ) : ℕ :=
  let removed : ℕ := 3
  have h1 : initial_total = 18 := by sorry
  have h2 : initial_blue = 6 := by sorry
  have h3 : final_probability = 1 / 5 := by sorry
  have h4 : (initial_blue - removed : ℚ) / (initial_total - removed) = final_probability := by sorry
  removed

#check blue_balls_removed

end blue_balls_removed_l2124_212468


namespace complex_division_equality_l2124_212486

theorem complex_division_equality : (1 - Complex.I) / Complex.I = -1 - Complex.I := by
  sorry

end complex_division_equality_l2124_212486


namespace problem_statement_l2124_212412

theorem problem_statement (a b c d : ℤ) 
  (h1 : a - b - c + d = 13) 
  (h2 : a + b - c - d = 5) : 
  (b - d)^2 = 16 := by
  sorry

end problem_statement_l2124_212412


namespace percentage_of_students_owning_only_cats_l2124_212476

theorem percentage_of_students_owning_only_cats
  (total_students : ℕ)
  (cat_owners : ℕ)
  (dog_owners : ℕ)
  (both_owners : ℕ)
  (h1 : total_students = 500)
  (h2 : cat_owners = 75)
  (h3 : dog_owners = 150)
  (h4 : both_owners = 25) :
  (cat_owners - both_owners) * 100 / total_students = 10 :=
by sorry

end percentage_of_students_owning_only_cats_l2124_212476


namespace reporter_average_words_per_minute_l2124_212462

/-- Calculates the average words per minute for a reporter given their pay structure and work conditions. -/
theorem reporter_average_words_per_minute 
  (word_pay : ℝ)
  (article_pay : ℝ)
  (num_articles : ℕ)
  (total_hours : ℝ)
  (hourly_earnings : ℝ)
  (h1 : word_pay = 0.1)
  (h2 : article_pay = 60)
  (h3 : num_articles = 3)
  (h4 : total_hours = 4)
  (h5 : hourly_earnings = 105) :
  (((hourly_earnings * total_hours - article_pay * num_articles) / word_pay) / (total_hours * 60)) = 10 := by
  sorry

#check reporter_average_words_per_minute

end reporter_average_words_per_minute_l2124_212462


namespace condition_relations_l2124_212418

-- Define the propositions
variable (A B C D : Prop)

-- Define the given conditions
axiom A_sufficient_D : A → D
axiom B_sufficient_C : B → C
axiom D_necessary_C : C → D
axiom D_sufficient_B : D → B

-- Theorem to prove
theorem condition_relations :
  (C → D) ∧ (A → B) := by sorry

end condition_relations_l2124_212418


namespace u_equals_fib_squared_l2124_212429

def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

def u : ℕ → ℤ
  | 0 => 4
  | 1 => 9
  | n + 2 => 3 * u (n + 1) - u n - 2 * (-1 : ℤ) ^ (n + 2)

theorem u_equals_fib_squared (n : ℕ) : u n = (fib (n + 2))^2 := by
  sorry

end u_equals_fib_squared_l2124_212429


namespace min_elements_in_set_l2124_212403

theorem min_elements_in_set (S : Type) [Fintype S] 
  (X : Fin 100 → Set S)
  (h_nonempty : ∀ i, Set.Nonempty (X i))
  (h_distinct : ∀ i j, i ≠ j → X i ≠ X j)
  (h_disjoint : ∀ i : Fin 99, Disjoint (X i) (X (Fin.succ i)))
  (h_not_union : ∀ i : Fin 99, (X i) ∪ (X (Fin.succ i)) ≠ Set.univ) :
  Fintype.card S ≥ 8 :=
sorry

end min_elements_in_set_l2124_212403
