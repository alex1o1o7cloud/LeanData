import Mathlib

namespace steve_email_percentage_l1506_150637

/-- Given Steve's email management scenario, prove that the percentage of emails
    moved to the work folder out of the remaining emails after trashing is 40%. -/
theorem steve_email_percentage :
  ∀ (initial_emails : ℕ) (emails_left : ℕ),
    initial_emails = 400 →
    emails_left = 120 →
    let emails_after_trash : ℕ := initial_emails / 2
    let emails_to_work : ℕ := emails_after_trash - emails_left
    (emails_to_work : ℚ) / emails_after_trash * 100 = 40 := by
  sorry

end steve_email_percentage_l1506_150637


namespace button_sequence_l1506_150694

theorem button_sequence (a : Fin 6 → ℕ) : 
  a 0 = 1 ∧ 
  (∀ i : Fin 5, a (i + 1) = 3 * a i) ∧ 
  a 4 = 81 ∧ 
  a 5 = 243 → 
  a 3 = 27 := by
sorry

end button_sequence_l1506_150694


namespace no_intersection_in_S_l1506_150676

-- Define the set S of polynomials
inductive S : (ℝ → ℝ) → Prop
  | base : S (λ x => x)
  | sub {f} : S f → S (λ x => x - f x)
  | add {f} : S f → S (λ x => x + (1 - x) * f x)

-- Define the theorem
theorem no_intersection_in_S :
  ∀ (f g : ℝ → ℝ), S f → S g → f ≠ g →
  ∀ x, 0 < x → x < 1 → f x ≠ g x :=
by sorry

end no_intersection_in_S_l1506_150676


namespace trajectory_equation_l1506_150680

/-- The trajectory of points equidistant from A(-1, 1, 0) and B(2, -1, -1) in 3D space -/
theorem trajectory_equation :
  ∀ (x y z : ℝ),
  (x + 1)^2 + (y - 1)^2 + z^2 = (x - 2)^2 + (y + 1)^2 + (z + 1)^2 →
  3*x - 2*y - z = 2 := by
sorry

end trajectory_equation_l1506_150680


namespace quadratic_equation_solution_l1506_150645

theorem quadratic_equation_solution (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 7 * x - 10 = 0 ↔ x = -2 ∨ x = 5/2) ∧ 
  (7^2 - 4 * k * (-10) > 0) ↔ 
  k = 2 := by sorry

end quadratic_equation_solution_l1506_150645


namespace smallest_four_digit_sum_16_l1506_150684

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ :=
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem smallest_four_digit_sum_16 :
  ∀ n : ℕ, is_four_digit n → digit_sum n = 16 → 1960 ≤ n :=
sorry

end smallest_four_digit_sum_16_l1506_150684


namespace multiplication_subtraction_equality_l1506_150678

theorem multiplication_subtraction_equality : (5 * 3) - 2 = 13 := by
  sorry

end multiplication_subtraction_equality_l1506_150678


namespace equation_equivalence_l1506_150607

theorem equation_equivalence :
  ∀ b c : ℝ,
  (∀ x : ℝ, |x - 3| = 4 ↔ x^2 + b*x + c = 0) →
  b = 1 ∧ c = -7 :=
by sorry

end equation_equivalence_l1506_150607


namespace abc_maximum_l1506_150624

theorem abc_maximum (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h1 : a * b + c^2 = (a + c) * (b + c)) (h2 : a + b + c = 3) :
  a * b * c ≤ 1 ∧ ∃ (a' b' c' : ℝ), a' * b' * c' = 1 ∧ 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧ 
    a' * b' + c'^2 = (a' + c') * (b' + c') ∧ 
    a' + b' + c' = 3 :=
by sorry

end abc_maximum_l1506_150624


namespace bucket_weight_l1506_150661

theorem bucket_weight (a b : ℝ) : ℝ :=
  let three_fourths_weight := a
  let one_third_weight := b
  let full_weight := (8 / 5) * a - (3 / 5) * b
  full_weight

#check bucket_weight

end bucket_weight_l1506_150661


namespace intersection_complement_equality_l1506_150693

universe u

def U : Set ℕ := {1, 2, 3, 4, 5}
def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {3, 4, 5}

theorem intersection_complement_equality :
  M ∩ (U \ N) = {1, 2} := by sorry

end intersection_complement_equality_l1506_150693


namespace quadratic_roots_expression_l1506_150634

theorem quadratic_roots_expression (m n : ℝ) : 
  m^2 + 3*m + 1 = 0 → n^2 + 3*n + 1 = 0 → m * n = 1 → (3*m + 1) / (m^3 * n) = -1 := by
  sorry

end quadratic_roots_expression_l1506_150634


namespace point_p_transformation_l1506_150647

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the rotation function
def rotate90ClockwiseAboutOrigin (p : Point2D) : Point2D :=
  { x := p.y, y := -p.x }

-- Define the reflection function
def reflectAcrossXAxis (p : Point2D) : Point2D :=
  { x := p.x, y := -p.y }

-- Define the composition of rotation and reflection
def rotateAndReflect (p : Point2D) : Point2D :=
  reflectAcrossXAxis (rotate90ClockwiseAboutOrigin p)

theorem point_p_transformation :
  let p : Point2D := { x := 3, y := -5 }
  rotateAndReflect p = { x := -5, y := 3 } := by sorry

end point_p_transformation_l1506_150647


namespace intersection_complement_equality_l1506_150686

def U : Set Nat := {0, 1, 2, 3, 4, 5}
def M : Set Nat := {0, 3, 5}
def N : Set Nat := {1, 4, 5}

theorem intersection_complement_equality : M ∩ (U \ N) = {0, 3} := by sorry

end intersection_complement_equality_l1506_150686


namespace dave_white_tshirt_packs_l1506_150602

/-- The number of T-shirts in a pack of white T-shirts -/
def white_pack_size : ℕ := 6

/-- The number of T-shirts in a pack of blue T-shirts -/
def blue_pack_size : ℕ := 4

/-- The number of packs of blue T-shirts Dave bought -/
def blue_packs : ℕ := 2

/-- The total number of T-shirts Dave bought -/
def total_tshirts : ℕ := 26

/-- The number of packs of white T-shirts Dave bought -/
def white_packs : ℕ := 3

theorem dave_white_tshirt_packs :
  white_packs * white_pack_size + blue_packs * blue_pack_size = total_tshirts :=
by sorry

end dave_white_tshirt_packs_l1506_150602


namespace people_who_got_off_train_l1506_150631

theorem people_who_got_off_train (initial_people : ℕ) (people_who_got_on : ℕ) (final_people : ℕ) :
  initial_people = 82 →
  people_who_got_on = 17 →
  final_people = 73 →
  ∃ (people_who_got_off : ℕ), 
    initial_people - people_who_got_off + people_who_got_on = final_people ∧
    people_who_got_off = 26 :=
by
  sorry

end people_who_got_off_train_l1506_150631


namespace fourth_task_end_time_l1506_150608

-- Define the start time of the first task
def start_time : Nat := 8 * 60  -- 8:00 AM in minutes

-- Define the end time of the second task
def end_second_task : Nat := 10 * 60 + 20  -- 10:20 AM in minutes

-- Define the number of tasks
def num_tasks : Nat := 4

-- Theorem to prove
theorem fourth_task_end_time :
  let total_time := end_second_task - start_time
  let task_duration := total_time / 2
  let end_time := end_second_task + task_duration * 2
  end_time = 12 * 60 + 40  -- 12:40 PM in minutes
  := by sorry

end fourth_task_end_time_l1506_150608


namespace trig_identity_l1506_150614

theorem trig_identity (α : ℝ) : 
  (Real.sin (α / 2))^6 - (Real.cos (α / 2))^6 = ((Real.sin α)^2 - 4) / 4 * Real.cos α := by
  sorry

end trig_identity_l1506_150614


namespace max_equal_distribution_l1506_150656

theorem max_equal_distribution (bags eyeliners scarves hairbands : ℕ) 
  (h1 : bags = 2923)
  (h2 : eyeliners = 3239)
  (h3 : scarves = 1785)
  (h4 : hairbands = 1379) :
  Nat.gcd bags (Nat.gcd eyeliners (Nat.gcd scarves hairbands)) = 1 := by
  sorry

end max_equal_distribution_l1506_150656


namespace prob_red_white_red_is_7_66_l1506_150638

-- Define the number of red and white marbles
def red_marbles : ℕ := 5
def white_marbles : ℕ := 7

-- Define the total number of marbles
def total_marbles : ℕ := red_marbles + white_marbles

-- Define the probability of drawing red, white, and red marbles in order
def prob_red_white_red : ℚ := (red_marbles : ℚ) / total_marbles *
                              (white_marbles : ℚ) / (total_marbles - 1) *
                              (red_marbles - 1 : ℚ) / (total_marbles - 2)

-- Theorem statement
theorem prob_red_white_red_is_7_66 :
  prob_red_white_red = 7 / 66 := by sorry

end prob_red_white_red_is_7_66_l1506_150638


namespace function_symmetry_l1506_150613

theorem function_symmetry (f : ℝ → ℝ) (t : ℝ) :
  (∀ x, f x = 3 * x + Real.sin x + 1) →
  f t = 2 →
  f (-t) = 0 := by
  sorry

end function_symmetry_l1506_150613


namespace z_in_second_quadrant_l1506_150659

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the equation
def equation (z : ℂ) : Prop := (1 - i) * z = 2 + 3 * i

-- Define the second quadrant
def second_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_second_quadrant :
  ∃ z : ℂ, equation z ∧ second_quadrant z :=
sorry

end z_in_second_quadrant_l1506_150659


namespace total_candy_collected_l1506_150617

/-- The number of candy pieces collected by Travis and his brother -/
def total_candy : ℕ := 68

/-- The number of people who collected candy -/
def num_people : ℕ := 2

/-- The number of candy pieces each person ate -/
def candy_eaten_per_person : ℕ := 4

/-- The number of candy pieces left after eating -/
def candy_left : ℕ := 60

/-- Theorem stating that the total candy collected equals 68 -/
theorem total_candy_collected :
  total_candy = candy_left + (num_people * candy_eaten_per_person) :=
by sorry

end total_candy_collected_l1506_150617


namespace number_problem_l1506_150672

theorem number_problem (n : ℚ) : (1/2 : ℚ) * (3/5 : ℚ) * n = 36 → n = 120 := by
  sorry

end number_problem_l1506_150672


namespace grapefruit_juice_percentage_l1506_150648

def total_volume : ℝ := 50
def orange_juice : ℝ := 20
def lemon_juice_percentage : ℝ := 35

theorem grapefruit_juice_percentage :
  (total_volume - orange_juice - (lemon_juice_percentage / 100) * total_volume) / total_volume * 100 = 25 := by
  sorry

end grapefruit_juice_percentage_l1506_150648


namespace divisible_by_nine_sequence_l1506_150689

theorem divisible_by_nine_sequence (N : ℕ) : 
  (∃ (seq : List ℕ), 
    seq.length = 1110 ∧ 
    (∀ n ∈ seq, n % 9 = 0) ∧
    (∀ n ∈ seq, N ≤ n ∧ n ≤ 10000) ∧
    (∀ m, N ≤ m ∧ m ≤ 10000 ∧ m % 9 = 0 → m ∈ seq)) →
  N = 27 := by
sorry

end divisible_by_nine_sequence_l1506_150689


namespace sum_of_squares_l1506_150649

theorem sum_of_squares (a b c : ℝ) 
  (h1 : a * b + b * c + c * a = 5)
  (h2 : a + b + c = 20) :
  a^2 + b^2 + c^2 = 390 := by
sorry

end sum_of_squares_l1506_150649


namespace negation_equivalence_l1506_150641

theorem negation_equivalence : 
  (¬(∀ x : ℝ, x^2 < 1 → -1 < x ∧ x < 1)) ↔ 
  (∀ x : ℝ, x^2 ≥ 1 → x ≥ 1 ∨ x ≤ -1) := by sorry

end negation_equivalence_l1506_150641


namespace percentage_equality_l1506_150685

theorem percentage_equality (x y : ℝ) (hx : x ≠ 0) :
  (0.4 * 0.5 * x = 0.2 * 0.3 * y) → y = (10/3) * x := by
sorry

end percentage_equality_l1506_150685


namespace complex_absolute_value_squared_l1506_150619

theorem complex_absolute_value_squared (z : ℂ) (h : z + Complex.abs z = 3 + 7*I) : Complex.abs z ^ 2 = 841 / 9 := by
  sorry

end complex_absolute_value_squared_l1506_150619


namespace remainder_divisibility_l1506_150626

theorem remainder_divisibility (x : ℤ) : x % 8 = 3 → x % 72 = 3 := by
  sorry

end remainder_divisibility_l1506_150626


namespace sum_six_to_thousand_l1506_150692

/-- Count of digit 6 occurrences in a number -/
def count_six (n : ℕ) : ℕ := sorry

/-- Sum of digit 6 occurrences from 1 to n -/
def sum_six_occurrences (n : ℕ) : ℕ := sorry

/-- The theorem stating that the sum of digit 6 occurrences from 1 to 1000 is 300 -/
theorem sum_six_to_thousand :
  sum_six_occurrences 1000 = 300 := by sorry

end sum_six_to_thousand_l1506_150692


namespace first_three_decimal_digits_l1506_150664

theorem first_three_decimal_digits (n : ℕ) (x : ℝ) : 
  n = 2005 → x = (10^n + 1)^(11/8) → 
  ∃ (k : ℕ), x = k + 0.375 + r ∧ 0 ≤ r ∧ r < 0.001 :=
sorry

end first_three_decimal_digits_l1506_150664


namespace triangle_properties_l1506_150618

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

theorem triangle_properties (t : Triangle) 
  (h1 : t.a^2 + t.b^2 = t.c^2 + t.a * t.b)
  (h2 : Real.sqrt 3 * t.c = 14 * Real.sin t.C)
  (h3 : t.a + t.b = 13) :
  t.C = π/3 ∧ t.c = 7 ∧ 
  (1/2 * t.a * t.b * Real.sin t.C = 10 * Real.sqrt 3) := by
  sorry

end triangle_properties_l1506_150618


namespace smallest_number_l1506_150677

theorem smallest_number (a b c d : ℚ) (ha : a = -2) (hb : b = -5/2) (hc : c = 0) (hd : d = 1/5) :
  b ≤ a ∧ b ≤ c ∧ b ≤ d :=
by sorry

end smallest_number_l1506_150677


namespace circle_radius_secant_l1506_150616

theorem circle_radius_secant (center P Q R : ℝ × ℝ) : 
  let distance := λ (a b : ℝ × ℝ) => Real.sqrt ((a.1 - b.1)^2 + (a.2 - b.2)^2)
  let radius := distance center Q
  distance center P = 15 ∧ 
  distance P Q = 10 ∧ 
  distance Q R = 8 ∧
  (∃ t : ℝ, 0 < t ∧ t < 1 ∧ R = (1 - t) • P + t • Q) →
  radius = 3 * Real.sqrt 5 := by
sorry

end circle_radius_secant_l1506_150616


namespace basketball_scores_theorem_l1506_150621

/-- Represents the scores of a basketball player in a series of games -/
structure BasketballScores where
  total_games : Nat
  sixth_game_score : Nat
  seventh_game_score : Nat
  eighth_game_score : Nat
  ninth_game_score : Nat
  first_five_avg : ℝ
  first_nine_avg : ℝ

/-- Theorem about basketball scores -/
theorem basketball_scores_theorem (scores : BasketballScores) 
  (h1 : scores.total_games = 10)
  (h2 : scores.sixth_game_score = 22)
  (h3 : scores.seventh_game_score = 15)
  (h4 : scores.eighth_game_score = 12)
  (h5 : scores.ninth_game_score = 19) :
  (scores.first_nine_avg = (5 * scores.first_five_avg + 68) / 9) ∧
  (∃ (min_y : ℝ), min_y = 12 ∧ ∀ y, y = scores.first_nine_avg → y ≥ min_y) ∧
  (scores.first_nine_avg > scores.first_five_avg → 
    ∃ (max_score : ℕ), max_score = 84 ∧ 
    ∀ s, s = (5 : ℝ) * scores.first_five_avg → s ≤ max_score) := by
  sorry

end basketball_scores_theorem_l1506_150621


namespace yellow_red_difference_after_border_l1506_150670

/-- Represents a hexagonal tile figure --/
structure HexFigure where
  red_tiles : ℕ
  yellow_tiles : ℕ

/-- Adds a border of yellow tiles to a hexagonal figure --/
def add_yellow_border (fig : HexFigure) : HexFigure :=
  { red_tiles := fig.red_tiles,
    yellow_tiles := fig.yellow_tiles + 18 }

/-- The initial hexagonal figure --/
def initial_figure : HexFigure :=
  { red_tiles := 15,
    yellow_tiles := 10 }

theorem yellow_red_difference_after_border :
  (add_yellow_border initial_figure).yellow_tiles - (add_yellow_border initial_figure).red_tiles = 13 :=
by sorry

end yellow_red_difference_after_border_l1506_150670


namespace polynomial_value_theorem_l1506_150681

theorem polynomial_value_theorem (P : Int → Int) (a b c d : Int) :
  (∀ x : Int, ∃ y : Int, P x = y) →  -- P has integer coefficients
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d) →  -- a, b, c, d are distinct
  (P a = 5 ∧ P b = 5 ∧ P c = 5 ∧ P d = 5) →  -- P(a) = P(b) = P(c) = P(d) = 5
  ¬ ∃ k : Int, P k = 8 :=  -- There does not exist an integer k such that P(k) = 8
by sorry

end polynomial_value_theorem_l1506_150681


namespace angle_ADF_measure_l1506_150633

-- Define the circle O and points A, B, C, D, E, F
variable (O : ℝ × ℝ) (A B C D E F : ℝ × ℝ)

-- Define the circle's radius
variable (r : ℝ)

-- Define the angle measure function
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

-- State the given conditions
axiom C_on_BE_extension : sorry
axiom CA_tangent : sorry
axiom DC_bisects_ACB : angle_measure C A D = angle_measure C B D
axiom DC_intersects_AE : sorry
axiom DC_intersects_AB : sorry

-- Define the theorem
theorem angle_ADF_measure :
  angle_measure A D F = 67.5 := sorry

end angle_ADF_measure_l1506_150633


namespace adams_dried_fruits_l1506_150667

/-- The problem of calculating the amount of dried fruits Adam bought --/
theorem adams_dried_fruits :
  ∀ (dried_fruits : ℝ),
  (3 : ℝ) * 12 + dried_fruits * 8 = 56 →
  dried_fruits = 2.5 := by
  sorry

end adams_dried_fruits_l1506_150667


namespace problem_solution_l1506_150646

-- Define the equation from the problem
def equation (n : ℕ) : Prop := 2^(2*n) = 2^n + 992

-- Define the constant term function
def constant_term (n : ℕ) : ℕ := Nat.choose (2*n) 2

-- Theorem statement
theorem problem_solution :
  (∃ n : ℕ, equation n ∧ n = 5) ∧
  constant_term 5 = 45 := by
sorry


end problem_solution_l1506_150646


namespace f_has_max_and_min_l1506_150654

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 3*x^2 - 9*x

-- Define the domain
def domain (x : ℝ) : Prop := -2 < x ∧ x < 4

-- Theorem statement
theorem f_has_max_and_min :
  ∃ (x_max x_min : ℝ),
    domain x_max ∧ domain x_min ∧
    (∀ x, domain x → f x ≤ f x_max) ∧
    (∀ x, domain x → f x_min ≤ f x) ∧
    f x_max = 5 ∧ f x_min = -27 ∧
    x_max = -1 ∧ x_min = 3 :=
sorry

end f_has_max_and_min_l1506_150654


namespace find_incorrect_value_l1506_150690

/-- Given a set of observations with an initial mean and a corrected mean after fixing one misrecorded value,
    this theorem proves that the original incorrect value can be determined. -/
theorem find_incorrect_value (n : ℕ) (m1 m2 x : ℚ) (hn : n > 0) :
  let y := n * m1 + x - n * m2
  y = 23 :=
by sorry

end find_incorrect_value_l1506_150690


namespace focus_directrix_distance_l1506_150665

-- Define the ellipse C₁
def ellipse_C1 (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 = 1

-- Define the parabola C₂
def parabola_C2 (x y : ℝ) : Prop :=
  y^2 = 4 * x

-- Theorem statement
theorem focus_directrix_distance :
  -- Conditions
  (ellipse_C1 (-2) 0) ∧
  (ellipse_C1 (Real.sqrt 2) ((Real.sqrt 2) / 2)) ∧
  (parabola_C2 3 (-2 * Real.sqrt 3)) ∧
  (parabola_C2 4 (-4)) →
  -- Conclusion
  ∃ (focus_x directrix_x : ℝ),
    -- Left focus of C₁
    focus_x = Real.sqrt 3 ∧
    -- Directrix of C₂
    directrix_x = -1 ∧
    -- Distance between left focus and directrix
    focus_x - directrix_x = Real.sqrt 3 - 1 :=
by sorry

end focus_directrix_distance_l1506_150665


namespace gravel_cost_proof_l1506_150600

/-- Calculates the cost of graveling two intersecting roads on a rectangular lawn. -/
def gravel_cost (lawn_length lawn_width road_width gravel_cost_per_sqm : ℕ) : ℕ :=
  let road_length_area := lawn_length * road_width
  let road_width_area := (lawn_width - road_width) * road_width
  let total_area := road_length_area + road_width_area
  total_area * gravel_cost_per_sqm

/-- Proves that the cost of graveling two intersecting roads on a rectangular lawn
    with given dimensions and costs is equal to 3900. -/
theorem gravel_cost_proof :
  gravel_cost 80 60 10 3 = 3900 := by
  sorry

end gravel_cost_proof_l1506_150600


namespace greatest_three_digit_number_l1506_150671

theorem greatest_three_digit_number : ∃ (n : ℕ), n = 953 ∧
  n ≤ 999 ∧
  ∃ (k : ℕ), n = 9 * k + 2 ∧
  ∃ (m : ℕ), n = 5 * m + 3 ∧
  ∃ (l : ℕ), n = 7 * l + 4 ∧
  ∀ (x : ℕ), x ≤ 999 → 
    (∃ (a b c : ℕ), x = 9 * a + 2 ∧ x = 5 * b + 3 ∧ x = 7 * c + 4) → 
    x ≤ n :=
by sorry

end greatest_three_digit_number_l1506_150671


namespace Q_roots_nature_l1506_150651

/-- The polynomial Q(x) = x^6 - 4x^5 + 3x^4 - 7x^3 - x^2 + x + 10 -/
def Q (x : ℝ) : ℝ := x^6 - 4*x^5 + 3*x^4 - 7*x^3 - x^2 + x + 10

/-- Theorem stating that Q(x) has at least one negative root and at least two positive roots -/
theorem Q_roots_nature :
  (∃ x : ℝ, x < 0 ∧ Q x = 0) ∧
  (∃ x y : ℝ, 0 < x ∧ 0 < y ∧ x ≠ y ∧ Q x = 0 ∧ Q y = 0) :=
sorry

end Q_roots_nature_l1506_150651


namespace parabola_intersection_l1506_150609

/-- Given a parabola y = x^2 and four points on it, if two lines formed by these points
    intersect on the y-axis, then the x-coordinate of the fourth point is determined by
    the x-coordinates of the other three points. -/
theorem parabola_intersection (a b c d : ℝ) : 
  (∃ l : ℝ, (a^2 = (b + a)*a + l ∧ b^2 = (b + a)*b + l) ∧ 
             (c^2 = (d + c)*c + l ∧ d^2 = (d + c)*d + l)) →
  d = a * b / c :=
sorry

end parabola_intersection_l1506_150609


namespace xyz_value_l1506_150658

theorem xyz_value (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (eq1 : x + 1/y = 5)
  (eq2 : y + 1/z = 2)
  (eq3 : z + 1/x = 8/3) :
  x * y * z = 8 + 3 * Real.sqrt 7 := by
sorry

end xyz_value_l1506_150658


namespace prism_21_edges_has_9_faces_l1506_150604

/-- Represents a prism with a given number of edges. -/
structure Prism where
  edges : ℕ

/-- Calculates the number of faces in a prism given its number of edges. -/
def num_faces (p : Prism) : ℕ :=
  (p.edges / 3) + 2

/-- Theorem stating that a prism with 21 edges has 9 faces. -/
theorem prism_21_edges_has_9_faces :
  ∀ (p : Prism), p.edges = 21 → num_faces p = 9 := by
  sorry

#eval num_faces { edges := 21 }

end prism_21_edges_has_9_faces_l1506_150604


namespace log_equation_implies_p_zero_l1506_150669

theorem log_equation_implies_p_zero (p q : ℝ) (hp : p > 0) (hq : q > 0) (hq2 : q + 2 > 0) :
  Real.log p - Real.log q = Real.log (p / (q + 2)) → p = 0 := by
  sorry

end log_equation_implies_p_zero_l1506_150669


namespace tanning_time_proof_l1506_150606

/-- Calculates the remaining tanning time for the last two weeks of a month. -/
def remaining_tanning_time (monthly_limit : ℕ) (week1_time : ℕ) (week2_time : ℕ) : ℕ :=
  monthly_limit - (week1_time + week2_time)

/-- Proves that given the specified tanning times, the remaining time is 45 minutes. -/
theorem tanning_time_proof : remaining_tanning_time 200 75 80 = 45 := by
  sorry

end tanning_time_proof_l1506_150606


namespace remainder_problem_l1506_150611

theorem remainder_problem (n : ℤ) (h : n % 5 = 3) : (4 * n + 2) % 5 = 4 := by
  sorry

end remainder_problem_l1506_150611


namespace talia_drives_16_miles_l1506_150601

/-- Represents the total distance Talia drives in a day -/
def total_distance (home_to_park park_to_grocery grocery_to_home : ℝ) : ℝ :=
  home_to_park + park_to_grocery + grocery_to_home

/-- Theorem stating that Talia drives 16 miles given the distances between locations -/
theorem talia_drives_16_miles :
  let home_to_park : ℝ := 5
  let park_to_grocery : ℝ := 3
  let grocery_to_home : ℝ := 8
  total_distance home_to_park park_to_grocery grocery_to_home = 16 := by
  sorry

end talia_drives_16_miles_l1506_150601


namespace cuboid_height_l1506_150653

/-- The height of a rectangular cuboid given its surface area, length, and width -/
theorem cuboid_height (surface_area length width height : ℝ) : 
  surface_area = 2 * length * width + 2 * length * height + 2 * width * height →
  surface_area = 442 →
  length = 7 →
  width = 8 →
  height = 11 := by
  sorry


end cuboid_height_l1506_150653


namespace grapefruit_orchards_l1506_150620

/-- Represents the number of orchards for each type of citrus fruit -/
structure CitrusOrchards where
  total : ℕ
  lemons : ℕ
  oranges : ℕ
  limes : ℕ
  grapefruits : ℕ
  mandarins : ℕ

/-- Theorem stating the number of grapefruit orchards given the conditions -/
theorem grapefruit_orchards (c : CitrusOrchards) : c.grapefruits = 6 :=
  by
  have h1 : c.total = 40 := sorry
  have h2 : c.lemons = 15 := sorry
  have h3 : c.oranges = 2 * c.lemons / 3 := sorry
  have h4 : c.limes = c.grapefruits := sorry
  have h5 : c.mandarins = c.grapefruits / 2 := sorry
  have h6 : c.total = c.lemons + c.oranges + c.limes + c.grapefruits + c.mandarins := sorry
  sorry

end grapefruit_orchards_l1506_150620


namespace triangle_side_perp_distance_relation_l1506_150691

/-- Represents a triangle with side lengths and perpendicular distances -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  h_a : ℝ
  h_b : ℝ
  h_c : ℝ

/-- Theorem stating the relationship between side lengths and perpendicular distances -/
theorem triangle_side_perp_distance_relation (t : Triangle) 
  (h_side : t.a < t.b ∧ t.b < t.c) : 
  t.h_a > t.h_b ∧ t.h_b > t.h_c := by
  sorry


end triangle_side_perp_distance_relation_l1506_150691


namespace inequality_proof_l1506_150629

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) :
  x + 1 / (2 * y) > y + 1 / x := by
  sorry

end inequality_proof_l1506_150629


namespace necessary_not_sufficient_condition_l1506_150603

theorem necessary_not_sufficient_condition : 
  (∀ x : ℝ, x - 1 = 0 → (x - 1) * (x - 2) = 0) ∧ 
  (∃ x : ℝ, (x - 1) * (x - 2) = 0 ∧ x - 1 ≠ 0) := by
  sorry

end necessary_not_sufficient_condition_l1506_150603


namespace square_remainder_l1506_150630

theorem square_remainder (n : ℤ) : n % 5 = 3 → (n^2) % 5 = 4 := by
  sorry

end square_remainder_l1506_150630


namespace quadratic_factorization_l1506_150699

theorem quadratic_factorization (a b c : ℤ) : 
  (∃ b c : ℤ, ∀ x : ℤ, (x - a) * (x - 6) + 1 = (x + b) * (x + c)) ↔ a = 8 := by
  sorry

end quadratic_factorization_l1506_150699


namespace divisibility_property_l1506_150697

theorem divisibility_property (n : ℕ) (h1 : n > 2) (h2 : Even n) :
  (n + 1) ∣ (n + 1)^(n - 1) := by
  sorry

end divisibility_property_l1506_150697


namespace stock_price_change_l1506_150679

theorem stock_price_change (total_stocks : ℕ) (higher_percentage : ℚ) : 
  total_stocks = 4200 →
  higher_percentage = 35/100 →
  ∃ (higher lower : ℕ),
    higher + lower = total_stocks ∧
    higher = (1 + higher_percentage) * lower ∧
    higher = 2412 := by
  sorry

end stock_price_change_l1506_150679


namespace line_segment_endpoint_l1506_150640

/-- Given a line segment from (2, 2) to (x, 6) with length 10 and x > 0, prove x = 2 + 2√21 -/
theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  (x - 2)^2 + 4^2 = 10^2 → 
  x = 2 + 2 * Real.sqrt 21 := by
sorry

end line_segment_endpoint_l1506_150640


namespace final_number_is_odd_l1506_150663

/-- The sum of integers from 1 to n -/
def sum_to_n (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The process of replacing two numbers with their absolute difference cubed -/
def replace_process (a b : ℤ) : ℤ := |a - b|^3

/-- The theorem stating that the final number after the replace process is odd -/
theorem final_number_is_odd (n : ℕ) (h : n = 2017) :
  ∃ (k : ℕ), Odd (sum_to_n n) ∧
  (∀ (a b : ℤ), Odd (a + b) ↔ Odd (replace_process a b)) →
  Odd k := by sorry

end final_number_is_odd_l1506_150663


namespace pizza_toppings_count_l1506_150666

theorem pizza_toppings_count (n : ℕ) (h : n = 8) : 
  (n) + (n.choose 2) = 36 := by sorry

end pizza_toppings_count_l1506_150666


namespace greatest_valid_integer_l1506_150623

def is_valid (n : ℕ) : Prop :=
  n < 150 ∧ Nat.gcd n 30 = 5

theorem greatest_valid_integer : 
  (∀ m : ℕ, is_valid m → m ≤ 145) ∧ is_valid 145 :=
sorry

end greatest_valid_integer_l1506_150623


namespace min_people_with_hat_and_glove_l1506_150673

theorem min_people_with_hat_and_glove (n : ℕ) (gloves hats both : ℕ) : 
  n > 0 → 
  gloves = (2 * n) / 5 → 
  hats = (3 * n) / 4 → 
  both = gloves + hats - n → 
  both ≥ 3 := by
sorry

end min_people_with_hat_and_glove_l1506_150673


namespace parabola_hyperbola_equations_l1506_150615

/-- Given a parabola and a hyperbola satisfying certain conditions, 
    prove their equations. -/
theorem parabola_hyperbola_equations :
  ∀ (a b : ℝ) (parabola hyperbola : ℝ → ℝ → Prop),
    a > 0 → b > 0 →
    (∀ x y, hyperbola x y ↔ x^2 / a^2 - y^2 / b^2 = 1) →
    (∃ f, f > 0 ∧ ∀ x y, parabola x y ↔ y^2 = 4 * f * x) →
    (∃ xf yf, hyperbola xf yf ∧ ∀ x y, parabola x y → (x - xf)^2 + y^2 = f^2) →
    parabola (3/2) (Real.sqrt 6) →
    hyperbola (3/2) (Real.sqrt 6) →
    (∀ x y, parabola x y ↔ y^2 = 4 * x) ∧
    (∀ x y, hyperbola x y ↔ 4 * x^2 - 4 * y^2 / 3 = 1) :=
by sorry

end parabola_hyperbola_equations_l1506_150615


namespace angle_terminal_side_trig_sum_l1506_150612

theorem angle_terminal_side_trig_sum (α : Real) :
  (∃ (x y : Real), x = -5 ∧ y = 12 ∧ x = Real.cos α * Real.sqrt (x^2 + y^2) ∧ y = Real.sin α * Real.sqrt (x^2 + y^2)) →
  Real.sin α + 2 * Real.cos α = 7/5 := by
  sorry

end angle_terminal_side_trig_sum_l1506_150612


namespace percent_decrease_cars_sold_car_sales_decrease_proof_l1506_150687

/-- Calculates the percent decrease in cars sold given the increase in total profit and average profit per car -/
theorem percent_decrease_cars_sold 
  (total_profit_increase : ℝ) 
  (avg_profit_per_car_increase : ℝ) : ℝ :=
  let new_total_profit_ratio := 1 + total_profit_increase
  let new_avg_profit_ratio := 1 + avg_profit_per_car_increase
  let cars_sold_ratio := new_total_profit_ratio / new_avg_profit_ratio
  (1 - cars_sold_ratio) * 100

/-- The percent decrease in cars sold is approximately 30% when total profit increases by 30% and average profit per car increases by 85.71% -/
theorem car_sales_decrease_proof : 
  abs (percent_decrease_cars_sold 0.30 0.8571 - 30) < 0.01 := by
  sorry

end percent_decrease_cars_sold_car_sales_decrease_proof_l1506_150687


namespace train_passing_time_specific_train_problem_l1506_150698

/-- The time taken for a faster train to completely pass a slower train -/
theorem train_passing_time (train_length : ℝ) (fast_speed slow_speed : ℝ) : ℝ :=
  let relative_speed := fast_speed - slow_speed
  let relative_speed_mps := relative_speed * (5 / 18)
  let total_distance := 2 * train_length
  total_distance / relative_speed_mps

/-- Proof of the specific train problem -/
theorem specific_train_problem :
  ∃ (t : ℝ), abs (t - train_passing_time 75 46 36) < 0.01 :=
sorry

end train_passing_time_specific_train_problem_l1506_150698


namespace classroom_size_l1506_150642

theorem classroom_size :
  ∀ (initial_students : ℕ),
  (0.4 * initial_students : ℝ) = (0.32 * (initial_students + 5) : ℝ) →
  initial_students = 20 :=
by
  sorry

end classroom_size_l1506_150642


namespace problem_solution_l1506_150625

noncomputable def f (a k : ℝ) (x : ℝ) : ℝ := k * a^x - a^(-x)

noncomputable def g (a : ℝ) (x : ℝ) : ℝ := a^(2*x) + a^(-2*x) - 4 * (f a 1 x)

theorem problem_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f := f a 1
  ∀ x : ℝ, f x = -f (-x) →
  f 1 > 0 →
  f 1 = 3/2 →
  (∀ x : ℝ, f (x^2 + 2*x) + f (x - 4) > 0 ↔ x < -4 ∨ x > 1) ∧
  (∃ m : ℝ, m = -2 ∧ ∀ x : ℝ, x ≥ 1 → g a x ≥ m) :=
by sorry

end problem_solution_l1506_150625


namespace complex_fraction_equality_l1506_150675

theorem complex_fraction_equality : 50 / (8 - 3/7) = 350/53 := by
  sorry

end complex_fraction_equality_l1506_150675


namespace total_pennies_donated_l1506_150605

def cassandra_pennies : ℕ := 5000
def james_difference : ℕ := 276

theorem total_pennies_donated (cassandra : ℕ) (james_diff : ℕ) 
  (h1 : cassandra = cassandra_pennies) 
  (h2 : james_diff = james_difference) : 
  cassandra + (cassandra - james_diff) = 9724 :=
by
  sorry

end total_pennies_donated_l1506_150605


namespace unique_determination_l1506_150650

-- Define the triangle types
inductive TriangleType
  | Isosceles
  | Equilateral
  | Right
  | Scalene

-- Define the given parts
inductive GivenParts
  | BaseAngleVertexAngle
  | VertexAngleBase
  | CircumscribedRadius
  | ArmInscribedRadius
  | TwoAnglesOneSide

-- Function to check if a combination uniquely determines a triangle
def uniquelyDetermines (t : TriangleType) (p : GivenParts) : Prop :=
  match t, p with
  | TriangleType.Isosceles, GivenParts.BaseAngleVertexAngle => False
  | TriangleType.Isosceles, GivenParts.VertexAngleBase => True
  | TriangleType.Equilateral, GivenParts.CircumscribedRadius => True
  | TriangleType.Right, GivenParts.ArmInscribedRadius => False
  | TriangleType.Scalene, GivenParts.TwoAnglesOneSide => True
  | _, _ => False

theorem unique_determination :
  ∀ (t : TriangleType) (p : GivenParts),
    (t = TriangleType.Isosceles ∧ p = GivenParts.BaseAngleVertexAngle) ↔ ¬(uniquelyDetermines t p) :=
sorry

end unique_determination_l1506_150650


namespace age_difference_l1506_150662

/-- Represents the ages of Ramesh, Mahesh, and Suresh -/
structure Ages where
  ramesh : ℕ
  mahesh : ℕ
  suresh : ℕ

/-- The ratio of present ages -/
def presentRatio (a : Ages) : Bool :=
  2 * a.mahesh = 5 * a.ramesh ∧ 5 * a.suresh = 8 * a.mahesh

/-- The ratio of ages after 15 years -/
def futureRatio (a : Ages) : Bool :=
  14 * (a.ramesh + 15) = 9 * (a.mahesh + 15) ∧
  21 * (a.mahesh + 15) = 14 * (a.suresh + 15)

/-- The theorem to be proved -/
theorem age_difference (a : Ages) :
  presentRatio a → futureRatio a → a.suresh - a.mahesh = 45 := by
  sorry

end age_difference_l1506_150662


namespace at_least_one_not_perfect_square_l1506_150688

theorem at_least_one_not_perfect_square (d : ℕ+) :
  ¬(∃ x y z : ℕ, (2 * d - 1 = x^2) ∧ (5 * d - 1 = y^2) ∧ (13 * d - 1 = z^2)) :=
by sorry

end at_least_one_not_perfect_square_l1506_150688


namespace geometric_series_sum_8_terms_l1506_150643

def geometric_series_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

theorem geometric_series_sum_8_terms :
  geometric_series_sum (1/4) (1/4) 8 = 65535/196608 := by
  sorry

end geometric_series_sum_8_terms_l1506_150643


namespace largest_perimeter_is_24_l1506_150627

/-- Represents a configuration of two regular polygons and a circle meeting at a point -/
structure ShapeConfiguration where
  n : ℕ  -- number of sides in each polygon
  polygonSideLength : ℝ
  circleRadius : ℝ
  polygonAngleSum : ℝ  -- sum of interior angles of polygons at meeting point
  circleAngle : ℝ  -- angle subtended by circle at meeting point

/-- The perimeter of the configuration, excluding the circle's circumference -/
def perimeter (config : ShapeConfiguration) : ℝ :=
  2 * config.n * config.polygonSideLength

/-- Theorem stating the largest possible perimeter for the given configuration -/
theorem largest_perimeter_is_24 (config : ShapeConfiguration) : 
  config.polygonSideLength = 2 ∧ 
  config.circleRadius = 1 ∧ 
  config.polygonAngleSum + config.circleAngle = 360 →
  ∃ (maxConfig : ShapeConfiguration), 
    perimeter maxConfig = 24 ∧ 
    ∀ (c : ShapeConfiguration), perimeter c ≤ perimeter maxConfig :=
by
  sorry

end largest_perimeter_is_24_l1506_150627


namespace z_in_third_quadrant_l1506_150655

def complex_number (a b : ℝ) : ℂ := Complex.mk a b

theorem z_in_third_quadrant :
  let i : ℂ := complex_number 0 1
  let z : ℂ := (1 - 2 * i) / i
  (z.re < 0) ∧ (z.im < 0) := by
  sorry

end z_in_third_quadrant_l1506_150655


namespace mountain_climbs_l1506_150683

/-- Proves that Boris needs to climb his mountain 4 times to match Hugo's total climb -/
theorem mountain_climbs (hugo_elevation : ℕ) (boris_difference : ℕ) (hugo_climbs : ℕ) : 
  hugo_elevation = 10000 →
  boris_difference = 2500 →
  hugo_climbs = 3 →
  (hugo_elevation * hugo_climbs) / (hugo_elevation - boris_difference) = 4 := by
sorry

end mountain_climbs_l1506_150683


namespace imaginary_part_of_complex_number_l1506_150660

theorem imaginary_part_of_complex_number (z : ℂ) :
  (z.re > 0) →
  (z.im = 2 * z.re) →
  (Complex.abs z = Real.sqrt 5) →
  z.im = 2 :=
by sorry

end imaginary_part_of_complex_number_l1506_150660


namespace coefficient_c_positive_l1506_150695

-- Define a quadratic trinomial
def quadratic_trinomial (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the condition for no roots
def no_roots (a b c : ℝ) : Prop := ∀ x, quadratic_trinomial a b c x ≠ 0

-- Theorem statement
theorem coefficient_c_positive
  (a b c : ℝ)
  (h1 : no_roots a b c)
  (h2 : a + b + c > 0) :
  c > 0 :=
sorry

end coefficient_c_positive_l1506_150695


namespace p_decreasing_zero_l1506_150610

/-- The probability that |T-H| = k after a game with 4m coins, 
    where T is the number of tails and H is the number of heads. -/
def p (m : ℕ) (k : ℕ) : ℝ := sorry

/-- The optimal strategy for the coin flipping game -/
def optimal_strategy : sorry := sorry

axiom p_zero_zero : p 0 0 = 1

axiom p_zero_pos : ∀ k : ℕ, k ≥ 1 → p 0 k = 0

/-- The main theorem: p_m(0) ≥ p_m+1(0) for all nonnegative integers m -/
theorem p_decreasing_zero : ∀ m : ℕ, p m 0 ≥ p (m + 1) 0 := by sorry

end p_decreasing_zero_l1506_150610


namespace equation_solution_l1506_150644

theorem equation_solution (y : ℝ) : y + 81 / (y - 3) = -12 ↔ y = -6 ∨ y = -3 := by
  sorry

end equation_solution_l1506_150644


namespace reflected_ray_equation_l1506_150628

/-- The equation of the reflected light ray given an incident ray and a reflecting line. -/
theorem reflected_ray_equation :
  -- Incident ray: y = 2x + 1
  let incident_ray (x y : ℝ) := y = 2 * x + 1
  -- Reflecting line: y = x
  let reflecting_line (x y : ℝ) := y = x
  -- Reflected ray equation
  let reflected_ray (x y : ℝ) := x - 2 * y - 1 = 0
  -- The theorem
  ∀ x y : ℝ, reflected_ray x y ↔ 
    ∃ m : ℝ, incident_ray m (2 * m + 1) ∧ 
             reflecting_line ((x + y) / 2) ((x + y) / 2) ∧
             (x - m = y - (2 * m + 1)) :=
by
  sorry

end reflected_ray_equation_l1506_150628


namespace arithmetic_sequence_condition_l1506_150636

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_condition 
  (a : ℕ → ℝ) (d : ℝ) (h : is_arithmetic_sequence a) :
  (d > 0 ↔ a 2 > a 1) :=
sorry

end arithmetic_sequence_condition_l1506_150636


namespace triangle_altitude_and_median_l1506_150696

/-- Triangle with vertices A(4,0), B(6,7), and C(0,3) -/
structure Triangle where
  A : ℝ × ℝ := (4, 0)
  B : ℝ × ℝ := (6, 7)
  C : ℝ × ℝ := (0, 3)

/-- Equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The altitude from A to BC -/
def altitude (t : Triangle) : LineEquation :=
  { a := 2, b := 7, c := -21 }

/-- The median from BC -/
def median (t : Triangle) : LineEquation :=
  { a := 5, b := 1, c := -20 }

theorem triangle_altitude_and_median (t : Triangle) :
  (altitude t = { a := 2, b := 7, c := -21 }) ∧
  (median t = { a := 5, b := 1, c := -20 }) := by
  sorry

end triangle_altitude_and_median_l1506_150696


namespace min_sum_squares_l1506_150632

theorem min_sum_squares (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3*x*y*z = 8) :
  ∃ (m : ℝ), (∀ a b c : ℝ, a^3 + b^3 + c^3 - 3*a*b*c = 8 → a^2 + b^2 + c^2 ≥ m) ∧
             (x^2 + y^2 + z^2 ≥ m) ∧
             m = 3 :=
sorry

end min_sum_squares_l1506_150632


namespace exponent_expression_equality_l1506_150657

theorem exponent_expression_equality : 3^(1^(0^8)) + ((3^1)^0)^8 = 4 := by
  sorry

end exponent_expression_equality_l1506_150657


namespace coin_collection_value_l1506_150635

theorem coin_collection_value (total_coins : ℕ) (two_dollar_coins : ℕ) : 
  total_coins = 275 →
  two_dollar_coins = 148 →
  (total_coins - two_dollar_coins) * 1 + two_dollar_coins * 2 = 423 := by
sorry

end coin_collection_value_l1506_150635


namespace part_I_part_II_l1506_150682

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

-- Define set N with parameter a
def N (a : ℝ) : Set ℝ := {x | a + 1 ≤ x ∧ x ≤ 2*a + 1}

-- Part I
theorem part_I : 
  M ∩ (U \ N 2) = {x : ℝ | -2 ≤ x ∧ x < 3} := by sorry

-- Part II
theorem part_II :
  ∀ a : ℝ, M ∪ N a = M → a ≤ 2 := by sorry

end part_I_part_II_l1506_150682


namespace function_property_center_of_symmetry_range_property_l1506_150674

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + 1 - a) / (a - x)

theorem function_property (a : ℝ) (x : ℝ) (h : x ≠ a) :
  f a x + f a (2 * a - x) + 2 = 0 := by sorry

theorem center_of_symmetry (a b : ℝ) 
  (h : ∀ x ≠ a, f a x + f a (6 - x) = 2 * b) :
  a + b = -4/7 := by sorry

theorem range_property (a : ℝ) :
  (∀ x ∈ Set.Icc (a + 1/2) (a + 1), f a x ∈ Set.Icc (-3) (-2)) ∧
  (∀ y ∈ Set.Icc (-3) (-2), ∃ x ∈ Set.Icc (a + 1/2) (a + 1), f a x = y) := by sorry

end function_property_center_of_symmetry_range_property_l1506_150674


namespace coefficient_a3_value_l1506_150639

/-- Given a polynomial expansion and sum of coefficients condition, prove a₃ = -5 -/
theorem coefficient_a3_value (a : ℝ) (a₀ a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ) :
  (∀ x, (1 + x) * (a - x)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  (a₀ + a₁ + a₂ + a₃ + a₄ + a₅ + a₆ + a₇ = 0) →
  a₃ = -5 := by sorry

end coefficient_a3_value_l1506_150639


namespace cubic_sum_l1506_150622

theorem cubic_sum (x y : ℝ) (h1 : 1/x + 1/y = 4) (h2 : x + y + x*y = 3) :
  x^3 + y^3 = 1188/125 := by sorry

end cubic_sum_l1506_150622


namespace rainfall_difference_l1506_150668

theorem rainfall_difference (sunday monday tuesday : ℝ) : 
  sunday = 4 ∧ 
  monday > sunday ∧ 
  tuesday = 2 * monday ∧ 
  sunday + monday + tuesday = 25 →
  monday - sunday = 3 := by
sorry

end rainfall_difference_l1506_150668


namespace no_solution_l1506_150652

def equation1 (x₁ x₂ x₃ : ℝ) : Prop := 2 * x₁ + 5 * x₂ - 4 * x₃ = 8
def equation2 (x₁ x₂ x₃ : ℝ) : Prop := 3 * x₁ + 15 * x₂ - 9 * x₃ = 5
def equation3 (x₁ x₂ x₃ : ℝ) : Prop := 5 * x₁ + 5 * x₂ - 7 * x₃ = 1

theorem no_solution : ¬∃ x₁ x₂ x₃ : ℝ, equation1 x₁ x₂ x₃ ∧ equation2 x₁ x₂ x₃ ∧ equation3 x₁ x₂ x₃ := by
  sorry

end no_solution_l1506_150652
