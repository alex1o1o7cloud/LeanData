import Mathlib

namespace greatest_divisor_of_sequence_l1396_139618

theorem greatest_divisor_of_sequence :
  ∃ (x : ℕ), x = 18 ∧ 
  (∀ y : ℕ, x ∣ (7^y + 12*y - 1)) ∧
  (∀ z : ℕ, (∀ y : ℕ, z ∣ (7^y + 12*y - 1)) → z ≤ x) :=
by sorry

end greatest_divisor_of_sequence_l1396_139618


namespace inequality_proof_l1396_139639

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  (a^2 + 1) * (b^3 + 2) * (c^6 + 5) ≥ 36 := by
  sorry

end inequality_proof_l1396_139639


namespace quadratic_discriminant_l1396_139603

theorem quadratic_discriminant (a b c : ℝ) (h : a ≠ 0) :
  (∃! x, a * x^2 + b * x + c = x - 2) ∧
  (∃! x, a * x^2 + b * x + c = 1 - x / 2) →
  b^2 - 4*a*c = -1/2 := by
sorry

end quadratic_discriminant_l1396_139603


namespace calculate_example_not_commutative_l1396_139620

-- Define the new operation
def otimes (a b : ℤ) : ℤ := a * b + a - b

-- Theorem 1: Calculate ((-2) ⊗ 5) ⊗ 6
theorem calculate_example : otimes (otimes (-2) 5) 6 = -125 := by
  sorry

-- Theorem 2: The operation is not commutative
theorem not_commutative : ∃ a b : ℤ, otimes a b ≠ otimes b a := by
  sorry

end calculate_example_not_commutative_l1396_139620


namespace charlie_seashells_l1396_139677

theorem charlie_seashells (c e : ℕ) : 
  c = e + 10 →  -- Charlie collected 10 more seashells than Emily
  e = c / 3 →   -- Emily collected one-third the number of seashells Charlie collected
  c = 15 :=     -- Charlie collected 15 seashells
by sorry

end charlie_seashells_l1396_139677


namespace bridge_length_bridge_length_calculation_l1396_139661

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed : ℝ) (crossing_time : ℝ) : ℝ :=
  let total_distance := train_speed * crossing_time / 3600 * 1000
  total_distance - train_length

/-- Proof of the bridge length calculation -/
theorem bridge_length_calculation : 
  bridge_length 110 45 30 = 265 := by
  sorry

end bridge_length_bridge_length_calculation_l1396_139661


namespace apple_distribution_l1396_139635

/-- Represents the number of apples each person receives when evenly distributing
    a given number of apples among a given number of people. -/
def apples_per_person (total_apples : ℕ) (num_people : ℕ) : ℕ :=
  total_apples / num_people

/-- Theorem stating that when 15 apples are evenly distributed among 3 people,
    each person receives 5 apples. -/
theorem apple_distribution :
  apples_per_person 15 3 = 5 := by
  sorry

end apple_distribution_l1396_139635


namespace min_distance_squared_l1396_139682

theorem min_distance_squared (a b c d : ℝ) : 
  b = a - 2 * Real.exp a → 
  c + d = 4 → 
  ∃ (min : ℝ), min = 18 ∧ ∀ (x y : ℝ), (a - x)^2 + (b - y)^2 ≥ min :=
sorry

end min_distance_squared_l1396_139682


namespace stratified_sampling_male_count_l1396_139605

theorem stratified_sampling_male_count 
  (total_students : ℕ) 
  (male_students : ℕ) 
  (female_students : ℕ) 
  (sample_size : ℕ) :
  total_students = male_students + female_students →
  total_students = 700 →
  male_students = 400 →
  female_students = 300 →
  sample_size = 35 →
  (male_students * sample_size) / total_students = 20 :=
by sorry

end stratified_sampling_male_count_l1396_139605


namespace prob_different_colors_value_l1396_139650

def blue_chips : ℕ := 6
def red_chips : ℕ := 5
def yellow_chips : ℕ := 4
def green_chips : ℕ := 3

def total_chips : ℕ := blue_chips + red_chips + yellow_chips + green_chips

def prob_different_colors : ℚ :=
  let prob_blue : ℚ := blue_chips / total_chips
  let prob_red : ℚ := red_chips / total_chips
  let prob_yellow : ℚ := yellow_chips / total_chips
  let prob_green : ℚ := green_chips / total_chips
  
  let prob_not_blue : ℚ := 1 - prob_blue
  let prob_not_red : ℚ := 1 - prob_red
  let prob_not_yellow : ℚ := 1 - prob_yellow
  let prob_not_green : ℚ := 1 - prob_green
  
  prob_blue * prob_not_blue +
  prob_red * prob_not_red +
  prob_yellow * prob_not_yellow +
  prob_green * prob_not_green

theorem prob_different_colors_value :
  prob_different_colors = 119 / 162 :=
sorry

end prob_different_colors_value_l1396_139650


namespace equation_solutions_l1396_139692

theorem equation_solutions : 
  let f (x : ℝ) := (x - 2) * (x - 3) * (x - 5) * (x - 6) * (x - 3) * (x - 5) * (x - 2)
  let g (x : ℝ) := (x - 3) * (x - 5) * (x - 3)
  let h (x : ℝ) := f x / g x
  ∀ x : ℝ, h x = 1 ↔ x = (-3 + Real.sqrt 5) / 2 ∨ x = (-3 - Real.sqrt 5) / 2 :=
by
  sorry


end equation_solutions_l1396_139692


namespace ellipse_triangle_area_l1396_139644

/-- The ellipse with equation 4x^2/49 + y^2/6 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (4 * p.1^2) / 49 + p.2^2 / 6 = 1}

/-- The foci of the ellipse -/
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

/-- A point P on the ellipse satisfying the given ratio condition -/
def P : ℝ × ℝ := sorry

/-- The distance between two points -/
def distance (p q : ℝ × ℝ) : ℝ := sorry

/-- The area of a triangle given its three vertices -/
def triangleArea (p q r : ℝ × ℝ) : ℝ := sorry

theorem ellipse_triangle_area :
  P ∈ Ellipse ∧ 
  distance P F1 / distance P F2 = 4/3 →
  triangleArea P F1 F2 = 6 := by sorry

end ellipse_triangle_area_l1396_139644


namespace percentage_loss_l1396_139611

theorem percentage_loss (cost_price selling_price : ℚ) : 
  cost_price = 2300 →
  selling_price = 1610 →
  (cost_price - selling_price) / cost_price * 100 = 30 := by
sorry

end percentage_loss_l1396_139611


namespace right_angle_on_circle_l1396_139653

/-- The circle C with equation (x - √3)² + (y - 1)² = 1 -/
def circle_C (x y : ℝ) : Prop := (x - Real.sqrt 3)^2 + (y - 1)^2 = 1

/-- The point A with coordinates (-t, 0) -/
def point_A (t : ℝ) : ℝ × ℝ := (-t, 0)

/-- The point B with coordinates (t, 0) -/
def point_B (t : ℝ) : ℝ × ℝ := (t, 0)

/-- Predicate to check if a point P forms a right angle with A and B -/
def forms_right_angle (P : ℝ × ℝ) (t : ℝ) : Prop :=
  let A := point_A t
  let B := point_B t
  (P.1 - A.1) * (P.1 - B.1) + (P.2 - A.2) * (P.2 - B.2) = 0

theorem right_angle_on_circle (t : ℝ) :
  t > 0 →
  (∃ P : ℝ × ℝ, circle_C P.1 P.2 ∧ forms_right_angle P t) →
  t ∈ Set.Icc 1 3 :=
sorry

end right_angle_on_circle_l1396_139653


namespace negation_of_existence_square_positive_negation_l1396_139649

theorem negation_of_existence (p : ℝ → Prop) :
  (¬ ∃ x > 0, p x) ↔ (∀ x > 0, ¬ p x) := by sorry

theorem square_positive_negation :
  (¬ ∃ x > 0, x^2 > 0) ↔ (∀ x > 0, x^2 ≤ 0) := by sorry

end negation_of_existence_square_positive_negation_l1396_139649


namespace no_natural_solution_for_equation_l1396_139688

theorem no_natural_solution_for_equation : ¬∃ (a b : ℕ), a^2 - 3*b^2 = 8 := by
  sorry

end no_natural_solution_for_equation_l1396_139688


namespace coin_jar_theorem_l1396_139671

/-- Represents the number of coins added or removed in each hour. 
    Positive numbers represent additions, negative numbers represent removals. -/
def coin_changes : List Int := [20, 30, 30, 40, -20, 50, 60, -15, 70, -25]

/-- The total number of hours -/
def total_hours : Nat := 10

/-- Calculates the final number of coins in the jar -/
def final_coin_count (changes : List Int) : Int :=
  changes.sum

/-- Theorem stating that the final number of coins in the jar is 240 -/
theorem coin_jar_theorem : 
  final_coin_count coin_changes = 240 := by
  sorry

end coin_jar_theorem_l1396_139671


namespace product_expansion_l1396_139689

theorem product_expansion (x : ℝ) : (3 * x + 4) * (2 * x^2 + 3 * x + 6) = 6 * x^3 + 17 * x^2 + 30 * x + 24 := by
  sorry

end product_expansion_l1396_139689


namespace star_calculation_l1396_139658

-- Define the * operation
def star (a b : ℚ) : ℚ := (a + 2*b) / 3

-- State the theorem
theorem star_calculation : star (star 4 6) 9 = 70 / 9 := by
  sorry

end star_calculation_l1396_139658


namespace watch_cost_price_l1396_139696

/-- The cost price of a watch satisfying certain selling conditions -/
theorem watch_cost_price : ∃ (cp : ℝ), 
  (cp * 0.79 = cp * 1.04 - 140) ∧ 
  cp = 560 := by
  sorry

end watch_cost_price_l1396_139696


namespace inequality_proof_l1396_139632

theorem inequality_proof (a b c : ℝ) (h1 : a ≥ b) (h2 : b ≥ c) (h3 : c > 0) :
  (a^2 - b^2)/c + (c^2 - b^2)/a + (a^2 - c^2)/b ≥ 3*a - 4*b + c :=
sorry

end inequality_proof_l1396_139632


namespace lola_poptarts_count_l1396_139642

/-- The number of mini cupcakes Lola baked -/
def lola_cupcakes : ℕ := 13

/-- The number of blueberry pies Lola baked -/
def lola_pies : ℕ := 8

/-- The number of mini cupcakes Lulu made -/
def lulu_cupcakes : ℕ := 16

/-- The number of pop tarts Lulu made -/
def lulu_poptarts : ℕ := 12

/-- The number of blueberry pies Lulu made -/
def lulu_pies : ℕ := 14

/-- The total number of pastries made by Lola and Lulu -/
def total_pastries : ℕ := 73

/-- The number of pop tarts Lola baked -/
def lola_poptarts : ℕ := 10

theorem lola_poptarts_count :
  lola_cupcakes + lola_poptarts + lola_pies + lulu_cupcakes + lulu_poptarts + lulu_pies = total_pastries :=
by sorry

end lola_poptarts_count_l1396_139642


namespace bob_questions_proof_l1396_139699

def question_rate (hour : Nat) : Nat :=
  match hour with
  | 0 => 13
  | n + 1 => 2 * question_rate n

def total_questions (hours : Nat) : Nat :=
  match hours with
  | 0 => 0
  | n + 1 => question_rate n + total_questions n

theorem bob_questions_proof :
  total_questions 3 = 91 :=
by sorry

end bob_questions_proof_l1396_139699


namespace circle_diameter_from_area_l1396_139655

theorem circle_diameter_from_area (A : ℝ) (d : ℝ) :
  A = 25 * Real.pi → d = (2 * (A / Real.pi).sqrt) → d = 10 := by
  sorry

end circle_diameter_from_area_l1396_139655


namespace exam_mean_score_l1396_139668

/-- Given an exam where a score of 58 is 2 standard deviations below the mean
    and a score of 98 is 3 standard deviations above the mean,
    prove that the mean score is 74. -/
theorem exam_mean_score (mean std_dev : ℝ) 
    (h1 : 58 = mean - 2 * std_dev)
    (h2 : 98 = mean + 3 * std_dev) : 
  mean = 74 := by
  sorry

end exam_mean_score_l1396_139668


namespace problem1_l1396_139624

theorem problem1 (x y : ℝ) (h : y ≠ 0) :
  ((x + 3 * y) * (x - 3 * y) - x^2) / (9 * y) = -y := by sorry

end problem1_l1396_139624


namespace max_length_sum_l1396_139656

/-- The length of a positive integer is the number of prime factors (not necessarily distinct) in its prime factorization. -/
def length (n : ℕ) : ℕ := sorry

/-- The maximum sum of lengths of x and y given the constraints. -/
theorem max_length_sum : 
  ∃ (x y : ℕ), 
    x > 1 ∧ 
    y > 1 ∧ 
    x + 3*y < 940 ∧ 
    ∀ (a b : ℕ), a > 1 → b > 1 → a + 3*b < 940 → length x + length y ≥ length a + length b ∧
    length x + length y = 15 :=
sorry

end max_length_sum_l1396_139656


namespace arithmetic_calculations_l1396_139610

theorem arithmetic_calculations : 
  ((62 + 38) / 4 = 25) ∧ 
  ((34 + 19) * 7 = 371) ∧ 
  (1500 - 125 * 8 = 500) := by
sorry

end arithmetic_calculations_l1396_139610


namespace extreme_points_sum_condition_l1396_139621

open Real

noncomputable def f (a x : ℝ) : ℝ := 1/2 * x^2 + a * log x - (a + 1) * x

noncomputable def F (a x : ℝ) : ℝ := f a x + (a - 1) * x

theorem extreme_points_sum_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x > 0 → F a x ≤ max (F a x₁) (F a x₂)) ∧
    F a x₁ + F a x₂ > -2/exp 1 - 2) →
  0 < a ∧ a < 1/exp 1 :=
sorry

end extreme_points_sum_condition_l1396_139621


namespace circle_under_translation_l1396_139613

/-- A parallel translation in a 2D plane. -/
structure ParallelTranslation where
  shift : ℝ × ℝ

/-- A circle in a 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The result of applying a parallel translation to a circle. -/
def translateCircle (c : Circle) (t : ParallelTranslation) : Circle :=
  { center := (c.center.1 + t.shift.1, c.center.2 + t.shift.2),
    radius := c.radius }

/-- Theorem: A circle remains a circle under parallel translation. -/
theorem circle_under_translation (c : Circle) (t : ParallelTranslation) :
  ∃ (c' : Circle), c' = translateCircle c t ∧ c'.radius = c.radius :=
by sorry

end circle_under_translation_l1396_139613


namespace triangle_abc_solutions_l1396_139674

theorem triangle_abc_solutions (a b : ℝ) (B : ℝ) :
  a = Real.sqrt 3 →
  b = Real.sqrt 2 →
  B = 45 * π / 180 →
  ∃ (A C c : ℝ),
    ((A = 60 * π / 180 ∧ C = 75 * π / 180 ∧ c = (Real.sqrt 6 + Real.sqrt 2) / 2) ∨
     (A = 120 * π / 180 ∧ C = 15 * π / 180 ∧ c = (Real.sqrt 6 - Real.sqrt 2) / 2)) ∧
    A + B + C = π ∧
    Real.sin A / a = Real.sin B / b ∧
    Real.sin C / c = Real.sin B / b :=
by sorry

end triangle_abc_solutions_l1396_139674


namespace sqrt_four_minus_2023_power_zero_equals_one_l1396_139645

theorem sqrt_four_minus_2023_power_zero_equals_one :
  Real.sqrt 4 - (2023 : ℝ) ^ 0 = 1 := by
  sorry

end sqrt_four_minus_2023_power_zero_equals_one_l1396_139645


namespace dividend_calculation_l1396_139607

theorem dividend_calculation (q d r x : ℝ) 
  (hq : q = -427.86)
  (hd : d = 52.7)
  (hr : r = -14.5)
  (hx : x = q * d + r) : 
  x = -22571.002 := by
sorry

end dividend_calculation_l1396_139607


namespace piggy_bank_theorem_specific_piggy_bank_case_l1396_139626

/-- Represents a configuration of piggy banks and their keys -/
structure PiggyBankConfig (n : ℕ) where
  keys : Fin n → Fin n
  injective : Function.Injective keys

/-- The probability of opening all remaining piggy banks given n total and k broken -/
def openProbability (n k : ℕ) : ℚ :=
  if k ≤ n then k / n else 0

theorem piggy_bank_theorem (n k : ℕ) (h : k ≤ n) :
  openProbability n k = k / n := by sorry

theorem specific_piggy_bank_case :
  openProbability 30 2 = 1 / 15 := by sorry

end piggy_bank_theorem_specific_piggy_bank_case_l1396_139626


namespace three_digit_congruence_solutions_l1396_139608

theorem three_digit_congruence_solutions : 
  let count := Finset.filter (fun x => 100 ≤ x ∧ x ≤ 999 ∧ (4573 * x + 502) % 23 = 1307 % 23) (Finset.range 1000)
  Finset.card count = 39 := by
  sorry

end three_digit_congruence_solutions_l1396_139608


namespace distance_from_T_to_S_l1396_139641

theorem distance_from_T_to_S (P Q : ℝ) : 
  let S := P + (3/4) * (Q - P)
  let T := P + (1/3) * (Q - P)
  S - T = 25 := by
sorry

end distance_from_T_to_S_l1396_139641


namespace tan_675_degrees_l1396_139646

theorem tan_675_degrees (m : ℤ) : 
  -180 < m ∧ m < 180 ∧ Real.tan (↑m * π / 180) = Real.tan (675 * π / 180) → m = 135 :=
by sorry

end tan_675_degrees_l1396_139646


namespace ellipse_properties_l1396_139684

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2/2 + y^2 = 1

-- Define the right focus
def right_focus : ℝ × ℝ := (1, 0)

-- Define the line l with slope m passing through the right focus
def line_l (m : ℝ) (x y : ℝ) : Prop := y = m * (x - 1)

-- Define the area of a triangle given three points
def triangle_area (A B P : ℝ × ℝ) : ℝ := sorry

-- Define the area of the incircle of a triangle
def incircle_area (A B F : ℝ × ℝ) : ℝ := sorry

theorem ellipse_properties :
  ∃ (P₁ P₂ : ℝ × ℝ),
    ellipse P₁.1 P₁.2 ∧ 
    ellipse P₂.1 P₂.2 ∧
    P₁ ≠ P₂ ∧
    (∀ (A B : ℝ × ℝ), 
      ellipse A.1 A.2 ∧ 
      ellipse B.1 B.2 ∧ 
      line_l 1 A.1 A.2 ∧ 
      line_l 1 B.1 B.2 →
      triangle_area A B P₁ = (2 * Real.sqrt 5 - 2) / 3 ∧
      triangle_area A B P₂ = (2 * Real.sqrt 5 - 2) / 3) ∧
    (∀ (P : ℝ × ℝ),
      ellipse P.1 P.2 ∧ 
      P ≠ P₁ ∧ 
      P ≠ P₂ →
      triangle_area A B P ≠ (2 * Real.sqrt 5 - 2) / 3) ∧
    (∃ (A B : ℝ × ℝ) (m : ℝ),
      ellipse A.1 A.2 ∧
      ellipse B.1 B.2 ∧
      line_l m A.1 A.2 ∧
      line_l m B.1 B.2 ∧
      incircle_area A B (-1, 0) = π / 8 ∧
      (∀ (C D : ℝ × ℝ) (n : ℝ),
        ellipse C.1 C.2 ∧
        ellipse D.1 D.2 ∧
        line_l n C.1 C.2 ∧
        line_l n D.1 D.2 →
        incircle_area C D (-1, 0) ≤ π / 8)) :=
sorry

end ellipse_properties_l1396_139684


namespace triangle_properties_l1396_139637

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the conditions
def altitude_BC (x y : ℝ) : Prop := x - 2*y - 1 = 0
def angle_bisector_A (y : ℝ) : Prop := y = 0
def point_B : ℝ × ℝ := (2, 1)

-- Define the theorem
theorem triangle_properties (ABC : Triangle) :
  altitude_BC ABC.A.1 ABC.A.2 ∧
  altitude_BC ABC.C.1 ABC.C.2 ∧
  angle_bisector_A ABC.A.2 ∧
  ABC.B = point_B →
  ABC.A = (1, 0) ∧
  ABC.C = (4, -3) ∧
  ∀ (x y : ℝ), y = x - 1 ↔ (x = ABC.A.1 ∧ y = ABC.A.2) ∨ (x = ABC.C.1 ∧ y = ABC.C.2) :=
by sorry


end triangle_properties_l1396_139637


namespace symmetric_points_sum_l1396_139601

/-- Two points P and Q in the Cartesian plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Symmetry about the x-axis -/
def symmetricAboutXAxis (p q : Point) : Prop :=
  p.x = q.x ∧ p.y = -q.y

/-- The theorem to be proved -/
theorem symmetric_points_sum (a b : ℝ) :
  let p : Point := ⟨a, 1⟩
  let q : Point := ⟨2, b⟩
  symmetricAboutXAxis p q → a + b = 1 := by
  sorry

end symmetric_points_sum_l1396_139601


namespace min_value_product_l1396_139670

theorem min_value_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 2) :
  (x + y) * (y + 3 * z) * (2 * x * z + 1) ≥ 16 * Real.sqrt 6 := by
  sorry

end min_value_product_l1396_139670


namespace card_arrangement_unique_l1396_139615

def CardArrangement (arrangement : List Nat) : Prop :=
  arrangement.length = 9 ∧
  arrangement.toFinset = Finset.range 9 ∧
  ∀ i, i + 2 < arrangement.length →
    ¬(arrangement[i]! < arrangement[i+1]! ∧ arrangement[i+1]! < arrangement[i+2]!) ∧
    ¬(arrangement[i]! > arrangement[i+1]! ∧ arrangement[i+1]! > arrangement[i+2]!)

theorem card_arrangement_unique :
  ∀ arrangement : List Nat,
    CardArrangement arrangement →
    arrangement[3]! = 5 ∧
    arrangement[5]! = 2 ∧
    arrangement[8]! = 9 :=
by sorry

end card_arrangement_unique_l1396_139615


namespace y_coordinate_relationship_l1396_139680

/-- A parabola defined by y = 2(x+1)² + c -/
structure Parabola where
  c : ℝ

/-- A point on a parabola -/
structure PointOnParabola (p : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y = 2 * (x + 1)^2 + p.c

/-- Theorem stating the relationship between y-coordinates of three points on the parabola -/
theorem y_coordinate_relationship (p : Parabola) 
  (A : PointOnParabola p) (B : PointOnParabola p) (C : PointOnParabola p)
  (hA : A.x = -2) (hB : B.x = 1) (hC : C.x = 2) :
  C.y > B.y ∧ B.y > A.y := by
  sorry

end y_coordinate_relationship_l1396_139680


namespace children_with_vip_seats_l1396_139669

/-- Proves the number of children with VIP seats in a concert hall -/
theorem children_with_vip_seats
  (total_attendees : ℕ)
  (children_percentage : ℚ)
  (vip_children_percentage : ℚ)
  (h1 : total_attendees = 400)
  (h2 : children_percentage = 75 / 100)
  (h3 : vip_children_percentage = 20 / 100) :
  ⌊(total_attendees : ℚ) * children_percentage * vip_children_percentage⌋ = 60 := by
  sorry

#check children_with_vip_seats

end children_with_vip_seats_l1396_139669


namespace stripe_area_on_cylinder_l1396_139634

/-- The area of a stripe on a cylindrical tank -/
theorem stripe_area_on_cylinder (d h w r : ℝ) (h1 : d = 20) (h2 : w = 4) (h3 : r = d / 2) :
  3 * (2 * π * r) * w = 240 * π := by
  sorry

end stripe_area_on_cylinder_l1396_139634


namespace short_story_pages_approx_l1396_139691

/-- Calculates the number of pages in each short story --/
def pages_per_short_story (stories_per_week : ℕ) (weeks : ℕ) (reams : ℕ) 
  (sheets_per_ream : ℕ) (pages_per_sheet : ℕ) : ℚ :=
  let total_sheets := reams * sheets_per_ream
  let total_pages := total_sheets * pages_per_sheet
  let total_stories := stories_per_week * weeks
  (total_pages : ℚ) / total_stories

theorem short_story_pages_approx : 
  let result := pages_per_short_story 3 12 3 500 2
  ∃ ε > 0, |result - 83.33| < ε := by
  sorry

end short_story_pages_approx_l1396_139691


namespace born_day_300_years_before_l1396_139640

/-- Represents days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Calculates the day of the week 300 years before a given Monday -/
def dayOfWeek300YearsBefore (endDay : DayOfWeek) : DayOfWeek :=
  match endDay with
  | DayOfWeek.Monday => DayOfWeek.Wednesday
  | _ => DayOfWeek.Monday  -- This case should never occur in our problem

/-- Theorem stating that 300 years before a Monday is a Wednesday -/
theorem born_day_300_years_before (endDay : DayOfWeek) 
  (h : endDay = DayOfWeek.Monday) : 
  dayOfWeek300YearsBefore endDay = DayOfWeek.Wednesday :=
by sorry

#check born_day_300_years_before

end born_day_300_years_before_l1396_139640


namespace hyperbola_asymptotes_l1396_139687

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 2 - y^2 / 4 = -1

-- Define the asymptote equation
def asymptote (x y : ℝ) : Prop := y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x

-- Theorem statement
theorem hyperbola_asymptotes :
  ∀ x y : ℝ, (∃ ε > 0, ∀ x' y' : ℝ, hyperbola x' y' ∧ x'^2 + y'^2 > 1/ε^2 →
    |y' - (Real.sqrt 2 * x')| < ε ∨ |y' - (-Real.sqrt 2 * x')| < ε) :=
sorry

end hyperbola_asymptotes_l1396_139687


namespace negation_of_proposition_l1396_139686

theorem negation_of_proposition (P : ℕ → Prop) :
  (∀ m : ℕ, 4^m ≥ 4*m) ↔ ¬(∃ m : ℕ, 4^m < 4*m) :=
by sorry

end negation_of_proposition_l1396_139686


namespace happy_valley_kennel_arrangement_l1396_139609

def num_chickens : Nat := 5
def num_dogs : Nat := 2
def num_cats : Nat := 5
def num_rabbits : Nat := 3
def total_animals : Nat := num_chickens + num_dogs + num_cats + num_rabbits

def animal_types : Nat := 4

theorem happy_valley_kennel_arrangement :
  (animal_types.factorial * num_chickens.factorial * num_dogs.factorial * 
   num_cats.factorial * num_rabbits.factorial) = 4147200 := by
  sorry

end happy_valley_kennel_arrangement_l1396_139609


namespace derivative_symmetry_l1396_139676

/-- Given a function f(x) = ax^4 + bx^2 + c where f'(1) = 2, prove that f'(-1) = -2 -/
theorem derivative_symmetry (a b c : ℝ) : 
  let f := fun (x : ℝ) => a * x^4 + b * x^2 + c
  (4 * a + 2 * b = 2) → 
  (fun (x : ℝ) => 4 * a * x^3 + 2 * b * x) (-1) = -2 := by
sorry

end derivative_symmetry_l1396_139676


namespace polynomial_factorization_l1396_139629

theorem polynomial_factorization (x m : ℝ) : 
  (x^2 + 6*x + 5 = (x+5)*(x+1)) ∧ (m^2 - m - 12 = (m+3)*(m-4)) := by
  sorry

end polynomial_factorization_l1396_139629


namespace geometric_sequence_max_value_l1396_139664

/-- A positive geometric sequence -/
def is_positive_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a n > 0 ∧ ∃ r > 0, ∀ k, a (k + 1) = r * a k

/-- The theorem statement -/
theorem geometric_sequence_max_value (a : ℕ → ℝ) 
  (h_geo : is_positive_geometric_sequence a)
  (h_condition : a 3 * a 6 + a 2 * a 7 = 2 * Real.exp 4) :
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ a 1 = x ∧ a 8 = y ∧ Real.log x * Real.log y ≤ 4) ∧
  (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ a 1 = x ∧ a 8 = y ∧ Real.log x * Real.log y = 4) :=
sorry

end geometric_sequence_max_value_l1396_139664


namespace negation_of_exists_greater_than_one_l1396_139638

theorem negation_of_exists_greater_than_one :
  (¬ ∃ x : ℝ, x > 1) ↔ (∀ x : ℝ, x ≤ 1) :=
by sorry

end negation_of_exists_greater_than_one_l1396_139638


namespace max_abs_sum_of_coeffs_l1396_139643

/-- A quadratic polynomial p(x) = ax^2 + bx + c -/
def QuadraticPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The condition that |p(x)| ≤ 1 for all x in [0,1] -/
def BoundedOnInterval (p : ℝ → ℝ) : Prop :=
  ∀ x, x ∈ Set.Icc 0 1 → |p x| ≤ 1

/-- The theorem stating that the maximum value of |a|+|b|+|c| is 4 -/
theorem max_abs_sum_of_coeffs (a b c : ℝ) :
  BoundedOnInterval (QuadraticPolynomial a b c) →
  |a| + |b| + |c| ≤ 4 :=
sorry

end max_abs_sum_of_coeffs_l1396_139643


namespace vector_statements_false_l1396_139663

open RealInnerProductSpace

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

def is_unit_vector (v : V) : Prop := ‖v‖ = 1

def parallel (v w : V) : Prop := ∃ (c : ℝ), v = c • w

theorem vector_statements_false (a₀ : V) (h : is_unit_vector a₀) :
  (∃ (a : V), a ≠ ‖a‖ • a₀) ∧
  (∃ (a : V), parallel a a₀ ∧ a ≠ ‖a‖ • a₀) ∧
  (∃ (a : V), parallel a a₀ ∧ is_unit_vector a ∧ a ≠ a₀) := by
  sorry

end vector_statements_false_l1396_139663


namespace striped_quadrilateral_area_l1396_139623

/-- Represents a quadrilateral cut from striped gift wrapping paper -/
structure StripedQuadrilateral where
  /-- The combined area of the grey stripes in the quadrilateral -/
  greyArea : ℝ
  /-- The stripes are equally wide -/
  equalStripes : Bool

/-- Theorem stating that if the grey stripes have an area of 10 in a quadrilateral
    cut from equally striped paper, then the total area of the quadrilateral is 20 -/
theorem striped_quadrilateral_area
  (quad : StripedQuadrilateral)
  (h1 : quad.greyArea = 10)
  (h2 : quad.equalStripes = true) :
  quad.greyArea * 2 = 20 := by
  sorry

#check striped_quadrilateral_area

end striped_quadrilateral_area_l1396_139623


namespace mcgillicuddy_kindergarten_count_l1396_139625

/-- Calculates the total number of students present in two kindergarten sessions -/
def total_students (morning_registered : ℕ) (morning_absent : ℕ) 
                   (afternoon_registered : ℕ) (afternoon_absent : ℕ) : ℕ :=
  (morning_registered - morning_absent) + (afternoon_registered - afternoon_absent)

/-- Theorem stating that the total number of students is 42 given the specified conditions -/
theorem mcgillicuddy_kindergarten_count : 
  total_students 25 3 24 4 = 42 := by
  sorry

#eval total_students 25 3 24 4

end mcgillicuddy_kindergarten_count_l1396_139625


namespace factorization_example_l1396_139606

/-- Represents factorization from left to right -/
def is_factorization_left_to_right (f : ℝ → ℝ) (g : ℝ → ℝ) (h : ℝ → ℝ) : Prop :=
  ∀ x, f x = g x * h x ∧ (∃ c, g x = c * x ∨ g x = x)

/-- The given equation represents factorization from left to right -/
theorem factorization_example :
  is_factorization_left_to_right
    (λ a : ℝ => 2 * a^2 + a)
    (λ a : ℝ => a)
    (λ a : ℝ => 2 * a + 1) := by
  sorry

end factorization_example_l1396_139606


namespace cos_30_minus_cos_60_l1396_139679

theorem cos_30_minus_cos_60 : Real.cos (30 * π / 180) - Real.cos (60 * π / 180) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end cos_30_minus_cos_60_l1396_139679


namespace shipping_percentage_above_50_l1396_139660

def flat_rate_shipping : Real := 5.00
def min_purchase_for_percentage : Real := 50.00

def shirt_price : Real := 12.00
def shirt_quantity : Nat := 3
def socks_price : Real := 5.00
def shorts_price : Real := 15.00
def shorts_quantity : Nat := 2
def swim_trunks_price : Real := 14.00

def total_purchase : Real := shirt_price * shirt_quantity + socks_price + shorts_price * shorts_quantity + swim_trunks_price
def total_bill : Real := 102.00

theorem shipping_percentage_above_50 :
  total_purchase > min_purchase_for_percentage →
  (total_bill - total_purchase) / total_purchase * 100 = 20 := by
sorry

end shipping_percentage_above_50_l1396_139660


namespace triangle_angle_sin_sum_bounds_triangle_angle_sin_sum_equality_condition_l1396_139657

open Real

/-- Triangle interior angles in radians -/
structure TriangleAngles where
  A : ℝ
  B : ℝ
  C : ℝ
  sum_eq_pi : A + B + C = π
  all_positive : 0 < A ∧ 0 < B ∧ 0 < C

theorem triangle_angle_sin_sum_bounds (t : TriangleAngles) :
  -2 < sin (3 * t.A) + sin (3 * t.B) + sin (3 * t.C) ∧
  sin (3 * t.A) + sin (3 * t.B) + sin (3 * t.C) ≤ 3/2 * Real.sqrt 3 :=
sorry

theorem triangle_angle_sin_sum_equality_condition (t : TriangleAngles) :
  sin (3 * t.A) + sin (3 * t.B) + sin (3 * t.C) = 3/2 * Real.sqrt 3 ↔
  t.A = 7*π/18 ∧ t.B = π/9 ∧ t.C = π/9 :=
sorry

end triangle_angle_sin_sum_bounds_triangle_angle_sin_sum_equality_condition_l1396_139657


namespace angle_in_third_quadrant_l1396_139698

def is_in_third_quadrant (α : ℝ) : Prop :=
  ∃ n : ℤ, 180 * (2 * n + 1) < α ∧ α < 180 * (2 * n + 1) + 90

theorem angle_in_third_quadrant (k : ℤ) (α : ℝ) 
  (h : (4 * k + 1) * 180 < α ∧ α < (4 * k + 1) * 180 + 60) : 
  is_in_third_quadrant α :=
sorry

end angle_in_third_quadrant_l1396_139698


namespace range_of_t_l1396_139628

theorem range_of_t (a b : ℝ) (h : a^2 + a*b + b^2 = 1) :
  let t := a^2 - a*b + b^2
  ∃ (x : ℝ), t = x ∧ 1/3 ≤ x ∧ x ≤ 3 ∧
  ∀ (y : ℝ), (∃ (a' b' : ℝ), a'^2 + a'*b' + b'^2 = 1 ∧ a'^2 - a'*b' + b'^2 = y) → 1/3 ≤ y ∧ y ≤ 3 :=
by sorry

end range_of_t_l1396_139628


namespace dream_car_gas_consumption_l1396_139627

/-- Calculates the total gas consumption for a car over two days -/
def total_gas_consumption (consumption_rate : ℝ) (miles_day1 : ℝ) (miles_day2 : ℝ) : ℝ :=
  consumption_rate * (miles_day1 + miles_day2)

/-- Proves that given the specified conditions, the total gas consumption is 4000 gallons -/
theorem dream_car_gas_consumption :
  let consumption_rate : ℝ := 4
  let miles_day1 : ℝ := 400
  let miles_day2 : ℝ := miles_day1 + 200
  total_gas_consumption consumption_rate miles_day1 miles_day2 = 4000 :=
by
  sorry

#eval total_gas_consumption 4 400 600

end dream_car_gas_consumption_l1396_139627


namespace base_6_addition_l1396_139681

/-- Addition of two numbers in base 6 -/
def add_base_6 (a b : ℕ) : ℕ :=
  sorry

/-- Conversion from base 10 to base 6 -/
def to_base_6 (n : ℕ) : ℕ :=
  sorry

/-- Conversion from base 6 to base 10 -/
def from_base_6 (n : ℕ) : ℕ :=
  sorry

theorem base_6_addition :
  add_base_6 (from_base_6 52301) (from_base_6 34122) = from_base_6 105032 :=
sorry

end base_6_addition_l1396_139681


namespace lexis_cement_is_10_l1396_139600

/-- The amount of cement (in tons) used for Lexi's street -/
def lexis_cement : ℝ := 15.1 - 5.1

/-- Theorem stating that the amount of cement used for Lexi's street is 10 tons -/
theorem lexis_cement_is_10 : lexis_cement = 10 := by
  sorry

end lexis_cement_is_10_l1396_139600


namespace intersection_constraint_l1396_139612

theorem intersection_constraint (a : ℝ) : 
  let A : Set ℝ := {a^2, a+1, -3}
  let B : Set ℝ := {a-3, 2*a-1, a^2+1}
  A ∩ B = {-3} → a = -1 ∨ a = 0 := by
sorry

end intersection_constraint_l1396_139612


namespace nathaniel_win_probability_l1396_139654

/-- A fair six-sided die -/
def FairDie : Type := Fin 6

/-- The game state -/
structure GameState :=
  (sum : ℕ)
  (currentPlayer : Bool)  -- true for Nathaniel, false for Obediah

/-- Check if a number is a multiple of 7 -/
def isMultipleOf7 (n : ℕ) : Bool :=
  n % 7 = 0

/-- The probability of Nathaniel winning the game -/
noncomputable def nathanielWinProbability : ℝ :=
  5 / 11

/-- Theorem: The probability of Nathaniel winning the game is 5/11 -/
theorem nathaniel_win_probability :
  nathanielWinProbability = 5 / 11 := by
  sorry

#check nathaniel_win_probability

end nathaniel_win_probability_l1396_139654


namespace smallest_five_digit_mod_9_4_l1396_139666

theorem smallest_five_digit_mod_9_4 : 
  ∀ n : ℕ, 
    10000 ≤ n ∧ n ≡ 4 [ZMOD 9] → 
    10003 ≤ n :=
by
  sorry

end smallest_five_digit_mod_9_4_l1396_139666


namespace window_installation_time_l1396_139631

theorem window_installation_time 
  (total_windows : ℕ) 
  (installed_windows : ℕ) 
  (remaining_time : ℕ) 
  (h1 : total_windows = 9)
  (h2 : installed_windows = 6)
  (h3 : remaining_time = 18)
  (h4 : installed_windows < total_windows) :
  (remaining_time : ℚ) / (total_windows - installed_windows : ℚ) = 6 := by
sorry

end window_installation_time_l1396_139631


namespace honda_red_percentage_l1396_139622

theorem honda_red_percentage (total_cars : ℕ) (honda_cars : ℕ) 
  (total_red_percentage : ℚ) (non_honda_red_percentage : ℚ) :
  total_cars = 9000 →
  honda_cars = 5000 →
  total_red_percentage = 60 / 100 →
  non_honda_red_percentage = 225 / 1000 →
  (honda_cars : ℚ) * (90 / 100) + 
    (total_cars - honda_cars : ℚ) * non_honda_red_percentage = 
    (total_cars : ℚ) * total_red_percentage :=
by sorry

end honda_red_percentage_l1396_139622


namespace no_real_roots_l1396_139630

theorem no_real_roots : ∀ x : ℝ, 4 * x^2 + 4 * x + (5/4) ≠ 0 := by
  sorry

end no_real_roots_l1396_139630


namespace puppies_per_cage_l1396_139619

/-- Given a pet store scenario with puppies and cages, calculate puppies per cage -/
theorem puppies_per_cage 
  (total_puppies : ℕ) 
  (sold_puppies : ℕ) 
  (num_cages : ℕ) 
  (h1 : total_puppies = 45)
  (h2 : sold_puppies = 39)
  (h3 : num_cages = 3)
  (h4 : sold_puppies < total_puppies) :
  (total_puppies - sold_puppies) / num_cages = 2 := by
sorry

end puppies_per_cage_l1396_139619


namespace incorrect_height_calculation_l1396_139695

/-- Proves that the incorrect height of a student is 151 cm given the conditions of the problem -/
theorem incorrect_height_calculation (n : ℕ) (initial_avg : ℝ) (actual_height : ℝ) (actual_avg : ℝ) :
  n = 30 ∧ 
  initial_avg = 175 ∧ 
  actual_height = 136 ∧ 
  actual_avg = 174.5 → 
  ∃ (incorrect_height : ℝ), 
    incorrect_height = 151 ∧ 
    n * initial_avg = (n - 1) * actual_avg + incorrect_height ∧
    n * actual_avg = (n - 1) * actual_avg + actual_height :=
by sorry

end incorrect_height_calculation_l1396_139695


namespace inequality_proof_l1396_139604

theorem inequality_proof (a b c d : ℝ) (h1 : b < a) (h2 : d < c) : a + c > b + d := by
  sorry

end inequality_proof_l1396_139604


namespace polynomial_simplification_l1396_139648

theorem polynomial_simplification (x : ℝ) : 
  (3*x - 4)*(x + 6) - (x + 3)*(3*x + 2) = 3*x - 30 := by
  sorry

end polynomial_simplification_l1396_139648


namespace boyGirlRatio_in_example_college_l1396_139683

/-- Represents the number of students in a college -/
structure College where
  total : ℕ
  girls : ℕ
  boys : ℕ
  total_eq : total = girls + boys

/-- The ratio of boys to girls in a college -/
def boyGirlRatio (c : College) : ℚ :=
  c.boys / c.girls

theorem boyGirlRatio_in_example_college :
  ∃ c : College, c.total = 600 ∧ c.girls = 200 ∧ boyGirlRatio c = 2 := by
  sorry

end boyGirlRatio_in_example_college_l1396_139683


namespace trig_identity_quadratic_equation_solution_l1396_139602

-- Part 1
theorem trig_identity : Real.cos (30 * π / 180) * Real.tan (60 * π / 180) - 2 * Real.sin (45 * π / 180) = 3/2 - Real.sqrt 2 := by
  sorry

-- Part 2
theorem quadratic_equation_solution (x : ℝ) : 3 * x^2 - 1 = -2 * x ↔ x = 1/3 ∨ x = -1 := by
  sorry

end trig_identity_quadratic_equation_solution_l1396_139602


namespace quadratic_function_inequality_l1396_139614

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : a > 0
  symmetry : ∀ x, a * x^2 + b * x + c = a * (2 - x)^2 + b * (2 - x) + c

/-- The theorem stating the relationship between f(2^x) and f(3^x) -/
theorem quadratic_function_inequality (f : QuadraticFunction) :
  ∀ x : ℝ, f.a * (3^x)^2 + f.b * (3^x) + f.c > f.a * (2^x)^2 + f.b * (2^x) + f.c :=
sorry

end quadratic_function_inequality_l1396_139614


namespace complex_imaginary_part_l1396_139652

theorem complex_imaginary_part (z : ℂ) : 
  z = -2 + I → Complex.im (z + z⁻¹) = 4/5 := by sorry

end complex_imaginary_part_l1396_139652


namespace supplementary_angles_ratio_l1396_139673

/-- Given two supplementary angles in a ratio of 5:3, the measure of the smaller angle is 67.5° -/
theorem supplementary_angles_ratio (angle1 angle2 : ℝ) : 
  angle1 + angle2 = 180 →  -- supplementary angles
  angle1 / angle2 = 5 / 3 →  -- ratio of 5:3
  min angle1 angle2 = 67.5 :=  -- smaller angle is 67.5°
by sorry

end supplementary_angles_ratio_l1396_139673


namespace quadratic_comparison_l1396_139647

/-- A quadratic function f(x) = x^2 - 2x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 - 2*x + m

theorem quadratic_comparison (m : ℝ) (y₁ y₂ : ℝ) 
  (h1 : f m (-1) = y₁)
  (h2 : f m 2 = y₂) :
  y₁ > y₂ := by
  sorry

end quadratic_comparison_l1396_139647


namespace expansion_terms_l1396_139636

-- Define the exponent
def n : ℕ := 2016

-- Define the function that represents the number of terms
def num_terms (n : ℕ) : ℕ :=
  4 * n + 1

-- Theorem statement
theorem expansion_terms : num_terms n = 4033 := by
  sorry

end expansion_terms_l1396_139636


namespace civil_servant_dispatch_l1396_139667

theorem civil_servant_dispatch (m n k : ℕ) (hm : m = 5) (hn : n = 4) (hk : k = 3) :
  (k.factorial * (Nat.choose (m + n) k - Nat.choose m k - Nat.choose n k)) = 420 :=
by sorry

end civil_servant_dispatch_l1396_139667


namespace false_statements_exist_l1396_139659

theorem false_statements_exist : ∃ (a b c d : ℝ),
  (a > b ∧ c ≠ 0 ∧ a * c ≤ b * c) ∧
  (a > b ∧ b > 0 ∧ c > d ∧ a * c ≤ b * d) ∧
  (∀ a b c : ℝ, a * c^2 > b * c^2 → a > b) ∧
  (∀ a b : ℝ, a < b ∧ b < 0 → a^2 > a * b ∧ a * b > b^2) :=
by sorry

end false_statements_exist_l1396_139659


namespace two_number_problem_l1396_139662

theorem two_number_problem (x y : ℝ) (h1 : x + y = 30) (h2 : 3 * y - 4 * x = 9) :
  |y - x| = 129 / 21 := by
sorry

end two_number_problem_l1396_139662


namespace function_inequality_l1396_139616

-- Define the functions f and g
variable (f g : ℝ → ℝ)

-- State the theorem
theorem function_inequality 
  (hf : Differentiable ℝ f) 
  (hg : Differentiable ℝ g)
  (h_deriv : ∀ x, deriv f x > deriv g x) : 
  f 2 + g 1 > f 1 + g 2 := by
  sorry

end function_inequality_l1396_139616


namespace parallel_planes_lines_relationship_l1396_139678

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the parallel relation for planes
variable (parallel : Plane → Plane → Prop)

-- Define the containment relation for lines in planes
variable (contained_in : Line → Plane → Prop)

-- Define the positional relationships between lines
variable (is_parallel : Line → Line → Prop)
variable (is_skew : Line → Line → Prop)

-- State the theorem
theorem parallel_planes_lines_relationship 
  (α β : Plane) (a b : Line) 
  (h1 : parallel α β) 
  (h2 : contained_in a α) 
  (h3 : contained_in b β) : 
  is_parallel a b ∨ is_skew a b :=
sorry

end parallel_planes_lines_relationship_l1396_139678


namespace distance_between_centers_l1396_139693

-- Define the circles
def circle_M (x y : ℝ) : Prop := x^2 + y^2 = 1
def circle_N (x y : ℝ) : Prop := x^2 + (y-2)^2 = 1

-- Define the centers of the circles
def center_M : ℝ × ℝ := (0, 0)
def center_N : ℝ × ℝ := (0, 2)

-- State the theorem
theorem distance_between_centers :
  let (x₁, y₁) := center_M
  let (x₂, y₂) := center_N
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 2 := by sorry

end distance_between_centers_l1396_139693


namespace equation_represents_pair_of_lines_l1396_139685

/-- The equation 9x^2 - 16y^2 = 0 represents a pair of straight lines -/
theorem equation_represents_pair_of_lines : 
  ∃ (m₁ m₂ : ℝ), ∀ (x y : ℝ), 9 * x^2 - 16 * y^2 = 0 ↔ (y = m₁ * x ∨ y = m₂ * x) :=
sorry

end equation_represents_pair_of_lines_l1396_139685


namespace x_value_proof_l1396_139617

theorem x_value_proof (x y : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_x_lt_y : x < y)
  (h_eq1 : Real.sqrt x + Real.sqrt y = 4)
  (h_eq2 : Real.sqrt (x + 2) + Real.sqrt (y + 2) = 5) :
  x = 49 / 36 := by
sorry

end x_value_proof_l1396_139617


namespace cylinder_volume_equality_l1396_139665

/-- Given two congruent cylinders with radius 10 inches and height 4 inches,
    where the radius of one cylinder and the height of the other are increased by x inches,
    prove that the only nonzero solution for equal volumes is x = 5. -/
theorem cylinder_volume_equality (x : ℝ) (hx : x ≠ 0) :
  π * (10 + x)^2 * 4 = π * 100 * (4 + x) → x = 5 := by
  sorry

end cylinder_volume_equality_l1396_139665


namespace calculator_display_l1396_139694

/-- The special key function -/
def f (x : ℚ) : ℚ := 1 / (1 - x)

/-- Applies the function n times to the initial value -/
def iterate_f (n : ℕ) (x : ℚ) : ℚ :=
  match n with
  | 0 => x
  | n + 1 => f (iterate_f n x)

theorem calculator_display : iterate_f 120 7 = 7 := by
  sorry

end calculator_display_l1396_139694


namespace cube_sum_geq_product_sum_l1396_139651

theorem cube_sum_geq_product_sum {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  a^3 + b^3 ≥ a^2*b + a*b^2 := by
  sorry

end cube_sum_geq_product_sum_l1396_139651


namespace triangle_properties_l1396_139633

theorem triangle_properties (A B C : ℝ) (a b c : ℝ) :
  A = π / 3 →
  (b + c) / (Real.sin B + Real.sin C) = 2 →
  a = Real.sqrt 3 ∧
  (a * b * Real.sin C / 2 = Real.sqrt 3 / 2 → a + b + c = 3 + Real.sqrt 3) :=
by sorry

end triangle_properties_l1396_139633


namespace vector_magnitude_l1396_139672

/-- Given two vectors a and b in ℝ², prove that if a = (1, -1) and a + b = (3, 1), 
    then the magnitude of b is 2√2. -/
theorem vector_magnitude (a b : ℝ × ℝ) : 
  a = (1, -1) → a + b = (3, 1) → ‖b‖ = 2 * Real.sqrt 2 := by
  sorry

end vector_magnitude_l1396_139672


namespace decagon_flip_impossible_l1396_139675

/-- Represents a point in the decagon configuration -/
structure Point where
  value : Int
  deriving Repr

/-- Represents the decagon configuration -/
structure DecagonConfig where
  points : List Point
  deriving Repr

/-- Represents an operation to flip signs -/
inductive FlipOperation
  | Side
  | Diagonal

/-- Applies a flip operation to the configuration -/
def applyFlip (config : DecagonConfig) (op : FlipOperation) : DecagonConfig :=
  sorry

/-- Checks if all points in the configuration are negative -/
def allNegative (config : DecagonConfig) : Bool :=
  sorry

/-- Theorem: It's impossible to make all points negative in a decagon configuration -/
theorem decagon_flip_impossible (initial : DecagonConfig) :
  ∀ (ops : List FlipOperation), ¬(allNegative (ops.foldl applyFlip initial)) :=
sorry

end decagon_flip_impossible_l1396_139675


namespace marble_difference_prove_marble_difference_l1396_139690

/-- The difference in marbles between Ed and Doug after a series of events -/
theorem marble_difference : ℤ → Prop :=
  fun initial_difference =>
    ∀ (doug_initial : ℤ) (doug_lost : ℤ) (susan_found : ℤ),
      initial_difference = 22 →
      doug_lost = 8 →
      susan_found = 5 →
      (doug_initial + initial_difference + susan_found) - (doug_initial - doug_lost) = 35

/-- Proof of the marble difference theorem -/
theorem prove_marble_difference : marble_difference 22 := by
  sorry

end marble_difference_prove_marble_difference_l1396_139690


namespace sweater_cost_l1396_139697

def original_savings : ℚ := 80

def makeup_fraction : ℚ := 3/4

theorem sweater_cost :
  let makeup_cost : ℚ := makeup_fraction * original_savings
  let sweater_cost : ℚ := original_savings - makeup_cost
  sweater_cost = 20 := by sorry

end sweater_cost_l1396_139697
