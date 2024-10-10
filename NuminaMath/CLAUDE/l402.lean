import Mathlib

namespace valentines_theorem_l402_40243

/-- The number of Valentines Mrs. Franklin initially had -/
def initial_valentines : ℕ := 58

/-- The number of additional Valentines Mrs. Franklin needs -/
def additional_valentines : ℕ := 16

/-- The number of students Mrs. Franklin has -/
def num_students : ℕ := 74

/-- Theorem stating that the initial number of Valentines plus the additional Valentines
    equals the total number of students -/
theorem valentines_theorem :
  initial_valentines + additional_valentines = num_students :=
by sorry

end valentines_theorem_l402_40243


namespace sufficient_but_not_necessary_l402_40248

theorem sufficient_but_not_necessary (x : ℝ) : 
  (∀ x, 2 < x ∧ x < 4 → Real.log x < Real.exp 1) ∧
  (∃ x, Real.log x < Real.exp 1 ∧ ¬(2 < x ∧ x < 4)) := by
sorry

end sufficient_but_not_necessary_l402_40248


namespace circle_tangent_to_y_axis_l402_40298

def circle_equation (x y : ℝ) := (x - 2)^2 + (y - 1)^2 = 4

def is_tangent_to_y_axis (equation : ℝ → ℝ → Prop) : Prop :=
  ∃ y : ℝ, equation 0 y ∧ ∀ x y : ℝ, x ≠ 0 → equation x y → (x - 0)^2 + (y - y)^2 > 0

theorem circle_tangent_to_y_axis :
  is_tangent_to_y_axis circle_equation ∧
  ∀ x y : ℝ, circle_equation x y → (x - 2)^2 + (y - 1)^2 = 4 :=
sorry

end circle_tangent_to_y_axis_l402_40298


namespace football_game_score_l402_40200

theorem football_game_score (total_points winning_margin : ℕ) 
  (h1 : total_points = 34) 
  (h2 : winning_margin = 14) : 
  ∃ (panthers_score cougars_score : ℕ), 
    panthers_score + cougars_score = total_points ∧ 
    cougars_score = panthers_score + winning_margin ∧ 
    panthers_score = 10 := by
  sorry

end football_game_score_l402_40200


namespace abs_equal_abs_neg_l402_40220

theorem abs_equal_abs_neg (x : ℝ) : |x| = |-x| := by sorry

end abs_equal_abs_neg_l402_40220


namespace arun_weight_upper_limit_l402_40261

/-- The upper limit of Arun's weight according to his own opinion -/
def U : ℝ := sorry

/-- Arun's actual weight -/
def arun_weight : ℝ := sorry

/-- Arun's opinion: his weight is greater than 62 kg but less than U -/
axiom arun_opinion : 62 < arun_weight ∧ arun_weight < U

/-- Arun's brother's opinion: Arun's weight is greater than 60 kg but less than 70 kg -/
axiom brother_opinion : 60 < arun_weight ∧ arun_weight < 70

/-- Arun's mother's opinion: Arun's weight cannot be greater than 65 kg -/
axiom mother_opinion : arun_weight ≤ 65

/-- The average of different probable weights of Arun is 64 kg -/
axiom average_weight : (62 + U) / 2 = 64

theorem arun_weight_upper_limit : U = 65 := by sorry

end arun_weight_upper_limit_l402_40261


namespace cyclic_n_gon_characterization_l402_40253

/-- A convex n-gon is cyclic if and only if there exist real numbers a_i and b_i
    for each vertex P_i such that for any i < j, the distance P_i P_j = |a_i b_j - a_j b_i|. -/
theorem cyclic_n_gon_characterization {n : ℕ} (P : Fin n → ℝ × ℝ) :
  (∃ (center : ℝ × ℝ) (radius : ℝ), ∀ i : Fin n, dist center (P i) = radius) ↔
  (∃ (a b : Fin n → ℝ), ∀ (i j : Fin n), i < j →
    dist (P i) (P j) = |a i * b j - a j * b i|) :=
by sorry

end cyclic_n_gon_characterization_l402_40253


namespace no_four_digit_numbers_sum_10_div_9_l402_40228

theorem no_four_digit_numbers_sum_10_div_9 : 
  ¬∃ (n : ℕ), 
    1000 ≤ n ∧ n < 10000 ∧ 
    (∃ (a b c d : ℕ), n = 1000*a + 100*b + 10*c + d ∧ a + b + c + d = 10) ∧
    n % 9 = 0 := by
  sorry

end no_four_digit_numbers_sum_10_div_9_l402_40228


namespace octal_to_decimal_conversion_l402_40277

-- Define the octal number
def octal_age : ℕ := 536

-- Define the decimal equivalent
def decimal_age : ℕ := 350

-- Theorem to prove the equivalence
theorem octal_to_decimal_conversion :
  (5 * 8^2 + 3 * 8^1 + 6 * 8^0) = decimal_age :=
by sorry

end octal_to_decimal_conversion_l402_40277


namespace problem_statement_l402_40266

theorem problem_statement (a b c : ℝ) (h : a + b + c = 0) :
  (a = 0 ∧ b = 0 ∧ c = 0 ↔ a * b + b * c + a * c = 0) ∧
  (a * b * c = 1 ∧ a ≥ b ∧ b ≥ c → c ≤ -Real.rpow 4 (1/3) / 2) :=
by sorry

end problem_statement_l402_40266


namespace rectangles_equal_perimeter_different_shape_l402_40289

/-- Two rectangles with equal perimeters can have different shapes -/
theorem rectangles_equal_perimeter_different_shape :
  ∃ (l₁ w₁ l₂ w₂ : ℝ), 
    l₁ > 0 ∧ w₁ > 0 ∧ l₂ > 0 ∧ w₂ > 0 ∧
    2 * (l₁ + w₁) = 2 * (l₂ + w₂) ∧
    l₁ / w₁ ≠ l₂ / w₂ :=
by sorry

end rectangles_equal_perimeter_different_shape_l402_40289


namespace symmetric_trapezoid_construction_l402_40296

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a trapezoid
structure Trapezoid where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ

-- Define symmetry for a trapezoid
def isSymmetricTrapezoid (t : Trapezoid) : Prop :=
  -- Add conditions for symmetry here
  sorry

-- Define the construction function
def constructSymmetricTrapezoid (c : Circle) (sideLength : ℝ) : Trapezoid :=
  sorry

-- Theorem statement
theorem symmetric_trapezoid_construction
  (c : Circle) (sideLength : ℝ) :
  isSymmetricTrapezoid (constructSymmetricTrapezoid c sideLength) :=
sorry

end symmetric_trapezoid_construction_l402_40296


namespace correct_selection_methods_l402_40244

/-- The number of members in the class committee -/
def total_members : ℕ := 5

/-- The number of roles to be filled -/
def roles_to_fill : ℕ := 3

/-- The number of members who cannot serve as the entertainment officer -/
def restricted_members : ℕ := 2

/-- The number of different selection methods -/
def selection_methods : ℕ := 36

/-- Theorem stating that the number of selection methods is correct -/
theorem correct_selection_methods :
  (total_members - restricted_members) * (total_members - 1) * (total_members - 2) = selection_methods :=
by sorry

end correct_selection_methods_l402_40244


namespace silent_reading_ratio_l402_40279

theorem silent_reading_ratio (total : ℕ) (board_games : ℕ) (homework : ℕ) 
  (h1 : total = 24)
  (h2 : board_games = total / 3)
  (h3 : homework = 4)
  : (total - board_games - homework) * 2 = total := by
  sorry

end silent_reading_ratio_l402_40279


namespace jimmy_card_distribution_l402_40226

theorem jimmy_card_distribution (initial_cards : ℕ) (remaining_cards : ℕ) 
  (h1 : initial_cards = 18)
  (h2 : remaining_cards = 9) :
  ∃ (cards_to_bob : ℕ), 
    cards_to_bob = 3 ∧ 
    initial_cards = remaining_cards + cards_to_bob + 2 * cards_to_bob :=
by
  sorry

end jimmy_card_distribution_l402_40226


namespace candy_distribution_l402_40230

theorem candy_distribution (n : ℕ) (total_candy : ℕ) : 
  total_candy = 120 →
  (∃ q : ℕ, total_candy = 2 * n + 2 * q) →
  n = 58 ∨ n = 60 :=
by sorry

end candy_distribution_l402_40230


namespace square_difference_equality_l402_40209

theorem square_difference_equality : 1004^2 - 998^2 - 1002^2 + 1000^2 = 8008 := by
  sorry

end square_difference_equality_l402_40209


namespace sector_area_l402_40292

/-- Given a circular sector with central angle 2π/3 and chord length 2√3, 
    its area is 4π/3 -/
theorem sector_area (θ : Real) (chord_length : Real) (area : Real) : 
  θ = 2 * Real.pi / 3 →
  chord_length = 2 * Real.sqrt 3 →
  area = 4 * Real.pi / 3 := by
  sorry

#check sector_area

end sector_area_l402_40292


namespace andrew_payment_l402_40223

/-- The total amount Andrew paid for grapes and mangoes -/
def total_amount (grape_quantity : ℕ) (grape_rate : ℕ) (mango_quantity : ℕ) (mango_rate : ℕ) : ℕ :=
  grape_quantity * grape_rate + mango_quantity * mango_rate

/-- Theorem stating that Andrew paid 1376 for his purchase -/
theorem andrew_payment :
  total_amount 14 54 10 62 = 1376 := by
  sorry

end andrew_payment_l402_40223


namespace remaining_cards_l402_40252

def initial_cards : ℕ := 13
def cards_given_away : ℕ := 9

theorem remaining_cards : initial_cards - cards_given_away = 4 := by
  sorry

end remaining_cards_l402_40252


namespace triangle_properties_l402_40265

theorem triangle_properties (a b c : ℝ) (A B C : ℝ) (R : ℝ) :
  -- Triangle ABC with sides a, b, c and angles A, B, C
  -- Vectors (a-b, 1) and (a-c, 2) are collinear
  (a - b) / (a - c) = 1 / 2 →
  -- Angle A is 120°
  A = 2 * π / 3 →
  -- Circumradius is 14
  R = 14 →
  -- Ratio a:b:c is 7:5:3
  ∃ (k : ℝ), a = 7 * k ∧ b = 5 * k ∧ c = 3 * k ∧
  -- Area of triangle ABC is 45√3
  1/2 * b * c * Real.sin A = 45 * Real.sqrt 3 := by
sorry

end triangle_properties_l402_40265


namespace triangle_perimeter_proof_l402_40269

theorem triangle_perimeter_proof (a b c : ℝ) (h1 : a = 7) (h2 : b = 10) (h3 : c = 15) :
  a + b + c = 32 := by
  sorry

end triangle_perimeter_proof_l402_40269


namespace triangle_tangent_sum_properties_l402_40273

noncomputable def tanHalfAngle (θ : Real) : Real := Real.tan (θ / 2)

def isAcute (θ : Real) : Prop := 0 < θ ∧ θ < Real.pi / 2

def isObtuse (θ : Real) : Prop := Real.pi / 2 < θ ∧ θ < Real.pi

theorem triangle_tangent_sum_properties
  (A B C : Real)
  (triangle_angles : A + B + C = Real.pi)
  (positive_angles : 0 < A ∧ 0 < B ∧ 0 < C) :
  let S := (tanHalfAngle A)^2 + (tanHalfAngle B)^2 + (tanHalfAngle C)^2
  let T := tanHalfAngle A + tanHalfAngle B + tanHalfAngle C
  -- Relationship between S and T
  (T^2 = S + 2) →
  -- 1. For acute triangles
  ((isAcute A ∧ isAcute B ∧ isAcute C) → S < 2) ∧
  -- 2. For obtuse triangles with obtuse angle ≥ 2arctan(4/3)
  ((isObtuse A ∨ isObtuse B ∨ isObtuse C) ∧
   (max A (max B C) ≥ 2 * Real.arctan (4/3)) → S ≥ 2) ∧
  -- 3. For obtuse triangles with obtuse angle < 2arctan(4/3)
  ((isObtuse A ∨ isObtuse B ∨ isObtuse C) ∧
   (max A (max B C) < 2 * Real.arctan (4/3)) →
   ∃ (A' B' C' : Real),
     A' + B' + C' = Real.pi ∧
     (isObtuse A' ∨ isObtuse B' ∨ isObtuse C') ∧
     max A' (max B' C') < 2 * Real.arctan (4/3) ∧
     (tanHalfAngle A')^2 + (tanHalfAngle B')^2 + (tanHalfAngle C')^2 > 2 ∧
     ∃ (A'' B'' C'' : Real),
       A'' + B'' + C'' = Real.pi ∧
       (isObtuse A'' ∨ isObtuse B'' ∨ isObtuse C'') ∧
       max A'' (max B'' C'') < 2 * Real.arctan (4/3) ∧
       (tanHalfAngle A'')^2 + (tanHalfAngle B'')^2 + (tanHalfAngle C'')^2 < 2) :=
by
  sorry

end triangle_tangent_sum_properties_l402_40273


namespace problem_solution_l402_40216

theorem problem_solution (x : ℝ) (h : 5 * x - 7 = 15 * x + 13) : 3 * (x + 4) = 6 := by
  sorry

end problem_solution_l402_40216


namespace sin_2x_eq_sin_x_solution_l402_40299

open Set
open Real

def solution_set : Set ℝ := {0, π, -π/3, π/3, 5*π/3}

theorem sin_2x_eq_sin_x_solution :
  {x : ℝ | x ∈ Ioo (-π) (2*π) ∧ sin (2*x) = sin x} = solution_set := by
  sorry

end sin_2x_eq_sin_x_solution_l402_40299


namespace line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l402_40242

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := x + y - 2 = 0
def l₂ (x y : ℝ) : Prop := 2*x + y + 2 = 0

-- Define the intersection point P
def P : ℝ × ℝ := (-4, 6)

-- Define line l₃
def l₃ (x y : ℝ) : Prop := x - 3*y - 1 = 0

-- Theorem for the first line
theorem line_through_P_and_origin :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  (∀ x y, a*x + b*y + c = 0 ↔ (x = P.1 ∧ y = P.2 ∨ x = 0 ∧ y = 0)) ∧
  a = 3 ∧ b = 2 ∧ c = 0 :=
sorry

-- Theorem for the second line
theorem line_through_P_perpendicular_to_l₃ :
  ∃ (a b c : ℝ), a ≠ 0 ∨ b ≠ 0 ∧
  (∀ x y, a*x + b*y + c = 0 ↔ (x = P.1 ∧ y = P.2)) ∧
  (a*(1 : ℝ) + b*(-3 : ℝ) = 0) ∧
  a = 3 ∧ b = 1 ∧ c = 6 :=
sorry

end line_through_P_and_origin_line_through_P_perpendicular_to_l₃_l402_40242


namespace inequality_proof_l402_40278

theorem inequality_proof (a b c d : ℝ) 
  (ha : 0 < a ∧ a ≤ 1) 
  (hb : 0 < b ∧ b ≤ 1) 
  (hc : 0 < c ∧ c ≤ 1) 
  (hd : 0 < d ∧ d ≤ 1) : 
  1 / (a^2 + b^2 + c^2 + d^2) ≥ 1/4 + (1-a)*(1-b)*(1-c)*(1-d) := by
sorry

end inequality_proof_l402_40278


namespace farm_horse_food_calculation_l402_40210

/-- Calculates the total amount of horse food needed daily on a farm -/
theorem farm_horse_food_calculation (sheep_count : ℕ) (sheep_to_horse_ratio : ℚ) (food_per_horse : ℕ) : 
  sheep_count = 48 →
  sheep_to_horse_ratio = 6 / 7 →
  food_per_horse = 230 →
  (sheep_count / sheep_to_horse_ratio : ℚ).num * food_per_horse = 12880 := by
  sorry

end farm_horse_food_calculation_l402_40210


namespace total_games_won_l402_40295

def bulls_games : ℕ := 70
def heat_games : ℕ := bulls_games + 5

theorem total_games_won : bulls_games + heat_games = 145 := by
  sorry

end total_games_won_l402_40295


namespace blocks_per_box_l402_40213

def total_blocks : ℕ := 12
def num_boxes : ℕ := 2

theorem blocks_per_box : total_blocks / num_boxes = 6 := by
  sorry

end blocks_per_box_l402_40213


namespace triangle_segment_sum_l402_40293

/-- Given a triangle ABC with vertices A(0,0), B(7,0), and C(3,4), and a line
    passing through (6-2√2, 3-√2) intersecting AC at P and BC at Q,
    if the area of triangle PQC is 14/3, then |CP| + |CQ| = 63. -/
theorem triangle_segment_sum (P Q : ℝ × ℝ) : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (7, 0)
  let C : ℝ × ℝ := (3, 4)
  let line_point : ℝ × ℝ := (6 - 2 * Real.sqrt 2, 3 - Real.sqrt 2)
  (∃ (t₁ t₂ : ℝ), 0 < t₁ ∧ t₁ < 1 ∧ 0 < t₂ ∧ t₂ < 1 ∧
    P = (t₁ * C.1 + (1 - t₁) * A.1, t₁ * C.2 + (1 - t₁) * A.2) ∧
    Q = (t₂ * C.1 + (1 - t₂) * B.1, t₂ * C.2 + (1 - t₂) * B.2) ∧
    ∃ (s : ℝ), P = (line_point.1 + s * (Q.1 - line_point.1), 
                    line_point.2 + s * (Q.2 - line_point.2))) →
  (abs (P.1 * Q.2 - P.2 * Q.1 + Q.1 * C.2 - Q.2 * C.1 + C.1 * P.2 - C.2 * P.1) / 2 = 14/3) →
  Real.sqrt ((C.1 - P.1)^2 + (C.2 - P.2)^2) + Real.sqrt ((C.1 - Q.1)^2 + (C.2 - Q.2)^2) = 63 :=
by sorry

end triangle_segment_sum_l402_40293


namespace min_m_value_l402_40236

/-- Given a function f(x) = 2^(|x-a|) where a ∈ ℝ, if f(1+x) = f(1-x) for all x ∈ ℝ 
    and f(x) is monotonically increasing on [m,+∞), then the minimum value of m is 1. -/
theorem min_m_value (a : ℝ) (f : ℝ → ℝ) (m : ℝ) 
    (h1 : ∀ x, f x = 2^(|x - a|))
    (h2 : ∀ x, f (1 + x) = f (1 - x))
    (h3 : MonotoneOn f (Set.Ici m)) :
  ∃ m₀ : ℝ, m₀ = 1 ∧ ∀ m' : ℝ, (∀ x ≥ m', MonotoneOn f (Set.Ici x)) → m' ≥ m₀ :=
sorry

end min_m_value_l402_40236


namespace square_fraction_count_l402_40214

theorem square_fraction_count : 
  ∃! (s : Finset Int), 
    (∀ n ∈ s, ∃ k : Int, (n : ℚ) / (20 - n) = k^2) ∧ 
    (∀ n : Int, n ∉ s → ¬∃ k : Int, (n : ℚ) / (20 - n) = k^2) ∧
    s.card = 4 := by
  sorry

end square_fraction_count_l402_40214


namespace margaret_egg_collection_l402_40287

/-- The number of groups Margaret's eggs can be organized into -/
def num_groups : ℕ := 5

/-- The number of eggs in each group -/
def eggs_per_group : ℕ := 7

/-- The total number of eggs in Margaret's collection -/
def total_eggs : ℕ := num_groups * eggs_per_group

theorem margaret_egg_collection : total_eggs = 35 := by
  sorry

end margaret_egg_collection_l402_40287


namespace abc_signs_l402_40257

theorem abc_signs (a b c : ℝ) 
  (h1 : (a > 0 ∧ b < 0 ∧ c = 0) ∨ 
        (a > 0 ∧ b = 0 ∧ c < 0) ∨ 
        (a < 0 ∧ b > 0 ∧ c = 0) ∨ 
        (a < 0 ∧ b = 0 ∧ c > 0) ∨ 
        (a = 0 ∧ b > 0 ∧ c < 0) ∨ 
        (a = 0 ∧ b < 0 ∧ c > 0))
  (h2 : a * b^2 * (a + c) * (b + c) < 0) :
  a > 0 ∧ b < 0 ∧ c = 0 := by
sorry

end abc_signs_l402_40257


namespace megan_snacks_l402_40217

/-- The number of snacks Megan has in a given time period -/
def num_snacks (snack_interval : ℕ) (total_minutes : ℕ) : ℕ :=
  total_minutes / snack_interval

theorem megan_snacks : num_snacks 20 220 = 11 := by
  sorry

end megan_snacks_l402_40217


namespace infinite_solutions_implies_integer_root_l402_40284

/-- A polynomial of degree 3 with integer coefficients -/
def IntPolynomial3 : Type := ℤ → ℤ

/-- The property that xP(x) = yP(y) has infinitely many solutions for distinct integers x and y -/
def HasInfiniteSolutions (P : IntPolynomial3) : Prop :=
  ∀ n : ℕ, ∃ (x y : ℤ), x ≠ y ∧ x * P x = y * P y ∧ (∀ m < n, x ≠ m ∧ y ≠ m)

/-- The existence of an integer root for a polynomial -/
def HasIntegerRoot (P : IntPolynomial3) : Prop :=
  ∃ k : ℤ, P k = 0

/-- Main theorem: If a polynomial of degree 3 with integer coefficients has infinitely many
    solutions for xP(x) = yP(y) with distinct integers x and y, then it has an integer root -/
theorem infinite_solutions_implies_integer_root (P : IntPolynomial3) 
  (h : HasInfiniteSolutions P) : HasIntegerRoot P := by
  sorry

end infinite_solutions_implies_integer_root_l402_40284


namespace system_solution_l402_40208

theorem system_solution :
  ∃ (a b c d : ℝ),
    (a + c = -1) ∧
    (a * c + b + d = -1) ∧
    (a * d + b * c = -5) ∧
    (b * d = 6) ∧
    ((a = -3 ∧ b = 2 ∧ c = 2 ∧ d = 3) ∨
     (a = 2 ∧ b = 3 ∧ c = -3 ∧ d = 2)) :=
by sorry

end system_solution_l402_40208


namespace sufficient_not_necessary_l402_40251

theorem sufficient_not_necessary (a : ℝ) : 
  (a < -1 → |a| > 1) ∧ ¬(|a| > 1 → a < -1) := by
  sorry

end sufficient_not_necessary_l402_40251


namespace father_current_age_l402_40286

/-- The father's age at the son's birth equals the son's current age -/
def father_age_at_son_birth (father_age_now son_age_now : ℕ) : Prop :=
  father_age_now - son_age_now = son_age_now

/-- The son's age 5 years ago was 26 -/
def son_age_five_years_ago (son_age_now : ℕ) : Prop :=
  son_age_now - 5 = 26

/-- Theorem stating that the father's current age is 62 years -/
theorem father_current_age :
  ∀ (father_age_now son_age_now : ℕ),
    father_age_at_son_birth father_age_now son_age_now →
    son_age_five_years_ago son_age_now →
    father_age_now = 62 :=
by
  sorry

end father_current_age_l402_40286


namespace hexagon_perimeter_l402_40225

-- Define the hexagon
structure Hexagon :=
  (AB : ℝ)
  (BC : ℝ)
  (CD : ℝ)
  (DE : ℝ)
  (EF : ℝ)
  (AC : ℝ)
  (AD : ℝ)
  (AE : ℝ)
  (AF : ℝ)

-- Define the theorem
theorem hexagon_perimeter (h : Hexagon) :
  h.AB = 1 →
  h.BC = 2 →
  h.CD = 2 →
  h.DE = 2 →
  h.EF = 3 →
  h.AC^2 = h.AB^2 + h.BC^2 →
  h.AD^2 = h.AC^2 + h.CD^2 →
  h.AE^2 = h.AD^2 + h.DE^2 →
  h.AF^2 = h.AE^2 + h.EF^2 →
  h.AB + h.BC + h.CD + h.DE + h.EF + h.AF = 10 + Real.sqrt 22 :=
by sorry

end hexagon_perimeter_l402_40225


namespace least_subtrahend_l402_40215

theorem least_subtrahend (n : Nat) : 
  (∀ (d : Nat), d ∈ [17, 19, 23] → (997 - n) % d = 3) →
  (∀ (m : Nat), m < n → ∃ (d : Nat), d ∈ [17, 19, 23] ∧ (997 - m) % d ≠ 3) →
  n = 3 := by
  sorry

end least_subtrahend_l402_40215


namespace isosceles_perpendicular_division_l402_40233

/-- An isosceles triangle with base 32 and legs 20 -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  base_eq : base = 32
  leg_eq : leg = 20

/-- The perpendicular from the apex to the base divides the base into two segments -/
def perpendicular_segments (t : IsoscelesTriangle) : ℝ × ℝ :=
  (7, 25)

/-- Theorem: The perpendicular from the apex of the isosceles triangle
    divides the base into segments of 7 and 25 units -/
theorem isosceles_perpendicular_division (t : IsoscelesTriangle) :
  perpendicular_segments t = (7, 25) := by
  sorry

end isosceles_perpendicular_division_l402_40233


namespace grandmothers_current_age_prove_grandmothers_age_l402_40234

/-- Given Yoojung's current age and her grandmother's future age, calculate the grandmother's current age. -/
theorem grandmothers_current_age (yoojung_current_age : ℕ) (yoojung_future_age : ℕ) (grandmother_future_age : ℕ) : ℕ :=
  grandmother_future_age - (yoojung_future_age - yoojung_current_age)

/-- Prove that given the conditions, the grandmother's current age is 55. -/
theorem prove_grandmothers_age :
  let yoojung_current_age := 5
  let yoojung_future_age := 10
  let grandmother_future_age := 60
  grandmothers_current_age yoojung_current_age yoojung_future_age grandmother_future_age = 55 := by
  sorry

end grandmothers_current_age_prove_grandmothers_age_l402_40234


namespace log_expression_equals_three_halves_l402_40276

theorem log_expression_equals_three_halves :
  (Real.log (Real.sqrt 27) + Real.log 8 - 3 * Real.log (Real.sqrt 10)) / Real.log 1.2 = 3/2 := by
  sorry

end log_expression_equals_three_halves_l402_40276


namespace rectangle_reconfiguration_l402_40275

/-- Given a 10 × 15 rectangle divided into two congruent polygons and reassembled into a new rectangle with length twice its width, the length of one side of the smaller rectangle formed by one of the polygons is 5√3. -/
theorem rectangle_reconfiguration (original_length original_width : ℝ)
  (new_length new_width z : ℝ) :
  original_length = 10 →
  original_width = 15 →
  original_length * original_width = new_length * new_width →
  new_length = 2 * new_width →
  z = new_length / 2 →
  z = 5 * Real.sqrt 3 := by
sorry

end rectangle_reconfiguration_l402_40275


namespace inequality_proof_l402_40264

theorem inequality_proof (a b c : ℝ) (h : a + b + c = 3) :
  1 / (5 * a^2 - 4 * a + 11) + 1 / (5 * b^2 - 4 * b + 11) + 1 / (5 * c^2 - 4 * c + 11) ≤ 1 / 4 := by
  sorry

end inequality_proof_l402_40264


namespace coin_arrangement_exists_l402_40259

/-- Represents a coin with mass, diameter, and minting year -/
structure Coin where
  mass : ℝ
  diameter : ℝ
  year : ℕ

/-- Represents a 3x3x3 arrangement of coins -/
def Arrangement := Fin 3 → Fin 3 → Fin 3 → Coin

/-- Checks if the arrangement satisfies the required conditions -/
def is_valid_arrangement (arr : Arrangement) : Prop :=
  ∀ i j k,
    (k < 2 → (arr i j k).mass < (arr i j (k+1)).mass) ∧
    (j < 2 → (arr i j k).diameter < (arr i (j+1) k).diameter) ∧
    (i < 2 → (arr i j k).year > (arr (i+1) j k).year)

theorem coin_arrangement_exists (coins : Fin 27 → Coin) 
  (h_distinct : ∀ i j, i ≠ j → coins i ≠ coins j) :
  ∃ (arr : Arrangement), is_valid_arrangement arr ∧ 
    ∀ i : Fin 27, ∃ x y z, arr x y z = coins i :=
sorry

end coin_arrangement_exists_l402_40259


namespace min_value_theorem_l402_40211

-- Define the function f
def f (x a b : ℝ) : ℝ := |x - a| + |x + b|

-- State the theorem
theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (hmin : ∀ x, f x a b ≥ 3) (hf_reaches_min : ∃ x, f x a b = 3) :
  (a^2 / b + b^2 / a) ≥ 3 ∧ ∃ a b, a > 0 ∧ b > 0 ∧ (a^2 / b + b^2 / a) = 3 :=
sorry

end min_value_theorem_l402_40211


namespace repeating_decimal_sum_diff_l402_40212

/-- Represents a repeating decimal -/
def RepeatingDecimal (numerator denominator : ℕ) : ℚ := numerator / denominator

theorem repeating_decimal_sum_diff (a b c : ℚ) :
  a = RepeatingDecimal 6 9 →
  b = RepeatingDecimal 2 9 →
  c = RepeatingDecimal 4 9 →
  a + b - c = 4 / 9 := by
  sorry

end repeating_decimal_sum_diff_l402_40212


namespace arith_seq_ratio_l402_40235

/-- Two arithmetic sequences and their properties -/
structure ArithSeqPair where
  a : ℕ → ℚ  -- First arithmetic sequence
  b : ℕ → ℚ  -- Second arithmetic sequence
  A : ℕ → ℚ  -- Sum of first n terms of a
  B : ℕ → ℚ  -- Sum of first n terms of b
  sum_ratio : ∀ n, A n / B n = (4 * n + 2 : ℚ) / (5 * n - 5 : ℚ)

/-- Main theorem -/
theorem arith_seq_ratio (seq : ArithSeqPair) : 
  (seq.a 5 + seq.a 13) / (seq.b 5 + seq.b 13) = 7 / 8 := by
  sorry

end arith_seq_ratio_l402_40235


namespace g_evaluation_l402_40241

def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x + 10

theorem g_evaluation : 5 * g 2 + 4 * g (-2) = 186 := by
  sorry

end g_evaluation_l402_40241


namespace regression_lines_intersection_l402_40268

/-- Represents a linear regression line -/
structure RegressionLine where
  slope : ℝ
  intercept : ℝ

/-- Checks if a point lies on a regression line -/
def lies_on (l : RegressionLine) (x y : ℝ) : Prop :=
  y = l.slope * x + l.intercept

theorem regression_lines_intersection
  (l₁ l₂ : RegressionLine)
  (s t : ℝ)
  (h₁ : ∀ (x y : ℝ), x = s ∧ y = t → lies_on l₁ x y)
  (h₂ : ∀ (x y : ℝ), x = s ∧ y = t → lies_on l₂ x y) :
  lies_on l₁ s t ∧ lies_on l₂ s t := by
  sorry


end regression_lines_intersection_l402_40268


namespace det_A_positive_iff_x_gt_one_l402_40227

/-- Definition of a 2x2 matrix A with elements dependent on x -/
def A (x : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![2, 3 - x; 1, x]

/-- Definition of determinant for 2x2 matrix -/
def det2x2 (M : Matrix (Fin 2) (Fin 2) ℝ) : ℝ :=
  M 0 0 * M 1 1 - M 0 1 * M 1 0

/-- Theorem stating that det(A) > 0 iff x > 1 -/
theorem det_A_positive_iff_x_gt_one :
  ∀ x : ℝ, det2x2 (A x) > 0 ↔ x > 1 := by
  sorry

end det_A_positive_iff_x_gt_one_l402_40227


namespace division_equality_l402_40254

theorem division_equality : (49 : ℝ) / 0.07 = 700 := by
  sorry

end division_equality_l402_40254


namespace max_value_of_trig_function_l402_40263

theorem max_value_of_trig_function :
  let f (x : ℝ) := Real.tan (x + 3 * Real.pi / 4) - Real.tan x + Real.sin (x + Real.pi / 4)
  ∀ x ∈ Set.Icc (-3 * Real.pi / 4) (-Real.pi / 2),
    f x ≤ 0 ∧ ∃ x₀ ∈ Set.Icc (-3 * Real.pi / 4) (-Real.pi / 2), f x₀ = 0 :=
by sorry

end max_value_of_trig_function_l402_40263


namespace average_speed_inequality_l402_40202

theorem average_speed_inequality (a b v : ℝ) (hab : a < b) (hv : v = (2 * a * b) / (a + b)) : 
  a < v ∧ v < Real.sqrt (a * b) := by sorry

end average_speed_inequality_l402_40202


namespace parallel_condition_l402_40258

/-- Two lines in the plane -/
structure Line2D where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition for two lines to be parallel -/
def parallel (l1 l2 : Line2D) : Prop :=
  l1.a * l2.b = l2.a * l1.b

/-- The first line: ax + y - a + 1 = 0 -/
def line1 (a : ℝ) : Line2D :=
  ⟨a, 1, -a + 1⟩

/-- The second line: 4x + ay - 2 = 0 -/
def line2 (a : ℝ) : Line2D :=
  ⟨4, a, -2⟩

/-- Statement: a = ±2 is a necessary but not sufficient condition for the lines to be parallel -/
theorem parallel_condition (a : ℝ) :
  (parallel (line1 a) (line2 a) → a = 2 ∨ a = -2) ∧
  ¬(a = 2 ∨ a = -2 → parallel (line1 a) (line2 a)) :=
sorry

end parallel_condition_l402_40258


namespace solution_set_f_geq_3_range_of_a_l402_40272

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |a * x + 1|

-- Theorem 1: Solution set for f(x) ≥ 3 when a = 1
theorem solution_set_f_geq_3 :
  {x : ℝ | f 1 x ≥ 3} = {x : ℝ | x ≤ -3/2 ∨ x ≥ 3/2} := by sorry

-- Theorem 2: Range of a when solution set for f(x) ≤ 3-x contains [-1, 1]
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-1) 1, f a x ≤ 3 - x) → a ∈ Set.Icc (-1) 1 := by sorry

end solution_set_f_geq_3_range_of_a_l402_40272


namespace zero_in_interval_l402_40297

def f (x : ℝ) := x^3 + 2*x - 2

theorem zero_in_interval :
  ∃ c ∈ Set.Icc 0 1, f c = 0 :=
by
  sorry

end zero_in_interval_l402_40297


namespace cousins_ages_sum_l402_40245

def is_single_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

theorem cousins_ages_sum :
  ∀ (a b c d e : ℕ),
    is_single_digit a ∧ is_single_digit b ∧ is_single_digit c ∧ is_single_digit d ∧ is_single_digit e →
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e →
    (a * b = 36 ∨ a * c = 36 ∨ a * d = 36 ∨ a * e = 36 ∨ b * c = 36 ∨ b * d = 36 ∨ b * e = 36 ∨ c * d = 36 ∨ c * e = 36 ∨ d * e = 36) →
    (a * b = 40 ∨ a * c = 40 ∨ a * d = 40 ∨ a * e = 40 ∨ b * c = 40 ∨ b * d = 40 ∨ b * e = 40 ∨ c * d = 40 ∨ c * e = 40 ∨ d * e = 40) →
    a + b + c + d + e = 33 :=
by
  sorry

#check cousins_ages_sum

end cousins_ages_sum_l402_40245


namespace cone_volume_l402_40231

/-- Given a cone with lateral surface area to base area ratio of 5:3 and height 4, 
    its volume is 12π. -/
theorem cone_volume (r : ℝ) (h : ℝ) (l : ℝ) : 
  h = 4 → l / r = 5 / 3 → l^2 = h^2 + r^2 → (1 / 3) * π * r^2 * h = 12 * π :=
by sorry

end cone_volume_l402_40231


namespace inheritance_solution_l402_40219

def inheritance_problem (x : ℝ) : Prop :=
  let federal_tax_rate := 0.25
  let state_tax_rate := 0.15
  let total_tax := 12000
  (federal_tax_rate * x) + (state_tax_rate * (1 - federal_tax_rate) * x) = total_tax

theorem inheritance_solution :
  ∃ x : ℝ, inheritance_problem x ∧ (round x = 33103) :=
sorry

end inheritance_solution_l402_40219


namespace committee_selections_theorem_l402_40247

/-- The number of ways to select a committee with at least one former member -/
def committee_selections_with_former (total_candidates : ℕ) (former_members : ℕ) (committee_size : ℕ) : ℕ :=
  Nat.choose total_candidates committee_size - Nat.choose (total_candidates - former_members) committee_size

/-- Theorem stating the number of committee selections with at least one former member -/
theorem committee_selections_theorem :
  committee_selections_with_former 15 6 4 = 1239 := by
  sorry

end committee_selections_theorem_l402_40247


namespace right_triangle_area_l402_40285

theorem right_triangle_area (a : ℝ) (h : a > 0) :
  ∃ (R r : ℝ), R > 0 ∧ r > 0 ∧ R = (5/2) * r ∧
  (∃ (b c : ℝ), b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2 ∧
    (1/2 * a * b = (Real.sqrt 21 * a^2) / 6 ∨
     1/2 * a * b = (Real.sqrt 19 * a^2) / 22)) :=
by sorry

end right_triangle_area_l402_40285


namespace min_area_triangle_containing_unit_square_l402_40288

/-- A triangle that contains a unit square. -/
structure TriangleContainingUnitSquare where
  /-- The area of the triangle. -/
  area : ℝ
  /-- The triangle contains a unit square. -/
  contains_unit_square : True

/-- The minimum area of a triangle containing a unit square is 2. -/
theorem min_area_triangle_containing_unit_square :
  ∀ t : TriangleContainingUnitSquare, t.area ≥ 2 ∧ ∃ t' : TriangleContainingUnitSquare, t'.area = 2 :=
by sorry

end min_area_triangle_containing_unit_square_l402_40288


namespace sum_of_squared_ratios_equals_two_thirds_l402_40260

theorem sum_of_squared_ratios_equals_two_thirds
  (x₁ x₂ x₃ y₁ y₂ y₃ : ℝ)
  (hx : x₁ + x₂ + x₃ = 0)
  (hy : y₁ + y₂ + y₃ = 0)
  (hxy : x₁*y₁ + x₂*y₂ + x₃*y₃ = 0)
  (hpos : (x₁^2 + x₂^2 + x₃^2) * (y₁^2 + y₂^2 + y₃^2) > 0) :
  x₁^2 / (x₁^2 + x₂^2 + x₃^2) + y₁^2 / (y₁^2 + y₂^2 + y₃^2) = 2/3 := by
  sorry

end sum_of_squared_ratios_equals_two_thirds_l402_40260


namespace area_eq_product_segments_l402_40249

/-- A right triangle with an inscribed circle -/
structure RightTriangleWithIncircle where
  /-- The length of one leg of the right triangle -/
  a : ℝ
  /-- The length of the other leg of the right triangle -/
  b : ℝ
  /-- The length of the hypotenuse of the right triangle -/
  c : ℝ
  /-- The radius of the inscribed circle -/
  r : ℝ
  /-- The length of one segment of the hypotenuse divided by the point of tangency -/
  m : ℝ
  /-- The length of the other segment of the hypotenuse divided by the point of tangency -/
  n : ℝ
  /-- The hypotenuse is the sum of its segments -/
  hyp_sum : c = m + n
  /-- The triangle satisfies the Pythagorean theorem -/
  pythagorean : a^2 + b^2 = c^2
  /-- All lengths are positive -/
  pos_a : a > 0
  pos_b : b > 0
  pos_c : c > 0
  pos_r : r > 0
  pos_m : m > 0
  pos_n : n > 0

/-- The area of a right triangle with an inscribed circle is equal to the product of the 
    lengths of the segments into which the hypotenuse is divided by the point of tangency 
    with the incircle -/
theorem area_eq_product_segments (t : RightTriangleWithIncircle) : 
  (1/2) * t.a * t.b = t.m * t.n := by
  sorry

end area_eq_product_segments_l402_40249


namespace exactly_two_rigid_motions_l402_40204

/-- Represents a point on a plane --/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a line on a plane --/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Represents the pattern on the line --/
inductive Pattern
  | Triangle
  | Square

/-- Represents a rigid motion transformation --/
inductive RigidMotion
  | Rotation (center : Point) (angle : ℝ)
  | Translation (dx : ℝ) (dy : ℝ)
  | ReflectionLine (l : Line)
  | ReflectionPerp (p : Point)

/-- The line with the pattern --/
def patternLine : Line := sorry

/-- The sequence of shapes along the line --/
def patternSequence : ℕ → Pattern := sorry

/-- Checks if a rigid motion preserves the pattern --/
def preservesPattern (rm : RigidMotion) : Prop := sorry

/-- The theorem to be proved --/
theorem exactly_two_rigid_motions :
  ∃! (s : Finset RigidMotion),
    s.card = 2 ∧ 
    (∀ rm ∈ s, preservesPattern rm) ∧
    (∀ rm, preservesPattern rm → rm ∈ s ∨ rm = RigidMotion.Translation 0 0) :=
sorry

end exactly_two_rigid_motions_l402_40204


namespace cattle_milk_production_l402_40239

theorem cattle_milk_production 
  (total_cattle : ℕ) 
  (male_percentage : ℚ) 
  (female_percentage : ℚ) 
  (num_male_cows : ℕ) 
  (avg_milk_per_cow : ℚ) : 
  male_percentage = 2/5 →
  female_percentage = 3/5 →
  num_male_cows = 50 →
  avg_milk_per_cow = 2 →
  (↑num_male_cows : ℚ) / male_percentage = ↑total_cattle →
  (↑total_cattle * female_percentage * avg_milk_per_cow : ℚ) = 150 := by
  sorry

end cattle_milk_production_l402_40239


namespace least_four_digit_solution_l402_40205

theorem least_four_digit_solution (x : ℕ) : 
  (x ≥ 1000 ∧ x < 10000) →
  (5 * x ≡ 15 [ZMOD 20]) →
  (3 * x + 7 ≡ 19 [ZMOD 8]) →
  (-3 * x + 2 ≡ x [ZMOD 14]) →
  (∀ y : ℕ, y ≥ 1000 ∧ y < x →
    ¬(5 * y ≡ 15 [ZMOD 20] ∧
      3 * y + 7 ≡ 19 [ZMOD 8] ∧
      -3 * y + 2 ≡ y [ZMOD 14])) →
  x = 1032 :=
by sorry

end least_four_digit_solution_l402_40205


namespace inequality_solution_set_l402_40238

theorem inequality_solution_set (x : ℝ) : 
  (x - 20) / (x + 16) ≤ 0 ↔ -16 < x ∧ x ≤ 20 :=
by sorry

end inequality_solution_set_l402_40238


namespace min_value_reciprocal_sum_l402_40246

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + 3 * b = 1) :
  1 / a + 3 / b ≥ 16 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ a₀ + 3 * b₀ = 1 ∧ 1 / a₀ + 3 / b₀ = 16 :=
by sorry

end min_value_reciprocal_sum_l402_40246


namespace square_plus_one_geq_two_abs_l402_40207

theorem square_plus_one_geq_two_abs (x : ℝ) : x^2 + 1 ≥ 2 * |x| := by
  sorry

end square_plus_one_geq_two_abs_l402_40207


namespace chocolate_ratio_l402_40270

theorem chocolate_ratio (initial_chocolates : ℕ) (num_sisters : ℕ) (given_to_mother : ℕ) (eaten_by_father : ℕ) (left_with_father : ℕ) :
  initial_chocolates = 20 →
  num_sisters = 4 →
  given_to_mother = 3 →
  eaten_by_father = 2 →
  left_with_father = 5 →
  ∃ (chocolates_per_person : ℕ) (given_to_father : ℕ),
    chocolates_per_person * (num_sisters + 1) = initial_chocolates ∧
    given_to_father = left_with_father + given_to_mother + eaten_by_father ∧
    given_to_father * 2 = initial_chocolates :=
by sorry

end chocolate_ratio_l402_40270


namespace balloon_count_sum_l402_40201

/-- The number of yellow balloons Fred has -/
def fred_balloons : ℕ := 5

/-- The number of yellow balloons Sam has -/
def sam_balloons : ℕ := 6

/-- The number of yellow balloons Mary has -/
def mary_balloons : ℕ := 7

/-- The total number of yellow balloons -/
def total_balloons : ℕ := 18

/-- Theorem stating that the sum of individual balloon counts equals the total -/
theorem balloon_count_sum :
  fred_balloons + sam_balloons + mary_balloons = total_balloons := by
  sorry

end balloon_count_sum_l402_40201


namespace least_divisible_by_240_sixty_cube_divisible_by_240_least_positive_integer_cube_divisible_by_240_l402_40291

theorem least_divisible_by_240 (a : ℕ) : a > 0 ∧ a^3 % 240 = 0 → a ≥ 60 := by
  sorry

theorem sixty_cube_divisible_by_240 : (60 : ℕ)^3 % 240 = 0 := by
  sorry

theorem least_positive_integer_cube_divisible_by_240 :
  ∃ (a : ℕ), a > 0 ∧ a^3 % 240 = 0 ∧ ∀ (b : ℕ), b > 0 ∧ b^3 % 240 = 0 → b ≥ a :=
by
  sorry

end least_divisible_by_240_sixty_cube_divisible_by_240_least_positive_integer_cube_divisible_by_240_l402_40291


namespace speed_difference_l402_40224

/-- Given a truck and a car traveling the same distance, prove the difference in their average speeds -/
theorem speed_difference (distance : ℝ) (truck_time car_time : ℝ) 
  (h1 : distance = 240)
  (h2 : truck_time = 8)
  (h3 : car_time = 5) : 
  (distance / car_time) - (distance / truck_time) = 18 := by
  sorry

end speed_difference_l402_40224


namespace cube_in_pyramid_l402_40232

/-- The edge length of a cube inscribed in a regular quadrilateral pyramid -/
theorem cube_in_pyramid (a h : ℝ) (ha : a > 0) (hh : h > 0) :
  ∃ x : ℝ, x > 0 ∧ 
    (a * h) / (a + h * Real.sqrt 2) ≤ x ∧
    x ≤ (a * h) / (a + h) :=
by sorry

end cube_in_pyramid_l402_40232


namespace corrected_mean_l402_40240

theorem corrected_mean (n : ℕ) (original_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 →
  original_mean = 40 →
  incorrect_value = 15 →
  correct_value = 45 →
  let total_sum := n * original_mean
  let difference := correct_value - incorrect_value
  let corrected_sum := total_sum + difference
  corrected_sum / n = 40.6 := by
  sorry

end corrected_mean_l402_40240


namespace rectangle_discrepancy_exists_l402_40206

/-- Represents a point in a 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a rectangle with sides parallel to the axes -/
structure Rectangle where
  x1 : ℝ
  y1 : ℝ
  x2 : ℝ
  y2 : ℝ

/-- The side length of the square -/
def squareSideLength : ℝ := 10^2019

/-- The total number of marked points -/
def totalPoints : ℕ := 10^4038

/-- A set of points marked in the square -/
def markedPoints : Set Point := sorry

/-- Calculates the area of a rectangle -/
def rectangleArea (r : Rectangle) : ℝ :=
  (r.x2 - r.x1) * (r.y2 - r.y1)

/-- Counts the number of points inside a rectangle -/
def pointsInRectangle (r : Rectangle) (points : Set Point) : ℕ := sorry

/-- The main theorem to be proved -/
theorem rectangle_discrepancy_exists :
  ∃ (r : Rectangle),
    r.x1 ≥ 0 ∧ r.y1 ≥ 0 ∧ r.x2 ≤ squareSideLength ∧ r.y2 ≤ squareSideLength ∧
    |rectangleArea r - (pointsInRectangle r markedPoints : ℝ)| ≥ 6 := by
  sorry

end rectangle_discrepancy_exists_l402_40206


namespace multiply_with_negative_l402_40281

theorem multiply_with_negative (a : ℝ) : 3 * a * (-2 * a) = -6 * a^2 := by
  sorry

end multiply_with_negative_l402_40281


namespace family_size_l402_40274

theorem family_size (planned_spending : ℝ) (orange_cost : ℝ) (savings_percentage : ℝ) :
  planned_spending = 15 →
  orange_cost = 1.5 →
  savings_percentage = 0.4 →
  (planned_spending * savings_percentage) / orange_cost = 4 :=
by sorry

end family_size_l402_40274


namespace inequality_conditions_l402_40283

theorem inequality_conditions (A B C : ℝ) :
  (∀ x y z : ℝ, A * (x - y) * (x - z) + B * (y - z) * (y - x) + C * (z - x) * (z - y) ≥ 0) ↔
  (A ≥ 0 ∧ B ≥ 0 ∧ C ≥ 0 ∧ A^2 + B^2 + C^2 ≤ 2*(A*B + B*C + C*A)) :=
by sorry

end inequality_conditions_l402_40283


namespace max_container_weight_for_transport_l402_40237

/-- Represents a container with a weight in tons -/
structure Container where
  weight : ℕ

/-- Represents a platform with a maximum load capacity -/
structure Platform where
  capacity : ℕ

/-- Represents a train with a number of platforms -/
structure Train where
  platforms : List Platform

/-- Checks if a given configuration of containers can be loaded onto a train -/
def canLoad (containers : List Container) (train : Train) : Prop :=
  sorry

/-- The main theorem stating that 26 is the maximum container weight that guarantees
    1500 tons can be transported -/
theorem max_container_weight_for_transport
  (total_weight : ℕ)
  (num_platforms : ℕ)
  (platform_capacity : ℕ)
  (h_total_weight : total_weight = 1500)
  (h_num_platforms : num_platforms = 25)
  (h_platform_capacity : platform_capacity = 80)
  : (∃ k : ℕ, k = 26 ∧
    (∀ containers : List Container,
      (∀ c ∈ containers, c.weight ≤ k ∧ c.weight > 0) →
      (containers.map (λ c => c.weight)).sum = total_weight →
      canLoad containers (Train.mk (List.replicate num_platforms (Platform.mk platform_capacity)))) ∧
    (∀ k' > k, ∃ containers : List Container,
      (∀ c ∈ containers, c.weight ≤ k' ∧ c.weight > 0) ∧
      (containers.map (λ c => c.weight)).sum = total_weight ∧
      ¬canLoad containers (Train.mk (List.replicate num_platforms (Platform.mk platform_capacity))))) :=
  sorry

end max_container_weight_for_transport_l402_40237


namespace existence_of_special_polynomial_l402_40280

theorem existence_of_special_polynomial (n : ℕ) (hn : n > 0) :
  ∃ P : Polynomial ℕ,
    (∀ (i : ℕ), Polynomial.coeff P i ∈ ({0, 1} : Set ℕ)) ∧
    (∀ (d : ℕ), d ≥ 2 → P.eval d ≠ 0 ∧ (P.eval d) % n = 0) :=
by sorry

end existence_of_special_polynomial_l402_40280


namespace history_paper_pages_l402_40271

/-- Calculates the total number of pages in a paper given the number of days and pages per day. -/
def total_pages (days : ℕ) (pages_per_day : ℕ) : ℕ :=
  days * pages_per_day

/-- Proves that a paper due in 3 days, requiring 27 pages per day, has 81 pages in total. -/
theorem history_paper_pages : total_pages 3 27 = 81 := by
  sorry

end history_paper_pages_l402_40271


namespace population_increase_rate_example_l402_40221

def population_increase_rate (initial_population final_population : ℕ) : ℚ :=
  (final_population - initial_population : ℚ) / initial_population * 100

theorem population_increase_rate_example :
  population_increase_rate 300 330 = 10 := by
sorry

end population_increase_rate_example_l402_40221


namespace band_tryouts_l402_40294

theorem band_tryouts (flutes clarinets trumpets pianists : ℕ) : 
  flutes = 20 →
  clarinets = 30 →
  pianists = 20 →
  (80 : ℚ) / 100 * flutes + 1 / 2 * clarinets + 1 / 3 * trumpets + 1 / 10 * pianists = 53 →
  trumpets = 60 :=
by sorry

end band_tryouts_l402_40294


namespace fraction_equality_l402_40229

theorem fraction_equality (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : (5*x - 2*y) / (2*x - 5*y) = 3) : 
  (2*x + 5*y) / (5*x - 2*y) = 31/63 := by
  sorry

end fraction_equality_l402_40229


namespace perpendicular_lines_to_plane_are_parallel_l402_40203

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_to_plane_are_parallel
  (α β γ : Plane) (m n : Line)
  (h_distinct_planes : α ≠ β ∧ β ≠ γ ∧ α ≠ γ)
  (h_distinct_lines : m ≠ n)
  (h_m_perp_α : perpendicular m α)
  (h_n_perp_α : perpendicular n α) :
  parallel m n :=
sorry

end perpendicular_lines_to_plane_are_parallel_l402_40203


namespace markus_bags_l402_40255

theorem markus_bags (mara_bags : ℕ) (mara_marbles_per_bag : ℕ) (markus_marbles_per_bag : ℕ) (markus_extra_marbles : ℕ) :
  mara_bags = 12 →
  mara_marbles_per_bag = 2 →
  markus_marbles_per_bag = 13 →
  markus_extra_marbles = 2 →
  (mara_bags * mara_marbles_per_bag + markus_extra_marbles) / markus_marbles_per_bag = 2 :=
by sorry

end markus_bags_l402_40255


namespace quadratic_solution_sum_l402_40218

theorem quadratic_solution_sum (c d : ℝ) : 
  (c^2 - 6*c + 13 = 25) ∧ 
  (d^2 - 6*d + 13 = 25) ∧ 
  (c ≥ d) →
  c + 2*d = 9 - Real.sqrt 21 := by
sorry

end quadratic_solution_sum_l402_40218


namespace carla_drink_problem_l402_40267

/-- The amount of water Carla drank in ounces -/
def water_amount : ℝ := 15

/-- The total amount of liquid Carla drank in ounces -/
def total_amount : ℝ := 54

/-- The amount of soda Carla drank in ounces -/
def soda_amount (x : ℝ) : ℝ := 3 * water_amount - x

theorem carla_drink_problem :
  ∃ x : ℝ, x = 6 ∧ water_amount + soda_amount x = total_amount := by sorry

end carla_drink_problem_l402_40267


namespace sphere_surface_area_circumscribed_cube_l402_40250

theorem sphere_surface_area_circumscribed_cube (edge_length : ℝ) 
  (h : edge_length = 2) : 
  4 * Real.pi * (edge_length * Real.sqrt 3 / 2) ^ 2 = 12 * Real.pi :=
by sorry

end sphere_surface_area_circumscribed_cube_l402_40250


namespace percentage_sum_theorem_l402_40282

theorem percentage_sum_theorem (X Y : ℝ) 
  (hX : 0.45 * X = 270) 
  (hY : 0.35 * Y = 210) : 
  0.75 * X + 0.55 * Y = 780 := by
sorry

end percentage_sum_theorem_l402_40282


namespace discounted_soda_price_l402_40256

/-- Calculates the price of discounted soda cans -/
theorem discounted_soda_price (regular_price : ℝ) (discount_percent : ℝ) (num_cans : ℕ) : 
  regular_price = 0.30 →
  discount_percent = 15 →
  num_cans = 72 →
  num_cans * (regular_price * (1 - discount_percent / 100)) = 18.36 :=
by sorry

end discounted_soda_price_l402_40256


namespace triangle_area_and_minimum_ratio_l402_40290

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The given condition 2sin²A + sin²B = sin²C -/
def satisfiesCondition (t : Triangle) : Prop :=
  2 * (Real.sin t.A)^2 + (Real.sin t.B)^2 = (Real.sin t.C)^2

theorem triangle_area_and_minimum_ratio (t : Triangle) 
  (h1 : satisfiesCondition t) 
  (h2 : t.b = 2 * t.a) 
  (h3 : t.b = 4) :
  -- Part 1: Area of triangle ABC is √15
  (1/2 * t.a * t.b * Real.sin t.C = Real.sqrt 15) ∧
  -- Part 2: Minimum value of c²/(ab) is 2√2, and c/a = 2 at this minimum
  (∀ t' : Triangle, satisfiesCondition t' → 
    t'.c^2 / (t'.a * t'.b) ≥ 2 * Real.sqrt 2) ∧
  (∃ t' : Triangle, satisfiesCondition t' ∧ 
    t'.c^2 / (t'.a * t'.b) = 2 * Real.sqrt 2 ∧ t'.c / t'.a = 2) :=
by sorry

end triangle_area_and_minimum_ratio_l402_40290


namespace x_minus_y_values_l402_40262

theorem x_minus_y_values (x y : ℝ) (h1 : |x| = 4) (h2 : |y| = 5) (h3 : x > y) :
  x - y = 9 ∨ x - y = 1 := by
  sorry

end x_minus_y_values_l402_40262


namespace circumscribed_equal_sides_equal_angles_inscribed_equal_sides_not_always_equal_angles_circumscribed_equal_angles_not_always_equal_sides_inscribed_equal_angles_equal_sides_l402_40222

/- Define the basic structures -/
structure Polygon :=
  (sides : ℕ)
  (isCircumscribed : Bool)
  (hasEqualSides : Bool)
  (hasEqualAngles : Bool)

/- Define the theorems to be proved -/
theorem circumscribed_equal_sides_equal_angles (p : Polygon) :
  p.isCircumscribed ∧ p.hasEqualSides → p.hasEqualAngles :=
sorry

theorem inscribed_equal_sides_not_always_equal_angles :
  ∃ p : Polygon, ¬p.isCircumscribed ∧ p.hasEqualSides ∧ ¬p.hasEqualAngles :=
sorry

theorem circumscribed_equal_angles_not_always_equal_sides :
  ∃ p : Polygon, p.isCircumscribed ∧ p.hasEqualAngles ∧ ¬p.hasEqualSides :=
sorry

theorem inscribed_equal_angles_equal_sides (p : Polygon) :
  ¬p.isCircumscribed ∧ p.hasEqualAngles → p.hasEqualSides :=
sorry

end circumscribed_equal_sides_equal_angles_inscribed_equal_sides_not_always_equal_angles_circumscribed_equal_angles_not_always_equal_sides_inscribed_equal_angles_equal_sides_l402_40222
