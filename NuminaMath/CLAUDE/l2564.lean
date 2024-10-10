import Mathlib

namespace initial_average_age_proof_l2564_256454

/-- Proves that the initial average age of a group is 16 years, given the specified conditions. -/
theorem initial_average_age_proof (initial_count : ℕ) (new_count : ℕ) (new_avg_age : ℚ) (final_avg_age : ℚ) :
  initial_count = 20 →
  new_count = 20 →
  new_avg_age = 15 →
  final_avg_age = 15.5 →
  (initial_count * initial_avg_age + new_count * new_avg_age) / (initial_count + new_count) = final_avg_age →
  initial_avg_age = 16 := by
  sorry

#check initial_average_age_proof

end initial_average_age_proof_l2564_256454


namespace fixed_point_of_line_l2564_256441

/-- The line equation mx + y - m - 1 = 0 passes through the point (1, 1) for all real m -/
theorem fixed_point_of_line (m : ℝ) : m * 1 + 1 - m - 1 = 0 := by
  sorry

end fixed_point_of_line_l2564_256441


namespace point_not_on_line_l2564_256430

/-- Given m > 2 and mb > 0, prove that (0, -2023) cannot lie on y = mx + b -/
theorem point_not_on_line (m b : ℝ) (hm : m > 2) (hmb : m * b > 0) :
  ¬ (∃ (x y : ℝ), x = 0 ∧ y = -2023 ∧ y = m * x + b) := by
  sorry

end point_not_on_line_l2564_256430


namespace monotonic_cubic_range_l2564_256424

/-- A cubic function parameterized by b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^3 - b*x^2 + 3*x - 5

/-- The derivative of f with respect to x -/
def f_deriv (b : ℝ) (x : ℝ) : ℝ := 3*x^2 - 2*b*x + 3

theorem monotonic_cubic_range (b : ℝ) :
  (∀ x : ℝ, Monotone (f b)) ↔ b ∈ Set.Icc (-3) 3 :=
sorry

end monotonic_cubic_range_l2564_256424


namespace greatest_integer_less_than_negative_seventeen_thirds_l2564_256477

theorem greatest_integer_less_than_negative_seventeen_thirds :
  ⌊-17/3⌋ = -6 := by sorry

end greatest_integer_less_than_negative_seventeen_thirds_l2564_256477


namespace blue_eyed_students_l2564_256464

theorem blue_eyed_students (total_students : ℕ) (blond_blue : ℕ) (neither : ℕ) :
  total_students = 30 →
  blond_blue = 6 →
  neither = 3 →
  ∃ (blue_eyes : ℕ),
    blue_eyes = 11 ∧
    2 * blue_eyes + (blue_eyes - blond_blue) + neither = total_students :=
by sorry

end blue_eyed_students_l2564_256464


namespace union_A_complement_B_l2564_256494

open Set

-- Define the universal set U as ℝ
def U : Set ℝ := univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- Theorem statement
theorem union_A_complement_B : 
  A ∪ (U \ B) = Iic 1 ∪ Ioi 2 := by sorry

end union_A_complement_B_l2564_256494


namespace simplify_expression_l2564_256435

theorem simplify_expression :
  (3 * Real.sqrt 8) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) =
  6 * Real.sqrt 6 + 6 * Real.sqrt 10 - 6 * Real.sqrt 14 := by
  sorry

end simplify_expression_l2564_256435


namespace sum_of_modified_integers_l2564_256499

theorem sum_of_modified_integers (P : ℤ) (x y : ℤ) (h : x + y = P) :
  3 * (x + 5) + 3 * (y + 5) = 3 * P + 30 := by
  sorry

end sum_of_modified_integers_l2564_256499


namespace plates_with_parents_is_eight_l2564_256480

/-- The number of plates used when Matt's parents join them -/
def plates_with_parents (total_plates : ℕ) (days_per_week : ℕ) (days_with_son : ℕ) (plates_per_person_with_son : ℕ) : ℕ :=
  (total_plates - days_with_son * plates_per_person_with_son * 2) / (days_per_week - days_with_son)

/-- Proof that the number of plates used when Matt's parents join them is 8 -/
theorem plates_with_parents_is_eight :
  plates_with_parents 38 7 3 1 = 8 := by
  sorry

end plates_with_parents_is_eight_l2564_256480


namespace axis_of_symmetry_is_one_l2564_256402

/-- Given two perpendicular lines and a quadratic function, prove that the axis of symmetry is x=1 -/
theorem axis_of_symmetry_is_one 
  (a b : ℝ) 
  (h1 : ∀ x y : ℝ, b * x + a * y = 0 → x - 2 * y + 2 = 0 → (b * 1 + a * 0) * (1 * 1 + 2 * 0) = -1) 
  (f : ℝ → ℝ) 
  (h2 : ∀ x : ℝ, f x = a * x^2 - b * x + a) : 
  ∃ p : ℝ, p = 1 ∧ ∀ x : ℝ, f (p + x) = f (p - x) :=
sorry

end axis_of_symmetry_is_one_l2564_256402


namespace initial_money_calculation_l2564_256469

def toy_car_price : ℕ := 11
def scarf_price : ℕ := 10
def beanie_price : ℕ := 14
def remaining_money : ℕ := 7

def total_spent : ℕ := 2 * toy_car_price + scarf_price + beanie_price

theorem initial_money_calculation :
  total_spent + remaining_money = 53 := by sorry

end initial_money_calculation_l2564_256469


namespace geometric_sequence_product_l2564_256493

theorem geometric_sequence_product (b : ℕ → ℝ) (q : ℝ) :
  (∀ n, b (n + 1) = q * b n) →
  ∀ n, (b n * b (n + 1) * b (n + 2)) * q^3 = (b (n + 1) * b (n + 2) * b (n + 3)) := by
  sorry

end geometric_sequence_product_l2564_256493


namespace tori_trash_count_l2564_256479

/-- The number of pieces of trash Tori picked up in the classrooms -/
def classroom_trash : ℕ := 344

/-- The number of pieces of trash Tori picked up outside the classrooms -/
def outside_trash : ℕ := 1232

/-- The total number of pieces of trash Tori picked up -/
def total_trash : ℕ := classroom_trash + outside_trash

theorem tori_trash_count : total_trash = 1576 := by
  sorry

end tori_trash_count_l2564_256479


namespace fourth_power_nested_sqrt_l2564_256496

theorem fourth_power_nested_sqrt : (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1)))^4 = 3 + 2 * Real.sqrt 2 := by
  sorry

end fourth_power_nested_sqrt_l2564_256496


namespace coin_flip_probability_l2564_256434

/-- The probability of getting exactly k heads in n flips of a fair coin -/
def prob_k_heads (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) * (1 / 2) ^ n

/-- The difference between the probability of 4 heads in 5 flips and 5 heads in 5 flips -/
def prob_difference : ℚ :=
  prob_k_heads 5 4 - prob_k_heads 5 5

theorem coin_flip_probability : prob_difference = 1 / 8 := by
  sorry

end coin_flip_probability_l2564_256434


namespace pet_groomer_problem_l2564_256438

theorem pet_groomer_problem (total_animals : ℕ) (cats : ℕ) (selected : ℕ) (prob : ℚ) :
  total_animals = 7 →
  cats = 2 →
  selected = 4 →
  prob = 2/7 →
  (Nat.choose cats cats * Nat.choose (total_animals - cats) (selected - cats)) / Nat.choose total_animals selected = prob →
  total_animals - cats = 5 := by
sorry

end pet_groomer_problem_l2564_256438


namespace bowling_balls_count_l2564_256453

theorem bowling_balls_count (red : ℕ) (green : ℕ) : 
  green = red + 6 →
  red + green = 66 →
  red = 30 := by
sorry

end bowling_balls_count_l2564_256453


namespace simplify_fraction_l2564_256412

theorem simplify_fraction : (150 : ℚ) / 450 = 1 / 3 := by sorry

end simplify_fraction_l2564_256412


namespace symmetric_circle_l2564_256443

/-- The equation of a circle symmetric to x^2 + y^2 = 4 with respect to the line x + y - 1 = 0 -/
theorem symmetric_circle (x y : ℝ) : 
  (∀ x y, x^2 + y^2 = 4 → x + y - 1 = 0 → (x-1)^2 + (y-1)^2 = 4) :=
by sorry

end symmetric_circle_l2564_256443


namespace two_digit_sum_problem_l2564_256427

/-- Given two-digit numbers ab and cd, and a three-digit number jjj,
    where a, b, c, and d are distinct positive integers,
    c = 9, and ab + cd = jjj, prove that cd = 98. -/
theorem two_digit_sum_problem (ab cd jjj : ℕ) (a b c d : ℕ) : 
  (10 ≤ ab) ∧ (ab < 100) →  -- ab is a two-digit number
  (10 ≤ cd) ∧ (cd < 100) →  -- cd is a two-digit number
  (100 ≤ jjj) ∧ (jjj < 1000) →  -- jjj is a three-digit number
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →  -- a, b, c, d are distinct
  0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d →  -- a, b, c, d are positive
  c = 9 →  -- given condition
  ab + cd = jjj →  -- sum equation
  cd = 98 := by
sorry

end two_digit_sum_problem_l2564_256427


namespace average_weight_problem_l2564_256472

theorem average_weight_problem (total_boys : ℕ) (group1_boys : ℕ) (group2_boys : ℕ) 
  (group2_avg_weight : ℚ) (total_avg_weight : ℚ) :
  total_boys = group1_boys + group2_boys →
  total_boys = 24 →
  group1_boys = 16 →
  group2_boys = 8 →
  group2_avg_weight = 45.15 →
  total_avg_weight = 48.55 →
  (group1_boys * (50.25 : ℚ) + group2_boys * group2_avg_weight) / total_boys = total_avg_weight :=
by sorry

end average_weight_problem_l2564_256472


namespace geometric_series_sum_l2564_256444

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h : r ≠ 1) :
  let series_sum := (a * (1 - r^n)) / (1 - r)
  let a := 1 / 5
  let r := -1 / 5
  let n := 5
  series_sum = 521 / 3125 := by
sorry

end geometric_series_sum_l2564_256444


namespace inscribed_circle_tangency_angles_l2564_256463

/-- A rhombus with an inscribed circle -/
structure RhombusWithInscribedCircle where
  /-- The measure of the acute angle of the rhombus in degrees -/
  acute_angle : ℝ
  /-- The assumption that the acute angle is 37 degrees -/
  acute_angle_is_37 : acute_angle = 37

/-- The angles formed by the points of tangency on the inscribed circle -/
def tangency_angles (r : RhombusWithInscribedCircle) : List ℝ :=
  [180 - r.acute_angle, r.acute_angle, 180 - r.acute_angle, r.acute_angle]

/-- Theorem stating the angles formed by the points of tangency -/
theorem inscribed_circle_tangency_angles (r : RhombusWithInscribedCircle) :
  tangency_angles r = [143, 37, 143, 37] := by
  sorry

end inscribed_circle_tangency_angles_l2564_256463


namespace hat_price_reduction_l2564_256447

theorem hat_price_reduction (original_price : ℝ) (first_reduction : ℝ) (second_reduction : ℝ) :
  original_price = 12 ∧ first_reduction = 0.2 ∧ second_reduction = 0.25 →
  original_price * (1 - first_reduction) * (1 - second_reduction) = 7.2 := by
sorry

end hat_price_reduction_l2564_256447


namespace car_profit_percent_l2564_256418

/-- Calculates the profit percent from buying, repairing, and selling a car -/
theorem car_profit_percent 
  (purchase_price : ℕ) 
  (mechanical_repairs : ℕ) 
  (bodywork : ℕ) 
  (interior_refurbishment : ℕ) 
  (taxes_and_fees : ℕ) 
  (selling_price : ℕ) 
  (h1 : purchase_price = 48000)
  (h2 : mechanical_repairs = 6000)
  (h3 : bodywork = 4000)
  (h4 : interior_refurbishment = 3000)
  (h5 : taxes_and_fees = 2000)
  (h6 : selling_price = 72900) :
  ∃ (profit_percent : ℚ), 
    abs (profit_percent - 15.71) < 0.01 ∧ 
    profit_percent = (selling_price - (purchase_price + mechanical_repairs + bodywork + interior_refurbishment + taxes_and_fees)) / 
                     (purchase_price + mechanical_repairs + bodywork + interior_refurbishment + taxes_and_fees) * 100 := by
  sorry


end car_profit_percent_l2564_256418


namespace undefined_fraction_min_x_l2564_256465

theorem undefined_fraction_min_x : 
  let f (x : ℝ) := (x - 3) / (6 * x^2 - 37 * x + 6)
  ∀ y < 1/6, ∃ ε > 0, ∀ x ∈ Set.Ioo (y - ε) (y + ε), f x ≠ 0⁻¹ :=
by sorry

end undefined_fraction_min_x_l2564_256465


namespace quadratic_sum_l2564_256417

/-- A quadratic function with vertex at (-2, 5) and specific points -/
def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^2 + e * x + f

theorem quadratic_sum (d e f : ℝ) :
  (∀ x, g d e f x = d * (x + 2)^2 + 5) →  -- vertex form
  g d e f 0 = -1 →                       -- g(0) = -1
  g d e f 1 = -4 →                       -- g(1) = -4
  d + e + 3 * f = 14 := by sorry

end quadratic_sum_l2564_256417


namespace outfits_count_l2564_256442

/-- The number of outfits with different colored shirts and hats -/
def num_outfits (red_shirts green_shirts blue_shirts : ℕ) 
  (pants : ℕ) (red_hats green_hats blue_hats : ℕ) : ℕ :=
  (red_shirts * (green_hats + blue_hats) * pants) +
  (green_shirts * (red_hats + blue_hats) * pants) +
  (blue_shirts * (red_hats + green_hats) * pants)

/-- Theorem stating the number of outfits given the specific quantities -/
theorem outfits_count : 
  num_outfits 6 4 5 7 9 7 6 = 1526 := by
  sorry

end outfits_count_l2564_256442


namespace max_equilateral_triangle_area_in_rectangle_l2564_256407

theorem max_equilateral_triangle_area_in_rectangle :
  ∀ (a b : ℝ),
  a = 10 ∧ b = 11 →
  ∃ (area : ℝ),
  area = 221 * Real.sqrt 3 - 330 ∧
  (∀ (triangle_area : ℝ),
    (∃ (x y : ℝ),
      0 ≤ x ∧ x ≤ a ∧
      0 ≤ y ∧ y ≤ b ∧
      triangle_area = (Real.sqrt 3 / 4) * (x^2 + y^2)) →
    triangle_area ≤ area) :=
by sorry

end max_equilateral_triangle_area_in_rectangle_l2564_256407


namespace grazing_problem_solution_l2564_256420

/-- Represents the grazing scenario with oxen and rent -/
structure GrazingScenario where
  a_oxen : ℕ
  a_months : ℕ
  b_oxen : ℕ
  b_months : ℕ
  c_oxen : ℕ
  c_months : ℕ
  total_rent : ℚ
  c_rent : ℚ

/-- Calculates the total oxen-months for a given scenario -/
def total_oxen_months (s : GrazingScenario) : ℕ :=
  s.a_oxen * s.a_months + s.b_oxen * s.b_months + s.c_oxen * s.c_months

/-- Theorem stating the solution to the grazing problem -/
theorem grazing_problem_solution (s : GrazingScenario) 
  (h1 : s.a_oxen = 10)
  (h2 : s.a_months = 7)
  (h3 : s.b_oxen = 12)
  (h4 : s.c_oxen = 15)
  (h5 : s.c_months = 3)
  (h6 : s.total_rent = 245)
  (h7 : s.c_rent = 62.99999999999999)
  : s.b_months = 5 := by
  sorry


end grazing_problem_solution_l2564_256420


namespace book_cost_proof_l2564_256495

/-- The original cost of a book before discount -/
def original_cost : ℝ := sorry

/-- The number of books bought -/
def num_books : ℕ := 10

/-- The discount per book -/
def discount_per_book : ℝ := 0.5

/-- The total amount paid -/
def total_paid : ℝ := 45

theorem book_cost_proof :
  original_cost = 5 :=
by
  sorry

end book_cost_proof_l2564_256495


namespace fencing_cost_l2564_256474

/-- Calculate the total cost of fencing a rectangular plot -/
theorem fencing_cost (length breadth perimeter cost_per_metre : ℝ) : 
  length = 200 →
  length = breadth + 20 →
  cost_per_metre = 26.5 →
  perimeter = 2 * (length + breadth) →
  perimeter * cost_per_metre = 20140 := by
  sorry

#check fencing_cost

end fencing_cost_l2564_256474


namespace obtuse_angle_range_l2564_256429

-- Define the vectors a and b
def a (x : ℝ) : Fin 3 → ℝ := ![x, 2, 0]
def b (x : ℝ) : Fin 3 → ℝ := ![3, 2 - x, x^2]

-- Define the dot product of two vectors
def dot_product (v w : Fin 3 → ℝ) : ℝ :=
  (v 0) * (w 0) + (v 1) * (w 1) + (v 2) * (w 2)

-- Define the condition for an obtuse angle
def is_obtuse_angle (v w : Fin 3 → ℝ) : Prop :=
  dot_product v w < 0

-- State the theorem
theorem obtuse_angle_range (x : ℝ) :
  is_obtuse_angle (a x) (b x) → x < -4 :=
sorry

end obtuse_angle_range_l2564_256429


namespace geometric_sequence_property_l2564_256406

def is_geometric_sequence (a₁ a₂ a₃ a₄ : ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ a₂ = a₁ * r ∧ a₃ = a₂ * r ∧ a₄ = a₃ * r

theorem geometric_sequence_property (a₁ a₂ a₃ a₄ : ℝ) 
  (h₁ : a₁ ≠ 0) (h₂ : a₂ ≠ 0) (h₃ : a₃ ≠ 0) (h₄ : a₄ ≠ 0) :
  (is_geometric_sequence a₁ a₂ a₃ a₄ → a₁ * a₄ = a₂ * a₃) ∧
  (∃ b₁ b₂ b₃ b₄ : ℝ, b₁ ≠ 0 ∧ b₂ ≠ 0 ∧ b₃ ≠ 0 ∧ b₄ ≠ 0 ∧
    b₁ * b₄ = b₂ * b₃ ∧ ¬is_geometric_sequence b₁ b₂ b₃ b₄) :=
by sorry

end geometric_sequence_property_l2564_256406


namespace smallest_prime_factor_in_C_l2564_256404

def C : Set Nat := {54, 56, 59, 63, 65}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧ (∃ (p : Nat), Nat.Prime p ∧ p ∣ n ∧
    ∀ (m : Nat) (q : Nat), m ∈ C → Nat.Prime q → q ∣ m → p ≤ q) ∧
  (∀ (m : Nat) (q : Nat), m ∈ C → Nat.Prime q → q ∣ m → 2 ≤ q) :=
by sorry

end smallest_prime_factor_in_C_l2564_256404


namespace marble_fraction_l2564_256484

theorem marble_fraction (total : ℝ) (h : total > 0) : 
  let initial_blue := (2/3) * total
  let initial_red := (1/3) * total
  let new_blue := 3 * initial_blue
  let new_total := new_blue + initial_red
  initial_red / new_total = 1/7 := by
  sorry

end marble_fraction_l2564_256484


namespace simplify_fraction_l2564_256421

theorem simplify_fraction : 
  1 / (1 / ((1/2)^0) + 1 / ((1/2)^1) + 1 / ((1/2)^2) + 1 / ((1/2)^3) + 1 / ((1/2)^4)) = 1 / 31 := by
  sorry

end simplify_fraction_l2564_256421


namespace complex_power_sum_l2564_256467

theorem complex_power_sum (z : ℂ) (h : z + 1/z = 2 * Real.cos (5 * π / 180)) :
  z^100 + 1/(z^100) = -2 * Real.cos (40 * π / 180) := by
  sorry

end complex_power_sum_l2564_256467


namespace simplify_square_roots_l2564_256400

theorem simplify_square_roots : (Real.sqrt 300 / Real.sqrt 75) - (Real.sqrt 200 / Real.sqrt 50) = 0 := by
  sorry

end simplify_square_roots_l2564_256400


namespace calculate_expression_factorize_polynomial_l2564_256471

-- Part 1
theorem calculate_expression : (1 / 3)⁻¹ - Real.sqrt 16 + (-2016)^0 = 0 := by sorry

-- Part 2
theorem factorize_polynomial (x : ℝ) : 3 * x^2 - 6 * x + 3 = 3 * (x - 1)^2 := by sorry

end calculate_expression_factorize_polynomial_l2564_256471


namespace max_points_top_four_teams_l2564_256437

/-- Represents a tournament with the given conditions -/
structure Tournament :=
  (num_teams : ℕ)
  (points_for_win : ℕ)
  (points_for_draw : ℕ)
  (points_for_loss : ℕ)

/-- Calculates the total number of games in the tournament -/
def total_games (t : Tournament) : ℕ :=
  t.num_teams * (t.num_teams - 1) / 2

/-- Represents the maximum possible points for top teams -/
def max_points_for_top_teams (t : Tournament) (num_top_teams : ℕ) : ℕ :=
  sorry

/-- Theorem stating the maximum possible points for each of the top four teams -/
theorem max_points_top_four_teams (t : Tournament) :
  t.num_teams = 7 →
  t.points_for_win = 3 →
  t.points_for_draw = 1 →
  t.points_for_loss = 0 →
  max_points_for_top_teams t 4 = 18 := by
  sorry

end max_points_top_four_teams_l2564_256437


namespace custom_mult_five_four_l2564_256458

-- Define the custom multiplication operation
def custom_mult (a b : ℤ) : ℤ := a^2 + a*b - b^2

-- Theorem statement
theorem custom_mult_five_four :
  custom_mult 5 4 = 29 := by
  sorry

end custom_mult_five_four_l2564_256458


namespace both_sports_fans_l2564_256416

/-- Represents the number of students who like basketball -/
def basketball_fans : ℕ := 7

/-- Represents the number of students who like cricket -/
def cricket_fans : ℕ := 8

/-- Represents the number of students who like either basketball or cricket or both -/
def total_fans : ℕ := 10

/-- Theorem stating that the number of students who like both basketball and cricket is 5 -/
theorem both_sports_fans : 
  basketball_fans + cricket_fans - total_fans = 5 := by sorry

end both_sports_fans_l2564_256416


namespace domino_double_cover_l2564_256476

/-- Represents a domino tile placement on a 2×2 square -/
inductive DominoPlacement
  | Horizontal
  | Vertical

/-- Represents a tiling of a 2n × 2m rectangle using 1 × 2 domino tiles -/
def Tiling (n m : ℕ) := Fin n → Fin m → DominoPlacement

/-- Checks if two tilings are complementary (non-overlapping) -/
def complementary (t1 t2 : Tiling n m) : Prop :=
  ∀ i j, t1 i j ≠ t2 i j

theorem domino_double_cover (n m : ℕ) :
  ∃ (t1 t2 : Tiling n m), complementary t1 t2 := by sorry

end domino_double_cover_l2564_256476


namespace shape_C_has_two_lines_of_symmetry_l2564_256426

-- Define a type for shapes
inductive Shape : Type
  | A
  | B
  | C
  | D

-- Define a function to count lines of symmetry
def linesOfSymmetry : Shape → ℕ
  | Shape.A => 4
  | Shape.B => 0
  | Shape.C => 2
  | Shape.D => 1

-- Theorem statement
theorem shape_C_has_two_lines_of_symmetry :
  linesOfSymmetry Shape.C = 2 ∧
  ∀ s : Shape, s ≠ Shape.C → linesOfSymmetry s ≠ 2 :=
by sorry

end shape_C_has_two_lines_of_symmetry_l2564_256426


namespace person_height_calculation_l2564_256422

/-- The height of a person used to determine the depth of water -/
def personHeight : ℝ := 6

/-- The depth of the water in feet -/
def waterDepth : ℝ := 60

/-- The relationship between the water depth and the person's height -/
def depthRelation : Prop := waterDepth = 10 * personHeight

theorem person_height_calculation : 
  depthRelation → personHeight = 6 := by sorry

end person_height_calculation_l2564_256422


namespace interest_rate_first_part_l2564_256497

/-- Given a total amount of 3200, divided into two parts where the first part is 800
    and the second part is at 5% interest rate, and the total annual interest is 144,
    prove that the interest rate of the first part is 3%. -/
theorem interest_rate_first_part (total : ℕ) (first_part : ℕ) (second_part : ℕ) 
  (second_rate : ℚ) (total_interest : ℕ) :
  total = 3200 →
  first_part = 800 →
  second_part = total - first_part →
  second_rate = 5 / 100 →
  total_interest = 144 →
  ∃ (first_rate : ℚ), 
    first_rate * first_part / 100 + second_rate * second_part = total_interest ∧
    first_rate = 3 / 100 := by
  sorry

end interest_rate_first_part_l2564_256497


namespace ron_sold_twelve_tickets_l2564_256498

/-- Represents the ticket sales problem with Ron and Kathy --/
structure TicketSales where
  ron_price : ℝ
  kathy_price : ℝ
  total_tickets : ℕ
  total_income : ℝ

/-- Theorem stating that Ron sold 12 tickets given the problem conditions --/
theorem ron_sold_twelve_tickets (ts : TicketSales) 
  (h1 : ts.ron_price = 2)
  (h2 : ts.kathy_price = 4.5)
  (h3 : ts.total_tickets = 20)
  (h4 : ts.total_income = 60) : 
  ∃ (ron_tickets : ℕ) (kathy_tickets : ℕ), 
    ron_tickets + kathy_tickets = ts.total_tickets ∧ 
    ron_tickets * ts.ron_price + kathy_tickets * ts.kathy_price = ts.total_income ∧
    ron_tickets = 12 := by
  sorry

end ron_sold_twelve_tickets_l2564_256498


namespace prob_at_most_two_cars_is_one_sixth_l2564_256459

/-- The number of cars in the metro train -/
def num_cars : ℕ := 6

/-- The number of deceased passengers -/
def num_deceased : ℕ := 4

/-- The probability that at most two cars have deceased passengers -/
def prob_at_most_two_cars : ℚ := 1 / 6

/-- Theorem stating that the probability of at most two cars having deceased passengers is 1/6 -/
theorem prob_at_most_two_cars_is_one_sixth :
  prob_at_most_two_cars = 1 / 6 := by sorry

end prob_at_most_two_cars_is_one_sixth_l2564_256459


namespace quadratic_factorization_l2564_256490

theorem quadratic_factorization (y : ℝ) : 16 * y^2 - 48 * y + 36 = (4 * y - 6)^2 := by
  sorry

end quadratic_factorization_l2564_256490


namespace task_completion_time_l2564_256448

theorem task_completion_time (a b c : ℝ) 
  (h1 : 1/a + 1/b = 1/2)
  (h2 : 1/b + 1/c = 1/4)
  (h3 : 1/c + 1/a = 5/12) :
  a = 3 := by
sorry

end task_completion_time_l2564_256448


namespace fifteen_sided_figure_area_main_theorem_l2564_256433

/-- The area of a fifteen-sided figure created by cutting off three right triangles
    from the corners of a 4 × 5 rectangle --/
theorem fifteen_sided_figure_area : ℝ → Prop :=
  λ area_result : ℝ =>
    let rectangle_width : ℝ := 4
    let rectangle_height : ℝ := 5
    let rectangle_area : ℝ := rectangle_width * rectangle_height
    let triangle_side : ℝ := 1
    let triangle_area : ℝ := (1 / 2) * triangle_side * triangle_side
    let num_triangles : ℕ := 3
    let total_removed_area : ℝ := (triangle_area : ℝ) * num_triangles
    let final_area : ℝ := rectangle_area - total_removed_area
    area_result = final_area ∧ area_result = 18.5

/-- The main theorem stating that the area of the fifteen-sided figure is 18.5 cm² --/
theorem main_theorem : fifteen_sided_figure_area 18.5 := by
  sorry

end fifteen_sided_figure_area_main_theorem_l2564_256433


namespace radical_product_simplification_l2564_256405

theorem radical_product_simplification (q : ℝ) (hq : q > 0) :
  Real.sqrt (42 * q) * Real.sqrt (7 * q) * Real.sqrt (3 * q) = 63 * q * Real.sqrt (2 * q) := by
  sorry

end radical_product_simplification_l2564_256405


namespace sin_cos_difference_74_14_l2564_256431

theorem sin_cos_difference_74_14 :
  Real.sin (74 * π / 180) * Real.cos (14 * π / 180) -
  Real.cos (74 * π / 180) * Real.sin (14 * π / 180) =
  Real.sqrt 3 / 2 := by
  sorry

end sin_cos_difference_74_14_l2564_256431


namespace smallest_undefined_inverse_l2564_256401

theorem smallest_undefined_inverse (b : ℕ) : 
  (b > 0) → 
  (¬ ∃ x, x * b ≡ 1 [ZMOD 75]) → 
  (¬ ∃ x, x * b ≡ 1 [ZMOD 90]) → 
  (∀ a < b, a > 0 → (∃ x, x * a ≡ 1 [ZMOD 75]) ∨ (∃ x, x * a ≡ 1 [ZMOD 90])) → 
  b = 15 := by
sorry

end smallest_undefined_inverse_l2564_256401


namespace max_value_sum_of_roots_l2564_256457

/-- Given that x and y are real numbers satisfying 3x² + 4y² = 48,
    the maximum value of √(x² + y² - 4x + 4) + √(x² + y² - 2x + 4y + 5) is 8 + √13 -/
theorem max_value_sum_of_roots (x y : ℝ) (h : 3 * x^2 + 4 * y^2 = 48) :
  (∃ (m : ℝ), ∀ (a b : ℝ), 3 * a^2 + 4 * b^2 = 48 →
    Real.sqrt (a^2 + b^2 - 4*a + 4) + Real.sqrt (a^2 + b^2 - 2*a + 4*b + 5) ≤ m) ∧
  (Real.sqrt (x^2 + y^2 - 4*x + 4) + Real.sqrt (x^2 + y^2 - 2*x + 4*y + 5) ≤ 8 + Real.sqrt 13) :=
by sorry

end max_value_sum_of_roots_l2564_256457


namespace vertex_below_x_axis_iff_k_less_than_4_l2564_256481

/-- A quadratic function of the form y = x^2 - 4x + k -/
def quadratic_function (x k : ℝ) : ℝ := x^2 - 4*x + k

/-- The x-coordinate of the vertex of the quadratic function -/
def vertex_x : ℝ := 2

/-- The y-coordinate of the vertex of the quadratic function -/
def vertex_y (k : ℝ) : ℝ := quadratic_function vertex_x k

/-- The vertex is below the x-axis if its y-coordinate is negative -/
def vertex_below_x_axis (k : ℝ) : Prop := vertex_y k < 0

theorem vertex_below_x_axis_iff_k_less_than_4 :
  ∀ k : ℝ, vertex_below_x_axis k ↔ k < 4 := by sorry

end vertex_below_x_axis_iff_k_less_than_4_l2564_256481


namespace fraction_ordering_l2564_256446

theorem fraction_ordering : 
  (5 : ℚ) / 19 < 7 / 21 ∧ 7 / 21 < 9 / 23 := by sorry

end fraction_ordering_l2564_256446


namespace quadratic_expression_value_l2564_256451

theorem quadratic_expression_value (x y : ℚ) 
  (eq1 : 2 * x + 5 * y = 20) 
  (eq2 : 5 * x + 2 * y = 26) : 
  20 * x^2 + 60 * x * y + 50 * y^2 = 59600 / 49 := by
  sorry

end quadratic_expression_value_l2564_256451


namespace output_is_76_l2564_256486

def function_machine (input : ℕ) : ℕ :=
  let step1 := input * 3
  if step1 ≤ 30 then
    (step1 + 10)
  else
    ((step1 - 7) * 2)

theorem output_is_76 : function_machine 15 = 76 := by
  sorry

end output_is_76_l2564_256486


namespace max_a_fourth_quadrant_l2564_256452

theorem max_a_fourth_quadrant (a : ℤ) : 
  let z : ℂ := (2 + a * Complex.I) / (1 + 2 * Complex.I)
  (z.re > 0 ∧ z.im < 0) → a ≤ 3 ∧ ∃ (a : ℤ), a = 3 ∧ 
    let z : ℂ := (2 + a * Complex.I) / (1 + 2 * Complex.I)
    (z.re > 0 ∧ z.im < 0) := by
  sorry

end max_a_fourth_quadrant_l2564_256452


namespace min_balls_for_single_color_l2564_256478

theorem min_balls_for_single_color (red green yellow blue white black : ℕ) 
  (h_red : red = 35)
  (h_green : green = 22)
  (h_yellow : yellow = 18)
  (h_blue : blue = 15)
  (h_white : white = 12)
  (h_black : black = 8) :
  let total := red + green + yellow + blue + white + black
  ∀ n : ℕ, n ≥ 87 → 
    ∃ color : ℕ, color ≥ 18 ∧ 
      (color ≤ red ∨ color ≤ green ∨ color ≤ yellow ∨ 
       color ≤ blue ∨ color ≤ white ∨ color ≤ black) ∧
    ∀ m : ℕ, m < 87 → 
      ¬(∃ color : ℕ, color ≥ 18 ∧ 
        (color ≤ red ∨ color ≤ green ∨ color ≤ yellow ∨ 
         color ≤ blue ∨ color ≤ white ∨ color ≤ black)) :=
by sorry

end min_balls_for_single_color_l2564_256478


namespace sqrt_2x_minus_1_meaningful_l2564_256450

theorem sqrt_2x_minus_1_meaningful (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 1) → x ≥ (1 / 2) := by sorry

end sqrt_2x_minus_1_meaningful_l2564_256450


namespace otimes_self_otimes_self_l2564_256408

/-- Custom operation ⊗ -/
def otimes (x y : ℝ) : ℝ := x^3 - y

/-- Theorem stating that h ⊗ (h ⊗ h) = h for any real h -/
theorem otimes_self_otimes_self (h : ℝ) : otimes h (otimes h h) = h := by
  sorry

end otimes_self_otimes_self_l2564_256408


namespace smallest_n_square_cube_l2564_256491

/-- A number is a perfect square if it's equal to some integer squared. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 2

/-- A number is a perfect cube if it's equal to some integer cubed. -/
def IsPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m ^ 3

/-- The smallest positive integer n such that 3n is a perfect square and 2n is a perfect cube is 108. -/
theorem smallest_n_square_cube : (
  ∀ n : ℕ, 
  n > 0 ∧ 
  IsPerfectSquare (3 * n) ∧ 
  IsPerfectCube (2 * n) → 
  n ≥ 108
) ∧ 
IsPerfectSquare (3 * 108) ∧ 
IsPerfectCube (2 * 108) := by
  sorry

end smallest_n_square_cube_l2564_256491


namespace modulus_of_complex_fraction_l2564_256470

theorem modulus_of_complex_fraction (i : ℂ) : i * i = -1 → Complex.abs ((3 - 4 * i) / i) = 5 := by
  sorry

end modulus_of_complex_fraction_l2564_256470


namespace triangle_inequality_l2564_256423

theorem triangle_inequality (a b c : ℝ) 
  (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : a + b > c) (h5 : b + c > a) (h6 : c + a > b)
  (h7 : a + b + c = 1) : 
  a^2 + b^2 + c^2 + 4*a*b*c < (1/2 : ℝ) := by
sorry

end triangle_inequality_l2564_256423


namespace chord_diameter_ratio_l2564_256428

/-- Given two concentric circles with radii R/2 and R, prove that if a chord of the larger circle
    is divided into three equal parts by the smaller circle, then the ratio of this chord to
    the diameter of the larger circle is 3√6/8. -/
theorem chord_diameter_ratio (R : ℝ) (h : R > 0) :
  ∃ (chord : ℝ), 
    (∃ (a : ℝ), chord = 3 * a ∧ 
      (∃ (x : ℝ), x^2 = 2 * a^2 ∧ x = R/2)) →
    chord / (2 * R) = 3 * Real.sqrt 6 / 8 := by
  sorry

end chord_diameter_ratio_l2564_256428


namespace extra_postage_count_l2564_256455

structure Envelope where
  length : Float
  height : Float
  thickness : Float

def requires_extra_postage (e : Envelope) : Bool :=
  let ratio := e.length / e.height
  ratio < 1.2 || ratio > 2.8 || e.thickness > 0.25

def envelopes : List Envelope := [
  { length := 7, height := 5, thickness := 0.2 },
  { length := 10, height := 2, thickness := 0.3 },
  { length := 7, height := 7, thickness := 0.1 },
  { length := 12, height := 4, thickness := 0.26 }
]

theorem extra_postage_count :
  (envelopes.filter requires_extra_postage).length = 3 := by
  sorry

end extra_postage_count_l2564_256455


namespace nesbitt_inequality_l2564_256440

theorem nesbitt_inequality {a b c : ℝ} (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (b + c) + b / (a + c) + c / (a + b) ≥ 3 / 2 ∧
  (a / (b + c) + b / (a + c) + c / (a + b) = 3 / 2 ↔ a = b ∧ b = c) :=
sorry

end nesbitt_inequality_l2564_256440


namespace smallest_positive_solution_sqrt_3x_eq_5x_l2564_256483

theorem smallest_positive_solution_sqrt_3x_eq_5x :
  ∃ (x : ℝ), x > 0 ∧ Real.sqrt (3 * x) = 5 * x ∧
  ∀ (y : ℝ), y > 0 → Real.sqrt (3 * y) = 5 * y → x ≤ y ∧
  x = 3 / 25 := by
  sorry

end smallest_positive_solution_sqrt_3x_eq_5x_l2564_256483


namespace beth_score_l2564_256468

/-- The score of a basketball game between two teams -/
structure BasketballScore where
  team1_player1 : ℕ  -- Beth's score
  team1_player2 : ℕ  -- Jan's score
  team2_player1 : ℕ  -- Judy's score
  team2_player2 : ℕ  -- Angel's score

/-- The conditions of the basketball game -/
def game_conditions (score : BasketballScore) : Prop :=
  score.team1_player2 = 10 ∧
  score.team2_player1 = 8 ∧
  score.team2_player2 = 11 ∧
  score.team1_player1 + score.team1_player2 = score.team2_player1 + score.team2_player2 + 3

/-- Theorem: Given the game conditions, Beth scored 12 points -/
theorem beth_score (score : BasketballScore) 
  (h : game_conditions score) : score.team1_player1 = 12 := by
  sorry


end beth_score_l2564_256468


namespace absolute_value_inequality_solution_set_l2564_256415

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |2*x - 3| > 1} = Set.Iio 1 ∪ Set.Ioi 2 := by sorry

end absolute_value_inequality_solution_set_l2564_256415


namespace right_triangle_check_l2564_256403

/-- Check if three numbers form a right-angled triangle --/
def is_right_triangle (a b c : ℝ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_check :
  ¬ is_right_triangle (1/3) (1/4) (1/5) ∧
  is_right_triangle 3 4 5 ∧
  ¬ is_right_triangle 2 3 4 ∧
  ¬ is_right_triangle 1 (Real.sqrt 3) 4 :=
by sorry

#check right_triangle_check

end right_triangle_check_l2564_256403


namespace f_monotonicity_and_range_l2564_256410

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log (x + 1) + (1/2) * a * x^2 - x

theorem f_monotonicity_and_range :
  (∀ x > -1, ∀ y ∈ (Set.Ioo (-1 : ℝ) (-1/2) ∪ Set.Ioi 0), x < y → f 2 x < f 2 y) ∧
  (∀ x > -1, ∀ y ∈ Set.Ioo (-1/2 : ℝ) 0, x < y → f 2 x > f 2 y) ∧
  (∀ a : ℝ, (∀ x > 0, f a x ≥ a * x - x) ↔ 0 ≤ a ∧ a ≤ 1) :=
sorry

end f_monotonicity_and_range_l2564_256410


namespace monday_hours_calculation_l2564_256436

def hourly_wage : ℝ := 10
def monday_tips : ℝ := 18
def tuesday_hours : ℝ := 5
def tuesday_tips : ℝ := 12
def wednesday_hours : ℝ := 7
def wednesday_tips : ℝ := 20
def total_earnings : ℝ := 240

theorem monday_hours_calculation (monday_hours : ℝ) :
  hourly_wage * monday_hours + monday_tips +
  hourly_wage * tuesday_hours + tuesday_tips +
  hourly_wage * wednesday_hours + wednesday_tips = total_earnings →
  monday_hours = 7 := by
sorry

end monday_hours_calculation_l2564_256436


namespace power_equality_l2564_256466

theorem power_equality (m : ℕ) : 9^4 = 3^m → m = 8 := by
  sorry

end power_equality_l2564_256466


namespace inequality_solution_implies_m_value_l2564_256414

/-- If the solution set of the inequality -1/2x^2 + 2x > mx is {x | 0 < x < 2}, then m = 1 -/
theorem inequality_solution_implies_m_value (m : ℝ) :
  (∀ x : ℝ, (-1/2 * x^2 + 2*x > m*x) ↔ (0 < x ∧ x < 2)) →
  m = 1 := by
sorry

end inequality_solution_implies_m_value_l2564_256414


namespace system_solution_unique_l2564_256419

theorem system_solution_unique :
  ∃! (x y : ℝ), 
    3 * x^2 + 4 * x * y + 12 * y^2 + 16 * y = -6 ∧
    x^2 - 12 * x * y + 4 * y^2 - 10 * x + 12 * y = -7 ∧
    x = 1/2 ∧ y = -3/4 := by
  sorry

end system_solution_unique_l2564_256419


namespace negative_fractions_comparison_l2564_256425

theorem negative_fractions_comparison : -2/3 < -1/2 := by sorry

end negative_fractions_comparison_l2564_256425


namespace max_product_with_sum_and_diff_l2564_256439

/-- Given two real numbers with a difference of 4 and a sum of 35, 
    their product is maximized when the numbers are 19.5 and 15.5 -/
theorem max_product_with_sum_and_diff (x y : ℝ) : 
  x - y = 4 → x + y = 35 → x * y ≤ 19.5 * 15.5 :=
by sorry

end max_product_with_sum_and_diff_l2564_256439


namespace log_product_l2564_256409

theorem log_product (x y : ℝ) (h1 : Real.log (x / 2) = 0.5) (h2 : Real.log (y / 5) = 0.1) :
  Real.log (x * y) = 1.6 := by
  sorry

end log_product_l2564_256409


namespace polynomial_division_quotient_l2564_256411

theorem polynomial_division_quotient :
  let dividend := 5 * X^5 - 3 * X^4 + 6 * X^3 - 8 * X^2 + 9 * X - 4
  let divisor := 4 * X^2 + 5 * X + 3
  let quotient := 5/4 * X^3 - 47/16 * X^2 + 257/64 * X - 1547/256
  dividend = divisor * quotient + (dividend % divisor) :=
by sorry

end polynomial_division_quotient_l2564_256411


namespace unique_sum_of_four_smallest_divisor_squares_l2564_256482

def is_sum_of_four_smallest_divisor_squares (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), 
    a < b ∧ b < c ∧ c < d ∧
    a ∣ n ∧ b ∣ n ∧ c ∣ n ∧ d ∣ n ∧
    (∀ x : ℕ, x ∣ n → x ≤ d) ∧
    n = a^2 + b^2 + c^2 + d^2

theorem unique_sum_of_four_smallest_divisor_squares : 
  ∀ n : ℕ, is_sum_of_four_smallest_divisor_squares n ↔ n = 30 := by
  sorry

end unique_sum_of_four_smallest_divisor_squares_l2564_256482


namespace square_difference_of_integers_l2564_256492

theorem square_difference_of_integers (x y : ℕ) 
  (h1 : x + y = 40) 
  (h2 : x - y = 14) : 
  x^2 - y^2 = 560 := by
  sorry

end square_difference_of_integers_l2564_256492


namespace min_value_quadratic_l2564_256449

theorem min_value_quadratic (x : ℝ) : 
  ∃ (m : ℝ), m = 702 ∧ ∀ x, 3 * x^2 - 18 * x + 729 ≥ m :=
by sorry

end min_value_quadratic_l2564_256449


namespace positive_root_negative_root_zero_root_l2564_256488

-- Define the equation
def equation (a b x : ℝ) : Prop := b + x = 4 * x + a

-- Theorem for positive root
theorem positive_root (a b : ℝ) : 
  b > a → ∃ x : ℝ, x > 0 ∧ equation a b x := by sorry

-- Theorem for negative root
theorem negative_root (a b : ℝ) : 
  b < a → ∃ x : ℝ, x < 0 ∧ equation a b x := by sorry

-- Theorem for zero root
theorem zero_root (a b : ℝ) : 
  b = a → ∃ x : ℝ, x = 0 ∧ equation a b x := by sorry

end positive_root_negative_root_zero_root_l2564_256488


namespace connection_duration_l2564_256460

/-- Calculates the number of days a client can be connected to the internet given the specified parameters. -/
def days_connected (initial_balance : ℚ) (payment : ℚ) (daily_cost : ℚ) (discontinuation_threshold : ℚ) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the client will be connected for 14 days. -/
theorem connection_duration :
  days_connected 0 7 (1/2) 5 = 14 :=
by sorry

end connection_duration_l2564_256460


namespace circle_radius_from_chords_l2564_256456

/-- Given a circle with two chords drawn from a single point, prove that the radius is 85/8 -/
theorem circle_radius_from_chords (chord1 chord2 midpoint_distance : ℝ) 
  (h1 : chord1 = 9)
  (h2 : chord2 = 17)
  (h3 : midpoint_distance = 5) : 
  ∃ (radius : ℝ), radius = 85 / 8 := by
  sorry

end circle_radius_from_chords_l2564_256456


namespace costume_cost_theorem_l2564_256461

/-- Calculates the total cost of materials for a costume --/
def costume_cost (skirt_length : ℝ) (skirt_width : ℝ) (num_skirts : ℕ) 
                 (skirt_cost_per_sqft : ℝ) (bodice_shirt_area : ℝ) 
                 (bodice_sleeve_area : ℝ) (bodice_cost_per_sqft : ℝ)
                 (bonnet_length : ℝ) (bonnet_width : ℝ) (bonnet_cost_per_sqft : ℝ)
                 (shoe_cover_length : ℝ) (shoe_cover_width : ℝ) 
                 (num_shoe_covers : ℕ) (shoe_cover_cost_per_sqft : ℝ) : ℝ :=
  let skirt_total_area := skirt_length * skirt_width * num_skirts
  let skirt_cost := skirt_total_area * skirt_cost_per_sqft
  let bodice_total_area := bodice_shirt_area + 2 * bodice_sleeve_area
  let bodice_cost := bodice_total_area * bodice_cost_per_sqft
  let bonnet_area := bonnet_length * bonnet_width
  let bonnet_cost := bonnet_area * bonnet_cost_per_sqft
  let shoe_cover_total_area := shoe_cover_length * shoe_cover_width * num_shoe_covers
  let shoe_cover_cost := shoe_cover_total_area * shoe_cover_cost_per_sqft
  skirt_cost + bodice_cost + bonnet_cost + shoe_cover_cost

/-- The total cost of materials for the costume is $479.63 --/
theorem costume_cost_theorem : 
  costume_cost 12 4 3 3 2 5 2.5 2.5 1.5 1.5 1 1.5 2 4 = 479.63 := by
  sorry

end costume_cost_theorem_l2564_256461


namespace plaster_cost_per_sq_meter_l2564_256475

/-- Represents the dimensions of a rectangular tank -/
structure TankDimensions where
  length : ℝ
  width : ℝ
  depth : ℝ

/-- Calculates the total surface area of a rectangular tank that needs to be plastered -/
def totalPlasterArea (d : TankDimensions) : ℝ :=
  2 * (d.length * d.depth + d.width * d.depth) + d.length * d.width

/-- Theorem: Given a rectangular tank with dimensions 25m x 12m x 6m and a total plastering cost of 223.2 paise, 
    the cost per square meter of plastering is 0.3 paise -/
theorem plaster_cost_per_sq_meter (tank : TankDimensions) 
  (h1 : tank.length = 25)
  (h2 : tank.width = 12)
  (h3 : tank.depth = 6)
  (total_cost : ℝ)
  (h4 : total_cost = 223.2) : 
  total_cost / totalPlasterArea tank = 0.3 := by
  sorry

end plaster_cost_per_sq_meter_l2564_256475


namespace sheridan_fish_problem_l2564_256489

/-- The problem of calculating Mrs. Sheridan's initial number of fish -/
theorem sheridan_fish_problem (fish_from_sister fish_total : ℕ) 
  (h1 : fish_from_sister = 47)
  (h2 : fish_total = 69)
  (h3 : fish_total = fish_from_sister + initial_fish) :
  initial_fish = 22 :=
by
  sorry

#check sheridan_fish_problem

end sheridan_fish_problem_l2564_256489


namespace puzzle_solution_l2564_256432

theorem puzzle_solution : 
  ∀ (S T U K : ℕ),
  (S ≠ T ∧ S ≠ U ∧ S ≠ K ∧ T ≠ U ∧ T ≠ K ∧ U ≠ K) →
  (100 ≤ T * 100 + U * 10 + K ∧ T * 100 + U * 10 + K < 1000) →
  (1000 ≤ S * 1000 + T * 100 + U * 10 + K ∧ S * 1000 + T * 100 + U * 10 + K < 10000) →
  (5 * (T * 100 + U * 10 + K) = S * 1000 + T * 100 + U * 10 + K) →
  (T * 100 + U * 10 + K = 250 ∨ T * 100 + U * 10 + K = 750) := by
sorry

end puzzle_solution_l2564_256432


namespace binary_multiplication_division_equality_l2564_256487

/-- Represents a binary number as a list of booleans, with the least significant bit first. -/
def BinaryNumber := List Bool

/-- Converts a natural number to its binary representation. -/
def toBinary (n : Nat) : BinaryNumber :=
  if n = 0 then [] else (n % 2 = 1) :: toBinary (n / 2)

/-- Converts a binary number to its decimal representation. -/
def toDecimal (b : BinaryNumber) : Nat :=
  b.foldl (fun acc digit => 2 * acc + if digit then 1 else 0) 0

/-- Multiplies two binary numbers. -/
def binaryMultiply (a b : BinaryNumber) : BinaryNumber :=
  toBinary (toDecimal a * toDecimal b)

/-- Divides a binary number by another binary number. -/
def binaryDivide (a b : BinaryNumber) : BinaryNumber :=
  toBinary (toDecimal a / toDecimal b)

theorem binary_multiplication_division_equality :
  let a := [false, true, false, true, true, false, true]  -- 1011010₂
  let b := [false, false, true, false, true, false, true] -- 1010100₂
  let c := [false, true, false, true]                     -- 1010₂
  binaryDivide (binaryMultiply a b) c = 
    [false, false, true, false, false, true, true, true, false, true] -- 1011100100₂
  := by sorry

end binary_multiplication_division_equality_l2564_256487


namespace rectangular_plot_breadth_l2564_256473

/-- 
A rectangular plot has an area that is 20 times its breadth,
and its length is 10 meters more than its breadth.
This theorem proves that the breadth of such a plot is 10 meters.
-/
theorem rectangular_plot_breadth : 
  ∀ (breadth length area : ℝ),
  area = 20 * breadth →
  length = breadth + 10 →
  area = length * breadth →
  breadth = 10 :=
by sorry

end rectangular_plot_breadth_l2564_256473


namespace pencil_sale_ratio_l2564_256462

theorem pencil_sale_ratio :
  ∀ (C S : ℚ),
  C > 0 → S > 0 →
  80 * C = 80 * S + 30 * S →
  (80 * C) / (80 * S) = 11 / 8 := by
sorry

end pencil_sale_ratio_l2564_256462


namespace tickets_bought_l2564_256485

theorem tickets_bought (ticket_cost : ℕ) (total_spent : ℕ) (h1 : ticket_cost = 44) (h2 : total_spent = 308) :
  total_spent / ticket_cost = 7 := by
sorry

end tickets_bought_l2564_256485


namespace debby_bottles_per_day_l2564_256445

/-- The number of bottles Debby bought -/
def total_bottles : ℕ := 8066

/-- The number of days the bottles lasted -/
def days_lasted : ℕ := 74

/-- The number of bottles Debby drank per day -/
def bottles_per_day : ℕ := total_bottles / days_lasted

theorem debby_bottles_per_day :
  bottles_per_day = 109 := by sorry

end debby_bottles_per_day_l2564_256445


namespace number_of_divisors_180_l2564_256413

theorem number_of_divisors_180 : Finset.card (Nat.divisors 180) = 18 := by
  sorry

end number_of_divisors_180_l2564_256413
