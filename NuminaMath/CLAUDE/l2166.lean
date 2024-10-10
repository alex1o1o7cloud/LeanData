import Mathlib

namespace division_problem_l2166_216607

theorem division_problem (x : ℝ) (h : (120 / x) - 15 = 5) : x = 6 := by
  sorry

end division_problem_l2166_216607


namespace percentage_to_pass_l2166_216654

/-- Calculates the percentage of total marks needed to pass an exam -/
theorem percentage_to_pass (marks_obtained : ℕ) (marks_short : ℕ) (total_marks : ℕ) :
  marks_obtained = 125 →
  marks_short = 40 →
  total_marks = 500 →
  (((marks_obtained + marks_short : ℚ) / total_marks) * 100 : ℚ) = 33 := by
  sorry

end percentage_to_pass_l2166_216654


namespace wrong_operation_correction_l2166_216602

theorem wrong_operation_correction (x : ℕ) : 
  x - 46 = 27 → x * 46 = 3358 := by
  sorry

end wrong_operation_correction_l2166_216602


namespace equation_solution_l2166_216601

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

/-- The fractional part function -/
noncomputable def frac (x : ℝ) : ℝ := x - floor x

/-- The set of solutions to the equation -/
def solution_set : Set ℝ := {29/12, 19/6, 97/24}

/-- The main theorem -/
theorem equation_solution :
  ∀ x : ℝ, (1 / (floor x : ℝ) + 1 / (floor (2*x) : ℝ) = frac x + 1/3) ↔ x ∈ solution_set := by
  sorry

end equation_solution_l2166_216601


namespace power_of_three_mod_five_l2166_216667

theorem power_of_three_mod_five : 3^2040 ≡ 1 [ZMOD 5] := by
  sorry

end power_of_three_mod_five_l2166_216667


namespace election_votes_l2166_216693

theorem election_votes :
  ∀ (V : ℕ) (geoff_votes : ℕ),
    geoff_votes = V / 100 →                     -- Geoff received 1% of votes
    geoff_votes + 3000 > V * 51 / 100 →         -- With 3000 more votes, Geoff would win
    geoff_votes + 3000 ≤ V * 51 / 100 + 1 →     -- Geoff needed exactly 3000 more votes to win
    V = 6000 := by
  sorry

end election_votes_l2166_216693


namespace muffin_fundraiser_l2166_216670

/-- Proves the number of muffin cases needed to raise $120 --/
theorem muffin_fundraiser (muffins_per_pack : ℕ) (packs_per_case : ℕ) 
  (price_per_muffin : ℚ) (fundraising_goal : ℚ) :
  muffins_per_pack = 4 →
  packs_per_case = 3 →
  price_per_muffin = 2 →
  fundraising_goal = 120 →
  (fundraising_goal / (muffins_per_pack * packs_per_case * price_per_muffin) : ℚ) = 5 := by
  sorry

end muffin_fundraiser_l2166_216670


namespace chord_length_perpendicular_bisector_l2166_216675

theorem chord_length_perpendicular_bisector (r : ℝ) (h : r = 10) : 
  let chord_length := 2 * (r^2 - (r/2)^2).sqrt
  chord_length = 10 * Real.sqrt 3 := by
  sorry

end chord_length_perpendicular_bisector_l2166_216675


namespace min_colors_correct_key_coloring_distinguishes_min_colors_optimal_l2166_216655

/-- The smallest number of colors needed to distinguish n keys arranged in a circle -/
def min_colors (n : ℕ) : ℕ :=
  if n ≤ 2 then n
  else if n ≤ 5 then 3
  else 2

/-- Theorem stating the minimum number of colors needed to distinguish n keys -/
theorem min_colors_correct (n : ℕ) :
  min_colors n = 
    if n ≤ 2 then n
    else if n ≤ 5 then 3
    else 2 :=
by
  sorry

/-- The coloring function that assigns colors to keys -/
def key_coloring (n : ℕ) : ℕ → Fin (min_colors n) :=
  sorry

/-- Theorem stating that the key_coloring function distinguishes all keys -/
theorem key_coloring_distinguishes (n : ℕ) :
  ∀ i j : Fin n, i ≠ j → 
    ∃ k : ℕ, (key_coloring n ((i + k) % n) ≠ key_coloring n ((j + k) % n)) ∨
            (key_coloring n ((n - i - k - 1) % n) ≠ key_coloring n ((n - j - k - 1) % n)) :=
by
  sorry

/-- Theorem stating that min_colors n is the smallest number that allows a distinguishing coloring -/
theorem min_colors_optimal (n : ℕ) :
  ∀ m : ℕ, m < min_colors n → 
    ¬∃ f : ℕ → Fin m, ∀ i j : Fin n, i ≠ j → 
      ∃ k : ℕ, (f ((i + k) % n) ≠ f ((j + k) % n)) ∨
              (f ((n - i - k - 1) % n) ≠ f ((n - j - k - 1) % n)) :=
by
  sorry

end min_colors_correct_key_coloring_distinguishes_min_colors_optimal_l2166_216655


namespace afternoon_emails_l2166_216660

theorem afternoon_emails (morning evening afternoon : ℕ) : 
  morning = 5 →
  morning = afternoon + 2 →
  afternoon = 7 := by sorry

end afternoon_emails_l2166_216660


namespace fruit_basket_count_l2166_216681

/-- The number of ways to choose items from a set of n identical items -/
def chooseOptions (n : ℕ) : ℕ := n + 1

/-- The number of different fruit baskets that can be created -/
def fruitBaskets (apples oranges : ℕ) : ℕ :=
  chooseOptions apples * chooseOptions oranges - 1

theorem fruit_basket_count :
  fruitBaskets 6 8 = 62 := by
  sorry

end fruit_basket_count_l2166_216681


namespace ngo_employees_proof_l2166_216645

/-- The number of literate employees in an NGO -/
def num_literate_employees : ℕ := 10

theorem ngo_employees_proof :
  let total_employees := num_literate_employees + 20
  let illiterate_wage_decrease := 300
  let total_wage_decrease := total_employees * 10
  illiterate_wage_decrease = total_wage_decrease →
  num_literate_employees = 10 := by
sorry

end ngo_employees_proof_l2166_216645


namespace square_garden_area_l2166_216619

/-- Represents a square garden -/
structure SquareGarden where
  side : ℝ
  area : ℝ
  perimeter : ℝ

/-- Theorem: The area of a square garden is 90.25 square feet given the conditions -/
theorem square_garden_area
  (garden : SquareGarden)
  (h1 : garden.perimeter = 38)
  (h2 : garden.area = 2 * garden.perimeter + 14.25)
  : garden.area = 90.25 := by
  sorry

end square_garden_area_l2166_216619


namespace pyramid_vertex_on_face_plane_l2166_216628

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a parallelepiped in 3D space -/
structure Parallelepiped where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- Represents a triangular pyramid in 3D space -/
structure TriangularPyramid where
  v1 : Point3D
  v2 : Point3D
  v3 : Point3D
  v4 : Point3D

/-- Checks if a point lies on a plane defined by three other points -/
def pointLiesOnPlane (p : Point3D) (p1 p2 p3 : Point3D) : Prop := sorry

/-- Main theorem: Each vertex of one pyramid lies on a face plane of the other pyramid -/
theorem pyramid_vertex_on_face_plane (p : Parallelepiped) : 
  let pyramid1 := TriangularPyramid.mk p.A p.B p.D p.D₁
  let pyramid2 := TriangularPyramid.mk p.A₁ p.B₁ p.C₁ p.C
  (pointLiesOnPlane pyramid1.v1 pyramid2.v1 pyramid2.v3 pyramid2.v4 ∧
   pointLiesOnPlane pyramid1.v2 pyramid2.v2 pyramid2.v3 pyramid2.v4 ∧
   pointLiesOnPlane pyramid1.v3 pyramid2.v1 pyramid2.v2 pyramid2.v4 ∧
   pointLiesOnPlane pyramid1.v4 pyramid2.v1 pyramid2.v2 pyramid2.v3) ∧
  (pointLiesOnPlane pyramid2.v1 pyramid1.v1 pyramid1.v3 pyramid1.v4 ∧
   pointLiesOnPlane pyramid2.v2 pyramid1.v2 pyramid1.v3 pyramid1.v4 ∧
   pointLiesOnPlane pyramid2.v3 pyramid1.v1 pyramid1.v2 pyramid1.v4 ∧
   pointLiesOnPlane pyramid2.v4 pyramid1.v1 pyramid1.v2 pyramid1.v3) := by
  sorry

end pyramid_vertex_on_face_plane_l2166_216628


namespace twelfth_term_of_sequence_l2166_216651

/-- An arithmetic sequence is defined by its first term and common difference -/
def arithmeticSequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

theorem twelfth_term_of_sequence (a₁ a₂ a₃ : ℚ) (h₁ : a₁ = 1/2) (h₂ : a₂ = 5/6) (h₃ : a₃ = 7/6) :
  arithmeticSequence a₁ (a₂ - a₁) 12 = 25/6 := by
  sorry

end twelfth_term_of_sequence_l2166_216651


namespace person_a_number_l2166_216656

theorem person_a_number : ∀ (A B : ℕ), 
  A < 10 → B < 10 →
  A + B = 8 →
  (10 * B + A) - (10 * A + B) = 18 →
  10 * A + B = 35 := by
sorry

end person_a_number_l2166_216656


namespace system_solutions_l2166_216648

def equation1 (x : ℝ) : ℝ := (x - 1) * (x - 2) * (x - 3)

def equation2 (x y : ℝ) : ℝ := 
  (|x - 1| + |y - 1|) * (|x - 2| + |y - 2|) * (|x - 3| + |y - 4|)

theorem system_solutions :
  ∀ (x y : ℝ), 
    (equation1 x = 0 ∧ equation2 x y = 0) ↔ 
    ((x = 1 ∧ y = 1) ∨ (x = 2 ∧ y = 2) ∨ (x = 3 ∧ y = 4)) :=
by sorry

end system_solutions_l2166_216648


namespace calculation_proof_l2166_216635

theorem calculation_proof : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |3034 - (1002 / 20.04) - 2983.95| < ε :=
by
  sorry

end calculation_proof_l2166_216635


namespace shaded_area_rectangle_triangle_l2166_216606

/-- Given a rectangle with width 8 and height 12, and a right triangle with base 6 and height 8,
    prove that the area of the shaded region formed by a segment connecting the top-left vertex
    of the rectangle to the farthest vertex of the triangle is 120 square units. -/
theorem shaded_area_rectangle_triangle (rectangle_width : ℝ) (rectangle_height : ℝ)
    (triangle_base : ℝ) (triangle_height : ℝ) :
  rectangle_width = 8 →
  rectangle_height = 12 →
  triangle_base = 6 →
  triangle_height = 8 →
  let rectangle_area := rectangle_width * rectangle_height
  let triangle_area := (1 / 2) * triangle_base * triangle_height
  let shaded_area := rectangle_area + triangle_area
  shaded_area = 120 := by
  sorry

end shaded_area_rectangle_triangle_l2166_216606


namespace inscribed_circle_radius_right_triangle_l2166_216639

/-- The radius of the inscribed circle in a right triangle with side lengths 3, 4, and 5 is 1 -/
theorem inscribed_circle_radius_right_triangle :
  let a : ℝ := 3
  let b : ℝ := 4
  let c : ℝ := 5
  let s : ℝ := (a + b + c) / 2
  let area : ℝ := (s * (s - a) * (s - b) * (s - c))^(1/2)
  a^2 + b^2 = c^2 → -- Pythagorean theorem to ensure it's a right triangle
  area / s = 1 := by
sorry


end inscribed_circle_radius_right_triangle_l2166_216639


namespace largest_59_double_l2166_216605

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 9 -/
def base10ToBase9 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) :=
      if m = 0 then acc
      else aux (m / 9) ((m % 9) :: acc)
    aux n []

/-- Checks if a number is a 5-9 double -/
def is59Double (m : Nat) : Prop :=
  let base5Digits := base10ToBase9 m
  let base9Value := base5ToBase10 base5Digits
  base9Value = 2 * m

theorem largest_59_double :
  ∀ n : Nat, n > 20 → ¬(is59Double n) ∧ is59Double 20 :=
sorry

end largest_59_double_l2166_216605


namespace age_ratio_proof_l2166_216678

/-- Given that Jacob is 24 years old now and Tony will be 18 years old in 6 years,
    prove that the ratio of Tony's current age to Jacob's current age is 1:2. -/
theorem age_ratio_proof (jacob_age : ℕ) (tony_future_age : ℕ) (years_until_future : ℕ) :
  jacob_age = 24 →
  tony_future_age = 18 →
  years_until_future = 6 →
  (tony_future_age - years_until_future) * 2 = jacob_age := by
sorry

end age_ratio_proof_l2166_216678


namespace investment_time_period_l2166_216638

/-- Proves that given a principal of 2000 invested at simple interest rates of 18% p.a. and 12% p.a.,
    if the difference in interest received is 240, then the time period of investment is 20 years. -/
theorem investment_time_period (principal : ℝ) (rate_high : ℝ) (rate_low : ℝ) (interest_diff : ℝ) :
  principal = 2000 →
  rate_high = 18 →
  rate_low = 12 →
  interest_diff = 240 →
  ∃ time : ℝ,
    principal * (rate_high / 100) * time - principal * (rate_low / 100) * time = interest_diff ∧
    time = 20 := by
  sorry

end investment_time_period_l2166_216638


namespace ratio_equation_solution_l2166_216692

theorem ratio_equation_solution : 
  let x : ℚ := 7 / 15
  (3 : ℚ) / 5 / ((6 : ℚ) / 7) = x / ((2 : ℚ) / 3) :=
by sorry

end ratio_equation_solution_l2166_216692


namespace apple_count_l2166_216677

theorem apple_count (red : ℕ) (green : ℕ) : 
  red = 16 → green = red + 12 → red + green = 44 := by
  sorry

end apple_count_l2166_216677


namespace parallelogram_height_l2166_216616

/-- A parallelogram with given area and base has a specific height -/
theorem parallelogram_height (area base height : ℝ) (h_area : area = 375) (h_base : base = 25) :
  area = base * height → height = 15 := by
  sorry

end parallelogram_height_l2166_216616


namespace number_equation_solution_l2166_216672

theorem number_equation_solution : ∃ x : ℝ, 46 + 3 * x = 109 ∧ x = 21 := by
  sorry

end number_equation_solution_l2166_216672


namespace imaginary_unit_power_fraction_l2166_216684

theorem imaginary_unit_power_fraction (i : ℂ) (h : i^2 = -1) : 
  i^2015 / (1 + i) = (-1 - i) / 2 := by sorry

end imaginary_unit_power_fraction_l2166_216684


namespace tank_capacity_proof_l2166_216685

/-- The capacity of a tank with specific inlet and outlet pipe characteristics -/
def tank_capacity : ℝ := 1280

/-- The time it takes for the outlet pipe to empty the full tank -/
def outlet_time : ℝ := 10

/-- The rate at which the inlet pipe fills the tank in litres per minute -/
def inlet_rate : ℝ := 8

/-- The additional time it takes to empty the tank when the inlet pipe is open -/
def additional_time : ℝ := 6

theorem tank_capacity_proof :
  tank_capacity = outlet_time * inlet_rate * 60 * (outlet_time + additional_time) / additional_time :=
by sorry

end tank_capacity_proof_l2166_216685


namespace brad_balloons_l2166_216610

theorem brad_balloons (total : ℕ) (red : ℕ) (green : ℕ) (blue : ℕ) 
  (h1 : total = 37)
  (h2 : red = 14)
  (h3 : green = 10)
  (h4 : total = red + green + blue) :
  blue = 13 := by
  sorry

end brad_balloons_l2166_216610


namespace greta_worked_40_hours_l2166_216631

/-- Greta's hourly rate in dollars -/
def greta_rate : ℝ := 12

/-- Lisa's hourly rate in dollars -/
def lisa_rate : ℝ := 15

/-- Number of hours Lisa would need to work to equal Greta's earnings -/
def lisa_hours : ℝ := 32

/-- Theorem stating that Greta worked 40 hours -/
theorem greta_worked_40_hours : 
  ∃ (greta_hours : ℝ), greta_hours * greta_rate = lisa_hours * lisa_rate ∧ greta_hours = 40 := by
  sorry

end greta_worked_40_hours_l2166_216631


namespace det_problem_l2166_216683

def det (a b d c : ℕ) : ℤ := a * c - b * d

theorem det_problem (b d : ℕ) (h : det 2 b d 4 = 2) : b + d = 5 ∨ b + d = 7 := by
  sorry

end det_problem_l2166_216683


namespace two_color_plane_division_l2166_216653

/-- A type representing a line in a plane. -/
structure Line where
  -- We don't need to specify the exact properties of a line for this problem

/-- A type representing a region in a plane. -/
structure Region where
  -- We don't need to specify the exact properties of a region for this problem

/-- A type representing a color (either red or blue). -/
inductive Color
  | Red
  | Blue

/-- A function that determines if two regions are adjacent. -/
def adjacent (r1 r2 : Region) : Prop :=
  sorry  -- The exact definition is not important for the statement

/-- A type representing a coloring of regions. -/
def Coloring := Region → Color

/-- A predicate that checks if a coloring is valid (no adjacent regions have the same color). -/
def valid_coloring (c : Coloring) : Prop :=
  ∀ r1 r2 : Region, adjacent r1 r2 → c r1 ≠ c r2

/-- The main theorem stating that for any set of lines dividing a plane,
    there exists a valid two-coloring of the resulting regions. -/
theorem two_color_plane_division (lines : Set Line) :
  ∃ c : Coloring, valid_coloring c :=
sorry

end two_color_plane_division_l2166_216653


namespace abs_neg_a_eq_three_l2166_216633

theorem abs_neg_a_eq_three (a : ℝ) : |(-a)| = 3 → a = 3 ∨ a = -3 := by
  sorry

end abs_neg_a_eq_three_l2166_216633


namespace total_frogs_in_lakes_l2166_216680

theorem total_frogs_in_lakes (lassie_frogs : ℕ) (crystal_percentage : ℚ) : 
  lassie_frogs = 45 →
  crystal_percentage = 80/100 →
  lassie_frogs + (crystal_percentage * lassie_frogs).floor = 81 :=
by
  sorry

end total_frogs_in_lakes_l2166_216680


namespace valid_number_is_composite_l2166_216695

def is_valid_pair (a b : ℕ) : Prop :=
  (a * 10 + b) % 17 = 0 ∨ (a * 10 + b) % 23 = 0

def contains_digits (n : ℕ) (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, ∃ k, n / (10^k) % 10 = d

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 10^1999 ∧ n < 10^2000) ∧
  (∀ i < 1999, is_valid_pair ((n / 10^i) % 10) ((n / 10^(i+1)) % 10)) ∧
  contains_digits n [1, 9, 8, 7]

theorem valid_number_is_composite (n : ℕ) (h : is_valid_number n) : 
  ¬(Nat.Prime n) := by
  sorry

end valid_number_is_composite_l2166_216695


namespace break_room_vacant_seats_l2166_216618

theorem break_room_vacant_seats :
  let total_tables : ℕ := 5
  let seats_per_table : ℕ := 8
  let occupied_tables : ℕ := 2
  let people_per_occupied_table : ℕ := 3
  let unusable_tables : ℕ := 1

  let usable_tables : ℕ := total_tables - unusable_tables
  let total_seats : ℕ := usable_tables * seats_per_table
  let occupied_seats : ℕ := occupied_tables * people_per_occupied_table

  total_seats - occupied_seats = 26 :=
by sorry

end break_room_vacant_seats_l2166_216618


namespace expression_equals_negative_one_l2166_216687

theorem expression_equals_negative_one (a x : ℝ) (ha : a ≠ 0) (hx1 : x ≠ a) (hx2 : x ≠ -2*a) :
  (((a / (2*a + x)) - (x / (a - x))) / ((x / (2*a + x)) + (a / (a - x)))) = -1 ↔ x = a / 2 :=
by sorry

end expression_equals_negative_one_l2166_216687


namespace count_valid_pairs_l2166_216662

def is_valid_pair (x y : ℕ) : Prop :=
  1 ≤ x ∧ x ≤ 23 ∧ 1 ≤ y ∧ y ≤ 23 ∧ (x^2 + y^2 + x + y) % 6 = 0

theorem count_valid_pairs :
  ∃ (S : Finset (ℕ × ℕ)), (∀ (p : ℕ × ℕ), p ∈ S ↔ is_valid_pair p.1 p.2) ∧ S.card = 225 := by
  sorry

end count_valid_pairs_l2166_216662


namespace lacson_unsold_sweet_potatoes_l2166_216676

/-- The number of sweet potatoes Mrs. Lacson has not yet sold -/
def sweet_potatoes_not_sold (total : ℕ) (sold_to_adams : ℕ) (sold_to_lenon : ℕ) : ℕ :=
  total - (sold_to_adams + sold_to_lenon)

/-- Theorem stating that Mrs. Lacson has 45 sweet potatoes not yet sold -/
theorem lacson_unsold_sweet_potatoes : 
  sweet_potatoes_not_sold 80 20 15 = 45 := by
  sorry

end lacson_unsold_sweet_potatoes_l2166_216676


namespace polynomial_sum_l2166_216641

def f (x : ℝ) : ℝ := -4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -6 * x^2 + 4 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 6 * x + 2

theorem polynomial_sum (x : ℝ) : f x + g x + h x = -4 * x^2 + 12 * x - 12 := by
  sorry

end polynomial_sum_l2166_216641


namespace min_games_for_2015_scores_l2166_216669

/-- Represents the scoring system for a football league -/
structure ScoringSystem where
  a : ℝ  -- Points for a win
  b : ℝ  -- Points for a draw
  h : a > b ∧ b > 0

/-- Calculates the number of possible scores after n games -/
def possibleScores (s : ScoringSystem) (n : ℕ) : ℕ :=
  sorry

/-- Theorem stating the minimum number of games for 2015 possible scores -/
theorem min_games_for_2015_scores (s : ScoringSystem) :
  (∀ m : ℕ, m < 62 → possibleScores s m < 2015) ∧
  possibleScores s 62 = 2015 :=
sorry

end min_games_for_2015_scores_l2166_216669


namespace dad_took_90_steps_l2166_216694

/-- The number of steps Dad takes for every 5 steps Masha takes -/
def dad_steps : ℕ := 3

/-- The number of steps Masha takes for every 5 steps Yasha takes -/
def masha_steps : ℕ := 3

/-- The total number of steps Masha and Yasha took together -/
def total_steps : ℕ := 400

/-- Theorem stating that Dad took 90 steps -/
theorem dad_took_90_steps : 
  ∃ (d m y : ℕ), 
    d * 5 = m * dad_steps ∧ 
    m * 5 = y * masha_steps ∧ 
    m + y = total_steps ∧ 
    d = 90 := by sorry

end dad_took_90_steps_l2166_216694


namespace f_satisfies_data_points_l2166_216630

/-- The function f that we want to prove satisfies the given data points -/
def f (x : ℤ) : ℤ := 2 * x^2 + 2 * x - 1

/-- The list of data points given in the table -/
def data_points : List (ℤ × ℤ) := [(1, 3), (2, 11), (3, 23), (4, 39), (5, 59)]

/-- Theorem stating that the function f satisfies all the given data points -/
theorem f_satisfies_data_points : ∀ (point : ℤ × ℤ), point ∈ data_points → f point.1 = point.2 := by
  sorry

#check f_satisfies_data_points

end f_satisfies_data_points_l2166_216630


namespace triangle_side_length_l2166_216661

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a = 3 →
  C = 2 * π / 3 →  -- 120° in radians
  S = (15 * Real.sqrt 3) / 4 →
  S = (1 / 2) * a * b * Real.sin C →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos C) →
  c = 2 := by
  sorry


end triangle_side_length_l2166_216661


namespace six_digit_number_divisibility_l2166_216665

/-- Represents a six-digit number with the given pattern -/
def SixDigitNumber (a b c : Nat) : Nat :=
  100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c

theorem six_digit_number_divisibility 
  (a b c : Nat) 
  (ha : a < 10) 
  (hb : b < 10) 
  (hc : c < 10) : 
  ∃ (k₁ k₂ k₃ : Nat), 
    SixDigitNumber a b c = 7 * k₁ ∧ 
    SixDigitNumber a b c = 13 * k₂ ∧ 
    SixDigitNumber a b c = 11 * k₃ := by
  sorry

#check six_digit_number_divisibility

end six_digit_number_divisibility_l2166_216665


namespace age_ratio_l2166_216634

/-- The ages of John, Mary, and Tonya satisfy certain conditions. -/
def AgeRelations (john mary tonya : ℕ) : Prop :=
  john = tonya / 2 ∧ tonya = 60 ∧ (john + mary + tonya) / 3 = 35

/-- The ratio of John's age to Mary's age is 2:1. -/
theorem age_ratio (john mary tonya : ℕ) 
  (h : AgeRelations john mary tonya) : john = 2 * mary := by
  sorry

end age_ratio_l2166_216634


namespace function_passes_through_point_l2166_216603

theorem function_passes_through_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 2)
  f 2 = 1 := by
  sorry

end function_passes_through_point_l2166_216603


namespace opposite_face_of_A_is_F_l2166_216600

-- Define the set of labels
inductive Label
| A | B | C | D | E | F

-- Define the structure of a cube face
structure CubeFace where
  label : Label

-- Define the structure of a cube
structure Cube where
  faces : List CubeFace
  adjacent : Label → List Label

-- Define the property of being opposite faces
def isOpposite (cube : Cube) (l1 l2 : Label) : Prop :=
  l1 ∉ cube.adjacent l2 ∧ l2 ∉ cube.adjacent l1

-- Theorem statement
theorem opposite_face_of_A_is_F (cube : Cube) 
  (h1 : cube.faces.length = 6)
  (h2 : ∀ l : Label, l ∈ (cube.faces.map CubeFace.label))
  (h3 : cube.adjacent Label.A = [Label.B, Label.C, Label.D, Label.E]) :
  isOpposite cube Label.A Label.F :=
sorry

end opposite_face_of_A_is_F_l2166_216600


namespace jackson_sandwiches_l2166_216644

/-- The number of peanut butter and jelly sandwiches Jackson ate during the school year -/
def sandwiches_eaten (weeks : ℕ) (missed_wednesdays : ℕ) (missed_fridays : ℕ) : ℕ :=
  (weeks - missed_wednesdays) + (weeks - missed_fridays)

/-- Theorem stating that Jackson ate 69 sandwiches during the school year -/
theorem jackson_sandwiches : sandwiches_eaten 36 1 2 = 69 := by
  sorry

end jackson_sandwiches_l2166_216644


namespace deborah_oranges_l2166_216697

theorem deborah_oranges (initial_oranges final_oranges susan_oranges : ℝ) 
  (h1 : initial_oranges = 55.0)
  (h2 : susan_oranges = 35.0)
  (h3 : final_oranges = 90)
  (h4 : final_oranges = initial_oranges + susan_oranges) :
  initial_oranges + susan_oranges - final_oranges = 0 :=
by sorry

end deborah_oranges_l2166_216697


namespace union_equals_reals_l2166_216668

def M : Set ℝ := {x : ℝ | |x| > 2}
def N : Set ℝ := {x : ℝ | x < 3}

theorem union_equals_reals : M ∪ N = Set.univ := by sorry

end union_equals_reals_l2166_216668


namespace ellipse_intersection_product_l2166_216624

/-- Definition of the ellipse C -/
def ellipse_C (x y : ℝ) : Prop := x^2/4 + y^2/3 = 1

/-- Definition of the right focus F2 -/
def right_focus : ℝ × ℝ := (1, 0)

/-- Definition of the left vertex A -/
def left_vertex : ℝ × ℝ := (-2, 0)

/-- Definition of a line passing through a point -/
def line_through (m : ℝ) (p : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - p.2 = m * (x - p.1)

/-- Definition of intersection of a line with x = 4 -/
def intersect_x_4 (m : ℝ) (p : ℝ × ℝ) : ℝ × ℝ :=
  (4, m * (4 - p.1) + p.2)

/-- Main theorem -/
theorem ellipse_intersection_product (l m n : ℝ) (P Q : ℝ × ℝ) :
  line_through l right_focus P.1 P.2 →
  line_through l right_focus Q.1 Q.2 →
  ellipse_C P.1 P.2 →
  ellipse_C Q.1 Q.2 →
  let M := intersect_x_4 m (left_vertex.1, left_vertex.2)
  let N := intersect_x_4 n (left_vertex.1, left_vertex.2)
  line_through m left_vertex P.1 P.2 →
  line_through n left_vertex Q.1 Q.2 →
  M.2 * N.2 = -9 :=
sorry

end ellipse_intersection_product_l2166_216624


namespace fraction_simplification_l2166_216617

theorem fraction_simplification (x : ℝ) : (x + 1) / 3 + (2 - 3 * x) / 2 = (8 - 7 * x) / 6 := by
  sorry

end fraction_simplification_l2166_216617


namespace sixth_fibonacci_is_eight_l2166_216657

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem sixth_fibonacci_is_eight :
  ∃ x, (fibonacci 0 = 1) ∧ 
       (fibonacci 1 = 1) ∧ 
       (fibonacci 2 = 2) ∧ 
       (fibonacci 3 = 3) ∧ 
       (fibonacci 4 = 5) ∧ 
       (fibonacci 5 = x) ∧ 
       (fibonacci 6 = 13) ∧ 
       (x = 8) := by
  sorry

end sixth_fibonacci_is_eight_l2166_216657


namespace totalSavingsIs4440_l2166_216636

-- Define the employees and their properties
structure Employee where
  name : String
  hourlyRate : ℚ
  hoursPerDay : ℚ
  savingRate : ℚ

-- Define the constants
def daysPerWeek : ℚ := 5
def numWeeks : ℚ := 4

-- Define the list of employees
def employees : List Employee := [
  ⟨"Robby", 10, 10, 2/5⟩,
  ⟨"Jaylen", 10, 8, 3/5⟩,
  ⟨"Miranda", 10, 10, 1/2⟩,
  ⟨"Alex", 12, 6, 1/3⟩,
  ⟨"Beth", 15, 4, 1/4⟩,
  ⟨"Chris", 20, 3, 3/4⟩
]

-- Calculate weekly savings for an employee
def weeklySavings (e : Employee) : ℚ :=
  e.hourlyRate * e.hoursPerDay * daysPerWeek * e.savingRate

-- Calculate total savings for all employees over the given number of weeks
def totalSavings : ℚ :=
  (employees.map weeklySavings).sum * numWeeks

-- Theorem statement
theorem totalSavingsIs4440 : totalSavings = 4440 := by
  sorry

end totalSavingsIs4440_l2166_216636


namespace expand_expression_solve_inequality_system_l2166_216699

-- Problem 1
theorem expand_expression (x : ℝ) : (2*x + 1)^2 + x*(x - 4) = 5*x^2 + 1 := by
  sorry

-- Problem 2
theorem solve_inequality_system (x : ℝ) :
  (3*x - 6 > 0 ∧ (5 - x)/2 < 1) ↔ x > 3 := by
  sorry

end expand_expression_solve_inequality_system_l2166_216699


namespace smallest_n_satisfies_conditions_count_non_seven_divisors_l2166_216647

def is_perfect_square (x : ℕ) : Prop := ∃ m : ℕ, x = m^2

def is_perfect_cube (x : ℕ) : Prop := ∃ m : ℕ, x = m^3

def is_perfect_seventh (x : ℕ) : Prop := ∃ m : ℕ, x = m^7

def smallest_n : ℕ := 2^6 * 3^10 * 7^14

theorem smallest_n_satisfies_conditions :
  is_perfect_square (smallest_n / 2) ∧
  is_perfect_cube (smallest_n / 3) ∧
  is_perfect_seventh (smallest_n / 7) := by sorry

theorem count_non_seven_divisors :
  (Finset.filter (fun d => ¬(d % 7 = 0)) (Nat.divisors smallest_n)).card = 77 := by sorry

end smallest_n_satisfies_conditions_count_non_seven_divisors_l2166_216647


namespace greatest_prime_factor_of_5_pow_7_plus_10_pow_6_l2166_216666

theorem greatest_prime_factor_of_5_pow_7_plus_10_pow_6 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (5^7 + 10^6) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (5^7 + 10^6) → q ≤ p :=
by sorry

end greatest_prime_factor_of_5_pow_7_plus_10_pow_6_l2166_216666


namespace function_properties_l2166_216612

def functional_equation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) + f (x - y) = f x * f y

theorem function_properties (f : ℝ → ℝ) 
  (h1 : f 1 = 1) 
  (h2 : functional_equation f) : 
  (f 0 = 2) ∧ 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x : ℝ, f (x + 6) = f x) :=
sorry

end function_properties_l2166_216612


namespace line_y_intercept_l2166_216691

/-- A straight line in the xy-plane with slope 4 passing through (50, 300) has y-intercept 100 -/
theorem line_y_intercept (m : ℝ) (x y b : ℝ) : 
  m = 4 → x = 50 → y = 300 → y = m * x + b → b = 100 := by sorry

end line_y_intercept_l2166_216691


namespace hyperbola_eccentricity_l2166_216689

/-- A hyperbola with foci on the x-axis and asymptotes y = ±√3x has eccentricity 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (h : b / a = Real.sqrt 3) :
  let e := Real.sqrt (1 + (b^2 / a^2))
  e = 2 := by sorry

end hyperbola_eccentricity_l2166_216689


namespace vector_magnitude_difference_l2166_216621

/-- Given two non-zero vectors in ℝ², if their sum is (-3, 6) and their difference is (-3, 2),
    then the difference of their squared magnitudes is 21. -/
theorem vector_magnitude_difference (a b : ℝ × ℝ) (ha : a ≠ (0, 0)) (hb : b ≠ (0, 0)) 
    (hsum : a.1 + b.1 = -3 ∧ a.2 + b.2 = 6) (hdiff : a.1 - b.1 = -3 ∧ a.2 - b.2 = 2) :
    a.1^2 + a.2^2 - (b.1^2 + b.2^2) = 21 := by
  sorry

end vector_magnitude_difference_l2166_216621


namespace james_brothers_cut_sixty_percent_fewer_trees_l2166_216637

/-- The percentage fewer trees James' brothers cut per day compared to James -/
def brothers_percentage_fewer (james_trees_per_day : ℕ) (total_days : ℕ) (james_solo_days : ℕ) (total_trees : ℕ) : ℚ :=
  let total_with_brothers := total_trees - james_trees_per_day * james_solo_days
  let daily_with_brothers := total_with_brothers / (total_days - james_solo_days)
  let brothers_trees_per_day := daily_with_brothers - james_trees_per_day
  (brothers_trees_per_day : ℚ) / james_trees_per_day * 100

theorem james_brothers_cut_sixty_percent_fewer_trees
  (h1 : brothers_percentage_fewer 20 5 2 196 = 60) :
  ∃ (james_trees_per_day : ℕ) (total_days : ℕ) (james_solo_days : ℕ) (total_trees : ℕ),
    james_trees_per_day = 20 ∧
    total_days = 5 ∧
    james_solo_days = 2 ∧
    total_trees = 196 ∧
    brothers_percentage_fewer james_trees_per_day total_days james_solo_days total_trees = 60 :=
by sorry

end james_brothers_cut_sixty_percent_fewer_trees_l2166_216637


namespace gcd_and_bezout_identity_l2166_216673

theorem gcd_and_bezout_identity :
  ∃ (d u v : ℤ), Int.gcd 663 182 = d ∧ d = 663 * u + 182 * v ∧ d = 13 :=
by sorry

end gcd_and_bezout_identity_l2166_216673


namespace specific_cyclic_quadrilateral_radii_l2166_216642

/-- A cyclic quadrilateral with given side lengths --/
structure CyclicQuadrilateral where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  cyclic : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0

/-- The radius of the circumscribed circle of a cyclic quadrilateral --/
def circumradius (q : CyclicQuadrilateral) : ℝ := sorry

/-- The radius of the inscribed circle of a cyclic quadrilateral --/
def inradius (q : CyclicQuadrilateral) : ℝ := sorry

/-- Theorem about the radii of circumscribed and inscribed circles for a specific cyclic quadrilateral --/
theorem specific_cyclic_quadrilateral_radii :
  ∃ (q : CyclicQuadrilateral),
    q.a = 36 ∧ q.b = 91 ∧ q.c = 315 ∧ q.d = 260 ∧
    circumradius q = 162.5 ∧
    inradius q = 140 / 3 := by sorry

end specific_cyclic_quadrilateral_radii_l2166_216642


namespace mary_shirts_left_l2166_216664

def blue_shirts : ℕ := 30
def brown_shirts : ℕ := 40
def red_shirts : ℕ := 20
def yellow_shirts : ℕ := 25

def blue_fraction : ℚ := 3/5
def brown_fraction : ℚ := 1/4
def red_fraction : ℚ := 2/3
def yellow_fraction : ℚ := 1/5

def shirts_left : ℕ := 69

theorem mary_shirts_left : 
  blue_shirts - Int.floor (blue_fraction * blue_shirts) +
  brown_shirts - Int.floor (brown_fraction * brown_shirts) +
  red_shirts - Int.floor (red_fraction * red_shirts) +
  yellow_shirts - Int.floor (yellow_fraction * yellow_shirts) = shirts_left := by
  sorry

end mary_shirts_left_l2166_216664


namespace rhombus_longer_diagonal_l2166_216625

/-- A rhombus with side length 34 units and shorter diagonal 32 units has a longer diagonal of 60 units. -/
theorem rhombus_longer_diagonal (side : ℝ) (shorter_diagonal : ℝ) (longer_diagonal : ℝ) : 
  side = 34 → shorter_diagonal = 32 → longer_diagonal = 60 := by
  sorry

end rhombus_longer_diagonal_l2166_216625


namespace largest_divisor_of_n_l2166_216609

theorem largest_divisor_of_n (n : ℕ) (hn : n > 0) (h_div : 360 ∣ n^3) :
  ∃ (w : ℕ), w = 30 ∧ w ∣ n ∧ ∀ (k : ℕ), k ∣ n → k ≤ w :=
sorry

end largest_divisor_of_n_l2166_216609


namespace ninth_grade_science_only_l2166_216611

/-- Represents the set of all ninth-grade students -/
def NinthGrade : Finset Nat := sorry

/-- Represents the set of students in the science class -/
def ScienceClass : Finset Nat := sorry

/-- Represents the set of students in the history class -/
def HistoryClass : Finset Nat := sorry

theorem ninth_grade_science_only :
  (NinthGrade.card = 120) →
  (ScienceClass.card = 85) →
  (HistoryClass.card = 75) →
  (NinthGrade = ScienceClass ∪ HistoryClass) →
  ((ScienceClass \ HistoryClass).card = 45) := by
  sorry

end ninth_grade_science_only_l2166_216611


namespace floor_of_7_8_l2166_216682

theorem floor_of_7_8 : ⌊(7.8 : ℝ)⌋ = 7 := by sorry

end floor_of_7_8_l2166_216682


namespace percentage_qualified_school_B_l2166_216620

/-- Percentage of students qualified from school A -/
def percentage_qualified_A : ℝ := 70

/-- Ratio of students appeared in school B compared to school A -/
def ratio_appeared_B_to_A : ℝ := 1.2

/-- Ratio of students qualified from school B compared to school A -/
def ratio_qualified_B_to_A : ℝ := 1.5

/-- Theorem: The percentage of students qualified from school B is 87.5% -/
theorem percentage_qualified_school_B :
  (ratio_qualified_B_to_A * percentage_qualified_A) / (ratio_appeared_B_to_A * 100) * 100 = 87.5 := by
  sorry

end percentage_qualified_school_B_l2166_216620


namespace tiangong_altitude_scientific_notation_l2166_216649

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem tiangong_altitude_scientific_notation :
  toScientificNotation 375000 = ScientificNotation.mk 3.75 5 (by norm_num) :=
sorry

end tiangong_altitude_scientific_notation_l2166_216649


namespace triangle_area_is_50_l2166_216698

/-- A square with side length 10 and lower left vertex at (0, 10) -/
structure Square where
  side : ℝ
  lower_left : ℝ × ℝ
  h_side : side = 10
  h_lower_left : lower_left = (0, 10)

/-- An isosceles triangle with base 10 on y-axis and lower right vertex at (0, 10) -/
structure IsoscelesTriangle where
  base : ℝ
  lower_right : ℝ × ℝ
  h_base : base = 10
  h_lower_right : lower_right = (0, 10)

/-- The area of the triangle formed by connecting the top vertex of the isosceles triangle
    to the top left vertex of the square -/
def triangle_area (s : Square) (t : IsoscelesTriangle) : ℝ :=
  sorry

/-- Theorem stating that the area of the formed triangle is 50 square units -/
theorem triangle_area_is_50 (s : Square) (t : IsoscelesTriangle) :
  triangle_area s t = 50 := by
  sorry

end triangle_area_is_50_l2166_216698


namespace solution_set_implies_a_value_l2166_216627

theorem solution_set_implies_a_value 
  (h : ∀ x : ℝ, -1 < x ∧ x < 2 ↔ -1/2 * x^2 + a * x > -1) : 
  a = 1/2 := by sorry

end solution_set_implies_a_value_l2166_216627


namespace remainder_problem_l2166_216646

theorem remainder_problem : (123456789012 : ℕ) % 252 = 228 := by
  sorry

end remainder_problem_l2166_216646


namespace consecutive_integers_sum_l2166_216622

theorem consecutive_integers_sum (n : ℤ) : 
  n * (n + 1) * (n + 2) = 504 → n + (n + 1) + (n + 2) = 24 := by
  sorry

end consecutive_integers_sum_l2166_216622


namespace total_eggs_supplied_in_week_l2166_216690

/-- Represents the number of eggs in a dozen --/
def dozen : ℕ := 12

/-- Represents the number of days in a week --/
def daysInWeek : ℕ := 7

/-- Represents the number of weekdays in a week --/
def weekdaysInWeek : ℕ := 5

/-- Represents the number of odd days in a week --/
def oddDaysInWeek : ℕ := 3

/-- Represents the number of even days in a week --/
def evenDaysInWeek : ℕ := 4

/-- Represents the daily egg supply to the first store --/
def firstStoreSupply : ℕ := 5 * dozen

/-- Represents the daily egg supply to the second store on weekdays --/
def secondStoreSupply : ℕ := 30

/-- Represents the egg supply to the third store on odd days --/
def thirdStoreOddSupply : ℕ := 25 * dozen

/-- Represents the egg supply to the third store on even days --/
def thirdStoreEvenSupply : ℕ := 15 * dozen

/-- Theorem stating the total number of eggs supplied in a week --/
theorem total_eggs_supplied_in_week :
  firstStoreSupply * daysInWeek +
  secondStoreSupply * weekdaysInWeek +
  thirdStoreOddSupply * oddDaysInWeek +
  thirdStoreEvenSupply * evenDaysInWeek = 2190 := by
  sorry

end total_eggs_supplied_in_week_l2166_216690


namespace line_tangent_to_parabola_l2166_216623

/-- The value of a for which the line x - y - 1 = 0 is tangent to the parabola y = ax² --/
theorem line_tangent_to_parabola :
  ∃! (a : ℝ), ∀ (x y : ℝ),
    (x - y - 1 = 0 ∧ y = a * x^2) →
    (∃! p : ℝ × ℝ, p.1 - p.2 - 1 = 0 ∧ p.2 = a * p.1^2) ∧
    a = 1/4 :=
sorry

end line_tangent_to_parabola_l2166_216623


namespace ratio_of_squares_difference_l2166_216643

theorem ratio_of_squares_difference : 
  (1523^2 - 1517^2) / (1530^2 - 1510^2) = 3/10 := by
  sorry

end ratio_of_squares_difference_l2166_216643


namespace least_three_digit_multiple_of_13_l2166_216688

theorem least_three_digit_multiple_of_13 : ∃ n : ℕ, 
  n = 104 ∧ 
  n % 13 = 0 ∧ 
  100 ≤ n ∧ n < 1000 ∧ 
  ∀ m : ℕ, (m % 13 = 0 ∧ 100 ≤ m ∧ m < 1000) → n ≤ m :=
sorry

end least_three_digit_multiple_of_13_l2166_216688


namespace systematic_sample_theorem_l2166_216608

/-- Represents a systematic sampling scenario -/
structure SystematicSample where
  population : ℕ
  sample_size : ℕ
  interval_start : ℕ
  interval_end : ℕ
  (population_positive : 0 < population)
  (sample_size_positive : 0 < sample_size)
  (sample_size_le_population : sample_size ≤ population)
  (interval_valid : interval_start ≤ interval_end)
  (interval_in_population : interval_end ≤ population)

/-- Calculates the number of selected individuals in the given interval -/
def selected_in_interval (s : SystematicSample) : ℕ :=
  ((s.interval_end - s.interval_start + 1) * s.sample_size + s.population - 1) / s.population

/-- Theorem stating that for the given systematic sample, 11 individuals are selected from the interval -/
theorem systematic_sample_theorem (s : SystematicSample) 
  (h1 : s.population = 640)
  (h2 : s.sample_size = 32)
  (h3 : s.interval_start = 161)
  (h4 : s.interval_end = 380) : 
  selected_in_interval s = 11 := by
  sorry

end systematic_sample_theorem_l2166_216608


namespace min_colors_for_distribution_centers_l2166_216696

theorem min_colors_for_distribution_centers : ∃ (n : ℕ),
  (n ≥ 5) ∧
  (n + n.choose 2 ≥ 12) ∧
  (∀ m : ℕ, m < n → m + m.choose 2 < 12) := by
  sorry

end min_colors_for_distribution_centers_l2166_216696


namespace function_values_unbounded_l2166_216632

/-- A function satisfying the given identity -/
def SatisfiesIdentity (f : ℤ × ℤ → ℤ) : Prop :=
  ∀ n m : ℤ, f (n, m) = (f (n - 1, m) + f (n + 1, m) + f (n, m - 1) + f (n, m + 1)) / 4

/-- The main theorem -/
theorem function_values_unbounded
  (f : ℤ × ℤ → ℤ)
  (h_satisfies : SatisfiesIdentity f)
  (h_nonconstant : ∃ (a b c d : ℤ), f (a, b) ≠ f (c, d)) :
  ∀ k : ℤ, (∃ n m : ℤ, f (n, m) > k) ∧ (∃ n m : ℤ, f (n, m) < k) :=
sorry

end function_values_unbounded_l2166_216632


namespace papayas_theorem_l2166_216686

def remaining_green_papayas (initial : ℕ) (friday_yellow : ℕ) : ℕ :=
  initial - friday_yellow - (2 * friday_yellow)

theorem papayas_theorem :
  remaining_green_papayas 14 2 = 8 := by
  sorry

end papayas_theorem_l2166_216686


namespace triangle_properties_l2166_216613

/-- An acute triangle with sides a, b, c -/
structure AcuteTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  acute : a > 0 ∧ b > 0 ∧ c > 0
  cosine_law : b^2 = a^2 + c^2 - a*c

/-- The perimeter of a triangle -/
def perimeter (t : AcuteTriangle) : ℝ := t.a + t.b + t.c

/-- The area of a triangle -/
def area (t : AcuteTriangle) : ℝ := sorry

theorem triangle_properties (t : AcuteTriangle) :
  (∃ angleA : ℝ, angleA = 60 * (π / 180) ∧ t.c = 2 → t.a = 2) ∧
  (area t = 2 * Real.sqrt 3 →
    6 * Real.sqrt 2 ≤ perimeter t ∧ perimeter t < 6 + 2 * Real.sqrt 3) :=
sorry

end triangle_properties_l2166_216613


namespace base_score_per_round_l2166_216671

theorem base_score_per_round 
  (total_rounds : ℕ) 
  (total_points : ℕ) 
  (bonus_points : ℕ) 
  (penalty_points : ℕ) 
  (h1 : total_rounds = 5)
  (h2 : total_points = 370)
  (h3 : bonus_points = 50)
  (h4 : penalty_points = 30) :
  (total_points - bonus_points + penalty_points) / total_rounds = 70 := by
sorry

end base_score_per_round_l2166_216671


namespace roberts_and_marias_ages_l2166_216640

theorem roberts_and_marias_ages (robert maria : ℕ) : 
  robert = maria + 8 →
  robert + 5 = 3 * (maria - 3) →
  robert + maria = 30 :=
by sorry

end roberts_and_marias_ages_l2166_216640


namespace hyperbola_property_l2166_216652

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 3 = 1

-- Define the foci
def left_focus : ℝ × ℝ := sorry
def right_focus : ℝ × ℝ := sorry

-- Define a line passing through the left focus
def line_through_left_focus (x y : ℝ) : Prop := sorry

-- Define the left branch of the hyperbola
def left_branch (x y : ℝ) : Prop := hyperbola x y ∧ x < 0

-- Define the intersection points
def point_M : ℝ × ℝ := sorry
def point_N : ℝ × ℝ := sorry

-- State the theorem
theorem hyperbola_property :
  hyperbola point_M.1 point_M.2 ∧
  hyperbola point_N.1 point_N.2 ∧
  left_branch point_M.1 point_M.2 ∧
  left_branch point_N.1 point_N.2 ∧
  line_through_left_focus point_M.1 point_M.2 ∧
  line_through_left_focus point_N.1 point_N.2
  →
  abs (dist point_M right_focus) + abs (dist point_N right_focus) - abs (dist point_M point_N) = 8 :=
sorry

end hyperbola_property_l2166_216652


namespace gcd_triple_characterization_l2166_216614

theorem gcd_triple_characterization (a b c : ℕ+) :
  (Nat.gcd a.val 20 = b.val ∧ Nat.gcd b.val 15 = c.val ∧ Nat.gcd a.val c.val = 5) ↔
  (∃ t : ℕ+, (a = 20 * t ∧ b = 20 ∧ c = 5) ∨
             (a = 20 * t - 10 ∧ b = 10 ∧ c = 5) ∨
             (a = 10 * t - 5 ∧ b = 5 ∧ c = 5)) :=
by sorry


end gcd_triple_characterization_l2166_216614


namespace function_is_zero_l2166_216663

/-- A function satisfying the given functional equation is the zero function -/
theorem function_is_zero (f : ℝ → ℝ) 
    (h : ∀ x y : ℝ, f (x * f y + 2 * x) = x * y + 2 * f x) : 
  ∀ x : ℝ, f x = 0 := by
  sorry

end function_is_zero_l2166_216663


namespace toy_sale_analysis_l2166_216604

-- Define the cost price
def cost_price : ℝ := 20

-- Define the maximum profit percentage
def max_profit_percentage : ℝ := 0.3

-- Define the linear relationship between weekly sales volume and selling price
def sales_volume (x : ℝ) : ℝ := -10 * x + 300

-- Define the profit function
def profit (x : ℝ) : ℝ := (x - cost_price) * sales_volume x

-- Theorem statement
theorem toy_sale_analysis :
  -- Part 1: Verify the linear relationship
  (sales_volume 22 = 80 ∧ sales_volume 24 = 60) ∧
  -- Part 2: Verify the selling price for 210 yuan profit
  (∃ x : ℝ, x ≤ cost_price * (1 + max_profit_percentage) ∧ profit x = 210 ∧ x = 23) ∧
  -- Part 3: Verify the maximum profit
  (∃ x : ℝ, x = 25 ∧ profit x = 250 ∧ ∀ y : ℝ, profit y ≤ profit x) := by
  sorry


end toy_sale_analysis_l2166_216604


namespace parallel_line_through_point_line_equation_proof_l2166_216629

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a point (x, y) lies on a given line -/
def Line.contains (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y + l.c = 0

/-- Check if two lines are parallel -/
def Line.parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

theorem parallel_line_through_point (given_line : Line) (point : ℝ × ℝ) :
  ∃ (parallel_line : Line),
    parallel_line.contains point.1 point.2 ∧
    Line.parallel parallel_line given_line :=
by
  sorry

/-- The main theorem to prove -/
theorem line_equation_proof :
  let given_line : Line := { a := 1, b := -2, c := -2 }
  let point : ℝ × ℝ := (1, 1)
  let parallel_line : Line := { a := 1, b := -2, c := 1 }
  parallel_line.contains point.1 point.2 ∧
  Line.parallel parallel_line given_line :=
by
  sorry

end parallel_line_through_point_line_equation_proof_l2166_216629


namespace multiplication_equality_l2166_216626

theorem multiplication_equality : 469157 * 9999 = 4691116843 := by
  sorry

end multiplication_equality_l2166_216626


namespace calculate_total_profit_total_profit_is_4600_l2166_216650

/-- Calculates the total profit given the investments, time periods, and Rajan's profit share -/
theorem calculate_total_profit (rajan_investment : ℕ) (rakesh_investment : ℕ) (mukesh_investment : ℕ)
  (rajan_months : ℕ) (rakesh_months : ℕ) (mukesh_months : ℕ) (rajan_profit : ℕ) : ℕ :=
  let rajan_share := rajan_investment * rajan_months
  let rakesh_share := rakesh_investment * rakesh_months
  let mukesh_share := mukesh_investment * mukesh_months
  let total_share := rajan_share + rakesh_share + mukesh_share
  let total_profit := (rajan_profit * total_share) / rajan_share
  total_profit

/-- Proves that the total profit is 4600 given the specific investments and Rajan's profit share -/
theorem total_profit_is_4600 :
  calculate_total_profit 20000 25000 15000 12 4 8 2400 = 4600 := by
  sorry

end calculate_total_profit_total_profit_is_4600_l2166_216650


namespace mathematicians_ages_l2166_216659

/-- Represents a mathematician --/
inductive Mathematician
| A
| B
| C

/-- Calculates the age of mathematician A or C given the base and smallest number --/
def calculate_age_A_C (base : ℕ) (smallest : ℕ) : ℕ :=
  smallest * base + (smallest + 2)

/-- Calculates the age of mathematician B given the base and smallest number --/
def calculate_age_B (base : ℕ) (smallest : ℕ) : ℕ :=
  smallest * base + (smallest + 1)

/-- Checks if the calculated age matches the product of the two largest numbers --/
def check_age_A_C (age : ℕ) (smallest : ℕ) : Prop :=
  age = (smallest + 4) * (smallest + 6)

/-- Checks if the calculated age matches the product of the next two consecutive numbers --/
def check_age_B (age : ℕ) (smallest : ℕ) : Prop :=
  age = (smallest + 2) * (smallest + 3)

theorem mathematicians_ages :
  ∃ (age_A age_B age_C : ℕ) (base_A base_B : ℕ) (smallest_A smallest_B smallest_C : ℕ),
    calculate_age_A_C base_A smallest_A = age_A ∧
    calculate_age_B base_B smallest_B = age_B ∧
    calculate_age_A_C base_A smallest_C = age_C ∧
    check_age_A_C age_A smallest_A ∧
    check_age_B age_B smallest_B ∧
    check_age_A_C age_C smallest_C ∧
    age_C < age_A ∧
    age_C < age_B ∧
    age_A = 48 ∧
    age_B = 56 ∧
    age_C = 35 ∧
    base_B = 10 :=
  by sorry

/-- Identifies the absent-minded mathematician --/
def absent_minded : Mathematician := Mathematician.B

end mathematicians_ages_l2166_216659


namespace molecular_weight_K2Cr2O7_is_296_l2166_216658

/-- The molecular weight of K2Cr2O7 in g/mole -/
def molecular_weight_K2Cr2O7 : ℝ := 296

/-- The number of moles given in the problem -/
def given_moles : ℝ := 4

/-- The total weight of the given moles in grams -/
def total_weight : ℝ := 1184

/-- Theorem stating that the molecular weight of K2Cr2O7 is 296 g/mole -/
theorem molecular_weight_K2Cr2O7_is_296 :
  molecular_weight_K2Cr2O7 = total_weight / given_moles :=
by sorry

end molecular_weight_K2Cr2O7_is_296_l2166_216658


namespace complex_fraction_simplification_l2166_216679

theorem complex_fraction_simplification :
  let numerator := (11/4) / ((11/10) + (10/3))
  let denominator := 5/2 - 4/3
  let left_fraction := numerator / denominator
  let right_fraction := 5/7 - ((13/6 + 9/2) * 3/8) / (11/4 - 3/2)
  left_fraction / right_fraction = -35/9 := by sorry

end complex_fraction_simplification_l2166_216679


namespace investment_ratio_l2166_216615

/-- Prove that given the conditions, the ratio of B's investment to C's investment is 2:3 -/
theorem investment_ratio (a_invest b_invest c_invest : ℚ) 
  (h1 : a_invest = 3 * b_invest)
  (h2 : ∃ f : ℚ, b_invest = f * c_invest)
  (h3 : b_invest / (a_invest + b_invest + c_invest) * 7700 = 1400) :
  b_invest / c_invest = 2 / 3 := by
  sorry

end investment_ratio_l2166_216615


namespace largest_three_digit_divisible_by_5_8_2_l2166_216674

theorem largest_three_digit_divisible_by_5_8_2 :
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ n % 5 = 0 ∧ n % 8 = 0 → n ≤ 960 :=
by sorry

end largest_three_digit_divisible_by_5_8_2_l2166_216674
