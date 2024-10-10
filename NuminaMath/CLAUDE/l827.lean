import Mathlib

namespace slope_range_l827_82732

/-- A line passing through (0,2) that intersects the circle (x-2)^2 + (y-2)^2 = 1 -/
structure IntersectingLine where
  slope : ℝ
  passes_through_origin : (0 : ℝ) = slope * 0 + 2
  intersects_circle : ∃ (x y : ℝ), y = slope * x + 2 ∧ (x - 2)^2 + (y - 2)^2 = 1

/-- The theorem stating the range of possible slopes for the intersecting line -/
theorem slope_range (l : IntersectingLine) : 
  l.slope ∈ Set.Icc (-Real.sqrt 3 / 3) (Real.sqrt 3 / 3) :=
sorry

end slope_range_l827_82732


namespace number_equation_solution_l827_82799

theorem number_equation_solution : ∃ x : ℚ, 3 + (1/2) * (1/3) * (1/5) * x = (1/15) * x ∧ x = 90 := by
  sorry

end number_equation_solution_l827_82799


namespace power_of_64_l827_82729

theorem power_of_64 : (64 : ℝ) ^ (5/3) = 1024 := by
  sorry

end power_of_64_l827_82729


namespace existence_of_m_n_l827_82737

theorem existence_of_m_n (k : ℕ) : 
  (∃ m n : ℕ, m * (m + k) = n * (n + 1)) ↔ k ≠ 2 ∧ k ≠ 3 :=
by sorry

end existence_of_m_n_l827_82737


namespace smallest_digit_for_divisibility_l827_82753

def is_divisible_by_3 (n : ℕ) : Prop :=
  n % 3 = 0

def digit_to_nat (d : ℕ) : ℕ :=
  526000 + d * 100 + 84

theorem smallest_digit_for_divisibility :
  ∃ (d : ℕ), d < 10 ∧ is_divisible_by_3 (digit_to_nat d) ∧
  ∀ (d' : ℕ), d' < d → ¬is_divisible_by_3 (digit_to_nat d') :=
by sorry

end smallest_digit_for_divisibility_l827_82753


namespace function_inequality_l827_82755

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f y < f x

theorem function_inequality (f : ℝ → ℝ) 
  (h_even : is_even_function f)
  (h_periodic : ∀ x, f (x + 4) = -f x)
  (h_decreasing : is_decreasing_on f 0 4) :
  f 13 < f 10 ∧ f 10 < f 15 := by
  sorry

end function_inequality_l827_82755


namespace fraction_problem_l827_82787

theorem fraction_problem (F : ℝ) :
  (0.4 * F * 150 = 36) → F = 0.6 := by
  sorry

end fraction_problem_l827_82787


namespace largest_n_for_integer_factors_l827_82717

def polynomial (n : ℤ) (x : ℤ) : ℤ := 3 * x^2 + n * x + 72

def has_integer_linear_factors (n : ℤ) : Prop :=
  ∃ (a b : ℤ), ∀ x, polynomial n x = (3*x + a) * (x + b)

theorem largest_n_for_integer_factors :
  (∃ n : ℤ, has_integer_linear_factors n) ∧
  (∀ m : ℤ, has_integer_linear_factors m → m ≤ 217) ∧
  has_integer_linear_factors 217 :=
sorry

end largest_n_for_integer_factors_l827_82717


namespace new_student_info_is_unique_l827_82756

-- Define the possible values for each attribute
inductive Surname
  | Ji | Zhang | Chen | Huang
deriving Repr, DecidableEq

inductive Gender
  | Male | Female
deriving Repr, DecidableEq

inductive Specialty
  | Singing | Dancing | Drawing
deriving Repr, DecidableEq

-- Define a structure for student information
structure StudentInfo where
  surname : Surname
  gender : Gender
  totalScore : Nat
  specialty : Specialty
deriving Repr

-- Define the information provided by each classmate
def classmate_A : StudentInfo := ⟨Surname.Ji, Gender.Male, 260, Specialty.Singing⟩
def classmate_B : StudentInfo := ⟨Surname.Zhang, Gender.Female, 220, Specialty.Dancing⟩
def classmate_C : StudentInfo := ⟨Surname.Chen, Gender.Male, 260, Specialty.Singing⟩
def classmate_D : StudentInfo := ⟨Surname.Huang, Gender.Female, 220, Specialty.Drawing⟩
def classmate_E : StudentInfo := ⟨Surname.Zhang, Gender.Female, 240, Specialty.Singing⟩

-- Define the correct information
def correct_info : StudentInfo := ⟨Surname.Huang, Gender.Male, 240, Specialty.Dancing⟩

-- Define a function to check if a piece of information is correct
def is_correct_piece (info : StudentInfo) (correct : StudentInfo) : Bool :=
  info.surname = correct.surname ∨ 
  info.gender = correct.gender ∨ 
  info.totalScore = correct.totalScore ∨ 
  info.specialty = correct.specialty

-- Theorem statement
theorem new_student_info_is_unique :
  (is_correct_piece classmate_A correct_info) ∧
  (is_correct_piece classmate_B correct_info) ∧
  (is_correct_piece classmate_C correct_info) ∧
  (is_correct_piece classmate_D correct_info) ∧
  (is_correct_piece classmate_E correct_info) ∧
  (∀ info : StudentInfo, 
    info ≠ correct_info → 
    (¬(is_correct_piece classmate_A info) ∨
     ¬(is_correct_piece classmate_B info) ∨
     ¬(is_correct_piece classmate_C info) ∨
     ¬(is_correct_piece classmate_D info) ∨
     ¬(is_correct_piece classmate_E info))) :=
by sorry

end new_student_info_is_unique_l827_82756


namespace sum_of_squares_geometric_progression_l827_82733

theorem sum_of_squares_geometric_progression
  (a r : ℝ) 
  (h1 : -1 < r ∧ r < 1) 
  (h2 : ∃ (S : ℝ), S = a / (1 - r)) : 
  ∃ (T : ℝ), T = a^2 / (1 - r^2) := by
  sorry

end sum_of_squares_geometric_progression_l827_82733


namespace derivative_sqrt_l827_82796

theorem derivative_sqrt (x : ℝ) (h : x > 0) :
  deriv (fun x => Real.sqrt x) x = 1 / (2 * Real.sqrt x) := by sorry

end derivative_sqrt_l827_82796


namespace min_value_theorem_l827_82739

-- Define the quadratic function
def f (a c x : ℝ) : ℝ := a * x^2 - 2 * x + c

-- State the theorem
theorem min_value_theorem (a c : ℝ) (h1 : a > 0) (h2 : c > 0) 
  (h3 : Set.range (f a c) = Set.Ici 0) :
  (∀ x : ℝ, 9 / a + 1 / c ≥ 6) ∧ (∃ x : ℝ, 9 / a + 1 / c = 6) :=
sorry

end min_value_theorem_l827_82739


namespace total_book_price_l827_82702

/-- Given the following conditions:
  - Total number of books: 90
  - Math books cost: $4 each
  - History books cost: $5 each
  - Number of math books: 60
  Prove that the total price of all books is $390 -/
theorem total_book_price (total_books : Nat) (math_book_price history_book_price : Nat) (math_books : Nat) :
  total_books = 90 →
  math_book_price = 4 →
  history_book_price = 5 →
  math_books = 60 →
  math_books * math_book_price + (total_books - math_books) * history_book_price = 390 := by
  sorry

#check total_book_price

end total_book_price_l827_82702


namespace area_of_ABCD_l827_82758

/-- Represents a rectangle with length and height -/
structure Rectangle where
  length : ℝ
  height : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.height

/-- Theorem: Area of rectangle ABCD -/
theorem area_of_ABCD (r1 r2 r3 : Rectangle) (ABCD : Rectangle) :
  area r1 + area r2 + area r3 = area ABCD →
  area r1 = 2 →
  ABCD.length = 5 →
  ABCD.height = 3 →
  area ABCD = 15 := by
  sorry

#check area_of_ABCD

end area_of_ABCD_l827_82758


namespace isosceles_triangle_angles_l827_82705

-- Define the triangle
structure IsoscelesTriangle where
  base_angle : Real
  inscribed_square_side : Real
  inscribed_circle_radius : Real

-- Define the conditions
def triangle_conditions (t : IsoscelesTriangle) : Prop :=
  t.inscribed_square_side / t.inscribed_circle_radius = 8 / 5

-- Theorem statement
theorem isosceles_triangle_angles (t : IsoscelesTriangle) 
  (h : triangle_conditions t) : 
  t.base_angle = 2 * Real.arctan (1 / 2) ∧ 
  π - 2 * t.base_angle = π - 4 * Real.arctan (1 / 2) := by
  sorry

#check isosceles_triangle_angles

end isosceles_triangle_angles_l827_82705


namespace symmetry_axis_l827_82715

-- Define a function f with the given property
def f : ℝ → ℝ := sorry

-- State the property of f
axiom f_property : ∀ x : ℝ, f x = f (3 - x)

-- Define the concept of an axis of symmetry
def is_axis_of_symmetry (a : ℝ) (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- Theorem statement
theorem symmetry_axis :
  is_axis_of_symmetry (3/2) f :=
sorry

end symmetry_axis_l827_82715


namespace max_value_of_function_l827_82795

theorem max_value_of_function (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2) :
  ∃ (max_y : ℝ), max_y = 5/2 ∧ 
  ∀ y : ℝ, y = 2^(2*x - 1) - 3 * 2^x + 5 → y ≤ max_y :=
by sorry

end max_value_of_function_l827_82795


namespace chessboard_coloring_theorem_l827_82712

/-- Represents a color (Red or Blue) -/
inductive Color
| Red
| Blue

/-- Represents a 4 x 7 chessboard coloring -/
def Coloring := Fin 4 → Fin 7 → Color

/-- Represents a rectangle on the chessboard -/
structure Rectangle where
  top_left : Fin 4 × Fin 7
  bottom_right : Fin 4 × Fin 7

/-- Check if a rectangle has all corners of the same color -/
def has_same_color_corners (c : Coloring) (r : Rectangle) : Prop :=
  let (t, l) := r.top_left
  let (b, r) := r.bottom_right
  c t l = c t r ∧ c t l = c b l ∧ c t l = c b r

/-- Main theorem: For any coloring of a 4 x 7 chessboard, 
    there exists a rectangle with four corners of the same color -/
theorem chessboard_coloring_theorem :
  ∀ (c : Coloring), ∃ (r : Rectangle), has_same_color_corners c r :=
sorry

end chessboard_coloring_theorem_l827_82712


namespace least_fourth_integer_l827_82700

theorem least_fourth_integer (a b c d : ℕ+) : 
  (a + b + c + d : ℚ) / 4 = 18 →
  a = 3 * b →
  b = c - 2 →
  (c : ℚ) = 1.5 * d →
  d ≥ 10 ∧ ∀ x : ℕ+, x < 10 → 
    ¬∃ a' b' c' : ℕ+, (a' + b' + c' + x : ℚ) / 4 = 18 ∧
                      a' = 3 * b' ∧
                      b' = c' - 2 ∧
                      (c' : ℚ) = 1.5 * x := by
  sorry

#check least_fourth_integer

end least_fourth_integer_l827_82700


namespace rectangle_circle_square_area_l827_82759

theorem rectangle_circle_square_area : 
  ∀ (r : ℝ) (l w : ℝ),
    r = 7 →  -- Circle radius
    l = 3 * w →  -- Rectangle length to width ratio
    2 * r = w →  -- Circle diameter equals rectangle width
    l * w + 2 * r^2 = 686 :=  -- Total area of rectangle and square
by
  sorry

end rectangle_circle_square_area_l827_82759


namespace angle_equivalence_same_quadrant_as_2016_l827_82777

theorem angle_equivalence (θ : ℝ) : 
  θ ≡ (θ % 360) [PMOD 360] :=
sorry

theorem same_quadrant_as_2016 : 
  (2016 : ℝ) % 360 = 216 :=
sorry

end angle_equivalence_same_quadrant_as_2016_l827_82777


namespace sum_of_coefficients_l827_82789

theorem sum_of_coefficients (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f (x + 4) = 4 * x^2 + 9 * x + 5) →
  (∀ x, f x = a * x^2 + b * x + c) →
  a + b + c = 14 := by
sorry

end sum_of_coefficients_l827_82789


namespace abs_rational_inequality_l827_82748

theorem abs_rational_inequality (x : ℝ) : 
  |((3 * x - 2) / (x - 2))| > 3 ↔ x ∈ Set.Ioo (4/3) 2 ∪ Set.Ioi 2 :=
sorry

end abs_rational_inequality_l827_82748


namespace unattainable_y_value_l827_82765

theorem unattainable_y_value (x : ℝ) (h : x ≠ -5/4) :
  ∀ y : ℝ, y = (2 - 3*x) / (4*x + 5) → y ≠ -3/4 := by
sorry

end unattainable_y_value_l827_82765


namespace mayoral_election_votes_l827_82744

theorem mayoral_election_votes (Z : ℕ) (hZ : Z = 25000) :
  let Y := (3 / 5 : ℚ) * Z
  let X := (8 / 5 : ℚ) * Y
  X = 24000 := by
  sorry

end mayoral_election_votes_l827_82744


namespace minimum_races_for_top_three_l827_82701

/-- Represents a horse in the race -/
structure Horse : Type :=
  (id : Nat)

/-- Represents a race with at most 5 horses -/
structure Race : Type :=
  (participants : Finset Horse)
  (size_constraint : participants.card ≤ 5)

/-- The set of all horses -/
def all_horses : Finset Horse := sorry

/-- The proposition that a given number of races is sufficient to determine the top 3 fastest horses -/
def can_determine_top_three (n : Nat) : Prop := sorry

/-- The proposition that a given number of races is necessary to determine the top 3 fastest horses -/
def is_necessary (n : Nat) : Prop := sorry

theorem minimum_races_for_top_three :
  (all_horses.card = 25) →
  (can_determine_top_three 7) ∧
  (∀ m : Nat, m < 7 → ¬(can_determine_top_three m)) ∧
  (is_necessary 7) :=
sorry

end minimum_races_for_top_three_l827_82701


namespace vector_parallel_and_perpendicular_l827_82722

def a (x : ℝ) : Fin 2 → ℝ := ![x, x + 2]
def b : Fin 2 → ℝ := ![1, 2]

theorem vector_parallel_and_perpendicular :
  (∃ (k : ℝ), a 2 = k • b) ∧
  (a (1/3) - b) • b = 0 := by sorry

end vector_parallel_and_perpendicular_l827_82722


namespace arithmetic_sequence_8th_term_l827_82775

/-- Given an arithmetic sequence of 25 terms with first term 7 and last term 98,
    prove that the 8th term is equal to 343/12. -/
theorem arithmetic_sequence_8th_term :
  ∀ (a : ℕ → ℚ),
    (∀ i j, a (i + 1) - a i = a (j + 1) - a j) →  -- arithmetic sequence
    (a 0 = 7) →                                   -- first term is 7
    (a 24 = 98) →                                 -- last term is 98
    (a 7 = 343 / 12) :=                           -- 8th term (index 7) is 343/12
by
  sorry

end arithmetic_sequence_8th_term_l827_82775


namespace spaceship_total_distance_l827_82723

/-- The distance traveled by a spaceship between three locations -/
def spaceship_distance (earth_to_x : ℝ) (x_to_y : ℝ) (y_to_earth : ℝ) : ℝ :=
  earth_to_x + x_to_y + y_to_earth

/-- Theorem: The total distance traveled by the spaceship is 0.7 light-years -/
theorem spaceship_total_distance :
  spaceship_distance 0.5 0.1 0.1 = 0.7 := by
  sorry

#eval spaceship_distance 0.5 0.1 0.1

end spaceship_total_distance_l827_82723


namespace marble_game_solution_l827_82740

/-- Represents a player in the game -/
inductive Player
| A
| B
| C

/-- Represents the game state -/
structure GameState where
  p : ℕ
  q : ℕ
  r : ℕ
  rounds : ℕ
  final_marbles : Player → ℕ
  b_last_round : ℕ

/-- The theorem statement -/
theorem marble_game_solution (g : GameState) 
  (h1 : g.p < g.q ∧ g.q < g.r)
  (h2 : g.rounds ≥ 2)
  (h3 : g.final_marbles Player.A = 20)
  (h4 : g.final_marbles Player.B = 10)
  (h5 : g.final_marbles Player.C = 9)
  (h6 : g.b_last_round = g.r) :
  ∃ (first_round : Player → ℕ), first_round Player.B = 4 := by
  sorry

end marble_game_solution_l827_82740


namespace two_numbers_difference_l827_82703

theorem two_numbers_difference (x y : ℕ) : 
  x ∈ Finset.range 38 ∧ 
  y ∈ Finset.range 38 ∧ 
  x < y ∧ 
  (Finset.sum (Finset.range 38) id) - x - y = x * y →
  y - x = 10 := by
sorry

end two_numbers_difference_l827_82703


namespace units_digit_G_1000_l827_82793

/-- The sequence G_n is defined as 3^(3^n) + 1 -/
def G (n : ℕ) : ℕ := 3^(3^n) + 1

/-- The units digit of a natural number -/
def units_digit (n : ℕ) : ℕ := n % 10

theorem units_digit_G_1000 :
  units_digit (3^(3^1000)) = 1 →
  units_digit (G 1000) = 2 :=
sorry

end units_digit_G_1000_l827_82793


namespace fraction_equality_l827_82767

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : a / b + (a + 6 * b) / (b + 6 * a) = 2) : a / b = 1 / 2 := by
  sorry

end fraction_equality_l827_82767


namespace fraction_comparison_l827_82792

theorem fraction_comparison (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (x^2 - y^2) / (x - y) > (x^2 + y^2) / (x + y) := by
  sorry

end fraction_comparison_l827_82792


namespace distance_between_vertices_l827_82763

-- Define the equation of the parabolas
def parabola_equation (x y : ℝ) : Prop :=
  Real.sqrt (x^2 + y^2) + abs (y - 3) = 5

-- Define the vertices of the parabolas
def vertex1 : ℝ × ℝ := (0, 4)
def vertex2 : ℝ × ℝ := (0, -1)

-- Theorem statement
theorem distance_between_vertices :
  parabola_equation vertex1.1 vertex1.2 ∧
  parabola_equation vertex2.1 vertex2.2 →
  Real.sqrt ((vertex1.1 - vertex2.1)^2 + (vertex1.2 - vertex2.2)^2) = 5 := by
  sorry


end distance_between_vertices_l827_82763


namespace find_n_l827_82724

theorem find_n : ∃ n : ℤ, (7 : ℝ) ^ (2 * n) = (1 / 49) ^ (n - 12) ∧ n = 6 := by sorry

end find_n_l827_82724


namespace root_sum_reciprocal_l827_82798

theorem root_sum_reciprocal (a b c : ℂ) : 
  (a^3 - a + 1 = 0) → 
  (b^3 - b + 1 = 0) → 
  (c^3 - c + 1 = 0) → 
  (1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) = -2) := by
sorry

end root_sum_reciprocal_l827_82798


namespace not_lucky_1994_l827_82721

/-- Represents a date in month/day/year format -/
structure Date where
  month : Nat
  day : Nat
  year : Nat

/-- Checks if a given date is valid -/
def isValidDate (d : Date) : Prop :=
  d.month ≥ 1 ∧ d.month ≤ 12 ∧ d.day ≥ 1 ∧ d.day ≤ 31

/-- Checks if a given year is lucky -/
def isLuckyYear (year : Nat) : Prop :=
  ∃ (d : Date), d.year = year ∧ isValidDate d ∧ (d.month * d.day = year % 100)

/-- Theorem stating that 1994 is not a lucky year -/
theorem not_lucky_1994 : ¬ isLuckyYear 1994 := by
  sorry


end not_lucky_1994_l827_82721


namespace projection_ratio_l827_82725

def projection_matrix : Matrix (Fin 2) (Fin 2) ℚ :=
  !![9/41, -20/41; -20/41, 16/41]

theorem projection_ratio :
  ∀ (a b : ℚ),
  (a ≠ 0) →
  (projection_matrix.vecMul ![a, b] = ![a, b]) →
  b / a = 8 / 5 := by
sorry

end projection_ratio_l827_82725


namespace correct_total_cost_l827_82709

/-- The cost of a single sandwich -/
def sandwich_cost : ℕ := 4

/-- The cost of a single soda -/
def soda_cost : ℕ := 3

/-- The number of sandwiches purchased -/
def num_sandwiches : ℕ := 4

/-- The number of sodas purchased -/
def num_sodas : ℕ := 5

/-- The total cost of the purchase -/
def total_cost : ℕ := sandwich_cost * num_sandwiches + soda_cost * num_sodas

theorem correct_total_cost : total_cost = 31 := by
  sorry

end correct_total_cost_l827_82709


namespace sphere_volume_increase_l827_82749

theorem sphere_volume_increase (r : ℝ) (h : r > 0) :
  (4 / 3 * Real.pi * (2 * r)^3) / (4 / 3 * Real.pi * r^3) = 8 := by
  sorry

end sphere_volume_increase_l827_82749


namespace golden_ratio_geometric_mean_l827_82791

/-- Golden ratio division of a line segment -/
structure GoldenRatioDivision (α : Type*) [LinearOrderedField α] where
  a : α -- length of the whole segment
  b : α -- length of the smaller segment
  h1 : 0 < b
  h2 : b < a
  h3 : (a - b) / b = b / a -- golden ratio condition

/-- Right triangle formed from a golden ratio division -/
def goldenRatioTriangle {α : Type*} [LinearOrderedField α] (d : GoldenRatioDivision α) :=
  { x : α // x^2 + d.b^2 = d.a^2 }

/-- The other leg of the golden ratio triangle is the geometric mean of the hypotenuse and the first leg -/
theorem golden_ratio_geometric_mean {α : Type*} [LinearOrderedField α] (d : GoldenRatioDivision α) :
  let t := goldenRatioTriangle d
  ∀ x : t, x.val * d.a = d.b * x.val :=
by sorry

end golden_ratio_geometric_mean_l827_82791


namespace union_equals_set_implies_m_equals_one_l827_82714

theorem union_equals_set_implies_m_equals_one :
  let A : Set ℝ := {-1, 2}
  let B : Set ℝ := {x | m * x + 1 = 0}
  ∀ m : ℝ, (A ∪ B = A) → m = 1 := by
sorry

end union_equals_set_implies_m_equals_one_l827_82714


namespace solution_set_f_range_of_m_l827_82797

-- Define the function f
def f (x : ℝ) : ℝ := 1 - |x - 2|

-- Theorem for the first part
theorem solution_set_f (x : ℝ) :
  f x > 1 - |x + 4| ↔ x > -1 :=
sorry

-- Theorem for the second part
theorem range_of_m (m : ℝ) :
  (∀ x ∈ Set.Ioo 2 (5/2), f x > |x - m|) ↔ m ∈ Set.Ico 2 3 :=
sorry

end solution_set_f_range_of_m_l827_82797


namespace one_fourths_in_two_thirds_l827_82747

theorem one_fourths_in_two_thirds : (2 : ℚ) / 3 / ((1 : ℚ) / 4) = 8 / 3 := by
  sorry

end one_fourths_in_two_thirds_l827_82747


namespace parabola_line_intersection_l827_82761

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space -/
structure Line where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Represents a parabola in 2D space -/
structure Parabola where
  a : ℝ  -- coefficient of x

def is_on_parabola (p : Point) (par : Parabola) : Prop :=
  p.y^2 = 4 * par.a * p.x

def is_on_line (p : Point) (l : Line) : Prop :=
  p.y = l.m * p.x + l.b

def is_focus (f : Point) (par : Parabola) : Prop :=
  f.x = par.a ∧ f.y = 0

def is_on_circle_diameter (a : Point) (p : Point) (q : Point) : Prop :=
  (p.x - a.x) * (q.x - a.x) + (p.y - a.y) * (q.y - a.y) = 0

theorem parabola_line_intersection 
  (par : Parabola) (l : Line) (f p q : Point) (h_focus : is_focus f par)
  (h_line_through_focus : is_on_line f l)
  (h_p_on_parabola : is_on_parabola p par) (h_p_on_line : is_on_line p l)
  (h_q_on_parabola : is_on_parabola q par) (h_q_on_line : is_on_line q l)
  (h_circle : is_on_circle_diameter ⟨-1, 1⟩ p q) :
  l.m = 1/2 ∧ l.b = -1 :=
sorry

end parabola_line_intersection_l827_82761


namespace savings_account_decrease_l827_82738

theorem savings_account_decrease (initial_balance : ℝ) (increase_percent : ℝ) (final_balance_percent : ℝ) :
  initial_balance = 125 →
  increase_percent = 25 →
  final_balance_percent = 100 →
  let increased_balance := initial_balance * (1 + increase_percent / 100)
  let final_balance := initial_balance * (final_balance_percent / 100)
  let decrease_amount := increased_balance - final_balance
  let decrease_percent := (decrease_amount / increased_balance) * 100
  decrease_percent = 20 := by
sorry

end savings_account_decrease_l827_82738


namespace solve_exponential_equation_l827_82720

theorem solve_exponential_equation :
  ∃ y : ℕ, (8 : ℝ)^4 = 2^y ∧ y = 12 :=
by
  sorry

end solve_exponential_equation_l827_82720


namespace unique_two_digit_multiple_l827_82751

theorem unique_two_digit_multiple : ∃! t : ℕ, 
  10 ≤ t ∧ t < 100 ∧ (13 * t) % 100 = 42 := by
  sorry

end unique_two_digit_multiple_l827_82751


namespace ellipse_equation_l827_82731

/-- An ellipse passing through (3, 0) with eccentricity √6/3 has standard equations x²/9 + y²/3 = 1 or x²/9 + y²/27 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let e : ℝ := Real.sqrt 6 / 3
  let passes_through : Prop := x^2 + y^2 = 9 ∧ y = 0
  let equation1 : Prop := x^2 / 9 + y^2 / 3 = 1
  let equation2 : Prop := x^2 / 9 + y^2 / 27 = 1
  passes_through → (equation1 ∨ equation2) :=
by sorry

end ellipse_equation_l827_82731


namespace increase_dimension_theorem_l827_82742

/-- Represents a rectangle with length and width --/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle --/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: If increasing both length and width of a rectangle by x feet
    increases its perimeter by 16 feet, then x must be 4 feet --/
theorem increase_dimension_theorem (r : Rectangle) (x : ℝ) :
  perimeter { length := r.length + x, width := r.width + x } - perimeter r = 16 →
  x = 4 := by
  sorry

end increase_dimension_theorem_l827_82742


namespace gcd_lcm_product_l827_82736

theorem gcd_lcm_product (a b : ℕ) (ha : a = 108) (hb : b = 250) :
  Nat.gcd a b * Nat.lcm a b = a * b := by
  sorry

end gcd_lcm_product_l827_82736


namespace parabola_shift_theorem_l827_82788

/-- Represents a parabola in the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a parabola horizontally and vertically -/
def shift_parabola (p : Parabola) (h : ℝ) (k : ℝ) : Parabola :=
  { a := p.a
  , b := -2 * p.a * h
  , c := p.a * h^2 + k }

theorem parabola_shift_theorem (original : Parabola) (h k : ℝ) :
  original.a = 3 ∧ original.b = 0 ∧ original.c = 0 ∧ h = 1 ∧ k = 2 →
  let shifted := shift_parabola original h k
  shifted.a = 3 ∧ shifted.b = -6 ∧ shifted.c = 5 :=
by sorry

end parabola_shift_theorem_l827_82788


namespace imaginary_part_of_complex_fraction_l827_82779

theorem imaginary_part_of_complex_fraction (i : ℂ) :
  i * i = -1 → Complex.im ((2 : ℂ) + i) / i = -2 := by
  sorry

end imaginary_part_of_complex_fraction_l827_82779


namespace complex_fraction_pure_imaginary_l827_82741

/-- A complex number z is pure imaginary if its real part is zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

/-- The problem statement -/
theorem complex_fraction_pure_imaginary (a : ℝ) :
  is_pure_imaginary ((a + 3 * Complex.I) / (1 + 2 * Complex.I)) → a = -6 := by
  sorry

end complex_fraction_pure_imaginary_l827_82741


namespace x_power_six_plus_reciprocal_l827_82783

theorem x_power_six_plus_reciprocal (x : ℝ) (h : x + 1/x = 3) : x^6 + 1/x^6 = 322 := by
  sorry

end x_power_six_plus_reciprocal_l827_82783


namespace max_teams_intramurals_l827_82735

/-- Represents the number of participants in each category -/
structure Participants where
  girls : Nat
  boys : Nat
  teenagers : Nat

/-- Represents the sports preferences for girls -/
structure GirlsPreferences where
  basketball : Nat
  volleyball : Nat
  soccer : Nat

/-- Represents the sports preferences for boys -/
structure BoysPreferences where
  basketball : Nat
  soccer : Nat

/-- Represents the sports preferences for teenagers -/
structure TeenagersPreferences where
  volleyball : Nat
  mixed_sports : Nat

/-- The main theorem statement -/
theorem max_teams_intramurals
  (total : Participants)
  (girls_pref : GirlsPreferences)
  (boys_pref : BoysPreferences)
  (teens_pref : TeenagersPreferences)
  (h1 : total.girls = 120)
  (h2 : total.boys = 96)
  (h3 : total.teenagers = 72)
  (h4 : girls_pref.basketball = 40)
  (h5 : girls_pref.volleyball = 50)
  (h6 : girls_pref.soccer = 30)
  (h7 : boys_pref.basketball = 48)
  (h8 : boys_pref.soccer = 48)
  (h9 : teens_pref.volleyball = 24)
  (h10 : teens_pref.mixed_sports = 48)
  (h11 : girls_pref.basketball + girls_pref.volleyball + girls_pref.soccer = total.girls)
  (h12 : boys_pref.basketball + boys_pref.soccer = total.boys)
  (h13 : teens_pref.volleyball + teens_pref.mixed_sports = total.teenagers) :
  ∃ (n : Nat), n = 24 ∧ 
    n ∣ total.girls ∧ 
    n ∣ total.boys ∧ 
    n ∣ total.teenagers ∧
    n ∣ girls_pref.basketball ∧
    n ∣ girls_pref.volleyball ∧
    n ∣ girls_pref.soccer ∧
    n ∣ boys_pref.basketball ∧
    n ∣ boys_pref.soccer ∧
    n ∣ teens_pref.volleyball ∧
    n ∣ teens_pref.mixed_sports ∧
    ∀ (m : Nat), (m > n) → 
      ¬(m ∣ total.girls ∧ 
        m ∣ total.boys ∧ 
        m ∣ total.teenagers ∧
        m ∣ girls_pref.basketball ∧
        m ∣ girls_pref.volleyball ∧
        m ∣ girls_pref.soccer ∧
        m ∣ boys_pref.basketball ∧
        m ∣ boys_pref.soccer ∧
        m ∣ teens_pref.volleyball ∧
        m ∣ teens_pref.mixed_sports) :=
by
  sorry

end max_teams_intramurals_l827_82735


namespace merchant_profit_percentage_l827_82730

def markup_percentage : ℝ := 0.30
def discount_percentage : ℝ := 0.10

theorem merchant_profit_percentage :
  let marked_price := 1 + markup_percentage
  let discounted_price := marked_price * (1 - discount_percentage)
  (discounted_price - 1) * 100 = 17 := by
sorry

end merchant_profit_percentage_l827_82730


namespace cat_dog_positions_l827_82728

/-- Represents the number of positions for the cat -/
def cat_positions : Nat := 4

/-- Represents the number of positions for the dog -/
def dog_positions : Nat := 6

/-- Represents the total number of moves -/
def total_moves : Nat := 317

/-- Calculates the final position of an animal given its number of positions and total moves -/
def final_position (positions : Nat) (moves : Nat) : Nat :=
  moves % positions

theorem cat_dog_positions :
  (final_position cat_positions total_moves = 0) ∧
  (final_position dog_positions total_moves = 5) := by
  sorry

end cat_dog_positions_l827_82728


namespace factory_production_l827_82781

/-- Calculates the number of toys produced per week in a factory -/
def toys_per_week (days_per_week : ℕ) (toys_per_day : ℕ) : ℕ :=
  days_per_week * toys_per_day

/-- Proves that the factory produces 5505 toys per week -/
theorem factory_production : toys_per_week 5 1101 = 5505 := by
  sorry

end factory_production_l827_82781


namespace tank_inlet_rate_l827_82768

/-- Given a tank with the following properties:
  * Capacity of 3600.000000000001 liters
  * Empties in 6 hours due to a leak
  * Empties in 8 hours when both the leak and inlet are open
  Prove that the rate at which the inlet pipe fills the tank is 150 liters per hour -/
theorem tank_inlet_rate (capacity : ℝ) (leak_time : ℝ) (combined_time : ℝ) :
  capacity = 3600.000000000001 →
  leak_time = 6 →
  combined_time = 8 →
  ∃ (inlet_rate : ℝ),
    inlet_rate = 150 ∧
    inlet_rate = (capacity / leak_time) - (capacity / combined_time) :=
by sorry

end tank_inlet_rate_l827_82768


namespace vector_parallel_sum_l827_82760

/-- Two vectors are parallel if their cross product is zero -/
def parallel (v w : ℝ × ℝ) : Prop :=
  v.1 * w.2 - v.2 * w.1 = 0

theorem vector_parallel_sum (m : ℝ) : 
  let a : ℝ × ℝ := (1, 1)
  let b : ℝ × ℝ := (3, m)
  parallel a (a.1 + b.1, a.2 + b.2) → m = 3 := by
sorry

end vector_parallel_sum_l827_82760


namespace molecular_weight_calculation_l827_82713

/-- The molecular weight of a compound given the total weight and number of moles -/
theorem molecular_weight_calculation (total_weight : ℝ) (num_moles : ℝ) 
  (h1 : total_weight = 525)
  (h2 : num_moles = 3)
  (h3 : total_weight > 0)
  (h4 : num_moles > 0) :
  total_weight / num_moles = 175 := by
sorry

end molecular_weight_calculation_l827_82713


namespace factor_expression_l827_82784

theorem factor_expression (x : ℝ) : 5*x*(x+2) + 11*(x+2) = (x+2)*(5*x+11) := by
  sorry

end factor_expression_l827_82784


namespace group_size_proof_l827_82708

theorem group_size_proof (total_collection : ℚ) (h1 : total_collection = 92.16) : ∃ n : ℕ, 
  (n : ℚ) * (n : ℚ) / 100 = total_collection ∧ n = 96 := by
  sorry

end group_size_proof_l827_82708


namespace alphabet_size_l827_82771

theorem alphabet_size (dot_and_line : ℕ) (line_no_dot : ℕ) (dot_no_line : ℕ)
  (h1 : dot_and_line = 20)
  (h2 : line_no_dot = 36)
  (h3 : dot_no_line = 4)
  : dot_and_line + line_no_dot + dot_no_line = 60 := by
  sorry

end alphabet_size_l827_82771


namespace opposite_expressions_l827_82745

theorem opposite_expressions (x : ℚ) : x = -3/2 → (3 + x/3 = -(x - 1)) := by
  sorry

end opposite_expressions_l827_82745


namespace cube_preserves_order_l827_82718

theorem cube_preserves_order (a b : ℝ) (h : a > b) : a^3 > b^3 := by
  sorry

end cube_preserves_order_l827_82718


namespace greater_number_with_hcf_and_product_l827_82719

theorem greater_number_with_hcf_and_product 
  (A B : ℕ+) 
  (hcf_condition : Nat.gcd A B = 11)
  (product_condition : A * B = 363) :
  max A B = 33 := by
  sorry

end greater_number_with_hcf_and_product_l827_82719


namespace max_value_ab_l827_82743

theorem max_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_perp : (1 : ℝ) * (1 : ℝ) + (2 * a - 1) * (-b) = 0) : 
  ∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 
  (1 : ℝ) * (1 : ℝ) + (2 * x - 1) * (-y) = 0 → 
  x * y ≤ a * b ∧ a * b ≤ (1/8 : ℝ) := by
sorry

end max_value_ab_l827_82743


namespace sara_survey_sara_survey_result_l827_82766

theorem sara_survey (total : ℕ) 
  (belief_rate : ℚ) 
  (zika_rate : ℚ) 
  (zika_believers : ℕ) : Prop :=
  belief_rate = 753/1000 →
  zika_rate = 602/1000 →
  zika_believers = 37 →
  ∃ (believers : ℕ),
    (believers : ℚ) = zika_believers / zika_rate ∧
    (total : ℚ) = (believers : ℚ) / belief_rate ∧
    total = 81

theorem sara_survey_result : 
  ∃ (total : ℕ), sara_survey total (753/1000) (602/1000) 37 :=
sorry

end sara_survey_sara_survey_result_l827_82766


namespace largest_common_divisor_of_cube_minus_self_l827_82794

def is_prime_square (n : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ n = p^2

theorem largest_common_divisor_of_cube_minus_self (n : ℕ) (h : is_prime_square n) :
  (∀ d : ℕ, d > 30 → ¬(d ∣ (n^3 - n))) ∧
  (30 ∣ (n^3 - n)) :=
sorry

end largest_common_divisor_of_cube_minus_self_l827_82794


namespace original_quantities_l827_82704

/-- The original planned quantities of products A and B -/
def original_plan (x y : ℕ) : Prop :=
  ∃ (a b : ℝ), 
    -- Original plan: spend 1500 yuan
    a * x + b * y = 1500 ∧
    -- New scenario 1
    (a + 1.5) * (x - 10) + (b + 1) * y = 1529 ∧
    -- New scenario 2
    (a + 1) * (x - 5) + (b + 1) * y = 1563.5 ∧
    -- Constraint
    205 < 2 * x + y ∧ 2 * x + y < 210

theorem original_quantities : 
  ∃ (x y : ℕ), original_plan x y ∧ x = 76 ∧ y = 55 := by
  sorry

end original_quantities_l827_82704


namespace yunas_grandfather_age_l827_82707

/-- Proves the age of Yuna's grandfather given the ages and age differences of family members. -/
theorem yunas_grandfather_age 
  (yuna_age : ℕ) 
  (father_age_diff : ℕ) 
  (grandfather_age_diff : ℕ) 
  (h1 : yuna_age = 9)
  (h2 : father_age_diff = 27)
  (h3 : grandfather_age_diff = 23) : 
  yuna_age + father_age_diff + grandfather_age_diff = 59 :=
by sorry

end yunas_grandfather_age_l827_82707


namespace power_of_power_l827_82772

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l827_82772


namespace jason_pepper_spray_l827_82770

def total_animals (raccoons : ℕ) (squirrel_multiplier : ℕ) : ℕ :=
  raccoons + raccoons * squirrel_multiplier

theorem jason_pepper_spray :
  total_animals 12 6 = 84 :=
by sorry

end jason_pepper_spray_l827_82770


namespace missing_number_proof_l827_82754

theorem missing_number_proof : ∃ n : ℝ, n * 120 = 173 * 240 ∧ n = 345.6 := by
  sorry

end missing_number_proof_l827_82754


namespace dog_grouping_ways_l827_82716

def total_dogs : ℕ := 15
def group_1_size : ℕ := 4
def group_2_size : ℕ := 6
def group_3_size : ℕ := 5

def duke_in_group_1 : Prop := True
def bella_in_group_2 : Prop := True

theorem dog_grouping_ways : 
  total_dogs = group_1_size + group_2_size + group_3_size →
  duke_in_group_1 →
  bella_in_group_2 →
  (Nat.choose (total_dogs - 2) (group_1_size - 1)) * 
  (Nat.choose (total_dogs - group_1_size - 1) (group_2_size - 1)) = 72072 := by
  sorry

end dog_grouping_ways_l827_82716


namespace quadratic_minimum_value_l827_82734

theorem quadratic_minimum_value (k : ℝ) : 
  (∀ x y : ℝ, 5*x^2 - 8*k*x*y + (4*k^2 + 3)*y^2 - 10*x - 6*y + 9 ≥ 0) ∧ 
  (∃ x y : ℝ, 5*x^2 - 8*k*x*y + (4*k^2 + 3)*y^2 - 10*x - 6*y + 9 = 0) →
  k = 1 := by sorry

end quadratic_minimum_value_l827_82734


namespace book_cost_theorem_l827_82774

/-- Proves that the total cost of books is 600 yuan given the problem conditions -/
theorem book_cost_theorem (total_children : ℕ) (paying_children : ℕ) (extra_payment : ℕ) :
  total_children = 12 →
  paying_children = 10 →
  extra_payment = 10 →
  (paying_children * extra_payment : ℕ) / (total_children - paying_children) * total_children = 600 :=
by
  sorry

#check book_cost_theorem

end book_cost_theorem_l827_82774


namespace chess_tournament_players_l827_82727

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- number of players excluding the lowest 8
  total_players : ℕ := n + 8
  
  -- Each player played exactly one game against each other player
  total_games : ℕ := total_players.choose 2
  
  -- Point distribution condition
  point_distribution : 
    2 * n.choose 2 + 56 = (total_players * (total_players - 1)) / 2

/-- The theorem stating that the total number of players in the tournament is 21 -/
theorem chess_tournament_players : 
  ∀ t : ChessTournament, t.total_players = 21 := by
  sorry

end chess_tournament_players_l827_82727


namespace expand_product_l827_82706

theorem expand_product (x : ℝ) : (3 * x + 4) * (x - 2) = 3 * x^2 - 2 * x - 8 := by
  sorry

end expand_product_l827_82706


namespace power_sum_equality_l827_82778

theorem power_sum_equality : 2^123 + 8^5 / 8^3 = 2^123 + 64 := by sorry

end power_sum_equality_l827_82778


namespace no_adjacent_same_probability_correct_probability_between_zero_and_one_l827_82785

def number_of_people : ℕ := 6
def die_sides : ℕ := 6

/-- The probability of no two adjacent people rolling the same number on a six-sided die 
    when six people are sitting around a circular table. -/
def no_adjacent_same_probability : ℚ :=
  625 / 1944

/-- Theorem stating that the calculated probability is correct. -/
theorem no_adjacent_same_probability_correct : 
  no_adjacent_same_probability = 625 / 1944 := by
  sorry

/-- Theorem stating that the probability is between 0 and 1. -/
theorem probability_between_zero_and_one :
  0 ≤ no_adjacent_same_probability ∧ no_adjacent_same_probability ≤ 1 := by
  sorry

end no_adjacent_same_probability_correct_probability_between_zero_and_one_l827_82785


namespace only_D_is_valid_assignment_l827_82710

-- Define what constitutes a valid assignment statement
def is_valid_assignment (s : String) : Prop :=
  ∃ (var : String) (expr : String), 
    s = var ++ "=" ++ expr ∧ 
    var ≠ expr ∧
    var.length = 1 ∧
    var.all Char.isLower

-- Define the given options
def option_A : String := "5=a"
def option_B : String := "a+2=a"
def option_C : String := "a=b=4"
def option_D : String := "a=2*a"

-- Theorem statement
theorem only_D_is_valid_assignment :
  ¬(is_valid_assignment option_A) ∧
  ¬(is_valid_assignment option_B) ∧
  ¬(is_valid_assignment option_C) ∧
  is_valid_assignment option_D :=
sorry

end only_D_is_valid_assignment_l827_82710


namespace transport_cost_calculation_l827_82773

/-- The problem statement for calculating transport cost --/
theorem transport_cost_calculation (purchase_price installation_cost sell_price : ℚ) : 
  purchase_price = 16500 →
  installation_cost = 250 →
  sell_price = 23100 →
  ∃ (labelled_price transport_cost : ℚ),
    purchase_price = labelled_price * (1 - 0.2) ∧
    sell_price = labelled_price * 1.1 + transport_cost + installation_cost ∧
    transport_cost = 162.5 := by
  sorry

end transport_cost_calculation_l827_82773


namespace triangle_area_fraction_l827_82782

/-- The area of a right triangle with vertices at (3,3), (5,5), and (3,5) on a 6 by 6 grid
    is 1/18 of the total area of the 6 by 6 square. -/
theorem triangle_area_fraction (grid_size : ℕ) (x1 y1 x2 y2 x3 y3 : ℕ) : 
  grid_size = 6 →
  x1 = 3 → y1 = 3 →
  x2 = 5 → y2 = 5 →
  x3 = 3 → y3 = 5 →
  (1 : ℚ) / 2 * (x2 - x1) * (y3 - y1) / (grid_size * grid_size) = 1 / 18 := by
  sorry

end triangle_area_fraction_l827_82782


namespace parabola_symmetry_l827_82752

def C₁ (x : ℝ) : ℝ := 2 * x^2 - 4 * x - 1

def C₂ (x : ℝ) : ℝ := 2 * (x - 3)^2 - 4 * (x - 3) - 1

def is_symmetry_line (f g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a - x) = g (a + x)

theorem parabola_symmetry :
  is_symmetry_line C₁ C₂ (5/2) :=
sorry

end parabola_symmetry_l827_82752


namespace no_integer_solutions_l827_82726

theorem no_integer_solutions : 
  ¬ ∃ (x y z : ℤ), x^1988 + y^1988 + z^1988 = 7^1990 := by
  sorry

end no_integer_solutions_l827_82726


namespace price_increase_percentage_l827_82757

theorem price_increase_percentage (original_price : ℝ) (increase_rate : ℝ) : 
  original_price = 200 →
  increase_rate = 0.1 →
  let new_price := original_price * (1 + increase_rate)
  (new_price - original_price) / original_price = increase_rate :=
by sorry

end price_increase_percentage_l827_82757


namespace leila_order_proof_l827_82776

/-- The number of chocolate cakes Leila ordered -/
def chocolate_cakes : ℕ := 3

/-- The cost of each chocolate cake -/
def chocolate_cake_cost : ℕ := 12

/-- The number of strawberry cakes Leila ordered -/
def strawberry_cakes : ℕ := 6

/-- The cost of each strawberry cake -/
def strawberry_cake_cost : ℕ := 22

/-- The total amount Leila should pay -/
def total_amount : ℕ := 168

theorem leila_order_proof :
  chocolate_cakes * chocolate_cake_cost + 
  strawberry_cakes * strawberry_cake_cost = total_amount :=
by sorry

end leila_order_proof_l827_82776


namespace semicircles_to_circle_area_ratio_l827_82711

theorem semicircles_to_circle_area_ratio : 
  let r₁ : ℝ := 10  -- radius of the larger circle
  let r₂ : ℝ := 8   -- radius of the first semicircle
  let r₃ : ℝ := 6   -- radius of the second semicircle
  let circle_area := π * r₁^2
  let semicircle_area_1 := (π * r₂^2) / 2
  let semicircle_area_2 := (π * r₃^2) / 2
  let combined_semicircle_area := semicircle_area_1 + semicircle_area_2
  (combined_semicircle_area / circle_area) = (1 / 2 : ℝ) :=
by sorry

end semicircles_to_circle_area_ratio_l827_82711


namespace spade_nested_operation_l827_82790

-- Define the spade operation
def spade (a b : ℤ) : ℤ := |a - b|

-- Theorem statement
theorem spade_nested_operation : spade 3 (spade 5 (spade 8 12)) = 2 := by
  sorry

end spade_nested_operation_l827_82790


namespace max_min_powers_l827_82769

theorem max_min_powers (a b : ℝ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 1) :
  let M := max (max (a^a) (a^b)) (max (b^a) (b^b))
  let m := min (min (a^a) (a^b)) (min (b^a) (b^b))
  M = b^a ∧ m = a^b := by
sorry

end max_min_powers_l827_82769


namespace g_neg_one_equals_neg_one_l827_82786

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of y being an odd function
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) + (-x)^2 = -(f x + x^2)

-- Define g in terms of f
def g (f : ℝ → ℝ) (x : ℝ) : ℝ := f x + 2

-- Theorem statement
theorem g_neg_one_equals_neg_one
  (h1 : is_odd_function f)
  (h2 : f 1 = 1) :
  g f (-1) = -1 := by
  sorry

end g_neg_one_equals_neg_one_l827_82786


namespace expected_value_of_win_l827_82762

def fair_8_sided_die := Finset.range 8

def win_amount (n : ℕ) : ℝ := 8 - n

theorem expected_value_of_win :
  Finset.sum fair_8_sided_die (λ n => (1 : ℝ) / 8 * win_amount n) = 3.5 := by
  sorry

end expected_value_of_win_l827_82762


namespace ratio_theorem_l827_82764

theorem ratio_theorem (a b c : ℝ) (h1 : b / a = 3) (h2 : c / b = 4) : 
  (2 * a + 3 * b) / (b + 2 * c) = 11 / 27 := by
  sorry

end ratio_theorem_l827_82764


namespace tyrah_pencils_l827_82750

theorem tyrah_pencils (sarah tim tyrah : ℕ) : 
  tyrah = 6 * sarah →
  tim = 8 * sarah →
  tim = 16 →
  tyrah = 12 :=
by sorry

end tyrah_pencils_l827_82750


namespace set_equality_l827_82746

theorem set_equality (A B C : Set α) 
  (h1 : A ∪ B ⊆ C) 
  (h2 : A ∪ C ⊆ B) 
  (h3 : B ∪ C ⊆ A) : 
  A = B ∧ B = C := by
sorry

end set_equality_l827_82746


namespace total_fish_bought_is_89_l827_82780

/-- Represents the number of fish bought on each visit -/
structure FishPurchase where
  goldfish : Nat
  bluefish : Nat
  greenfish : Nat
  purplefish : Nat
  redfish : Nat

/-- Calculates the total number of fish in a purchase -/
def totalFish (purchase : FishPurchase) : Nat :=
  purchase.goldfish + purchase.bluefish + purchase.greenfish + purchase.purplefish + purchase.redfish

/-- Theorem: The total number of fish Roden bought is 89 -/
theorem total_fish_bought_is_89 
  (visit1 : FishPurchase := { goldfish := 15, bluefish := 7, greenfish := 0, purplefish := 0, redfish := 0 })
  (visit2 : FishPurchase := { goldfish := 10, bluefish := 12, greenfish := 5, purplefish := 0, redfish := 0 })
  (visit3 : FishPurchase := { goldfish := 3, bluefish := 7, greenfish := 9, purplefish := 0, redfish := 0 })
  (visit4 : FishPurchase := { goldfish := 4, bluefish := 8, greenfish := 6, purplefish := 2, redfish := 1 }) :
  totalFish visit1 + totalFish visit2 + totalFish visit3 + totalFish visit4 = 89 := by
  sorry


end total_fish_bought_is_89_l827_82780
