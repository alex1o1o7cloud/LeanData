import Mathlib

namespace robs_baseball_cards_l3866_386664

theorem robs_baseball_cards 
  (rob_doubles : ℕ) 
  (rob_total : ℕ) 
  (jess_doubles : ℕ) 
  (h1 : rob_doubles = rob_total / 3)
  (h2 : jess_doubles = 5 * rob_doubles)
  (h3 : jess_doubles = 40) : 
  rob_total = 24 := by
sorry

end robs_baseball_cards_l3866_386664


namespace remainder_3249_div_82_l3866_386612

theorem remainder_3249_div_82 : 3249 % 82 = 51 := by sorry

end remainder_3249_div_82_l3866_386612


namespace flower_shop_expenses_flower_shop_weekly_expenses_l3866_386637

/-- Weekly expenses for running a flower shop -/
theorem flower_shop_expenses (rent : ℝ) (utility_rate : ℝ) (hours_per_day : ℕ) 
  (days_per_week : ℕ) (employees_per_shift : ℕ) (hourly_wage : ℝ) : ℝ :=
  let utilities := rent * utility_rate
  let employee_hours := hours_per_day * days_per_week * employees_per_shift
  let employee_wages := employee_hours * hourly_wage
  rent + utilities + employee_wages

/-- Proof of the flower shop's weekly expenses -/
theorem flower_shop_weekly_expenses :
  flower_shop_expenses 1200 0.2 16 5 2 12.5 = 3440 := by
  sorry

end flower_shop_expenses_flower_shop_weekly_expenses_l3866_386637


namespace min_obtuse_triangle_l3866_386679

-- Define the initial angles of the triangle
def α₀ : Real := 60.001
def β₀ : Real := 60
def γ₀ : Real := 59.999

-- Define a function to calculate the nth angle
def angle (n : Nat) (initial : Real) : Real :=
  (-2)^n * (initial - 60) + 60

-- Define a predicate for an obtuse triangle
def is_obtuse (α β γ : Real) : Prop :=
  α > 90 ∨ β > 90 ∨ γ > 90

-- State the theorem
theorem min_obtuse_triangle :
  ∃ (n : Nat), (∀ k < n, ¬is_obtuse (angle k α₀) (angle k β₀) (angle k γ₀)) ∧
               is_obtuse (angle n α₀) (angle n β₀) (angle n γ₀) ∧
               n = 15 := by
  sorry


end min_obtuse_triangle_l3866_386679


namespace quadratic_root_implies_k_l3866_386693

theorem quadratic_root_implies_k (k : ℝ) : 
  (∃ x : ℝ, x^2 - 2*k*x + k^2 = 0 ∧ x = -1) → k = -1 := by
  sorry

end quadratic_root_implies_k_l3866_386693


namespace translation_of_sine_graph_l3866_386699

open Real

theorem translation_of_sine_graph (θ φ : ℝ) : 
  (abs θ < π / 2) →
  (0 < φ) →
  (φ < π) →
  (sin θ = 1 / 2) →
  (sin (θ - 2 * φ) = 1 / 2) →
  (φ = 2 * π / 3) :=
by sorry

end translation_of_sine_graph_l3866_386699


namespace concert_attendance_l3866_386684

theorem concert_attendance (total_students : ℕ) (total_attendees : ℕ)
  (h_total : total_students = 1500)
  (h_attendees : total_attendees = 900) :
  ∃ (girls boys girls_attended : ℕ),
    girls + boys = total_students ∧
    (3 * girls + 2 * boys = 5 * total_attendees) ∧
    girls_attended = 643 ∧
    4 * girls_attended = 3 * girls :=
by sorry

end concert_attendance_l3866_386684


namespace table_sum_theorem_l3866_386613

-- Define a 3x3 table as a function from (Fin 3 × Fin 3) to ℕ
def Table := Fin 3 → Fin 3 → ℕ

-- Define the property that the table contains numbers from 1 to 9
def containsOneToNine (t : Table) : Prop :=
  ∀ n : ℕ, 1 ≤ n ∧ n ≤ 9 → ∃ i j : Fin 3, t i j = n

-- Define the sum of a diagonal
def diagonalSum (t : Table) (d : Bool) : ℕ :=
  if d then t 0 0 + t 1 1 + t 2 2
  else t 0 2 + t 1 1 + t 2 0

-- Define the sum of five specific cells
def fiveCellSum (t : Table) : ℕ :=
  t 0 1 + t 1 0 + t 1 1 + t 1 2 + t 2 1

theorem table_sum_theorem (t : Table) 
  (h1 : containsOneToNine t)
  (h2 : diagonalSum t true = 7)
  (h3 : diagonalSum t false = 21) :
  fiveCellSum t = 25 := by
  sorry


end table_sum_theorem_l3866_386613


namespace equation_properties_l3866_386695

variable (a : ℝ)
variable (z : ℂ)

def has_real_solution (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 - (a + Complex.I)*x - (Complex.I + 2) = 0

def has_imaginary_solution (a : ℝ) : Prop :=
  ∃ y : ℝ, y ≠ 0 ∧ (Complex.I*y)^2 - (a + Complex.I)*(Complex.I*y) - (Complex.I + 2) = 0

theorem equation_properties :
  (has_real_solution a ↔ a = 1) ∧
  ¬(has_imaginary_solution a) := by sorry

end equation_properties_l3866_386695


namespace sailboat_canvas_area_l3866_386638

/-- The total area of canvas needed for a model sailboat with three sails -/
theorem sailboat_canvas_area
  (rect_length : ℝ)
  (rect_width : ℝ)
  (tri1_base : ℝ)
  (tri1_height : ℝ)
  (tri2_base : ℝ)
  (tri2_height : ℝ)
  (h_rect_length : rect_length = 5)
  (h_rect_width : rect_width = 8)
  (h_tri1_base : tri1_base = 3)
  (h_tri1_height : tri1_height = 4)
  (h_tri2_base : tri2_base = 4)
  (h_tri2_height : tri2_height = 6) :
  rect_length * rect_width +
  (tri1_base * tri1_height) / 2 +
  (tri2_base * tri2_height) / 2 = 58 := by
sorry


end sailboat_canvas_area_l3866_386638


namespace derivative_zero_not_sufficient_nor_necessary_l3866_386631

-- Define a real-valued function
variable (f : ℝ → ℝ)
-- Define a real number x
variable (x : ℝ)

-- Define what it means for a function to have an extremum at a point
def has_extremum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∃ ε > 0, ∀ y ∈ Set.Ioo (x - ε) (x + ε), f y ≤ f x ∨ f y ≥ f x

-- Define the statement to be proved
theorem derivative_zero_not_sufficient_nor_necessary :
  ¬(∀ f x, (deriv f x = 0 → has_extremum f x) ∧ (has_extremum f x → deriv f x = 0)) :=
sorry

end derivative_zero_not_sufficient_nor_necessary_l3866_386631


namespace four_terms_after_substitution_l3866_386609

-- Define the expression with a variable for the asterisk
def expression (a : ℝ → ℝ) : ℝ → ℝ := λ x => (x^4 - 3)^2 + (x^3 + a x)^2

-- Define the proposed replacement for the asterisk
def replacement : ℝ → ℝ := λ x => x^3 + 3*x

-- The theorem to prove
theorem four_terms_after_substitution :
  ∃ (c₁ c₂ c₃ c₄ : ℝ) (n₁ n₂ n₃ n₄ : ℕ), 
    (∀ x, expression replacement x = c₁ * x^n₁ + c₂ * x^n₂ + c₃ * x^n₃ + c₄ * x^n₄) ∧
    (n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₁ ≠ n₄ ∧ n₂ ≠ n₃ ∧ n₂ ≠ n₄ ∧ n₃ ≠ n₄) :=
by
  sorry

end four_terms_after_substitution_l3866_386609


namespace rational_function_pair_l3866_386688

theorem rational_function_pair (f g : ℚ → ℚ)
  (h1 : ∀ x y : ℚ, f (g x - g y) = f (g x) - y)
  (h2 : ∀ x y : ℚ, g (f x - f y) = g (f x) - y) :
  ∃ c : ℚ, (∀ x : ℚ, f x = c * x) ∧ (∀ x : ℚ, g x = x / c) :=
sorry

end rational_function_pair_l3866_386688


namespace fraction_subtraction_result_l3866_386600

theorem fraction_subtraction_result : (18 : ℚ) / 42 - 3 / 8 - 1 / 12 = -5 / 168 := by
  sorry

end fraction_subtraction_result_l3866_386600


namespace division_problem_l3866_386624

theorem division_problem : (102 / 6) / 3 = 5 + 2/3 := by sorry

end division_problem_l3866_386624


namespace tournament_balls_used_l3866_386619

/-- A tennis tournament with specified conditions -/
structure TennisTournament where
  rounds : Nat
  games_per_round : List Nat
  cans_per_game : Nat
  balls_per_can : Nat

/-- Calculate the total number of tennis balls used in the tournament -/
def total_balls_used (t : TennisTournament) : Nat :=
  (t.games_per_round.sum * t.cans_per_game * t.balls_per_can)

/-- Theorem stating the total number of tennis balls used in the specific tournament -/
theorem tournament_balls_used :
  let t : TennisTournament := {
    rounds := 4,
    games_per_round := [8, 4, 2, 1],
    cans_per_game := 5,
    balls_per_can := 3
  }
  total_balls_used t = 225 := by sorry

end tournament_balls_used_l3866_386619


namespace problem_solution_l3866_386653

theorem problem_solution (x y m n a b : ℝ) : 
  x = (Real.sqrt 3 - 1) / 2 →
  y = (Real.sqrt 3 + 1) / 2 →
  m = 1 / x - 1 / y →
  n = y / x + x / y →
  Real.sqrt a - Real.sqrt b = n + 2 →
  Real.sqrt (a * b) = m →
  m = 2 ∧ n = 4 ∧ Real.sqrt a + Real.sqrt b = 2 * Real.sqrt 11 := by
  sorry

end problem_solution_l3866_386653


namespace inequality_solution_l3866_386680

theorem inequality_solution (x : ℝ) : 
  2 - 1 / (2 * x + 3) < 4 ↔ x < -7/4 ∨ x > -3/2 :=
by sorry

end inequality_solution_l3866_386680


namespace cornelias_asian_countries_l3866_386630

theorem cornelias_asian_countries 
  (total_countries : ℕ) 
  (european_countries : ℕ) 
  (south_american_countries : ℕ) 
  (h1 : total_countries = 42)
  (h2 : european_countries = 20)
  (h3 : south_american_countries = 10)
  (h4 : 2 * (total_countries - european_countries - south_american_countries) / 2 = 
       total_countries - european_countries - south_american_countries) :
  (total_countries - european_countries - south_american_countries) / 2 = 6 := by
sorry

end cornelias_asian_countries_l3866_386630


namespace parallelogram_area_l3866_386677

/-- Represents a parallelogram ABCD with given properties -/
structure Parallelogram where
  perimeter : ℝ
  height_BC : ℝ
  height_CD : ℝ
  perimeter_positive : perimeter > 0
  height_BC_positive : height_BC > 0
  height_CD_positive : height_CD > 0

/-- The area of the parallelogram ABCD is 280 cm² -/
theorem parallelogram_area (ABCD : Parallelogram)
  (h_perimeter : ABCD.perimeter = 75)
  (h_height_BC : ABCD.height_BC = 14)
  (h_height_CD : ABCD.height_CD = 16) :
  ∃ (area : ℝ), area = 280 ∧ (∃ (base : ℝ), base * ABCD.height_BC = area ∧ base * ABCD.height_CD = area) :=
sorry

end parallelogram_area_l3866_386677


namespace negation_of_proposition_l3866_386641

theorem negation_of_proposition :
  (¬ (∀ x : ℝ, x ≥ 0 → x^2 - x + 1 ≥ 0)) ↔ (∃ x : ℝ, x ≥ 0 ∧ x^2 - x + 1 < 0) :=
by sorry

end negation_of_proposition_l3866_386641


namespace marks_increase_ratio_class_marks_double_l3866_386606

/-- Given a class of students, prove that if their marks are increased by a certain ratio,
    the ratio of new marks to original marks can be determined by the new and old averages. -/
theorem marks_increase_ratio (n : ℕ) (old_avg new_avg : ℚ) :
  n > 0 →
  old_avg > 0 →
  new_avg > old_avg →
  (n * new_avg) / (n * old_avg) = new_avg / old_avg := by sorry

/-- In a class of 11 students, if the average marks increase from 36 to 72,
    prove that the ratio of new marks to original marks is 2. -/
theorem class_marks_double :
  let n : ℕ := 11
  let old_avg : ℚ := 36
  let new_avg : ℚ := 72
  (n * new_avg) / (n * old_avg) = 2 := by sorry

end marks_increase_ratio_class_marks_double_l3866_386606


namespace inequality_solution_set_l3866_386662

theorem inequality_solution_set : 
  {x : ℝ | 5 - x^2 > 4*x} = Set.Ioo (-5 : ℝ) 1 := by sorry

end inequality_solution_set_l3866_386662


namespace chicken_count_l3866_386678

/-- The number of chickens Colten has -/
def colten_chickens : ℕ := 37

/-- The number of chickens Skylar has -/
def skylar_chickens : ℕ := 3 * colten_chickens - 4

/-- The number of chickens Quentin has -/
def quentin_chickens : ℕ := 2 * skylar_chickens + 25

/-- The total number of chickens -/
def total_chickens : ℕ := quentin_chickens + skylar_chickens + colten_chickens

theorem chicken_count : total_chickens = 383 := by
  sorry

end chicken_count_l3866_386678


namespace quadratic_roots_sum_product_l3866_386657

theorem quadratic_roots_sum_product (x₁ x₂ : ℝ) : 
  x₁^2 - 2*x₁ - 5 = 0 → x₂^2 - 2*x₂ - 5 = 0 → x₁ + x₂ + 3*x₁*x₂ = -13 := by
  sorry

end quadratic_roots_sum_product_l3866_386657


namespace joan_initial_balloons_l3866_386645

/-- The number of balloons Joan lost -/
def lost_balloons : ℕ := 2

/-- The number of balloons Joan currently has -/
def current_balloons : ℕ := 7

/-- The initial number of balloons Joan had -/
def initial_balloons : ℕ := current_balloons + lost_balloons

theorem joan_initial_balloons : initial_balloons = 9 := by
  sorry

end joan_initial_balloons_l3866_386645


namespace quadratic_inequality_solution_sets_l3866_386640

def quadratic_inequality (a b c : ℝ) (x : ℝ) := a * x^2 + b * x + c > 0

def solution_set (a b c : ℝ) := {x : ℝ | quadratic_inequality a b c x}

theorem quadratic_inequality_solution_sets
  (a b c : ℝ) (h : solution_set a b c = {x : ℝ | -2 < x ∧ x < 1}) :
  {x : ℝ | c * x^2 + a * x + b ≥ 0} = {x : ℝ | x ≤ -1/2 ∨ x ≥ 1} :=
sorry

end quadratic_inequality_solution_sets_l3866_386640


namespace complex_equation_solution_l3866_386665

theorem complex_equation_solution (a b : ℝ) : 
  (a : ℂ) + 3 * I = (b + I) * I → a = -1 ∧ b = 3 := by
  sorry

end complex_equation_solution_l3866_386665


namespace gcf_lcm_sum_18_27_36_l3866_386647

theorem gcf_lcm_sum_18_27_36 : 
  let X := Nat.gcd 18 (Nat.gcd 27 36)
  let Y := Nat.lcm 18 (Nat.lcm 27 36)
  X + Y = 117 := by
sorry

end gcf_lcm_sum_18_27_36_l3866_386647


namespace fraction_decomposition_l3866_386652

theorem fraction_decomposition (x A B : ℚ) : 
  (7 * x - 18) / (3 * x^2 - 5 * x - 2) = A / (x + 1) + B / (3 * x - 2) →
  A = -4/7 ∧ B = 61/7 := by
  sorry

end fraction_decomposition_l3866_386652


namespace crazy_silly_school_books_read_l3866_386607

/-- Represents the 'crazy silly school' series -/
structure CrazySillySchool where
  total_books : Nat
  total_movies : Nat
  books_read : Nat
  movies_watched : Nat

/-- Theorem: In the 'crazy silly school' series, if all available books and movies are consumed,
    and the number of movies watched is 2 more than the number of books read,
    then the number of books read is 8. -/
theorem crazy_silly_school_books_read
  (css : CrazySillySchool)
  (h1 : css.total_books = 8)
  (h2 : css.total_movies = 10)
  (h3 : css.movies_watched = css.books_read + 2)
  (h4 : css.books_read = css.total_books)
  (h5 : css.movies_watched = css.total_movies) :
  css.books_read = 8 := by
  sorry

#check crazy_silly_school_books_read

end crazy_silly_school_books_read_l3866_386607


namespace quadratic_radical_equivalence_l3866_386659

-- Define what it means for two quadratic radicals to be of the same type
def same_type_quadratic_radical (a b : ℝ) : Prop :=
  ∃ (k : ℕ), k > 1 ∧ (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ a = k * x^2 ∧ b = k * y^2)

-- State the theorem
theorem quadratic_radical_equivalence :
  same_type_quadratic_radical (m + 1) 8 → m = 1 :=
by
  sorry

end quadratic_radical_equivalence_l3866_386659


namespace fraction_to_decimal_l3866_386692

theorem fraction_to_decimal :
  (3 : ℚ) / 40 = 0.075 := by sorry

end fraction_to_decimal_l3866_386692


namespace bills_difference_l3866_386635

/-- The number of bills each person had at the beginning -/
structure Bills where
  geric : ℕ
  kyla : ℕ
  jessa : ℕ

/-- The conditions of the problem -/
def problem_conditions (b : Bills) : Prop :=
  b.geric = 2 * b.kyla ∧
  b.geric = 16 ∧
  b.jessa - 3 = 7

/-- The theorem to prove -/
theorem bills_difference (b : Bills) 
  (h : problem_conditions b) : b.jessa - b.kyla = 2 := by
  sorry

end bills_difference_l3866_386635


namespace collinear_points_sum_l3866_386650

/-- Three points in 3D space are collinear if they all lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop := sorry

/-- The main theorem: if the given points are collinear, then 2a + b = 8. -/
theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → 2 * a + b = 8 := by
  sorry

end collinear_points_sum_l3866_386650


namespace fold_sum_theorem_l3866_386642

/-- Represents a point on the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a fold of a piece of graph paper -/
structure Fold where
  p1 : Point
  p2 : Point
  p3 : Point
  p4 : Point

/-- The sum of the x and y coordinates of the fourth point in a fold -/
def fourthPointSum (f : Fold) : ℝ :=
  f.p4.x + f.p4.y

/-- Theorem stating that for the given fold, the sum of x and y coordinates of the fourth point is 13 -/
theorem fold_sum_theorem (f : Fold) 
  (h1 : f.p1 = ⟨0, 4⟩) 
  (h2 : f.p2 = ⟨5, 0⟩) 
  (h3 : f.p3 = ⟨9, 6⟩) : 
  fourthPointSum f = 13 := by
  sorry

end fold_sum_theorem_l3866_386642


namespace relationship_between_x_and_y_l3866_386669

theorem relationship_between_x_and_y (x y : ℝ) 
  (h1 : 2 * x - y > x + 1) 
  (h2 : x + 2 * y < 2 * y - 3) : 
  x < -3 ∧ y < -4 ∧ x > y + 1 := by
  sorry

end relationship_between_x_and_y_l3866_386669


namespace square_side_length_l3866_386646

theorem square_side_length (s AF DH BG AE : ℝ) (area_EFGH : ℝ) 
  (h1 : AF = 7)
  (h2 : DH = 4)
  (h3 : BG = 5)
  (h4 : AE = 1)
  (h5 : area_EFGH = 78)
  (h6 : s > 0)
  (h7 : s * s = ((area_EFGH - (AF - DH) * (BG - AE)) * 2) + area_EFGH) :
  s = 12 := by
  sorry

end square_side_length_l3866_386646


namespace multiples_6_10_not_5_8_empty_l3866_386602

theorem multiples_6_10_not_5_8_empty : 
  {n : ℤ | 1 ≤ n ∧ n ≤ 300 ∧ 6 ∣ n ∧ 10 ∣ n ∧ ¬(5 ∣ n) ∧ ¬(8 ∣ n)} = ∅ := by
  sorry

end multiples_6_10_not_5_8_empty_l3866_386602


namespace road_length_proof_l3866_386690

/-- The total length of the road in meters -/
def total_length : ℝ := 1000

/-- The length repaired in the first week in meters -/
def first_week_repair : ℝ := 0.2 * total_length

/-- The length repaired in the second week in meters -/
def second_week_repair : ℝ := 0.25 * total_length

/-- The length repaired in the third week in meters -/
def third_week_repair : ℝ := 480

/-- The length remaining unrepaired in meters -/
def remaining_length : ℝ := 70

theorem road_length_proof :
  first_week_repair + second_week_repair + third_week_repair + remaining_length = total_length := by
  sorry

end road_length_proof_l3866_386690


namespace system_solution_l3866_386610

theorem system_solution (x y : ℝ) : 
  x^2 + y^2 ≤ 2 ∧ 
  81 * x^4 - 18 * x^2 * y^2 + y^4 - 360 * x^2 - 40 * y^2 + 400 = 0 ↔ 
  ((x = -3 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5) ∨
   (x = -3 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5) ∨
   (x = 3 / Real.sqrt 5 ∧ y = -1 / Real.sqrt 5) ∨
   (x = 3 / Real.sqrt 5 ∧ y = 1 / Real.sqrt 5)) :=
by sorry

end system_solution_l3866_386610


namespace even_sum_condition_l3866_386616

theorem even_sum_condition (m n : ℤ) : 
  (∃ k l : ℤ, m = 2 * k ∧ n = 2 * l) → (∃ p : ℤ, m + n = 2 * p) ∧
  ¬(∀ q : ℤ, m + n = 2 * q → ∃ r s : ℤ, m = 2 * r ∧ n = 2 * s) :=
by sorry

end even_sum_condition_l3866_386616


namespace lagrange_interpolation_polynomial_l3866_386611

def P (x : ℝ) : ℝ := x^3 - 2*x^2 + 3*x - 5

theorem lagrange_interpolation_polynomial :
  P (-1) = -11 ∧ P 1 = -3 ∧ P 2 = 1 ∧ P 3 = 13 :=
by sorry

end lagrange_interpolation_polynomial_l3866_386611


namespace complex_magnitude_problem_l3866_386604

theorem complex_magnitude_problem (s : ℝ) (w : ℂ) 
  (h1 : |s| < 3) 
  (h2 : w + 2 / w = s) : 
  Complex.abs w = Real.sqrt 2 := by
  sorry

end complex_magnitude_problem_l3866_386604


namespace star_equation_solution_l3866_386672

def star (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

theorem star_equation_solution :
  ∀ A : ℝ, star A 6 = 31 → A = 10.5 := by
  sorry

end star_equation_solution_l3866_386672


namespace two_cars_speed_l3866_386691

/-- Two cars traveling in the same direction with given conditions -/
theorem two_cars_speed (t v₁ S₁ S₂ : ℝ) (h_t : t = 30) (h_v₁ : v₁ = 25)
  (h_S₁ : S₁ = 100) (h_S₂ : S₂ = 400) :
  ∃ v₂ : ℝ, (v₂ = 35 ∨ v₂ = 15) ∧
  (S₂ - S₁) / t = |v₂ - v₁| :=
by sorry

end two_cars_speed_l3866_386691


namespace drug_price_reduction_equation_l3866_386667

/-- Represents the price reduction scenario for a drug -/
def PriceReductionScenario (initial_price final_price : ℝ) (x : ℝ) : Prop :=
  initial_price * (1 - x)^2 = final_price

/-- Theorem stating the equation for the given drug price reduction scenario -/
theorem drug_price_reduction_equation :
  PriceReductionScenario 140 35 x ↔ 140 * (1 - x)^2 = 35 := by
  sorry

end drug_price_reduction_equation_l3866_386667


namespace product_of_divisors_implies_n_l3866_386670

/-- The product of all positive divisors of a natural number -/
def divisor_product (n : ℕ) : ℕ := sorry

/-- The number of positive divisors of a natural number -/
def divisor_count (n : ℕ) : ℕ := sorry

theorem product_of_divisors_implies_n (N : ℕ) :
  divisor_product N = 2^120 * 3^60 * 5^90 → N = 18000 := by sorry

end product_of_divisors_implies_n_l3866_386670


namespace marbles_remaining_l3866_386614

def initial_marbles : ℕ := 47
def shared_marbles : ℕ := 42

theorem marbles_remaining : initial_marbles - shared_marbles = 5 := by
  sorry

end marbles_remaining_l3866_386614


namespace hare_jumps_to_12th_cell_l3866_386615

/-- The number of ways a hare can reach the nth cell -/
def hare_jumps : ℕ → ℕ
| 0 => 0  -- No ways to reach the 0th cell (not part of the strip)
| 1 => 1  -- One way to be at the 1st cell (starting position)
| 2 => 1  -- One way to reach the 2nd cell (single jump from 1st)
| (n + 3) => hare_jumps (n + 2) + hare_jumps (n + 1)

/-- The number of ways a hare can jump from the 1st cell to the 12th cell is 144 -/
theorem hare_jumps_to_12th_cell : hare_jumps 12 = 144 := by
  sorry

end hare_jumps_to_12th_cell_l3866_386615


namespace farm_animal_difference_l3866_386603

theorem farm_animal_difference (goats chickens ducks pigs : ℕ) : 
  goats = 66 →
  chickens = 2 * goats →
  ducks = (goats + chickens) / 2 →
  pigs = ducks / 3 →
  goats - pigs = 33 := by
sorry

end farm_animal_difference_l3866_386603


namespace complex_addition_simplification_l3866_386636

theorem complex_addition_simplification :
  (-5 + 3*Complex.I) + (2 - 7*Complex.I) = -3 - 4*Complex.I :=
by sorry

end complex_addition_simplification_l3866_386636


namespace fishing_competition_l3866_386686

/-- Fishing Competition Problem -/
theorem fishing_competition (days : ℕ) (jackson_per_day : ℕ) (george_per_day : ℕ) (total_catch : ℕ) :
  days = 5 →
  jackson_per_day = 6 →
  george_per_day = 8 →
  total_catch = 90 →
  ∃ (jonah_per_day : ℕ),
    jonah_per_day = 4 ∧
    total_catch = days * (jackson_per_day + george_per_day + jonah_per_day) :=
by
  sorry


end fishing_competition_l3866_386686


namespace power_two_half_equals_two_l3866_386698

theorem power_two_half_equals_two : 2^(2/2) = 2 := by
  sorry

end power_two_half_equals_two_l3866_386698


namespace apple_bag_weight_l3866_386629

theorem apple_bag_weight (empty_weight loaded_weight : ℕ) (num_bags : ℕ) : 
  empty_weight = 500 →
  loaded_weight = 1700 →
  num_bags = 20 →
  (loaded_weight - empty_weight) / num_bags = 60 :=
by sorry

end apple_bag_weight_l3866_386629


namespace salary_percentage_increase_l3866_386687

theorem salary_percentage_increase 
  (initial_salary final_salary : ℝ) 
  (h1 : initial_salary = 50)
  (h2 : final_salary = 90) : 
  (final_salary - initial_salary) / initial_salary * 100 = 80 := by
  sorry

end salary_percentage_increase_l3866_386687


namespace domino_reconstruction_theorem_l3866_386654

/-- Represents a 2x1 domino with color information -/
inductive Domino
| WhiteWhite
| BlueBlue
| WhiteBlue
| BlueWhite

/-- Represents an 8x8 grid -/
def Grid := List (List Bool)

/-- Counts the number of blue cells in a grid -/
def countBlue (g : Grid) : Nat := sorry

/-- Divides a grid into 2x1 dominoes -/
def divideToDominoes (g : Grid) : List Domino := sorry

/-- Reconstructs an 8x8 grid from a list of dominoes -/
def reconstructGrid (dominoes : List Domino) : Grid := sorry

/-- Checks if two grids have the same blue pattern -/
def samePattern (g1 g2 : Grid) : Bool := sorry

theorem domino_reconstruction_theorem (g1 g2 : Grid) 
  (h : countBlue g1 = countBlue g2) :
  ∃ (d1 d2 : List Domino), 
    d1 = divideToDominoes g1 ∧ 
    d2 = divideToDominoes g2 ∧ 
    samePattern (reconstructGrid (d1 ++ d2)) g1 ∧
    samePattern (reconstructGrid (d1 ++ d2)) g2 := by
  sorry

end domino_reconstruction_theorem_l3866_386654


namespace max_pairs_for_marcella_l3866_386634

/-- Given an initial number of shoe pairs and a number of individual shoes lost,
    calculate the maximum number of complete pairs remaining. -/
def max_pairs_remaining (initial_pairs : ℕ) (shoes_lost : ℕ) : ℕ :=
  initial_pairs - shoes_lost

/-- Theorem: With 23 initial pairs and 9 individual shoes lost,
    the maximum number of complete pairs remaining is 14. -/
theorem max_pairs_for_marcella :
  max_pairs_remaining 23 9 = 14 := by
  sorry

end max_pairs_for_marcella_l3866_386634


namespace password_guess_probabilities_l3866_386682

/-- The probability of guessing the correct last digit of a 6-digit password in no more than 2 attempts -/
def guess_probability (total_digits : ℕ) (max_attempts : ℕ) : ℚ :=
  1 / total_digits + (1 - 1 / total_digits) * (1 / (total_digits - 1))

/-- The probability of guessing the correct last digit of a 6-digit password in no more than 2 attempts, given that the last digit is even -/
def guess_probability_even (total_even_digits : ℕ) (max_attempts : ℕ) : ℚ :=
  1 / total_even_digits + (1 - 1 / total_even_digits) * (1 / (total_even_digits - 1))

theorem password_guess_probabilities :
  (guess_probability 10 2 = 1/5) ∧ (guess_probability_even 5 2 = 2/5) :=
sorry

end password_guess_probabilities_l3866_386682


namespace line_tangent_to_parabola_l3866_386633

/-- A line y = 3x + c is tangent to the parabola y^2 = 12x if and only if c = 3 -/
theorem line_tangent_to_parabola (c : ℝ) : 
  (∃ x y : ℝ, y = 3 * x + c ∧ y^2 = 12 * x ∧ 
   ∀ x' y' : ℝ, y' = 3 * x' + c ∧ y'^2 = 12 * x' → x' = x ∧ y' = y) ↔ 
  c = 3 :=
sorry

end line_tangent_to_parabola_l3866_386633


namespace root_implies_a_range_l3866_386668

theorem root_implies_a_range :
  ∀ a : ℝ, (∃ x : ℝ, 9^(-|x - 2|) - 4 * 3^(-|x - 2|) - a = 0) → -3 ≤ a ∧ a < 0 := by
  sorry

end root_implies_a_range_l3866_386668


namespace one_fourth_between_fractions_l3866_386661

theorem one_fourth_between_fractions :
  let start := (1 : ℚ) / 5
  let finish := (4 : ℚ) / 5
  let one_fourth_way := (3 * start + 1 * finish) / (3 + 1)
  one_fourth_way = (7 : ℚ) / 20 := by sorry

end one_fourth_between_fractions_l3866_386661


namespace merchant_discount_percentage_l3866_386621

/-- Proves that if a merchant marks up goods by 60% and makes a 20% profit after offering a discount, then the discount percentage is 25%. -/
theorem merchant_discount_percentage 
  (markup_percentage : ℝ) 
  (profit_percentage : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : markup_percentage = 60) 
  (h2 : profit_percentage = 20) : 
  discount_percentage = 25 := by
  sorry

end merchant_discount_percentage_l3866_386621


namespace siblings_ages_l3866_386656

/-- Represents the ages of three siblings --/
structure SiblingsAges where
  david : ℕ
  yuan : ℕ
  maria : ℕ

/-- Conditions for the siblings' ages --/
def validAges (ages : SiblingsAges) : Prop :=
  ages.yuan = ages.david + 7 ∧
  ages.yuan = 2 * ages.david ∧
  ages.maria = ages.david + 4 ∧
  2 * ages.maria = ages.yuan

theorem siblings_ages :
  ∃ (ages : SiblingsAges), validAges ages ∧ ages.david = 7 ∧ ages.maria = 11 := by
  sorry

end siblings_ages_l3866_386656


namespace set_A_equivalence_l3866_386651

theorem set_A_equivalence : 
  {x : ℚ | (x + 1) * (x - 2/3) * (x^2 - 2) = 0} = {-1, 2/3} := by
  sorry

end set_A_equivalence_l3866_386651


namespace two_squares_same_plus_signs_l3866_386689

/-- Represents a cell in the 8x8 table -/
structure Cell :=
  (row : Fin 8)
  (col : Fin 8)

/-- Represents the 8x8 table with plus signs -/
def Table := Cell → Bool

/-- Represents a 4x4 square within the 8x8 table -/
structure Square :=
  (top_left_row : Fin 5)
  (top_left_col : Fin 5)

/-- Counts the number of plus signs in a given 4x4 square -/
def count_plus_signs (t : Table) (s : Square) : Nat :=
  sorry

theorem two_squares_same_plus_signs (t : Table) :
  ∃ s1 s2 : Square, s1 ≠ s2 ∧ count_plus_signs t s1 = count_plus_signs t s2 :=
sorry

end two_squares_same_plus_signs_l3866_386689


namespace line_through_point_l3866_386697

/-- Given a line equation bx + (b-1)y = b+3 that passes through the point (3, -7), prove that b = 4/5 -/
theorem line_through_point (b : ℚ) : 
  (b * 3 + (b - 1) * (-7) = b + 3) → b = 4/5 := by
  sorry

end line_through_point_l3866_386697


namespace dairy_water_mixture_l3866_386632

theorem dairy_water_mixture (pure_dairy : ℝ) (profit_percentage : ℝ) 
  (h1 : pure_dairy > 0)
  (h2 : profit_percentage = 25) : 
  let total_mixture := pure_dairy * (1 + profit_percentage / 100)
  let water_added := total_mixture - pure_dairy
  (water_added / total_mixture) * 100 = 20 := by
  sorry

end dairy_water_mixture_l3866_386632


namespace hyperbola_focal_length_l3866_386685

/-- The focal length of the hyperbola x²/10 - y²/2 = 1 is 4√3 -/
theorem hyperbola_focal_length : 
  ∃ (f : ℝ), f = 4 * Real.sqrt 3 ∧ 
  f = 2 * Real.sqrt ((10 : ℝ) + 2) ∧
  ∀ (x y : ℝ), x^2 / 10 - y^2 / 2 = 1 → 
    ∃ (c : ℝ), c = Real.sqrt ((10 : ℝ) + 2) ∧ 
    f = 2 * c :=
by sorry

end hyperbola_focal_length_l3866_386685


namespace remainder_sum_l3866_386605

theorem remainder_sum (c d : ℤ) 
  (hc : c % 60 = 53)
  (hd : d % 45 = 28) : 
  (c + d) % 15 = 6 := by
sorry

end remainder_sum_l3866_386605


namespace gcd_lcm_product_24_60_l3866_386660

theorem gcd_lcm_product_24_60 : Nat.gcd 24 60 * Nat.lcm 24 60 = 1440 := by
  sorry

end gcd_lcm_product_24_60_l3866_386660


namespace min_h_10_l3866_386608

/-- A function is stringent if f(x) + f(y) > 2y^2 for all positive integers x and y -/
def Stringent (f : ℕ+ → ℤ) : Prop :=
  ∀ x y : ℕ+, f x + f y > 2 * y.val ^ 2

/-- The sum of h from 1 to 15 -/
def SumH (h : ℕ+ → ℤ) : ℤ :=
  (Finset.range 15).sum (fun i => h ⟨i + 1, Nat.succ_pos i⟩)

theorem min_h_10 (h : ℕ+ → ℤ) (stringent_h : Stringent h) 
    (min_sum : ∀ g : ℕ+ → ℤ, Stringent g → SumH g ≥ SumH h) : 
    h ⟨10, by norm_num⟩ ≥ 136 := by
  sorry

end min_h_10_l3866_386608


namespace trivia_team_tryouts_l3866_386627

/-- 
Given 8 schools, where 17 students didn't get picked for each team,
and 384 total students make the teams, prove that 65 students tried out
for the trivia teams in each school.
-/
theorem trivia_team_tryouts (
  num_schools : ℕ) 
  (students_not_picked : ℕ) 
  (total_students_picked : ℕ) 
  (h1 : num_schools = 8)
  (h2 : students_not_picked = 17)
  (h3 : total_students_picked = 384) :
  num_schools * (65 - students_not_picked) = total_students_picked := by
  sorry

end trivia_team_tryouts_l3866_386627


namespace tom_trout_catch_l3866_386673

/-- Proves that Tom's catch equals 48 trout given the conditions -/
theorem tom_trout_catch (melanie_catch : ℕ) (tom_multiplier : ℕ) 
  (h1 : melanie_catch = 16)
  (h2 : tom_multiplier = 3) : 
  melanie_catch * tom_multiplier = 48 := by
  sorry

end tom_trout_catch_l3866_386673


namespace teds_age_l3866_386620

theorem teds_age (t s : ℕ) : t = 3 * s - 20 → t + s = 76 → t = 52 := by
  sorry

end teds_age_l3866_386620


namespace burger_cost_is_87_l3866_386639

/-- The cost of Uri's purchase in cents -/
def uri_cost : ℕ := 385

/-- The cost of Gen's purchase in cents -/
def gen_cost : ℕ := 360

/-- The number of burgers Uri bought -/
def uri_burgers : ℕ := 3

/-- The number of sodas Uri bought -/
def uri_sodas : ℕ := 2

/-- The number of burgers Gen bought -/
def gen_burgers : ℕ := 2

/-- The number of sodas Gen bought -/
def gen_sodas : ℕ := 3

/-- The cost of a burger in cents -/
def burger_cost : ℕ := 87

theorem burger_cost_is_87 :
  uri_burgers * burger_cost + uri_sodas * ((uri_cost - uri_burgers * burger_cost) / uri_sodas) = uri_cost ∧
  gen_burgers * burger_cost + gen_sodas * ((uri_cost - uri_burgers * burger_cost) / uri_sodas) = gen_cost :=
by sorry

end burger_cost_is_87_l3866_386639


namespace smallest_N_and_digit_sum_l3866_386671

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

theorem smallest_N_and_digit_sum :
  ∃ N : ℕ, 
    (∀ k : ℕ, k < N → k * (k + 1) ≤ 10^6) ∧
    N * (N + 1) > 10^6 ∧
    N = 1000 ∧
    sum_of_digits N = 1 := by
  sorry

end smallest_N_and_digit_sum_l3866_386671


namespace abs_eq_zero_iff_eq_seven_fifths_l3866_386625

theorem abs_eq_zero_iff_eq_seven_fifths (x : ℝ) : |5*x - 7| = 0 ↔ x = 7/5 := by sorry

end abs_eq_zero_iff_eq_seven_fifths_l3866_386625


namespace spinner_sectors_area_l3866_386622

/-- Represents a circular spinner with win and lose sectors. -/
structure Spinner :=
  (radius : ℝ)
  (win_prob : ℝ)
  (lose_prob : ℝ)

/-- Calculates the area of a circular sector given the total area and probability. -/
def sector_area (total_area : ℝ) (probability : ℝ) : ℝ :=
  total_area * probability

theorem spinner_sectors_area (s : Spinner) 
  (h1 : s.radius = 12)
  (h2 : s.win_prob = 1/3)
  (h3 : s.lose_prob = 1/2) :
  let total_area := Real.pi * s.radius^2
  sector_area total_area s.win_prob = 48 * Real.pi ∧
  sector_area total_area s.lose_prob = 72 * Real.pi := by
sorry

end spinner_sectors_area_l3866_386622


namespace no_1989_digit_number_sum_equals_product_l3866_386681

theorem no_1989_digit_number_sum_equals_product : ¬ ∃ (n : ℕ), 
  (n ≥ 10^1988 ∧ n < 10^1989) ∧  -- n has 1989 digits
  (∃ (d₁ d₂ d₃ : ℕ), d₁ < 1989 ∧ d₂ < 1989 ∧ d₃ < 1989 ∧ 
    d₁ ≠ d₂ ∧ d₁ ≠ d₃ ∧ d₂ ≠ d₃ ∧
    (n / 10^d₁ % 10 = 5) ∧ (n / 10^d₂ % 10 = 5) ∧ (n / 10^d₃ % 10 = 5)) ∧  -- at least three digits are 5
  (List.sum (List.map (λ i => n / 10^i % 10) (List.range 1989)) = 
   List.prod (List.map (λ i => n / 10^i % 10) (List.range 1989))) :=  -- sum of digits equals product of digits
by sorry

end no_1989_digit_number_sum_equals_product_l3866_386681


namespace brie_blouses_l3866_386644

/-- The number of blouses Brie has -/
def num_blouses : ℕ := sorry

/-- The number of skirts Brie has -/
def num_skirts : ℕ := 6

/-- The number of slacks Brie has -/
def num_slacks : ℕ := 8

/-- The percentage of blouses in the hamper -/
def blouse_hamper_percent : ℚ := 75 / 100

/-- The percentage of skirts in the hamper -/
def skirt_hamper_percent : ℚ := 50 / 100

/-- The percentage of slacks in the hamper -/
def slack_hamper_percent : ℚ := 25 / 100

/-- The total number of clothes to be washed -/
def clothes_to_wash : ℕ := 14

theorem brie_blouses : 
  num_blouses = 12 := by sorry

end brie_blouses_l3866_386644


namespace smallest_marble_count_l3866_386663

/-- Represents the number of marbles of each color in the urn -/
structure MarbleCount where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ
  yellow : ℕ

/-- Calculates the total number of marbles in the urn -/
def totalMarbles (m : MarbleCount) : ℕ :=
  m.red + m.white + m.blue + m.green + m.yellow

/-- Calculates the probability of drawing 5 red marbles -/
def probFiveRed (m : MarbleCount) : ℚ :=
  (m.red.choose 5 : ℚ) / (totalMarbles m).choose 5

/-- Calculates the probability of drawing 1 white and 4 red marbles -/
def probOneWhiteFourRed (m : MarbleCount) : ℚ :=
  ((m.white.choose 1 * m.red.choose 4) : ℚ) / (totalMarbles m).choose 5

/-- Calculates the probability of drawing 1 white, 1 blue, and 3 red marbles -/
def probOneWhiteOneBlueTwoRed (m : MarbleCount) : ℚ :=
  ((m.white.choose 1 * m.blue.choose 1 * m.red.choose 3) : ℚ) / (totalMarbles m).choose 5

/-- Calculates the probability of drawing 1 white, 1 blue, 1 green, and 2 red marbles -/
def probOneWhiteOneBlueOneGreenTwoRed (m : MarbleCount) : ℚ :=
  ((m.white.choose 1 * m.blue.choose 1 * m.green.choose 1 * m.red.choose 2) : ℚ) / (totalMarbles m).choose 5

/-- Calculates the probability of drawing one marble of each color -/
def probOneEachColor (m : MarbleCount) : ℚ :=
  ((m.white.choose 1 * m.blue.choose 1 * m.green.choose 1 * m.yellow.choose 1 * m.red.choose 1) : ℚ) / (totalMarbles m).choose 5

/-- Checks if all probabilities are equal -/
def allProbabilitiesEqual (m : MarbleCount) : Prop :=
  probFiveRed m = probOneWhiteFourRed m ∧
  probFiveRed m = probOneWhiteOneBlueTwoRed m ∧
  probFiveRed m = probOneWhiteOneBlueOneGreenTwoRed m ∧
  probFiveRed m = probOneEachColor m

/-- The main theorem stating that 33 is the smallest number of marbles satisfying the conditions -/
theorem smallest_marble_count : 
  ∃ (m : MarbleCount), totalMarbles m = 33 ∧ allProbabilitiesEqual m ∧
  ∀ (m' : MarbleCount), totalMarbles m' < 33 → ¬allProbabilitiesEqual m' :=
sorry

end smallest_marble_count_l3866_386663


namespace activity_popularity_order_l3866_386696

def soccer_popularity : ℚ := 13/40
def swimming_popularity : ℚ := 9/24
def baseball_popularity : ℚ := 11/30
def hiking_popularity : ℚ := 3/10

def activity_order : List String := ["Swimming", "Baseball", "Soccer", "Hiking"]

theorem activity_popularity_order :
  swimming_popularity > baseball_popularity ∧
  baseball_popularity > soccer_popularity ∧
  soccer_popularity > hiking_popularity :=
by sorry

end activity_popularity_order_l3866_386696


namespace distinct_triangles_count_l3866_386683

/-- The number of vertices in our geometric solid -/
def n : ℕ := 10

/-- The number of vertices needed to form a triangle -/
def k : ℕ := 3

/-- Combination function -/
def combination (n k : ℕ) : ℕ := 
  Nat.choose n k

theorem distinct_triangles_count : combination n k = 120 := by
  sorry

end distinct_triangles_count_l3866_386683


namespace unique_x_with_square_conditions_l3866_386643

theorem unique_x_with_square_conditions : ∃! (x : ℕ), 
  x > 0 ∧ 
  (∃ (n : ℕ), 2 * x + 1 = n^2) ∧ 
  (∀ (k : ℕ), (2 * x + 2 ≤ k) ∧ (k ≤ 3 * x + 2) → ¬∃ (m : ℕ), k = m^2) ∧
  x = 4 :=
by sorry

end unique_x_with_square_conditions_l3866_386643


namespace arithmetic_geometric_sequence_ratio_l3866_386675

theorem arithmetic_geometric_sequence_ratio
  (a : ℕ → ℝ)  -- arithmetic sequence
  (d : ℝ)      -- common difference
  (h1 : d ≠ 0) -- d is non-zero
  (h2 : ∀ n, a (n + 1) = a n + d)  -- definition of arithmetic sequence
  (h3 : (a 9) ^ 2 = a 5 * a 15)    -- a_5, a_9, a_15 form geometric sequence
  : (a 9) / (a 5) = 3 / 2 :=
by sorry

end arithmetic_geometric_sequence_ratio_l3866_386675


namespace definite_integral_exp_abs_x_l3866_386628

theorem definite_integral_exp_abs_x : 
  ∫ x in (-2)..4, Real.exp (|x|) = Real.exp 2 - Real.exp 1 := by sorry

end definite_integral_exp_abs_x_l3866_386628


namespace ordering_of_expressions_l3866_386648

/-- Given a = e^0.1 - 1, b = sin 0.1, and c = ln 1.1, prove that c < b < a -/
theorem ordering_of_expressions :
  let a : ℝ := Real.exp 0.1 - 1
  let b : ℝ := Real.sin 0.1
  let c : ℝ := Real.log 1.1
  c < b ∧ b < a := by sorry

end ordering_of_expressions_l3866_386648


namespace set_union_equality_l3866_386674

theorem set_union_equality (a : ℝ) : 
  let A : Set ℝ := {1, a}
  let B : Set ℝ := {a^2}
  A ∪ B = A → a = -1 ∨ a = 0 := by
sorry

end set_union_equality_l3866_386674


namespace perpendicular_parallel_perpendicular_unique_perpendicular_plane_l3866_386601

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_line_line : Line → Line → Prop)
variable (passes_through : Plane → Line → Prop)

-- Theorem 1
theorem perpendicular_parallel_perpendicular
  (a b : Line) (α : Plane) :
  parallel a α → perpendicular_line_plane b α →
  perpendicular_line_line a b :=
sorry

-- Theorem 2
theorem unique_perpendicular_plane
  (a b : Line) :
  perpendicular_line_line a b →
  ∃! p : Plane, passes_through p b ∧ perpendicular_line_plane a p :=
sorry

end perpendicular_parallel_perpendicular_unique_perpendicular_plane_l3866_386601


namespace gilda_marbles_l3866_386623

theorem gilda_marbles (M : ℝ) (h : M > 0) : 
  let remaining_after_pedro : ℝ := 0.70 * M
  let remaining_after_ebony : ℝ := 0.85 * remaining_after_pedro
  let remaining_after_jimmy : ℝ := 0.80 * remaining_after_ebony
  let final_remaining : ℝ := 0.90 * remaining_after_jimmy
  final_remaining / M = 0.4284 := by
sorry

end gilda_marbles_l3866_386623


namespace quadratic_radical_condition_l3866_386618

theorem quadratic_radical_condition (x : ℝ) : Real.sqrt ((x - 3)^2) = x - 3 ↔ x ≥ 3 := by
  sorry

end quadratic_radical_condition_l3866_386618


namespace x_power_3a_minus_2b_l3866_386655

theorem x_power_3a_minus_2b (x a b : ℝ) (h1 : x^a = 3) (h2 : x^b = 4) :
  x^(3*a - 2*b) = 27/16 := by
sorry

end x_power_3a_minus_2b_l3866_386655


namespace cara_family_age_difference_l3866_386676

/-- The age difference between Cara's grandmother and Cara's mom -/
def age_difference (cara_age mom_age grandma_age : ℕ) : ℕ :=
  grandma_age - mom_age

theorem cara_family_age_difference :
  ∀ (cara_age mom_age grandma_age : ℕ),
    cara_age = 40 →
    mom_age = cara_age + 20 →
    grandma_age = 75 →
    age_difference cara_age mom_age grandma_age = 15 :=
by
  sorry

end cara_family_age_difference_l3866_386676


namespace number_guessing_game_l3866_386658

theorem number_guessing_game (a b c : ℕ) : 
  a > 0 ∧ a < 10 ∧ b > 0 ∧ b < 10 ∧ c > 0 ∧ c < 10 →
  ((2 * a + 2) * 5 + b) * 10 + c = 567 →
  a = 4 ∧ b = 6 ∧ c = 7 :=
by sorry

end number_guessing_game_l3866_386658


namespace y_satisfies_differential_equation_l3866_386617

-- Define the function y
noncomputable def y (x : ℝ) : ℝ :=
  Real.sqrt ((Real.log ((1 + Real.exp x) / 2))^2 + 1)

-- State the theorem
theorem y_satisfies_differential_equation (x : ℝ) :
  (1 + Real.exp x) * y x * (deriv y x) = Real.exp x := by
  sorry

end y_satisfies_differential_equation_l3866_386617


namespace doughnuts_per_staff_member_l3866_386649

theorem doughnuts_per_staff_member 
  (total_doughnuts : ℕ) 
  (staff_members : ℕ) 
  (doughnuts_left : ℕ) 
  (h1 : total_doughnuts = 50) 
  (h2 : staff_members = 19) 
  (h3 : doughnuts_left = 12) : 
  (total_doughnuts - doughnuts_left) / staff_members = 2 :=
sorry

end doughnuts_per_staff_member_l3866_386649


namespace inequality_and_minimum_value_l3866_386626

theorem inequality_and_minimum_value 
  (a b x y : ℝ) 
  (ha : 0 < a) 
  (hb : 0 < b) 
  (hx : 0 < x) 
  (hy : 0 < y) 
  (hxy : x + y = 1) : 
  (a + b ≥ 2 * Real.sqrt (a * b)) ∧ 
  (∃ (min : ℝ), min = 9 ∧ ∀ (z w : ℝ), 0 < z → 0 < w → z + w = 1 → 1/z + 4/w ≥ min) := by
  sorry

end inequality_and_minimum_value_l3866_386626


namespace ellipse_line_slope_l3866_386694

/-- The slope of a line passing through the right focus of an ellipse -/
theorem ellipse_line_slope (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := Real.sqrt 3 / 2
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let F := (Real.sqrt (a^2 - b^2), 0)
  ∀ (A B : ℝ × ℝ),
    ellipse A.1 A.2 ∧ 
    ellipse B.1 B.2 ∧ 
    A.2 > 0 ∧ 
    B.2 < 0 ∧
    (A.1 - F.1, A.2 - F.2) = 3 • (F.1 - B.1, F.2 - B.2) →
    (A.2 - B.2) / (A.1 - B.1) = -Real.sqrt 2 :=
by sorry

end ellipse_line_slope_l3866_386694


namespace name_tag_paper_perimeter_l3866_386666

theorem name_tag_paper_perimeter :
  ∀ (num_students : ℕ) (tag_side_length : ℝ) (paper_width : ℝ) (unused_width : ℝ),
    num_students = 24 →
    tag_side_length = 4 →
    paper_width = 34 →
    unused_width = 2 →
    (paper_width - unused_width) / tag_side_length * tag_side_length * 
      (num_students / ((paper_width - unused_width) / tag_side_length)) = 
      paper_width - unused_width →
    2 * (paper_width + (num_students / ((paper_width - unused_width) / tag_side_length)) * tag_side_length) = 92 := by
  sorry

end name_tag_paper_perimeter_l3866_386666
