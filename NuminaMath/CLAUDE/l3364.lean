import Mathlib

namespace bob_questions_theorem_l3364_336402

/-- Represents the number of questions Bob creates in each hour -/
def questions_per_hour : Fin 3 → ℕ
  | 0 => 13
  | 1 => 13 * 2
  | 2 => 13 * 2 * 2

/-- The total number of questions Bob creates in three hours -/
def total_questions : ℕ := (questions_per_hour 0) + (questions_per_hour 1) + (questions_per_hour 2)

theorem bob_questions_theorem :
  total_questions = 91 := by
  sorry

end bob_questions_theorem_l3364_336402


namespace unique_positive_integer_solution_l3364_336498

theorem unique_positive_integer_solution :
  ∃! (x : ℕ), x > 0 ∧ (3 * x : ℤ) - 5 < 1 :=
by sorry

end unique_positive_integer_solution_l3364_336498


namespace first_nonzero_digit_of_1_over_137_l3364_336470

theorem first_nonzero_digit_of_1_over_137 :
  ∃ (n : ℕ) (r : ℚ), (1 : ℚ) / 137 = n / 10^(n.succ) + r ∧ 0 < r ∧ r < 1 / 10^n ∧ n = 7 :=
by sorry

end first_nonzero_digit_of_1_over_137_l3364_336470


namespace ram_selection_probability_l3364_336430

theorem ram_selection_probability
  (p_ravi : ℝ)
  (p_both : ℝ)
  (h_ravi : p_ravi = 1 / 5)
  (h_both : p_both = 0.05714285714285714)
  (h_independent : ∀ p_ram : ℝ, p_both = p_ram * p_ravi) :
  ∃ p_ram : ℝ, p_ram = 2 / 7 :=
by sorry

end ram_selection_probability_l3364_336430


namespace shooting_probability_l3364_336426

/-- The probability of hitting a shot -/
def shooting_accuracy : ℚ := 9/10

/-- The probability of hitting two consecutive shots -/
def two_consecutive_hits : ℚ := 1/2

/-- The probability of hitting the next shot given that the first shot was hit -/
def next_shot_probability : ℚ := 5/9

theorem shooting_probability :
  shooting_accuracy = 9/10 →
  two_consecutive_hits = 1/2 →
  next_shot_probability = two_consecutive_hits / shooting_accuracy :=
by sorry

end shooting_probability_l3364_336426


namespace vacation_pictures_l3364_336432

theorem vacation_pictures (zoo museum amusement beach deleted : ℕ) 
  (h1 : zoo = 120)
  (h2 : museum = 34)
  (h3 : amusement = 25)
  (h4 : beach = 21)
  (h5 : deleted = 73) :
  zoo + museum + amusement + beach - deleted = 127 := by
  sorry

end vacation_pictures_l3364_336432


namespace problem_1_problem_2_problem_3_l3364_336403

-- Define the variables and conditions
variable (a b c : ℝ)
variable (h1 : 2 * a + b = c)
variable (h2 : c ≠ 0)

-- Theorem 1
theorem problem_1 : (2 * a + b - c - 1)^2023 = -1 := by sorry

-- Theorem 2
theorem problem_2 : (10 * c) / (4 * a + 2 * b) = 5 := by sorry

-- Theorem 3
theorem problem_3 : (2 * a + b) * 3 = c + 4 * a + 2 * b := by sorry

end problem_1_problem_2_problem_3_l3364_336403


namespace count_special_numbers_l3364_336429

def is_valid_three_digit_number (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

def reverse_number (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  ones * 100 + tens * 10 + hundreds

theorem count_special_numbers :
  ∃ (S : Finset ℕ), 
    (∀ n ∈ S, is_valid_three_digit_number n ∧ is_valid_three_digit_number (n - 297) ∧ 
               n - 297 = reverse_number n) ∧
    S.card = 60 :=
sorry

end count_special_numbers_l3364_336429


namespace sqrt_abs_sum_zero_implies_sum_power_l3364_336400

theorem sqrt_abs_sum_zero_implies_sum_power (a b : ℝ) :
  Real.sqrt (a - 2) + |b + 1| = 0 → (a + b)^2023 = 1 := by
sorry

end sqrt_abs_sum_zero_implies_sum_power_l3364_336400


namespace kathryn_remaining_money_l3364_336433

/-- Calculates the remaining money for Kathryn after expenses --/
def remaining_money (rent : ℕ) (salary : ℕ) : ℕ :=
  let food_travel : ℕ := 2 * rent
  let rent_share : ℕ := rent / 2
  let total_expenses : ℕ := rent_share + food_travel
  salary - total_expenses

/-- Proves that Kathryn's remaining money after expenses is $2000 --/
theorem kathryn_remaining_money :
  remaining_money 1200 5000 = 2000 := by
  sorry

#eval remaining_money 1200 5000

end kathryn_remaining_money_l3364_336433


namespace one_point_one_billion_scientific_notation_l3364_336482

/-- Expresses 1.1 billion in scientific notation -/
theorem one_point_one_billion_scientific_notation :
  (1.1 * 10^9 : ℝ) = 1100000000 := by
  sorry

end one_point_one_billion_scientific_notation_l3364_336482


namespace max_ratio_two_digit_integers_l3364_336483

theorem max_ratio_two_digit_integers (x y : ℕ) : 
  10 ≤ x ∧ x < 100 ∧ 10 ≤ y ∧ y < 100 →  -- x and y are two-digit positive integers
  (x + y) / 2 = 55 →                     -- mean of x and y is 55
  ∃ (z : ℕ), x * y = z ^ 2 →             -- product xy is a square number
  ∀ (a b : ℕ), 
    10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧
    (a + b) / 2 = 55 ∧
    (∃ (w : ℕ), a * b = w ^ 2) →
    x / y ≥ a / b →
  x / y ≤ 9 :=
by sorry

end max_ratio_two_digit_integers_l3364_336483


namespace hilt_pencil_cost_l3364_336459

/-- The cost of a pencil given total money and number of pencils that can be bought --/
def pencil_cost (total_money : ℚ) (num_pencils : ℕ) : ℚ :=
  total_money / num_pencils

theorem hilt_pencil_cost :
  pencil_cost 50 10 = 5 := by
  sorry

end hilt_pencil_cost_l3364_336459


namespace difference_of_squares_l3364_336461

theorem difference_of_squares (a b : ℝ) : (a + b) * (a - b) = a^2 - b^2 := by
  sorry

end difference_of_squares_l3364_336461


namespace average_age_increase_l3364_336414

theorem average_age_increase (initial_count : ℕ) (replaced_count : ℕ) (age1 age2 : ℕ) (women_avg_age : ℚ) : 
  initial_count = 7 →
  replaced_count = 2 →
  age1 = 18 →
  age2 = 22 →
  women_avg_age = 30.5 →
  (2 * women_avg_age - (age1 + age2 : ℚ)) / initial_count = 3 := by
  sorry

end average_age_increase_l3364_336414


namespace fraction_addition_l3364_336448

theorem fraction_addition (a : ℝ) (ha : a ≠ 0) : 3 / a + 2 / a = 5 / a := by
  sorry

end fraction_addition_l3364_336448


namespace equation_solution_l3364_336465

theorem equation_solution : 
  ∃ x : ℝ, (x + 1) / (x - 1) = 1 / (x - 2) + 1 ↔ x = 3 := by sorry

end equation_solution_l3364_336465


namespace polynomial_division_theorem_l3364_336407

theorem polynomial_division_theorem (x : ℝ) : 
  12 * x^3 + 18 * x^2 + 27 * x + 17 = 
  (4 * x + 3) * (3 * x^2 + 2.25 * x + 5/16) + 29/16 := by
sorry

end polynomial_division_theorem_l3364_336407


namespace minimum_fourth_quarter_score_l3364_336438

def required_average : ℚ := 85
def num_quarters : ℕ := 4
def first_quarter : ℚ := 82
def second_quarter : ℚ := 77
def third_quarter : ℚ := 78

theorem minimum_fourth_quarter_score :
  let total_required := required_average * num_quarters
  let sum_first_three := first_quarter + second_quarter + third_quarter
  let minimum_fourth := total_required - sum_first_three
  minimum_fourth = 103 ∧
  (first_quarter + second_quarter + third_quarter + minimum_fourth) / num_quarters ≥ required_average :=
by sorry

end minimum_fourth_quarter_score_l3364_336438


namespace parabola_parameter_values_l3364_336489

/-- Represents a point on a parabola -/
structure ParabolaPoint where
  x : ℝ
  y : ℝ

/-- Represents a parabola y² = 2px -/
structure Parabola where
  p : ℝ
  hp : p > 0

/-- The point satisfies the parabola equation -/
def on_parabola (point : ParabolaPoint) (parabola : Parabola) : Prop :=
  point.y^2 = 2 * parabola.p * point.x

/-- The distance from the point to the directrix (x = -p/2) is 10 -/
def distance_to_directrix (point : ParabolaPoint) (parabola : Parabola) : Prop :=
  point.x + parabola.p / 2 = 10

/-- The distance from the point to the axis of symmetry (y-axis) is 6 -/
def distance_to_axis (point : ParabolaPoint) : Prop :=
  point.y = 6 ∨ point.y = -6

theorem parabola_parameter_values
  (parabola : Parabola)
  (point : ParabolaPoint)
  (h_on_parabola : on_parabola point parabola)
  (h_directrix : distance_to_directrix point parabola)
  (h_axis : distance_to_axis point) :
  parabola.p = 2 ∨ parabola.p = 18 := by
  sorry

end parabola_parameter_values_l3364_336489


namespace sin_2alpha_value_l3364_336449

theorem sin_2alpha_value (α : ℝ) (h : Real.cos (α + π / 4) = 1 / 3) : 
  Real.sin (2 * α) = 7 / 9 := by
  sorry

end sin_2alpha_value_l3364_336449


namespace line_slope_intercept_sum_l3364_336443

/-- A line with slope 6 passing through (4, -3) and intersecting y = -x + 1 has m + b = -21 -/
theorem line_slope_intercept_sum (m b : ℝ) : 
  m = 6 ∧ 
  -3 = 6 * 4 + b ∧ 
  ∃ x y : ℝ, y = 6 * x + b ∧ y = -x + 1 →
  m + b = -21 := by
  sorry

end line_slope_intercept_sum_l3364_336443


namespace bee_colony_fraction_l3364_336405

theorem bee_colony_fraction (initial_bees : ℕ) (daily_loss : ℕ) (days : ℕ) :
  initial_bees = 80000 →
  daily_loss = 1200 →
  days = 50 →
  (initial_bees - daily_loss * days) / initial_bees = 1 / 4 := by
sorry

end bee_colony_fraction_l3364_336405


namespace inscribed_cube_side_length_l3364_336460

/-- A cone with a circular base of radius 1 and height 3 --/
structure Cone :=
  (base_radius : ℝ := 1)
  (height : ℝ := 3)

/-- A cube inscribed in a cone such that four vertices lie on the base and four on the sloping sides --/
structure InscribedCube :=
  (cone : Cone)
  (side_length : ℝ)
  (four_vertices_on_base : Prop)
  (four_vertices_on_slope : Prop)

/-- The side length of the inscribed cube is 3√2 / (3 + √2) --/
theorem inscribed_cube_side_length (cube : InscribedCube) :
  cube.side_length = 3 * Real.sqrt 2 / (3 + Real.sqrt 2) := by
  sorry

end inscribed_cube_side_length_l3364_336460


namespace zachary_crunches_l3364_336472

/-- Proves that Zachary did 58 crunches given the problem conditions -/
theorem zachary_crunches : 
  ∀ (zachary_pushups zachary_crunches david_pushups david_crunches : ℕ),
  zachary_pushups = 46 →
  david_pushups = zachary_pushups + 38 →
  david_crunches = zachary_crunches - 62 →
  zachary_crunches = zachary_pushups + 12 →
  zachary_crunches = 58 := by
  sorry

end zachary_crunches_l3364_336472


namespace choir_size_after_new_members_l3364_336491

theorem choir_size_after_new_members (original : Nat) (new : Nat) : 
  original = 36 → new = 9 → original + new = 45 := by
  sorry

end choir_size_after_new_members_l3364_336491


namespace determinant_of_2x2_matrix_l3364_336406

theorem determinant_of_2x2_matrix :
  let A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -2; 3, 5]
  Matrix.det A = 41 := by
  sorry

end determinant_of_2x2_matrix_l3364_336406


namespace correct_stratified_sample_l3364_336420

/-- Represents the number of students in each grade --/
structure GradePopulation where
  freshmen : ℕ
  sophomores : ℕ
  seniors : ℕ

/-- Represents the number of students sampled from each grade --/
structure SampleSize where
  freshmen : ℕ
  sophomores : ℕ
  seniors : ℕ

/-- Calculates the stratified sample size for each grade --/
def stratifiedSample (pop : GradePopulation) (totalSample : ℕ) : SampleSize :=
  let totalPop := pop.freshmen + pop.sophomores + pop.seniors
  { freshmen := (totalSample * pop.freshmen) / totalPop,
    sophomores := (totalSample * pop.sophomores) / totalPop,
    seniors := (totalSample * pop.seniors) / totalPop }

/-- Theorem stating the correct stratified sample sizes for the given population --/
theorem correct_stratified_sample :
  let pop : GradePopulation := { freshmen := 900, sophomores := 1200, seniors := 600 }
  let sample := stratifiedSample pop 135
  sample.freshmen = 45 ∧ sample.sophomores = 60 ∧ sample.seniors = 30 := by
  sorry

end correct_stratified_sample_l3364_336420


namespace sequence_representation_l3364_336446

theorem sequence_representation (q : ℕ → ℕ) 
  (h_increasing : ∀ n, q n < q (n + 1))
  (h_bound : ∀ n, q n < 2 * n) :
  ∀ m : ℕ, ∃ k l : ℕ, q k = m ∨ q l - q k = m :=
sorry

end sequence_representation_l3364_336446


namespace gcd_960_1632_l3364_336437

theorem gcd_960_1632 : Nat.gcd 960 1632 = 96 := by
  sorry

end gcd_960_1632_l3364_336437


namespace equation_solver_l3364_336468

theorem equation_solver (m : ℕ) (p : ℝ) 
  (h1 : ((1^m) / (5^m)) * ((1^16) / (4^16)) = 1 / (2*(p^31)))
  (h2 : m = 31) : 
  p = 10 := by
  sorry

end equation_solver_l3364_336468


namespace geometric_sum_five_terms_l3364_336423

/-- Given a geometric sequence with first term a and common ratio r,
    find n such that the sum of the first n terms is equal to s. -/
def find_n_for_geometric_sum (a r s : ℚ) : ℕ :=
  sorry

theorem geometric_sum_five_terms
  (a r : ℚ)
  (h_a : a = 1/3)
  (h_r : r = 1/3)
  (h_sum : (a * (1 - r^5)) / (1 - r) = 80/243) :
  find_n_for_geometric_sum a r (80/243) = 5 :=
sorry

end geometric_sum_five_terms_l3364_336423


namespace triangular_frame_is_stable_bicycle_frame_triangle_stability_l3364_336441

/-- A bicycle frame is a structure used in bicycles. -/
structure BicycleFrame where
  shape : Type

/-- A triangle is a geometric shape with three sides. -/
inductive Triangle : Type

/-- Stability is a property that can be possessed by structures. -/
class Stable (α : Type) where
  is_stable : α → Prop

/-- A bicycle frame made in the shape of a triangle -/
def triangular_frame : BicycleFrame := { shape := Triangle }

/-- The theorem stating that a triangular bicycle frame is stable -/
theorem triangular_frame_is_stable :
  Stable Triangle → Stable (triangular_frame.shape) :=
by
  sorry

/-- The main theorem proving that a bicycle frame made in the shape of a triangle is stable -/
theorem bicycle_frame_triangle_stability :
  Stable (triangular_frame.shape) :=
by
  sorry

end triangular_frame_is_stable_bicycle_frame_triangle_stability_l3364_336441


namespace circle_symmetric_l3364_336424

-- Define a circle
def Circle : Type := Unit

-- Define axisymmetric property
def isAxisymmetric (shape : Type) : Prop := sorry

-- Define centrally symmetric property
def isCentrallySymmetric (shape : Type) : Prop := sorry

-- Theorem stating that a circle is both axisymmetric and centrally symmetric
theorem circle_symmetric : isAxisymmetric Circle ∧ isCentrallySymmetric Circle := by
  sorry

end circle_symmetric_l3364_336424


namespace cuboids_painted_count_l3364_336458

/-- The number of outer faces on a single cuboid -/
def faces_per_cuboid : ℕ := 6

/-- The total number of painted faces -/
def total_painted_faces : ℕ := 60

/-- The number of cuboids painted -/
def num_cuboids : ℕ := total_painted_faces / faces_per_cuboid

theorem cuboids_painted_count : num_cuboids = 10 := by
  sorry

end cuboids_painted_count_l3364_336458


namespace friends_who_ate_bread_l3364_336469

theorem friends_who_ate_bread (loaves : ℕ) (slices_per_loaf : ℕ) (slices_per_friend : ℕ) :
  loaves = 4 →
  slices_per_loaf = 15 →
  slices_per_friend = 6 →
  (loaves * slices_per_loaf) % slices_per_friend = 0 →
  (loaves * slices_per_loaf) / slices_per_friend = 10 := by
  sorry

end friends_who_ate_bread_l3364_336469


namespace place_mat_length_l3364_336473

theorem place_mat_length (r : ℝ) (n : ℕ) (w : ℝ) (x : ℝ) : 
  r = 4 →
  n = 6 →
  w = 1 →
  (x + 2 * Real.sqrt 3 - 1/2)^2 = 63/4 →
  x = (3 * Real.sqrt 7 - Real.sqrt 3) / 2 :=
sorry

end place_mat_length_l3364_336473


namespace remaining_quarters_count_l3364_336440

def initial_amount : ℚ := 40
def pizza_cost : ℚ := 2.75
def soda_cost : ℚ := 1.5
def jeans_cost : ℚ := 11.5

def remaining_money : ℚ := initial_amount - (pizza_cost + soda_cost + jeans_cost)

def quarters_in_dollar : ℕ := 4

theorem remaining_quarters_count : 
  (remaining_money * quarters_in_dollar).floor = 97 := by sorry

end remaining_quarters_count_l3364_336440


namespace uncommon_card_cost_is_half_dollar_l3364_336485

/-- The cost of an uncommon card in Tom's deck -/
def uncommon_card_cost : ℚ :=
  let rare_cards : ℕ := 19
  let uncommon_cards : ℕ := 11
  let common_cards : ℕ := 30
  let rare_card_cost : ℚ := 1
  let common_card_cost : ℚ := 1/4
  let total_deck_cost : ℚ := 32
  (total_deck_cost - rare_cards * rare_card_cost - common_cards * common_card_cost) / uncommon_cards

theorem uncommon_card_cost_is_half_dollar : uncommon_card_cost = 1/2 := by
  sorry

end uncommon_card_cost_is_half_dollar_l3364_336485


namespace no_real_solutions_l3364_336492

theorem no_real_solutions :
  ¬∃ y : ℝ, (8 * y^2 + 47 * y + 5) / (4 * y + 15) = 4 * y + 2 :=
by sorry

end no_real_solutions_l3364_336492


namespace translation_theorem_l3364_336494

/-- A point in the Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point horizontally -/
def translateHorizontal (p : Point) (dx : ℝ) : Point :=
  { x := p.x + dx, y := p.y }

/-- Translate a point vertically -/
def translateVertical (p : Point) (dy : ℝ) : Point :=
  { x := p.x, y := p.y + dy }

/-- The main theorem -/
theorem translation_theorem (m n : ℝ) :
  let p := Point.mk m n
  let p' := translateVertical (translateHorizontal p 2) 1
  p'.x = m + 2 ∧ p'.y = n + 1 := by
  sorry

end translation_theorem_l3364_336494


namespace log_equation_solution_l3364_336428

theorem log_equation_solution (x : ℝ) (h1 : x > 0) (h2 : x ≠ 1) :
  (Real.log x / Real.log 4) * (Real.log 8 / Real.log x) = Real.log 8 / Real.log 4 :=
by sorry

end log_equation_solution_l3364_336428


namespace imaginary_part_of_z_l3364_336415

theorem imaginary_part_of_z (z : ℂ) (h : (1 + Complex.I) * z = 1 - z) :
  Complex.im z = -1/5 := by sorry

end imaginary_part_of_z_l3364_336415


namespace girls_in_college_l3364_336410

theorem girls_in_college (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 600)
  (h2 : boys_ratio = 8)
  (h3 : girls_ratio = 4) :
  (girls_ratio * total_students) / (boys_ratio + girls_ratio) = 200 :=
sorry

end girls_in_college_l3364_336410


namespace integer_solutions_of_equation_l3364_336481

theorem integer_solutions_of_equation :
  {(x, y) : ℤ × ℤ | (x^2 - y^2)^2 = 16*y + 1} =
  {(1, 0), (-1, 0), (4, 3), (-4, 3), (4, 5), (-4, 5)} := by
  sorry

end integer_solutions_of_equation_l3364_336481


namespace triangle_side_range_l3364_336421

theorem triangle_side_range (a b c : ℝ) (A B C : ℝ) :
  c = Real.sqrt 2 →
  a * Real.cos C = c * Real.sin A →
  0 < A ∧ A < π →
  0 < B ∧ B < π →
  0 < C ∧ C < π →
  A + B + C = π →
  Real.sqrt 2 < b ∧ b < 2 :=
sorry

end triangle_side_range_l3364_336421


namespace triangle_sides_theorem_l3364_336413

def triangle_exists (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

theorem triangle_sides_theorem (x : ℕ+) :
  triangle_exists 8 11 (x.val ^ 2) ↔ x.val = 2 ∨ x.val = 3 ∨ x.val = 4 := by
  sorry

end triangle_sides_theorem_l3364_336413


namespace carries_strawberry_harvest_l3364_336447

/-- Calculates the expected strawberry harvest from a rectangular garden. -/
def strawberry_harvest (length width planting_density yield_per_plant : ℕ) : ℕ :=
  length * width * planting_density * yield_per_plant

/-- Proves that Carrie's garden will yield 7200 strawberries. -/
theorem carries_strawberry_harvest :
  strawberry_harvest 10 12 5 12 = 7200 := by
  sorry

end carries_strawberry_harvest_l3364_336447


namespace expected_ones_value_l3364_336451

/-- The number of magnets --/
def n : ℕ := 50

/-- The probability of a difference of 1 between two randomly chosen numbers --/
def p : ℚ := 49 / 1225

/-- The number of pairs of consecutive magnets --/
def num_pairs : ℕ := n - 1

/-- The expected number of times the difference 1 occurs --/
def expected_ones : ℚ := num_pairs * p

theorem expected_ones_value : expected_ones = 49 / 25 := by sorry

end expected_ones_value_l3364_336451


namespace cone_volume_from_semicircle_l3364_336419

/-- The volume of a cone whose development diagram is a semicircle with radius 2 -/
theorem cone_volume_from_semicircle (r : Real) (l : Real) (h : Real) : 
  l = 2 → 
  2 * π * r = π * 2 → 
  h^2 + r^2 = l^2 → 
  (1/3 : Real) * π * r^2 * h = (Real.sqrt 3 * π) / 3 := by
  sorry

end cone_volume_from_semicircle_l3364_336419


namespace max_cities_is_107_l3364_336411

/-- The maximum number of cities that can be visited in a specific sequence -/
def max_cities : ℕ := 107

/-- The total number of cities in the country -/
def total_cities : ℕ := 110

/-- A function representing the number of roads for each city in the sequence -/
def roads_for_city (k : ℕ) : ℕ := k

/-- Theorem stating that the maximum number of cities that can be visited in the specific sequence is 107 -/
theorem max_cities_is_107 :
  ∀ N : ℕ, 
  (∀ k : ℕ, 2 ≤ k → k ≤ N → roads_for_city k = k) →
  N ≤ total_cities →
  N ≤ max_cities :=
sorry

end max_cities_is_107_l3364_336411


namespace pet_shop_legs_l3364_336467

/-- The total number of legs in a pet shop with birds, dogs, snakes, and spiders -/
def total_legs (num_birds num_dogs num_snakes num_spiders : ℕ) 
               (bird_legs dog_legs snake_legs spider_legs : ℕ) : ℕ :=
  num_birds * bird_legs + num_dogs * dog_legs + num_snakes * snake_legs + num_spiders * spider_legs

/-- Theorem stating that the total number of legs in the given pet shop scenario is 34 -/
theorem pet_shop_legs : 
  total_legs 3 5 4 1 2 4 0 8 = 34 := by
  sorry

end pet_shop_legs_l3364_336467


namespace range_of_b_l3364_336409

theorem range_of_b (b : ℝ) : 
  (∀ a : ℝ, a ≤ -1 → a * 2 * b - b - 3 * a ≥ 0) → 
  b ∈ Set.Iic 1 :=
sorry

end range_of_b_l3364_336409


namespace clock_angle_at_7_clock_angle_at_7_is_150_l3364_336479

/-- The smaller angle formed by the hands of a clock at 7 o'clock is 150 degrees. -/
theorem clock_angle_at_7 : ℝ :=
  let total_hours : ℕ := 12
  let total_degrees : ℝ := 360
  let hours_at_7 : ℕ := 7
  let angle_per_hour : ℝ := total_degrees / total_hours
  let hour_hand_angle : ℝ := angle_per_hour * hours_at_7
  let smaller_angle : ℝ := total_degrees - hour_hand_angle
  smaller_angle

/-- The theorem states that the smaller angle formed by the hands of a clock at 7 o'clock is 150 degrees. -/
theorem clock_angle_at_7_is_150 : clock_angle_at_7 = 150 := by
  sorry

end clock_angle_at_7_clock_angle_at_7_is_150_l3364_336479


namespace binomial_and_factorial_l3364_336455

theorem binomial_and_factorial : 
  (Nat.choose 10 5 = 252) ∧ (Nat.factorial (Nat.choose 10 5 - 5) = Nat.factorial 247) := by
  sorry

end binomial_and_factorial_l3364_336455


namespace max_stores_visited_is_three_l3364_336457

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  num_stores : ℕ
  total_visits : ℕ
  num_shoppers : ℕ
  two_store_visitors : ℕ

/-- The given shopping scenario -/
def given_scenario : ShoppingScenario :=
  { num_stores := 8
  , total_visits := 22
  , num_shoppers := 12
  , two_store_visitors := 8 }

/-- The maximum number of stores visited by any single person -/
def max_stores_visited (scenario : ShoppingScenario) : ℕ :=
  3

/-- Theorem stating that the maximum number of stores visited by any single person is 3 -/
theorem max_stores_visited_is_three (scenario : ShoppingScenario) 
  (h1 : scenario.num_stores = given_scenario.num_stores)
  (h2 : scenario.total_visits = given_scenario.total_visits)
  (h3 : scenario.num_shoppers = given_scenario.num_shoppers)
  (h4 : scenario.two_store_visitors = given_scenario.two_store_visitors)
  (h5 : scenario.two_store_visitors * 2 + (scenario.num_shoppers - scenario.two_store_visitors) ≤ scenario.total_visits)
  : max_stores_visited scenario = 3 := by
  sorry

end max_stores_visited_is_three_l3364_336457


namespace nitin_rank_last_l3364_336486

def class_size : ℕ := 58
def nitin_rank_start : ℕ := 24

theorem nitin_rank_last : class_size - nitin_rank_start + 1 = 35 := by
  sorry

end nitin_rank_last_l3364_336486


namespace expression_evaluation_l3364_336427

/-- Proves that the given expression evaluates to -3/2 when x = -1/2 and y = 3 -/
theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := 3
  3 * (2 * x^2 * y - x * y^2) - 2 * (-2 * y^2 * x + x^2 * y) = -3/2 := by
  sorry

end expression_evaluation_l3364_336427


namespace perpendicular_line_equation_l3364_336490

/-- A line passing through (-1, 2) and perpendicular to 2x - 3y + 4 = 0 has the equation 3x + 2y - 1 = 0 -/
theorem perpendicular_line_equation :
  ∀ (l : Set (ℝ × ℝ)),
  ((-1, 2) ∈ l) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ ∃ (t : ℝ), x = -1 + 3*t ∧ y = 2 - 2*t) →
  (∀ (x y : ℝ), (x, y) ∈ l ↔ 3*x + 2*y - 1 = 0) :=
by sorry

end perpendicular_line_equation_l3364_336490


namespace sequence_properties_l3364_336497

def a (n : ℕ+) : ℚ := (3 * n - 2) / (3 * n + 1)

theorem sequence_properties :
  (a 10 = 28 / 31) ∧
  (a 3 = 7 / 10) ∧
  (∀ n : ℕ+, 0 < a n ∧ a n < 1) := by
  sorry

end sequence_properties_l3364_336497


namespace difference_of_percentages_l3364_336463

-- Define the percentage
def percentage : ℚ := 25 / 100

-- Define the two amounts in pence (to avoid floating-point issues)
def amount1 : ℕ := 3700  -- £37 in pence
def amount2 : ℕ := 1700  -- £17 in pence

-- Theorem statement
theorem difference_of_percentages :
  (percentage * amount1 - percentage * amount2 : ℚ) = 500 := by
  sorry

end difference_of_percentages_l3364_336463


namespace range_of_a_l3364_336488

def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

theorem range_of_a (a : ℝ) (ha : a ∈ A) : a ∈ Set.Icc (-1) 3 := by
  sorry

end range_of_a_l3364_336488


namespace triangle_area_extension_l3364_336435

/-- Given a triangle ABC with area 36 and base BC of length 7, and an extended triangle BCD
    with CD of length 30, prove that the area of BCD is 1080/7. -/
theorem triangle_area_extension (h : ℝ) : 
  36 = (1/2) * 7 * h →  -- Area of ABC
  (1/2) * 30 * h = 1080/7 := by
  sorry

end triangle_area_extension_l3364_336435


namespace magnitude_a_minus_b_equals_5_l3364_336477

def vector_a : ℝ × ℝ := (-1, 1)
def vector_b : ℝ × ℝ := (3, -2)

theorem magnitude_a_minus_b_equals_5 :
  Real.sqrt ((vector_a.1 - vector_b.1)^2 + (vector_a.2 - vector_b.2)^2) = 5 := by
  sorry

end magnitude_a_minus_b_equals_5_l3364_336477


namespace perfect_square_trinomial_l3364_336450

theorem perfect_square_trinomial (m : ℝ) : 
  (∃ a : ℝ, ∀ x : ℝ, x^2 + 2*(m+1)*x + 25 = (x + a)^2) → 
  (m = 4 ∨ m = -6) :=
by sorry

end perfect_square_trinomial_l3364_336450


namespace solve_equation_l3364_336418

theorem solve_equation (n : ℕ) : 2^(2*n) + 2^(2*n) + 2^(2*n) + 2^(2*n) = 4^18 → n = 9 := by
  sorry

end solve_equation_l3364_336418


namespace mixture_volume_l3364_336454

/-- Given a mixture of milk and water, prove that the initial volume is 145 liters -/
theorem mixture_volume (initial_milk : ℝ) (initial_water : ℝ) : 
  initial_milk / initial_water = 3 / 2 →
  initial_milk / (initial_water + 58) = 3 / 4 →
  initial_milk + initial_water = 145 := by
sorry

end mixture_volume_l3364_336454


namespace simplify_expression_l3364_336499

theorem simplify_expression (y : ℝ) : 3*y + 4*y^2 + 2 - (7 - 3*y - 4*y^2) = 8*y^2 + 6*y - 5 := by
  sorry

end simplify_expression_l3364_336499


namespace picture_arrangements_l3364_336401

/-- The number of people in the initial group -/
def initial_group_size : ℕ := 4

/-- The number of people combined into one unit -/
def combined_unit_size : ℕ := 2

/-- The effective number of units to arrange -/
def effective_units : ℕ := initial_group_size - combined_unit_size + 1

theorem picture_arrangements :
  (effective_units).factorial = 6 := by
  sorry

end picture_arrangements_l3364_336401


namespace special_quadratic_property_l3364_336471

/-- A quadratic function f(x) = x^2 + ax + b satisfying specific conditions -/
def special_quadratic (a b : ℝ) : ℝ → ℝ := fun x ↦ x^2 + a*x + b

/-- Theorem: If f(f(0)) = f(f(1)) = 0 and f(0) ≠ f(1), then f(2) = 3 -/
theorem special_quadratic_property (a b : ℝ) :
  let f := special_quadratic a b
  (f (f 0) = 0) → (f (f 1) = 0) → (f 0 ≠ f 1) → (f 2 = 3) := by
  sorry

end special_quadratic_property_l3364_336471


namespace unique_solution_l3364_336422

-- Define the equation
def equation (x : ℝ) : Prop :=
  2021 * x = 2022 * (x^2021)^(1/2021) - 1

-- Theorem statement
theorem unique_solution :
  ∃! x : ℝ, x ≥ 0 ∧ equation x :=
sorry

end unique_solution_l3364_336422


namespace perpendicular_lines_intersection_l3364_336452

theorem perpendicular_lines_intersection (a b c d : ℝ) : 
  (∀ x y, a * x - 2 * y = d) →  -- First line equation
  (∀ x y, 2 * x + b * y = c) →  -- Second line equation
  (a * 2 - 2 * (-3) = d) →      -- Lines intersect at (2, -3)
  (2 * 2 + b * (-3) = c) →      -- Lines intersect at (2, -3)
  (a * b = -4) →                -- Perpendicular lines condition
  (d = 12) :=                   -- Conclusion
by sorry

end perpendicular_lines_intersection_l3364_336452


namespace cylindrical_fortress_pi_l3364_336439

/-- Given a cylindrical fortress with circumference 38 feet and height 11 feet,
    if its volume is calculated as V = (1/12) * (circumference^2 * height),
    then the implied value of π is 3. -/
theorem cylindrical_fortress_pi (circumference height : ℝ) (π : ℝ) : 
  circumference = 38 →
  height = 11 →
  (1/12) * (circumference^2 * height) = π * (circumference / (2 * π))^2 * height →
  π = 3 := by
  sorry

end cylindrical_fortress_pi_l3364_336439


namespace expression_equals_sixteen_ten_to_five_hundred_l3364_336466

theorem expression_equals_sixteen_ten_to_five_hundred :
  (3^500 + 4^501)^2 - (3^500 - 4^501)^2 = 16 * 10^500 := by
  sorry

end expression_equals_sixteen_ten_to_five_hundred_l3364_336466


namespace farm_hens_count_l3364_336464

/-- Given a farm with roosters and hens, where the number of hens is 5 less than 9 times
    the number of roosters, and the total number of chickens is 75, prove that there are 67 hens. -/
theorem farm_hens_count (roosters hens : ℕ) : 
  hens = 9 * roosters - 5 →
  hens + roosters = 75 →
  hens = 67 := by
sorry

end farm_hens_count_l3364_336464


namespace sum_of_integers_l3364_336445

theorem sum_of_integers (x y : ℕ+) 
  (h_diff : x - y = 18) 
  (h_prod : x * y = 72) : 
  x + y = 2 * Real.sqrt 153 := by
  sorry

end sum_of_integers_l3364_336445


namespace largest_number_is_482_l3364_336412

/-- Represents a systematic sample from a range of products -/
structure SystematicSample where
  total_products : Nat
  first_number : Nat
  second_number : Nat

/-- Calculates the largest number in a systematic sample -/
def largest_number (s : SystematicSample) : Nat :=
  let interval := s.second_number - s.first_number
  let sample_size := s.total_products / interval
  s.first_number + interval * (sample_size - 1)

/-- Theorem stating that for the given systematic sample, the largest number is 482 -/
theorem largest_number_is_482 :
  let s : SystematicSample := ⟨500, 7, 32⟩
  largest_number s = 482 := by sorry

end largest_number_is_482_l3364_336412


namespace area_between_concentric_circles_with_tangent_chord_l3364_336475

/-- The area between two concentric circles with a tangent chord -/
theorem area_between_concentric_circles_with_tangent_chord 
  (r : ℝ) -- radius of the smaller circle
  (c : ℝ) -- length of the chord of the larger circle
  (h1 : r = 40) -- given radius of the smaller circle
  (h2 : c = 120) -- given length of the chord
  : ∃ (A : ℝ), A = 3600 * Real.pi ∧ A = Real.pi * ((c / 2)^2 - r^2) := by
  sorry

end area_between_concentric_circles_with_tangent_chord_l3364_336475


namespace existence_of_numbers_l3364_336487

theorem existence_of_numbers : ∃ (a b c d : ℕ), 
  (a : ℚ) / b + (c : ℚ) / d = 1 ∧ (a : ℚ) / d + (c : ℚ) / b = 2008 := by
  sorry

end existence_of_numbers_l3364_336487


namespace xiaoxiao_reading_plan_l3364_336495

/-- Given a book with a total number of pages, pages already read, and days to finish,
    calculate the average number of pages to read per day. -/
def averagePagesPerDay (totalPages pagesRead daysToFinish : ℕ) : ℚ :=
  (totalPages - pagesRead : ℚ) / daysToFinish

/-- Theorem stating that for a book with 160 pages, 60 pages read, and 5 days to finish,
    the average number of pages to read per day is 20. -/
theorem xiaoxiao_reading_plan :
  averagePagesPerDay 160 60 5 = 20 := by
  sorry

end xiaoxiao_reading_plan_l3364_336495


namespace no_real_solutions_l3364_336444

theorem no_real_solutions : ¬∃ (x y : ℝ), 9^(x^2 + y) + 9^(x + y^2) = (1:ℝ)/3 := by
  sorry

end no_real_solutions_l3364_336444


namespace field_ratio_l3364_336493

theorem field_ratio (l w : ℝ) (h1 : ∃ k : ℕ, l = k * w) 
  (h2 : l = 36) (h3 : 81 = (1/8) * (l * w)) : l / w = 2 := by
  sorry

end field_ratio_l3364_336493


namespace power_equation_solution_l3364_336425

theorem power_equation_solution (m : ℕ) : 2^m = 2 * 16^2 * 4^3 → m = 15 := by
  sorry

end power_equation_solution_l3364_336425


namespace valerie_light_bulbs_l3364_336462

theorem valerie_light_bulbs :
  let total_budget : ℕ := 60
  let small_bulb_cost : ℕ := 8
  let large_bulb_cost : ℕ := 12
  let small_bulb_count : ℕ := 3
  let remaining_money : ℕ := 24
  let large_bulb_count : ℕ := (total_budget - remaining_money - small_bulb_cost * small_bulb_count) / large_bulb_cost
  large_bulb_count = 3 :=
by
  sorry

end valerie_light_bulbs_l3364_336462


namespace gasoline_canister_detonation_probability_l3364_336474

/-- The probability of detonating a gasoline canister -/
theorem gasoline_canister_detonation_probability :
  let n : ℕ := 5  -- number of available shots
  let p : ℚ := 2/3  -- probability of hitting the target
  let q : ℚ := 1 - p  -- probability of missing the target
  -- Assumption: shots are independent (implied by using binomial probability)
  -- Assumption: first successful hit causes a leak, second causes detonation (implied by the problem setup)
  232/243 = 1 - (q^n + n * q^(n-1) * p) :=
by sorry

end gasoline_canister_detonation_probability_l3364_336474


namespace problem_solution_l3364_336431

theorem problem_solution (a b c d : ℝ) 
  (h1 : a + b - c - d = 3)
  (h2 : a * b - 3 * b * c + c * d - 3 * d * a = 4)
  (h3 : 3 * a * b - b * c + 3 * c * d - d * a = 5) :
  11 * (a - c)^2 + 17 * (b - d)^2 = 63 := by
  sorry

end problem_solution_l3364_336431


namespace soldier_hit_target_l3364_336456

theorem soldier_hit_target (p q : Prop) : 
  (p ∨ q) ↔ (∃ shot : Fin 2, shot.val = 0 ∧ p ∨ shot.val = 1 ∧ q) :=
by sorry

end soldier_hit_target_l3364_336456


namespace T_equals_x_plus_one_to_fourth_l3364_336478

theorem T_equals_x_plus_one_to_fourth (x : ℝ) : 
  (x + 2)^4 - 4*(x + 2)^3 + 6*(x + 2)^2 - 4*(x + 2) + 1 = (x + 1)^4 := by
  sorry

end T_equals_x_plus_one_to_fourth_l3364_336478


namespace smallest_m_for_positive_integer_solutions_l3364_336476

theorem smallest_m_for_positive_integer_solutions :
  ∃ (m : ℤ), m = -1 ∧
  (∀ k : ℤ, k < m →
    ¬∃ (x y : ℤ), x > 0 ∧ y > 0 ∧ x + y = 2*k + 7 ∧ x - 2*y = 4*k - 3) ∧
  (∃ (x y : ℤ), x > 0 ∧ y > 0 ∧ x + y = 2*m + 7 ∧ x - 2*y = 4*m - 3) :=
by sorry

end smallest_m_for_positive_integer_solutions_l3364_336476


namespace fraction_bounds_l3364_336434

theorem fraction_bounds (x y : ℝ) (h : x^2*y^2 + x*y + 1 = 3*y^2) :
  let F := (y - x) / (x + 4*y)
  0 ≤ F ∧ F ≤ 4 := by sorry

end fraction_bounds_l3364_336434


namespace circle_properties_l3364_336453

-- Define the circle's equation
def circle_equation (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 2

-- Define the lines
def line1 (x y : ℝ) : Prop := x + y - 1 = 0
def line2 (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the center of the circle
def center : ℝ × ℝ := (0, 1)

theorem circle_properties :
  ∀ x y : ℝ,
    (line1 x y ∧ line2 x y → (x, y) = center) ∧
    circle_equation 1 0 ∧
    (∀ a b : ℝ, (a - center.1)^2 + (b - center.2)^2 = 2 ↔ circle_equation a b) :=
by sorry

end circle_properties_l3364_336453


namespace det_of_matrix_l3364_336442

def matrix : Matrix (Fin 3) (Fin 3) ℝ :=
  !![2, -1, 4;
     0,  6, -3;
     3,  0,  1]

theorem det_of_matrix : Matrix.det matrix = -51 := by
  sorry

end det_of_matrix_l3364_336442


namespace missing_number_proof_l3364_336404

theorem missing_number_proof (some_number : ℤ) : 
  (|4 - some_number * (3 - 12)| - |5 - 11| = 70) → some_number = 8 :=
by
  sorry

end missing_number_proof_l3364_336404


namespace cherry_cost_weight_relationship_l3364_336417

/-- The relationship between the cost of cherries and their weight -/
theorem cherry_cost_weight_relationship (x y : ℝ) :
  (∀ w, w * 16 = w * (y / x)) → y = 16 * x :=
by sorry

end cherry_cost_weight_relationship_l3364_336417


namespace B_power_101_l3364_336408

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_101 : B^101 = B^2 := by sorry

end B_power_101_l3364_336408


namespace welders_problem_l3364_336436

/-- The number of days needed to complete the order with all welders -/
def total_days : ℝ := 3

/-- The number of welders that leave after the first day -/
def leaving_welders : ℕ := 12

/-- The number of additional days needed by remaining welders to complete the order -/
def remaining_days : ℝ := 3.0000000000000004

/-- The initial number of welders -/
def initial_welders : ℕ := 36

theorem welders_problem :
  (initial_welders - leaving_welders : ℝ) / initial_welders * remaining_days = 2 / 3 :=
sorry

end welders_problem_l3364_336436


namespace highway_mileage_calculation_l3364_336484

/-- Calculates the highway mileage of a car given total distance, city distance, city mileage, and total gas used. -/
theorem highway_mileage_calculation 
  (total_highway_distance : ℝ) 
  (total_city_distance : ℝ) 
  (city_mileage : ℝ) 
  (total_gas_used : ℝ) 
  (h1 : total_highway_distance = 210)
  (h2 : total_city_distance = 54)
  (h3 : city_mileage = 18)
  (h4 : total_gas_used = 9) :
  (total_highway_distance / (total_gas_used - total_city_distance / city_mileage)) = 35 := by
sorry

end highway_mileage_calculation_l3364_336484


namespace wire_cutting_l3364_336416

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 70 →
  ratio = 3 / 7 →
  shorter_length + (shorter_length / ratio) = total_length →
  shorter_length = 21 :=
by
  sorry

end wire_cutting_l3364_336416


namespace intersection_line_of_circles_l3364_336480

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + (y - 2)^2 = 9

-- Define the line
def line (x y : ℝ) : Prop := x + 2*y - 1 = 0

-- Theorem statement
theorem intersection_line_of_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → line x y :=
by sorry

end intersection_line_of_circles_l3364_336480


namespace first_year_payment_is_twenty_l3364_336496

/-- Calculates the first year payment given the total payment and yearly increases -/
def firstYearPayment (totalPayment : ℚ) (secondYearIncrease thirdYearIncrease fourthYearIncrease : ℚ) : ℚ :=
  (totalPayment - (secondYearIncrease + (secondYearIncrease + thirdYearIncrease) + 
   (secondYearIncrease + thirdYearIncrease + fourthYearIncrease))) / 4

/-- Theorem stating that the first year payment is 20.00 given the problem conditions -/
theorem first_year_payment_is_twenty :
  firstYearPayment 96 2 3 4 = 20 := by
  sorry

#eval firstYearPayment 96 2 3 4

end first_year_payment_is_twenty_l3364_336496
