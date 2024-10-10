import Mathlib

namespace additional_machines_for_half_time_l3068_306879

/-- Represents the number of machines needed to complete a job in a given time -/
def machines_needed (initial_machines : ℕ) (initial_days : ℕ) (new_days : ℕ) : ℕ :=
  initial_machines * initial_days / new_days

/-- Proof that 95 additional machines are needed to complete the job in half the time -/
theorem additional_machines_for_half_time (initial_machines : ℕ) (initial_days : ℕ) 
    (h1 : initial_machines = 5) (h2 : initial_days = 20) :
  machines_needed initial_machines initial_days (initial_days / 2) - initial_machines = 95 := by
  sorry

#eval machines_needed 5 20 10 - 5  -- Should output 95

end additional_machines_for_half_time_l3068_306879


namespace rectangle_area_formula_l3068_306887

/-- Represents a rectangle with a specific ratio of length to width and diagonal length. -/
structure Rectangle where
  ratio_length : ℝ
  ratio_width : ℝ
  diagonal : ℝ
  ratio_condition : ratio_length / ratio_width = 5 / 2

/-- The theorem stating that the area of the rectangle can be expressed as (10/29)d^2 -/
theorem rectangle_area_formula (rect : Rectangle) : 
  ∃ (length width : ℝ), 
    length / width = rect.ratio_length / rect.ratio_width ∧
    length * width = (10/29) * rect.diagonal^2 := by
  sorry


end rectangle_area_formula_l3068_306887


namespace cannot_form_triangle_l3068_306842

/-- Triangle Inequality Theorem: A triangle can be formed if the sum of the lengths 
    of any two sides is greater than the length of the remaining side -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Proof that the set of line segments (5, 8, 2) cannot form a triangle -/
theorem cannot_form_triangle : ¬ triangle_inequality 5 8 2 := by
  sorry

end cannot_form_triangle_l3068_306842


namespace f_2x_equals_x_plus_1_over_x_minus_1_l3068_306841

theorem f_2x_equals_x_plus_1_over_x_minus_1 
  (x : ℝ) 
  (h : x^2 ≠ 4) : 
  let f := fun (y : ℝ) => (y + 2) / (y - 2)
  f (2 * x) = (x + 1) / (x - 1) := by
  sorry

end f_2x_equals_x_plus_1_over_x_minus_1_l3068_306841


namespace consecutive_integers_sum_l3068_306892

theorem consecutive_integers_sum (a b c d : ℤ) : 
  (b = a + 1) → (c = b + 1) → (d = c + 1) → (a + b + c + d = 274) → (b = 68) := by
  sorry

end consecutive_integers_sum_l3068_306892


namespace absolute_value_inequality_l3068_306829

theorem absolute_value_inequality (a : ℝ) :
  (∀ x : ℝ, |x - 3| + |x + 1| > a) → a < 4 := by
  sorry

end absolute_value_inequality_l3068_306829


namespace unique_right_triangle_exists_l3068_306894

theorem unique_right_triangle_exists : ∃! (a : ℝ), 
  a > 0 ∧ 
  let b := 2 * a
  let c := Real.sqrt (a^2 + b^2)
  (a + b + c) - (1/2 * a * b) = c :=
by sorry

end unique_right_triangle_exists_l3068_306894


namespace set_operations_l3068_306801

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x - 4 < 0}
def B : Set ℝ := {x | 0 < x ∧ x < 5}

-- Define the universe set U
def U : Set ℝ := Set.univ

-- Theorem statements
theorem set_operations :
  (A ∩ B = {x | 0 < x ∧ x < 2}) ∧
  (Set.compl A = {x | x ≥ 2}) ∧
  (Set.compl A ∩ B = {x | 2 ≤ x ∧ x < 5}) := by sorry

end set_operations_l3068_306801


namespace min_value_of_f_range_of_m_l3068_306811

noncomputable section

open Real MeasureTheory

-- Define the functions
def f (a : ℝ) (x : ℝ) : ℝ := log x + a / x - 1
def g (x : ℝ) : ℝ := x + 1 / x

-- Part 1
theorem min_value_of_f :
  (∀ x > 0, f 2 x ≥ log 2) ∧ (∃ x > 0, f 2 x = log 2) := by sorry

-- Part 2
theorem range_of_m :
  let f' := f (-1)
  {m : ℝ | ∃ x ∈ Set.Icc 1 (Real.exp 1), g x < m * (f' x + 1)} =
  Set.Ioi ((Real.exp 2 + 1) / (Real.exp 1 - 1)) ∪ Set.Iio (-2) := by sorry

end

end min_value_of_f_range_of_m_l3068_306811


namespace cubic_factorization_l3068_306850

theorem cubic_factorization (t : ℝ) : t^3 - 125 = (t - 5) * (t^2 + 5*t + 25) := by
  sorry

end cubic_factorization_l3068_306850


namespace max_power_sum_l3068_306873

theorem max_power_sum (c d : ℕ) : 
  d > 1 → 
  c^d < 630 → 
  (∀ (x y : ℕ), y > 1 → x^y < 630 → c^d ≥ x^y) → 
  c + d = 27 := by
sorry

end max_power_sum_l3068_306873


namespace travel_distance_ratio_l3068_306817

/-- Proves that if a person travels 400 km every odd month and x times 400 km every even month,
    and the total distance traveled in 24 months is 14400 km, then x = 2. -/
theorem travel_distance_ratio (x : ℝ) : 
  (12 * 400 + 12 * (400 * x) = 14400) → x = 2 := by
  sorry

end travel_distance_ratio_l3068_306817


namespace complex_product_real_imag_parts_l3068_306836

/-- If z = (2i-1)/i is a complex number with real part a and imaginary part b, then ab = 2 -/
theorem complex_product_real_imag_parts : 
  let z : ℂ := (2 * Complex.I - 1) / Complex.I
  let a : ℝ := z.re
  let b : ℝ := z.im
  a * b = 2 := by sorry

end complex_product_real_imag_parts_l3068_306836


namespace extended_triangle_similarity_l3068_306867

/-- Triangle ABC with given side lengths --/
structure Triangle :=
  (AB : ℝ)
  (BC : ℝ)
  (CA : ℝ)

/-- Extended triangle PABC --/
structure ExtendedTriangle extends Triangle :=
  (PC : ℝ)

/-- Similarity of triangles PAB and PCA --/
def is_similar (t : ExtendedTriangle) : Prop :=
  t.PC / t.AB = t.CA / t.PC

theorem extended_triangle_similarity (t : ExtendedTriangle) 
  (h1 : t.AB = 8)
  (h2 : t.BC = 7)
  (h3 : t.CA = 6)
  (h4 : is_similar t) :
  t.PC = 9 := by
  sorry

end extended_triangle_similarity_l3068_306867


namespace value_range_of_sum_product_l3068_306852

theorem value_range_of_sum_product (x : ℝ) : 
  ∃ (a b c : ℝ), a + b + c = 1 ∧ a^2 * b + b^2 * c + c^2 * a = x :=
sorry

end value_range_of_sum_product_l3068_306852


namespace incorrect_expression_l3068_306812

theorem incorrect_expression (a b : ℝ) : 
  2 * ((a^2 + b^2) - a*b) ≠ (a + b)^2 - 2*a*b := by
  sorry

end incorrect_expression_l3068_306812


namespace garden_dimensions_l3068_306851

/-- Represents a rectangular garden with given perimeter and length-to-breadth ratio -/
structure RectangularGarden where
  perimeter : ℝ
  length_breadth_ratio : ℝ
  length : ℝ
  breadth : ℝ
  diagonal : ℝ
  perimeter_eq : perimeter = 2 * (length + breadth)
  ratio_eq : length = length_breadth_ratio * breadth

/-- Theorem about the dimensions of a specific rectangular garden -/
theorem garden_dimensions (g : RectangularGarden) 
  (h_perimeter : g.perimeter = 500)
  (h_ratio : g.length_breadth_ratio = 3/2) :
  g.length = 150 ∧ g.diagonal = Real.sqrt 32500 := by
  sorry

#check garden_dimensions

end garden_dimensions_l3068_306851


namespace min_sum_squares_on_line_l3068_306861

/-- Given points M(1, 0) and N(-1, 0), and P(x, y) on the line 2x - y - 1 = 0,
    the minimum value of PM^2 + PN^2 is 2/5, achieved when P is at (1/5, -3/5) -/
theorem min_sum_squares_on_line :
  let M : ℝ × ℝ := (1, 0)
  let N : ℝ × ℝ := (-1, 0)
  let P : ℝ → ℝ × ℝ := fun x => (x, 2*x - 1)
  let dist_squared (a b : ℝ × ℝ) : ℝ := (a.1 - b.1)^2 + (a.2 - b.2)^2
  let sum_dist_squared (x : ℝ) : ℝ := dist_squared (P x) M + dist_squared (P x) N
  ∀ x : ℝ, sum_dist_squared x ≥ 2/5 ∧ 
    (sum_dist_squared (1/5) = 2/5 ∧ P (1/5) = (1/5, -3/5)) := by
  sorry

end min_sum_squares_on_line_l3068_306861


namespace cubic_function_property_l3068_306846

/-- Given a cubic function f(x) = ax³ + bx + 2 where f(-12) = 3, prove that f(12) = 1 -/
theorem cubic_function_property (a b : ℝ) (f : ℝ → ℝ) 
  (h1 : ∀ x, f x = a * x^3 + b * x + 2)
  (h2 : f (-12) = 3) : 
  f 12 = 1 := by sorry

end cubic_function_property_l3068_306846


namespace performance_selection_ways_l3068_306831

/-- The number of students who can sing -/
def num_singers : ℕ := 3

/-- The number of students who can dance -/
def num_dancers : ℕ := 2

/-- The number of students who can both sing and dance -/
def num_both : ℕ := 1

/-- The total number of students -/
def total_students : ℕ := 6

/-- The number of students to be selected for singing -/
def singers_to_select : ℕ := 2

/-- The number of students to be selected for dancing -/
def dancers_to_select : ℕ := 1

/-- The number of ways to select the required students for the performance -/
def num_ways : ℕ := Nat.choose (num_singers + num_both) singers_to_select * num_dancers - 1

theorem performance_selection_ways :
  num_ways = Nat.choose (num_singers + num_both) singers_to_select * num_dancers - 1 :=
by sorry

end performance_selection_ways_l3068_306831


namespace log_sum_cubes_l3068_306848

-- Define lg as the base-10 logarithm
noncomputable def lg (x : ℝ) := Real.log x / Real.log 10

-- State the theorem
theorem log_sum_cubes (h : lg 2 + lg 5 = 1) :
  (lg 2)^3 + 3*(lg 2)*(lg 5) + (lg 5)^3 = 1 := by sorry

end log_sum_cubes_l3068_306848


namespace fraction_equals_zero_l3068_306849

theorem fraction_equals_zero (x : ℝ) (h : 5 * x ≠ 0) :
  (x - 6) / (5 * x) = 0 ↔ x = 6 := by sorry

end fraction_equals_zero_l3068_306849


namespace positive_trig_expressions_l3068_306845

theorem positive_trig_expressions :
  (Real.sin (305 * π / 180) * Real.cos (460 * π / 180) > 0) ∧
  (Real.cos (378 * π / 180) * Real.sin (1100 * π / 180) > 0) ∧
  (Real.tan (188 * π / 180) * Real.cos (158 * π / 180) ≤ 0) ∧
  (Real.tan (400 * π / 180) * Real.tan (470 * π / 180) ≤ 0) :=
by sorry

end positive_trig_expressions_l3068_306845


namespace largest_three_digit_multiple_of_9_with_digit_sum_18_l3068_306859

/-- A function that returns the sum of digits of a natural number -/
def digit_sum (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is a three-digit number -/
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem largest_three_digit_multiple_of_9_with_digit_sum_18 :
  ∀ n : ℕ, is_three_digit n → n % 9 = 0 → digit_sum n = 18 → n ≤ 990 :=
by sorry

end largest_three_digit_multiple_of_9_with_digit_sum_18_l3068_306859


namespace great_eighteen_games_l3068_306871

/-- Great Eighteen Soccer League -/
structure SoccerLeague where
  divisions : Nat
  teams_per_division : Nat
  intra_division_games : Nat
  inter_division_games : Nat

/-- Calculate the total number of games in the league -/
def total_games (league : SoccerLeague) : Nat :=
  let total_teams := league.divisions * league.teams_per_division
  let intra_games := (league.divisions * league.teams_per_division * (league.teams_per_division - 1) * league.intra_division_games) / 2
  let inter_games := (total_teams * (total_teams - league.teams_per_division) * league.inter_division_games) / 2
  intra_games + inter_games

/-- Theorem: The Great Eighteen Soccer League has 351 scheduled games -/
theorem great_eighteen_games :
  let league := SoccerLeague.mk 3 6 3 2
  total_games league = 351 := by
  sorry

end great_eighteen_games_l3068_306871


namespace max_students_distribution_l3068_306897

theorem max_students_distribution (pens pencils erasers notebooks rulers : ℕ) 
  (h1 : pens = 3528) 
  (h2 : pencils = 3920) 
  (h3 : erasers = 3150) 
  (h4 : notebooks = 5880) 
  (h5 : rulers = 4410) : 
  Nat.gcd pens (Nat.gcd pencils (Nat.gcd erasers (Nat.gcd notebooks rulers))) = 2 := by
  sorry

end max_students_distribution_l3068_306897


namespace even_function_implies_even_g_l3068_306823

/-- A function f : ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem even_function_implies_even_g
  (f g : ℝ → ℝ)
  (h1 : ∀ x, f x - x^2 = g x)
  (h2 : IsEven f) :
  IsEven g := by
  sorry

end even_function_implies_even_g_l3068_306823


namespace arithmetic_sequence_property_l3068_306802

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a₁ d : ℝ), ∀ n, a n = a₁ + (n - 1) * d

/-- The sum of specific terms in the sequence equals 120 -/
def sum_condition (a : ℕ → ℝ) : Prop :=
  a 4 + a 6 + a 8 + a 10 + a 12 = 120

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h1 : arithmetic_sequence a) (h2 : sum_condition a) : 
  2 * a 10 - a 12 = 24 := by
  sorry

end arithmetic_sequence_property_l3068_306802


namespace normal_distribution_std_dev_l3068_306825

/-- A normal distribution with mean 54 and 3 standard deviations below the mean greater than 47 has a standard deviation less than 2.33 -/
theorem normal_distribution_std_dev (σ : ℝ) 
  (h1 : 54 - 3 * σ > 47) : 
  σ < 2.33 := by
sorry

end normal_distribution_std_dev_l3068_306825


namespace prime_even_intersection_l3068_306824

-- Define the set of prime numbers
def P : Set ℕ := {n : ℕ | Nat.Prime n}

-- Define the set of even numbers
def Q : Set ℕ := {n : ℕ | ∃ k : ℕ, n = 2 * k}

-- Theorem statement
theorem prime_even_intersection :
  P ∩ Q = {2} :=
sorry

end prime_even_intersection_l3068_306824


namespace min_value_theorem_l3068_306880

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 1) :
  ∀ z : ℝ, z = x^2 / (x + 2) + y^2 / (y + 1) → z ≥ 1/4 :=
by sorry

end min_value_theorem_l3068_306880


namespace no_grades_four_or_five_l3068_306826

theorem no_grades_four_or_five (n : ℕ) (x : ℕ) : 
  (5 : ℕ) ≠ 0 → -- There are 5 problems
  n * x + (x + 1) = 25 → -- Total problems solved
  9 ≤ n + 1 → -- At least 9 students (including Peter)
  x + 1 ≤ 5 → -- Maximum grade is 5
  ¬(x = 3 ∨ x = 4) :=
by sorry

end no_grades_four_or_five_l3068_306826


namespace consecutive_sum_product_l3068_306862

theorem consecutive_sum_product (a b c : ℕ) : 
  (a + 1 = b) → (b + 1 = c) → (a + b + c = 48) → (a * c = 255) := by
  sorry

end consecutive_sum_product_l3068_306862


namespace binomial_coefficient_x_squared_l3068_306864

theorem binomial_coefficient_x_squared (x : ℝ) : 
  (Finset.range 11).sum (fun k => Nat.choose 10 k * x^(10 - k) * (1/x)^k) = 
  210 * x^2 + (Finset.range 11).sum (fun k => if k ≠ 4 then Nat.choose 10 k * x^(10 - k) * (1/x)^k else 0) :=
sorry

end binomial_coefficient_x_squared_l3068_306864


namespace inequality_equivalence_l3068_306889

def inequality_solution (x : ℝ) : Prop :=
  x ∈ Set.Iio 0 ∪ Set.Ioo 0 5 ∪ Set.Ioi 5

theorem inequality_equivalence :
  ∀ x : ℝ, (x^2 / (x - 5)^2 > 0) ↔ inequality_solution x :=
by sorry

end inequality_equivalence_l3068_306889


namespace tax_deduction_percentage_l3068_306843

theorem tax_deduction_percentage (weekly_income : ℝ) (water_bill : ℝ) (tithe_percentage : ℝ) (remaining_amount : ℝ)
  (h1 : weekly_income = 500)
  (h2 : water_bill = 55)
  (h3 : tithe_percentage = 10)
  (h4 : remaining_amount = 345)
  (h5 : remaining_amount = weekly_income - (weekly_income * (tithe_percentage / 100)) - water_bill - (weekly_income * (tax_percentage / 100))) :
  tax_percentage = 10 := by
  sorry


end tax_deduction_percentage_l3068_306843


namespace andrews_donation_l3068_306876

/-- The age when Andrew started donating -/
def start_age : ℕ := 11

/-- Andrew's current age -/
def current_age : ℕ := 29

/-- The amount Andrew donates each year in thousands -/
def yearly_donation : ℕ := 7

/-- Calculate the total amount Andrew has donated -/
def total_donation : ℕ := (current_age - start_age) * yearly_donation

/-- Theorem stating that Andrew's total donation is 126k -/
theorem andrews_donation : total_donation = 126 := by sorry

end andrews_donation_l3068_306876


namespace max_comic_books_l3068_306810

theorem max_comic_books (cost : ℚ) (budget : ℚ) (h1 : cost = 25/20) (h2 : budget = 10) :
  ⌊budget / cost⌋ = 8 := by
sorry

end max_comic_books_l3068_306810


namespace simplify_expression_l3068_306883

theorem simplify_expression (a : ℚ) : ((2 * a + 6) - 3 * a) / 2 = -a / 2 + 3 := by
  sorry

end simplify_expression_l3068_306883


namespace negation_equivalence_l3068_306816

/-- Represents the property of being an honor student -/
def is_honor_student (x : Type) : Prop := sorry

/-- Represents the property of receiving a scholarship -/
def receives_scholarship (x : Type) : Prop := sorry

/-- The original statement: All honor students receive scholarships -/
def all_honor_students_receive_scholarships : Prop :=
  ∀ x, is_honor_student x → receives_scholarship x

/-- The negation of the original statement -/
def negation_of_statement : Prop :=
  ¬(∀ x, is_honor_student x → receives_scholarship x)

/-- The proposed equivalent negation: Some honor students do not receive scholarships -/
def some_honor_students_dont_receive_scholarships : Prop :=
  ∃ x, is_honor_student x ∧ ¬receives_scholarship x

/-- Theorem stating that the negation of the original statement is equivalent to 
    "Some honor students do not receive scholarships" -/
theorem negation_equivalence :
  negation_of_statement ↔ some_honor_students_dont_receive_scholarships := by sorry

end negation_equivalence_l3068_306816


namespace floor_inequality_l3068_306886

theorem floor_inequality (α β : ℝ) : 
  ⌊2 * α⌋ + ⌊2 * β⌋ ≥ ⌊α⌋ + ⌊β⌋ + ⌊α + β⌋ := by
  sorry

end floor_inequality_l3068_306886


namespace julia_internet_speed_l3068_306870

-- Define the given conditions
def songs_downloaded : ℕ := 7200
def download_time_minutes : ℕ := 30
def song_size_mb : ℕ := 5

-- Define the internet speed calculation function
def calculate_internet_speed (songs : ℕ) (time_minutes : ℕ) (size_mb : ℕ) : ℚ :=
  (songs * size_mb : ℚ) / (time_minutes * 60 : ℚ)

-- Theorem statement
theorem julia_internet_speed :
  calculate_internet_speed songs_downloaded download_time_minutes song_size_mb = 20 := by
  sorry


end julia_internet_speed_l3068_306870


namespace star_value_l3068_306819

theorem star_value (x : ℤ) : 45 - (28 - (37 - (15 - x))) = 59 → x = -154 := by
  sorry

end star_value_l3068_306819


namespace longest_chord_line_eq_l3068_306803

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle in 2D space -/
structure Circle where
  equation : ℝ → ℝ → Prop

/-- Represents a line in 2D space -/
structure Line where
  equation : ℝ → ℝ → Prop

/-- Given a circle and a point inside it, returns the line containing the longest chord passing through the point -/
def longestChordLine (c : Circle) (m : Point) : Line :=
  sorry

/-- The theorem stating that the longest chord line passing through M(3, -1) in the given circle has the equation x + 2y - 2 = 0 -/
theorem longest_chord_line_eq (c : Circle) (m : Point) :
  c.equation = (fun x y => x^2 + y^2 - 4*x + y - 2 = 0) →
  m = ⟨3, -1⟩ →
  (longestChordLine c m).equation = (fun x y => x + 2*y - 2 = 0) :=
by sorry

end longest_chord_line_eq_l3068_306803


namespace match_duration_l3068_306857

theorem match_duration (goals_per_interval : ℝ) (interval_duration : ℝ) (total_goals : ℝ) :
  goals_per_interval = 2 →
  interval_duration = 15 →
  total_goals = 16 →
  (total_goals / goals_per_interval) * interval_duration = 120 :=
by
  sorry

#check match_duration

end match_duration_l3068_306857


namespace line_perp_plane_iff_planes_perp_l3068_306878

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perpPlanes : Plane → Plane → Prop)
variable (perpLinePlane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

variable (α β : Plane)
variable (l : Line)

-- State the theorem
theorem line_perp_plane_iff_planes_perp 
  (h_intersect : α ≠ β) 
  (h_subset : subset l α) :
  perpLinePlane l β ↔ perpPlanes α β := by
  sorry

end line_perp_plane_iff_planes_perp_l3068_306878


namespace min_rotation_regular_pentagon_l3068_306809

/-- The angle of rotation for a regular pentagon to overlap with itself -/
def pentagon_rotation_angle : ℝ := 72

/-- A regular pentagon has 5 sides -/
def pentagon_sides : ℕ := 5

/-- Theorem: The minimum angle of rotation for a regular pentagon to overlap with itself is 72 degrees -/
theorem min_rotation_regular_pentagon :
  pentagon_rotation_angle = 360 / pentagon_sides :=
sorry

end min_rotation_regular_pentagon_l3068_306809


namespace root_equation_property_l3068_306818

theorem root_equation_property (α β : ℝ) : 
  (α^2 + α - 1 = 0) → 
  (β^2 + β - 1 = 0) → 
  α^2 + 2*β^2 + β = 4 := by
sorry

end root_equation_property_l3068_306818


namespace xiao_yu_better_l3068_306815

/-- The number of optional questions -/
def total_questions : ℕ := 8

/-- The number of questions randomly selected -/
def selected_questions : ℕ := 4

/-- The probability of Xiao Ming correctly answering a single question -/
def xiao_ming_prob : ℚ := 3/4

/-- The number of questions Xiao Yu can correctly complete -/
def xiao_yu_correct : ℕ := 6

/-- The number of questions Xiao Yu cannot complete -/
def xiao_yu_incorrect : ℕ := 2

/-- The probability of Xiao Ming correctly completing at least 3 questions -/
def xiao_ming_at_least_three : ℚ :=
  Nat.choose selected_questions 3 * xiao_ming_prob^3 * (1 - xiao_ming_prob) +
  Nat.choose selected_questions 4 * xiao_ming_prob^4

/-- The probability of Xiao Yu correctly completing at least 3 questions -/
def xiao_yu_at_least_three : ℚ :=
  (Nat.choose xiao_yu_correct 3 * Nat.choose xiao_yu_incorrect 1 +
   Nat.choose xiao_yu_correct 4 * Nat.choose xiao_yu_incorrect 0) /
  Nat.choose total_questions selected_questions

/-- Theorem stating that Xiao Yu has a higher probability of correctly completing at least 3 questions -/
theorem xiao_yu_better : xiao_yu_at_least_three > xiao_ming_at_least_three := by
  sorry

end xiao_yu_better_l3068_306815


namespace maia_daily_work_requests_l3068_306898

/-- The number of client requests Maia receives daily -/
def daily_requests : ℕ := 6

/-- The number of days Maia works -/
def work_days : ℕ := 5

/-- The number of client requests remaining after the work period -/
def remaining_requests : ℕ := 10

/-- The number of client requests Maia works on each day -/
def daily_work_requests : ℕ := (daily_requests * work_days - remaining_requests) / work_days

theorem maia_daily_work_requests :
  daily_work_requests = 4 := by sorry

end maia_daily_work_requests_l3068_306898


namespace right_triangle_hypotenuse_l3068_306832

theorem right_triangle_hypotenuse (leg : ℝ) (angle : ℝ) (h1 : leg = 15) (h2 : angle = 45) :
  let hypotenuse := leg * Real.sqrt 2
  hypotenuse = 15 * Real.sqrt 2 := by sorry

end right_triangle_hypotenuse_l3068_306832


namespace jar_size_proof_l3068_306890

/-- Proves that the size of the second jar type is 1/2 gallon given the problem conditions -/
theorem jar_size_proof (total_water : ℚ) (total_jars : ℕ) 
  (h1 : total_water = 28)
  (h2 : total_jars = 48)
  (h3 : ∃ (n : ℕ), n * (1 + x + 1/4) = total_jars ∧ n * (1 + x + 1/4) = total_water)
  : x = 1/2 := by
  sorry

#check jar_size_proof

end jar_size_proof_l3068_306890


namespace min_of_three_exists_l3068_306853

theorem min_of_three_exists : ∃ (f : ℝ → ℝ → ℝ → ℝ), 
  ∀ (a b c : ℝ), f a b c ≤ a ∧ f a b c ≤ b ∧ f a b c ≤ c ∧ 
  (∀ (m : ℝ), m ≤ a ∧ m ≤ b ∧ m ≤ c → f a b c ≥ m) :=
sorry

end min_of_three_exists_l3068_306853


namespace range_of_a_l3068_306840

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 2 * |x - 1| + |x - a| ≥ 2) ↔ a ∈ Set.Iic (-1) ∪ Set.Ici 3 :=
by sorry

end range_of_a_l3068_306840


namespace G_fraction_difference_l3068_306821

/-- G is defined as the infinite repeating decimal 0.871871871... -/
def G : ℚ := 871 / 999

/-- The difference between the denominator and numerator when G is expressed as a fraction in lowest terms -/
def denominator_numerator_difference : ℕ := 999 - 871

theorem G_fraction_difference : denominator_numerator_difference = 128 := by
  sorry

end G_fraction_difference_l3068_306821


namespace ratio_expression_value_l3068_306863

theorem ratio_expression_value (A B C : ℚ) (h : A / B = 3 / 2 ∧ B / C = 2 / 5) :
  (4 * A + 3 * B) / (5 * C - 2 * A) = 18 / 19 := by
  sorry

end ratio_expression_value_l3068_306863


namespace purely_imaginary_complex_number_l3068_306830

theorem purely_imaginary_complex_number (a : ℝ) :
  let z : ℂ := (a^2 - 2*a) + (a - 2)*Complex.I
  (∀ x : ℝ, z = x*Complex.I) ↔ a = 0 := by
  sorry

end purely_imaginary_complex_number_l3068_306830


namespace max_product_with_constraints_l3068_306891

theorem max_product_with_constraints (a b : ℕ) :
  a + b = 100 →
  a % 5 = 2 →
  b % 6 = 3 →
  a * b ≤ 2331 ∧ ∃ (a' b' : ℕ), a' + b' = 100 ∧ a' % 5 = 2 ∧ b' % 6 = 3 ∧ a' * b' = 2331 :=
by sorry

end max_product_with_constraints_l3068_306891


namespace expression_evaluation_l3068_306854

theorem expression_evaluation (x : ℝ) (h : x = 2) : 
  (2*x - 1)^2 + (x + 3)*(x - 3) - 4*(x - 1) = 0 := by
  sorry

end expression_evaluation_l3068_306854


namespace unique_tournaments_eq_fib_l3068_306896

/-- Represents a sequence of scores in descending order -/
def ScoreSequence (n : ℕ) := { a : Fin n → ℕ // ∀ i j, i ≤ j → a i ≥ a j }

/-- Represents a tournament outcome -/
structure Tournament (n : ℕ) where
  scores : ScoreSequence n
  team_scores : Fin n → ℕ

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The number of unique tournament outcomes for n teams -/
def uniqueTournaments (n : ℕ) : ℕ := sorry

/-- Theorem stating that the number of unique tournament outcomes is the (n+1)th Fibonacci number -/
theorem unique_tournaments_eq_fib (n : ℕ) : uniqueTournaments n = fib (n + 1) := by sorry

end unique_tournaments_eq_fib_l3068_306896


namespace number_of_hens_l3068_306844

theorem number_of_hens (total_animals : ℕ) (total_feet : ℕ) (hen_feet : ℕ) (cow_feet : ℕ) 
  (h1 : total_animals = 44)
  (h2 : total_feet = 140)
  (h3 : hen_feet = 2)
  (h4 : cow_feet = 4) :
  ∃ (hens cows : ℕ), 
    hens + cows = total_animals ∧ 
    hen_feet * hens + cow_feet * cows = total_feet ∧
    hens = 18 := by
  sorry

end number_of_hens_l3068_306844


namespace valid_arrangement_5_cubes_valid_arrangement_6_cubes_l3068_306877

/-- A cube in 3D space --/
structure Cube where
  position : ℝ × ℝ × ℝ

/-- An arrangement of cubes in 3D space --/
def Arrangement (n : ℕ) := Fin n → Cube

/-- Predicate to check if two cubes share a polygonal face --/
def SharesFace (c1 c2 : Cube) : Prop := sorry

/-- Predicate to check if an arrangement is valid (each cube shares a face with every other) --/
def ValidArrangement (arr : Arrangement n) : Prop :=
  ∀ i j, i ≠ j → SharesFace (arr i) (arr j)

/-- Theorem stating the existence of a valid arrangement for 5 cubes --/
theorem valid_arrangement_5_cubes : ∃ (arr : Arrangement 5), ValidArrangement arr := sorry

/-- Theorem stating the existence of a valid arrangement for 6 cubes --/
theorem valid_arrangement_6_cubes : ∃ (arr : Arrangement 6), ValidArrangement arr := sorry

end valid_arrangement_5_cubes_valid_arrangement_6_cubes_l3068_306877


namespace quadrilateral_area_theorem_l3068_306827

-- Define the triangle and its division
structure DividedTriangle where
  total_area : ℝ
  triangle1_area : ℝ
  triangle2_area : ℝ
  triangle3_area : ℝ
  quadrilateral_area : ℝ
  division_valid : 
    total_area = triangle1_area + triangle2_area + triangle3_area + quadrilateral_area

-- State the theorem
theorem quadrilateral_area_theorem (t : DividedTriangle) 
  (h1 : t.triangle1_area = 4)
  (h2 : t.triangle2_area = 9)
  (h3 : t.triangle3_area = 9) :
  t.quadrilateral_area = 36 := by
  sorry


end quadrilateral_area_theorem_l3068_306827


namespace f_properties_l3068_306805

def f (a x : ℝ) : ℝ := x^2 + |x - a| - 1

theorem f_properties (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = 0 ∧
  (∀ x, f a x ≥ -a - 5/4) ∧ (a ≤ -1/2 → ∃ x, f a x = -a - 5/4) ∧
  (∀ x, f a x ≥ a^2 - 1) ∧ (-1/2 < a → a ≤ 1/2 → ∃ x, f a x = a^2 - 1) ∧
  (∀ x, f a x ≥ a - 5/4) ∧ (1/2 < a → ∃ x, f a x = a - 5/4) :=
by sorry

end f_properties_l3068_306805


namespace emma_sister_age_relationship_l3068_306814

/-- Emma's current age -/
def emma_age : ℕ := 7

/-- Age difference between Emma and her sister -/
def age_difference : ℕ := 9

/-- Emma's age when her sister is 56 -/
def emma_future_age : ℕ := 47

/-- Emma's sister's age when Emma is 47 -/
def sister_future_age : ℕ := 56

/-- Theorem stating the relationship between Emma's age and her sister's age -/
theorem emma_sister_age_relationship (x : ℕ) :
  x ≥ age_difference →
  emma_future_age = sister_future_age - age_difference →
  x - age_difference = emma_age + (x - sister_future_age) :=
by
  sorry

end emma_sister_age_relationship_l3068_306814


namespace line_and_circle_problem_l3068_306847

-- Define the line l
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 = 0

-- Define the line m
def line_m (k : ℝ) (x y : ℝ) : Prop := x - k * y + 2 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

-- Define parallel lines
def parallel (k : ℝ) : Prop := ∃ (c : ℝ), ∀ x y, line_l k x y ↔ line_m k (x + c) (y + c * k)

-- Define tangent line to circle
def tangent (k : ℝ) : Prop := ∃! (x y : ℝ), line_l k x y ∧ circle_C x y

theorem line_and_circle_problem :
  (∀ k, parallel k → (k = 1 ∨ k = -1)) ∧
  (∀ k, tangent k → k = 1) :=
sorry

end line_and_circle_problem_l3068_306847


namespace car_rental_theorem_l3068_306882

/-- Represents a car rental company's pricing model -/
structure CarRental where
  totalVehicles : ℕ
  baseRentalFee : ℕ
  feeIncrement : ℕ
  rentedMaintCost : ℕ
  nonRentedMaintCost : ℕ

/-- Calculates the number of rented vehicles given a rental fee -/
def rentedVehicles (cr : CarRental) (rentalFee : ℕ) : ℕ :=
  cr.totalVehicles - (rentalFee - cr.baseRentalFee) / cr.feeIncrement

/-- Calculates the monthly revenue given a rental fee -/
def monthlyRevenue (cr : CarRental) (rentalFee : ℕ) : ℕ :=
  let rented := rentedVehicles cr rentalFee
  rentalFee * rented - cr.rentedMaintCost * rented - cr.nonRentedMaintCost * (cr.totalVehicles - rented)

/-- The main theorem about the car rental company -/
theorem car_rental_theorem (cr : CarRental) 
    (h1 : cr.totalVehicles = 100)
    (h2 : cr.baseRentalFee = 3000)
    (h3 : cr.feeIncrement = 60)
    (h4 : cr.rentedMaintCost = 160)
    (h5 : cr.nonRentedMaintCost = 60) :
  (rentedVehicles cr 3900 = 85) ∧
  (∃ maxRevenue : ℕ, maxRevenue = 324040 ∧ 
    ∀ fee, monthlyRevenue cr fee ≤ maxRevenue) ∧
  (∃ maxFee : ℕ, maxFee = 4560 ∧
    monthlyRevenue cr maxFee = 324040 ∧
    ∀ fee, monthlyRevenue cr fee ≤ monthlyRevenue cr maxFee) :=
  sorry


end car_rental_theorem_l3068_306882


namespace horizontal_asymptote_of_f_l3068_306875

noncomputable def f (x : ℝ) : ℝ := 
  (15 * x^4 + 6 * x^3 + 7 * x^2 + 4 * x + 5) / (5 * x^5 + 3 * x^3 + 9 * x^2 + 2 * x + 4)

theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ N, ∀ x, x > N → |f x| < ε :=
by
  sorry

end horizontal_asymptote_of_f_l3068_306875


namespace student_pairing_fraction_l3068_306872

theorem student_pairing_fraction (t s : ℕ) (ht : t > 0) (hs : s > 0) :
  (t / 4 : ℚ) = (3 * s / 7 : ℚ) →
  (3 * s / 7 : ℚ) / ((t : ℚ) + (s : ℚ)) = 3 / 19 := by
sorry

end student_pairing_fraction_l3068_306872


namespace equilateral_triangle_perimeter_l3068_306834

/-- Given an equilateral triangle where the area is twice the length of one of its sides,
    prove that its perimeter is 8√3 units. -/
theorem equilateral_triangle_perimeter (s : ℝ) (h : s > 0) : 
  (s^2 * Real.sqrt 3) / 4 = 2 * s → 3 * s = 8 * Real.sqrt 3 := by sorry

end equilateral_triangle_perimeter_l3068_306834


namespace quadratic_function_properties_l3068_306869

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_function_properties
  (a b c : ℝ)
  (ha : a > 0)
  (x₁ x₂ : ℝ)
  (hroots : ∀ x, f a b c x - x = 0 ↔ x = x₁ ∨ x = x₂)
  (horder : 0 < x₁ ∧ x₁ < x₂ ∧ x₂ < 1/a) :
  (∀ x, 0 < x ∧ x < x₁ → x < f a b c x ∧ f a b c x < x₁) ∧
  (-b / (2*a) < x₁ / 2) :=
by sorry

end quadratic_function_properties_l3068_306869


namespace room_area_ratio_problem_l3068_306874

/-- Proof of room area ratio problem -/
theorem room_area_ratio_problem (original_length original_width increase : ℕ) 
  (total_area : ℕ) (num_equal_rooms : ℕ) :
  let new_length : ℕ := original_length + increase
  let new_width : ℕ := original_width + increase
  let equal_room_area : ℕ := new_length * new_width
  let total_equal_rooms_area : ℕ := num_equal_rooms * equal_room_area
  let largest_room_area : ℕ := total_area - total_equal_rooms_area
  original_length = 13 ∧ 
  original_width = 18 ∧ 
  increase = 2 ∧
  total_area = 1800 ∧
  num_equal_rooms = 4 →
  largest_room_area / equal_room_area = 2 := by
sorry

end room_area_ratio_problem_l3068_306874


namespace disjoint_subsets_bound_l3068_306888

theorem disjoint_subsets_bound (m : ℕ) (A B : Finset ℕ) : 
  A ⊆ Finset.range m → 
  B ⊆ Finset.range m → 
  A ∩ B = ∅ → 
  A.sum id = B.sum id → 
  (A.card : ℝ) < m / Real.sqrt 2 ∧ (B.card : ℝ) < m / Real.sqrt 2 := by
  sorry

end disjoint_subsets_bound_l3068_306888


namespace snacks_expenditure_l3068_306838

theorem snacks_expenditure (total : ℝ) (movies books music ice_cream snacks : ℝ) :
  total = 50 ∧
  movies = (1/4) * total ∧
  books = (1/8) * total ∧
  music = (1/4) * total ∧
  ice_cream = (1/5) * total ∧
  snacks = total - (movies + books + music + ice_cream) →
  snacks = 8.75 := by sorry

end snacks_expenditure_l3068_306838


namespace ceiling_sum_equality_l3068_306813

theorem ceiling_sum_equality : 
  ⌈Real.sqrt (16/9 : ℝ)⌉ + ⌈(16/9 : ℝ)⌉ + ⌈((16/9 : ℝ)^2)⌉ = 8 := by
  sorry

end ceiling_sum_equality_l3068_306813


namespace smallest_absolute_value_is_zero_l3068_306828

theorem smallest_absolute_value_is_zero : 
  ∀ q : ℚ, |0| ≤ |q| :=
by
  sorry

end smallest_absolute_value_is_zero_l3068_306828


namespace marta_average_earnings_l3068_306804

/-- Represents Marta's work and earnings on her grandparent's farm --/
structure FarmWork where
  total_collected : ℕ
  task_a_rate : ℕ
  task_b_rate : ℕ
  task_c_rate : ℕ
  tips : ℕ
  task_a_hours : ℕ
  task_b_hours : ℕ
  task_c_hours : ℕ

/-- Calculates the average hourly earnings including tips --/
def average_hourly_earnings (work : FarmWork) : ℚ :=
  work.total_collected / (work.task_a_hours + work.task_b_hours + work.task_c_hours)

/-- Theorem stating that Marta's average hourly earnings, including tips, is $16 per hour --/
theorem marta_average_earnings :
  let work := FarmWork.mk 240 12 10 8 50 3 5 7
  average_hourly_earnings work = 16 := by
  sorry


end marta_average_earnings_l3068_306804


namespace average_speed_calculation_l3068_306860

theorem average_speed_calculation (total_distance : ℝ) (first_half_distance : ℝ) (second_half_distance : ℝ) 
  (first_half_speed : ℝ) (second_half_speed : ℝ) : 
  total_distance = 50 →
  first_half_distance = 25 →
  second_half_distance = 25 →
  first_half_speed = 66 →
  second_half_speed = 33 →
  (total_distance / ((first_half_distance / first_half_speed) + (second_half_distance / second_half_speed))) = 44 := by
  sorry

end average_speed_calculation_l3068_306860


namespace max_value_sqrt_sum_l3068_306856

theorem max_value_sqrt_sum (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : ∀ m : ℝ, 2 ≤ m ∧ m ≤ 3 → a + b ≤ m^2 - 2*m + 6) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧
    (∀ m : ℝ, 2 ≤ m ∧ m ≤ 3 → x + y ≤ m^2 - 2*m + 6) ∧
    Real.sqrt (x + 1) + Real.sqrt (y + 1) = 4 ∧
    (∀ c d : ℝ, c > 0 → d > 0 →
      (∀ m : ℝ, 2 ≤ m ∧ m ≤ 3 → c + d ≤ m^2 - 2*m + 6) →
      Real.sqrt (c + 1) + Real.sqrt (d + 1) ≤ 4) := by
sorry

end max_value_sqrt_sum_l3068_306856


namespace field_area_change_l3068_306806

theorem field_area_change (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let new_length := 1.35 * a
  let new_width := 0.86 * b
  let initial_area := a * b
  let new_area := new_length * new_width
  (new_area - initial_area) / initial_area = 0.161 := by
sorry

end field_area_change_l3068_306806


namespace boys_without_calculators_l3068_306858

theorem boys_without_calculators (total_students : ℕ) (total_boys : ℕ) (students_with_calculators : ℕ) (girls_with_calculators : ℕ) 
  (h1 : total_students = 40)
  (h2 : total_boys = 20)
  (h3 : students_with_calculators = 30)
  (h4 : girls_with_calculators = 18) :
  total_boys - (students_with_calculators - girls_with_calculators) = 8 := by
  sorry

end boys_without_calculators_l3068_306858


namespace volume_removed_percentage_l3068_306807

/-- Proves that removing six 4 cm cubes from a 20 cm × 15 cm × 10 cm box removes 12.8% of its volume -/
theorem volume_removed_percentage (box_length box_width box_height cube_side : ℝ) 
  (num_cubes_removed : ℕ) : 
  box_length = 20 → 
  box_width = 15 → 
  box_height = 10 → 
  cube_side = 4 → 
  num_cubes_removed = 6 → 
  (num_cubes_removed * cube_side^3) / (box_length * box_width * box_height) * 100 = 12.8 := by
  sorry

end volume_removed_percentage_l3068_306807


namespace square_perimeter_l3068_306855

theorem square_perimeter (total_area overlap_area circle_area : ℝ) : 
  total_area = 2018 →
  overlap_area = 137 →
  circle_area = 1371 →
  ∃ (square_side : ℝ), 
    square_side > 0 ∧ 
    square_side^2 = total_area - (circle_area - overlap_area) ∧
    4 * square_side = 112 :=
by sorry

end square_perimeter_l3068_306855


namespace matrix_power_identity_l3068_306899

variable {n : ℕ}

/-- Prove that for n×n complex matrices A, B, and C, if A^2 = B^2 = C^2 and B^3 = ABC + 2I, then A^6 = I. -/
theorem matrix_power_identity 
  (A B C : Matrix (Fin n) (Fin n) ℂ) 
  (h1 : A ^ 2 = B ^ 2)
  (h2 : B ^ 2 = C ^ 2)
  (h3 : B ^ 3 = A * B * C + 2 • 1) : 
  A ^ 6 = 1 := by
sorry

end matrix_power_identity_l3068_306899


namespace element_in_set_l3068_306865

theorem element_in_set : ∀ (M : Set ℕ), M = {0, 1, 2} → 0 ∈ M := by
  sorry

end element_in_set_l3068_306865


namespace rectangle_area_ratio_l3068_306881

/-- Given two rectangles A and B with sides proportional to a constant k, 
    prove that their area ratio is 4:1 -/
theorem rectangle_area_ratio 
  (k : ℝ) 
  (a b c d : ℝ) 
  (h_pos : k > 0) 
  (h_ka : a = k * a) 
  (h_kb : b = k * b) 
  (h_kc : c = k * c) 
  (h_kd : d = k * d) 
  (h_ratio : a / c = b / d) 
  (h_val : a / c = 2 / 5) : 
  (a * b) / (c * d) = 4 := by
  sorry

end rectangle_area_ratio_l3068_306881


namespace inequality_system_solution_l3068_306808

theorem inequality_system_solution :
  ∀ x : ℝ, (x + 2 < 3 * x ∧ (5 - x) / 2 + 1 < 0) ↔ x > 7 := by
  sorry

end inequality_system_solution_l3068_306808


namespace correct_raisin_distribution_l3068_306800

/-- The number of raisins received by each person -/
structure RaisinDistribution where
  bryce : ℕ
  carter : ℕ
  alice : ℕ

/-- The conditions of the raisin distribution problem -/
def valid_distribution (d : RaisinDistribution) : Prop :=
  d.bryce = d.carter + 10 ∧
  d.carter = d.bryce / 2 ∧
  d.alice = 2 * d.carter

/-- The theorem stating the correct raisin distribution -/
theorem correct_raisin_distribution :
  ∃ (d : RaisinDistribution), valid_distribution d ∧ d.bryce = 20 ∧ d.carter = 10 ∧ d.alice = 20 :=
by
  sorry

end correct_raisin_distribution_l3068_306800


namespace decagon_diagonal_intersections_eq_choose_l3068_306822

/-- The number of interior intersection points of diagonals in a regular decagon -/
def decagon_diagonal_intersections : ℕ :=
  Nat.choose 10 4

/-- Theorem: The number of interior intersection points of diagonals in a regular decagon
    is equal to the number of ways to choose 4 vertices out of 10 -/
theorem decagon_diagonal_intersections_eq_choose :
  decagon_diagonal_intersections = Nat.choose 10 4 := by
  sorry

end decagon_diagonal_intersections_eq_choose_l3068_306822


namespace total_pencils_l3068_306895

/-- Given that each child has 2 pencils and there are 15 children, 
    prove that the total number of pencils is 30. -/
theorem total_pencils (pencils_per_child : ℕ) (num_children : ℕ) 
  (h1 : pencils_per_child = 2) (h2 : num_children = 15) : 
  pencils_per_child * num_children = 30 := by
  sorry

end total_pencils_l3068_306895


namespace contrapositive_square_sum_zero_l3068_306868

theorem contrapositive_square_sum_zero (m n : ℝ) :
  (¬(mn = 0) → ¬(m^2 + n^2 = 0)) ↔ (m^2 + n^2 = 0 → mn = 0) := by sorry

end contrapositive_square_sum_zero_l3068_306868


namespace cheburashka_count_l3068_306839

/-- Represents the number of characters in a row -/
def n : ℕ := 16

/-- Represents the total number of Krakozyabras -/
def total_krakozyabras : ℕ := 29

/-- Represents the number of Cheburashkas -/
def num_cheburashkas : ℕ := 11

theorem cheburashka_count :
  (2 * (n - 1) = total_krakozyabras) ∧
  (num_cheburashkas * 2 + (num_cheburashkas - 1) * 2 + num_cheburashkas = n) :=
by sorry

#check cheburashka_count

end cheburashka_count_l3068_306839


namespace smallest_student_group_l3068_306833

theorem smallest_student_group (n : ℕ) : 
  (n % 6 = 3) ∧ 
  (n % 7 = 4) ∧ 
  (n % 8 = 5) ∧ 
  (n % 9 = 2) ∧ 
  (∀ m : ℕ, m < n → ¬(m % 6 = 3 ∧ m % 7 = 4 ∧ m % 8 = 5 ∧ m % 9 = 2)) → 
  n = 765 := by
sorry

end smallest_student_group_l3068_306833


namespace expression_value_l3068_306820

theorem expression_value (x y : ℝ) (h : x / (2 * y) = 3 / 2) :
  (7 * x + 5 * y) / (x - 2 * y) = 26 := by
sorry

end expression_value_l3068_306820


namespace boat_travel_time_l3068_306835

/-- Proves that a boat traveling upstream for 1.5 hours will take 1 hour to travel the same distance downstream, given the boat's speed in still water and the stream's speed. -/
theorem boat_travel_time 
  (boat_speed : ℝ) 
  (stream_speed : ℝ) 
  (upstream_time : ℝ) 
  (h1 : boat_speed = 15) 
  (h2 : stream_speed = 3) 
  (h3 : upstream_time = 1.5) : 
  (boat_speed - stream_speed) * upstream_time / (boat_speed + stream_speed) = 1 := by
  sorry

#check boat_travel_time

end boat_travel_time_l3068_306835


namespace sons_age_l3068_306885

/-- Given a father and son where the father is 26 years older than the son,
    and in two years the father's age will be twice the son's age,
    prove that the son's current age is 24 years. -/
theorem sons_age (son_age father_age : ℕ) 
  (h1 : father_age = son_age + 26)
  (h2 : father_age + 2 = 2 * (son_age + 2)) : 
  son_age = 24 := by
  sorry

end sons_age_l3068_306885


namespace ellipse_equation_from_line_l3068_306837

/-- The standard equation of an ellipse -/
structure EllipseEquation where
  a : ℝ
  b : ℝ
  h : a > 0 ∧ b > 0

/-- A line passing through a focus and vertex of an ellipse -/
structure EllipseLine where
  slope : ℝ
  intercept : ℝ

/-- The theorem statement -/
theorem ellipse_equation_from_line (l : EllipseLine) 
  (h1 : l.slope = 1/2 ∧ l.intercept = -1) 
  (h2 : ∃ (f v : ℝ × ℝ), l.slope * f.1 + l.intercept = f.2 ∧ 
                          l.slope * v.1 + l.intercept = v.2) 
  (h3 : l.slope * 0 + l.intercept = 1) :
  ∃ (e1 e2 : EllipseEquation), 
    (e1.a^2 = 5 ∧ e1.b^2 = 1) ∨ 
    (e2.a^2 = 5 ∧ e2.b^2 = 4) := by
  sorry

end ellipse_equation_from_line_l3068_306837


namespace agnes_hourly_rate_l3068_306893

/-- Proves that Agnes's hourly rate is $15 given the conditions of the problem -/
theorem agnes_hourly_rate : 
  ∀ (mila_rate : ℝ) (agnes_weekly_hours : ℝ) (mila_equal_hours : ℝ) (weeks_per_month : ℝ),
  mila_rate = 10 →
  agnes_weekly_hours = 8 →
  mila_equal_hours = 48 →
  weeks_per_month = 4 →
  (agnes_weekly_hours * weeks_per_month * (mila_rate * mila_equal_hours / (agnes_weekly_hours * weeks_per_month))) = 15 :=
by sorry

end agnes_hourly_rate_l3068_306893


namespace derivative_inequality_l3068_306866

theorem derivative_inequality (f : ℝ → ℝ) (f' : ℝ → ℝ) (h1 : ∀ x, HasDerivAt f (f' x) x)
  (h2 : ∀ x, f' x + f x > 0) (x₁ x₂ : ℝ) (h3 : x₁ < x₂) :
  Real.exp x₁ * f x₁ < Real.exp x₂ * f x₂ := by
  sorry

end derivative_inequality_l3068_306866


namespace polynomial_factorization_l3068_306884

theorem polynomial_factorization (y : ℤ) :
  5 * (y + 4) * (y + 7) * (y + 9) * (y + 11) - 4 * y^2 =
  (y + 1) * (y + 9) * (5 * y^2 + 33 * y + 441) :=
by sorry

end polynomial_factorization_l3068_306884
