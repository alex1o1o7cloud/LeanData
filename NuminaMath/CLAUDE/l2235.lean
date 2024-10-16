import Mathlib

namespace NUMINAMATH_CALUDE_two_digit_number_insertion_theorem_l2235_223568

theorem two_digit_number_insertion_theorem :
  ∃! (S : Finset Nat),
    (∀ n ∈ S, 10 ≤ n ∧ n < 100) ∧
    (∀ n ∉ S, ¬(10 ≤ n ∧ n < 100)) ∧
    (∀ n ∈ S,
      ∃ d : Nat,
      d < 10 ∧
      (100 * (n / 10) + 10 * d + (n % 10) = 9 * n)) ∧
    S.card = 4 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_number_insertion_theorem_l2235_223568


namespace NUMINAMATH_CALUDE_marble_difference_l2235_223586

/-- The number of marbles each person has -/
structure Marbles where
  merill : ℕ
  elliot : ℕ
  selma : ℕ

/-- The conditions of the marble problem -/
def marble_problem (m : Marbles) : Prop :=
  m.merill = 30 ∧ m.selma = 50 ∧ m.merill = 2 * m.elliot

/-- The theorem to prove -/
theorem marble_difference (m : Marbles) (h : marble_problem m) :
  m.selma - (m.merill + m.elliot) = 5 := by
  sorry

end NUMINAMATH_CALUDE_marble_difference_l2235_223586


namespace NUMINAMATH_CALUDE_mean_score_remaining_students_l2235_223521

theorem mean_score_remaining_students 
  (n : ℕ) 
  (h1 : n > 20) 
  (h2 : (15 : ℝ) * 10 = (15 : ℝ) * mean_first_15)
  (h3 : (5 : ℝ) * 16 = (5 : ℝ) * mean_next_5)
  (h4 : ((15 : ℝ) * mean_first_15 + (5 : ℝ) * mean_next_5 + (n - 20 : ℝ) * mean_remaining) / n = 11) :
  mean_remaining = (11 * n - 230) / (n - 20) := by
  sorry

end NUMINAMATH_CALUDE_mean_score_remaining_students_l2235_223521


namespace NUMINAMATH_CALUDE_light_bulb_probabilities_l2235_223506

/-- Represents the number of light bulbs in the box -/
def total_bulbs : ℕ := 6

/-- Represents the number of defective light bulbs -/
def defective_bulbs : ℕ := 2

/-- Represents the number of good light bulbs -/
def good_bulbs : ℕ := 4

/-- Represents the number of bulbs selected -/
def selected_bulbs : ℕ := 2

/-- Calculates the probability of selecting two defective bulbs -/
def prob_two_defective : ℚ := 1 / 15

/-- Calculates the probability of selecting exactly one defective bulb -/
def prob_one_defective : ℚ := 8 / 15

theorem light_bulb_probabilities :
  (total_bulbs = defective_bulbs + good_bulbs) ∧
  (selected_bulbs = 2) →
  (prob_two_defective = 1 / 15) ∧
  (prob_one_defective = 8 / 15) := by
  sorry

end NUMINAMATH_CALUDE_light_bulb_probabilities_l2235_223506


namespace NUMINAMATH_CALUDE_athlete_arrangements_l2235_223542

/-- The number of athletes and tracks -/
def n : ℕ := 6

/-- Function to calculate the number of arrangements where A, B, and C are not adjacent -/
def arrangements_not_adjacent : ℕ := sorry

/-- Function to calculate the number of arrangements where there is one person between A and B -/
def arrangements_one_between : ℕ := sorry

/-- Function to calculate the number of arrangements where A is not on first or second track, and B is on fifth or sixth track -/
def arrangements_restricted : ℕ := sorry

/-- Theorem stating the correct number of arrangements for each scenario -/
theorem athlete_arrangements :
  arrangements_not_adjacent = 144 ∧
  arrangements_one_between = 192 ∧
  arrangements_restricted = 144 :=
sorry

end NUMINAMATH_CALUDE_athlete_arrangements_l2235_223542


namespace NUMINAMATH_CALUDE_probability_of_two_in_three_elevenths_l2235_223517

/-- The decimal representation of 3/11 as a sequence of digits -/
def decimalRep : ℕ → Fin 10
  | 0 => 2
  | 1 => 7
  | n + 2 => decimalRep n

/-- The period of the decimal representation of 3/11 -/
def period : ℕ := 2

/-- Count of digit 2 in one period of the decimal representation -/
def countOfTwo : ℕ := 1

theorem probability_of_two_in_three_elevenths :
  (countOfTwo : ℚ) / (period : ℚ) = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_probability_of_two_in_three_elevenths_l2235_223517


namespace NUMINAMATH_CALUDE_seventh_power_of_complex_l2235_223525

theorem seventh_power_of_complex (z : ℂ) : 
  z = (Real.sqrt 3 + Complex.I) / 2 → z^7 = -Real.sqrt 3 / 2 - Complex.I / 2 := by
  sorry

end NUMINAMATH_CALUDE_seventh_power_of_complex_l2235_223525


namespace NUMINAMATH_CALUDE_range_of_b_l2235_223565

theorem range_of_b (a b : ℝ) (h1 : 0 ≤ a + b ∧ a + b < 1) (h2 : 2 ≤ a - b ∧ a - b < 3) :
  -3/2 < b ∧ b < -1/2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_b_l2235_223565


namespace NUMINAMATH_CALUDE_linear_regression_equation_l2235_223560

/-- Represents a linear regression model --/
structure LinearRegression where
  slope : ℝ
  intercept : ℝ

/-- Checks if two variables are positively correlated --/
def positively_correlated (x y : ℝ → ℝ) : Prop := sorry

/-- Calculates the sample mean of a variable --/
def sample_mean (x : ℝ → ℝ) : ℝ := sorry

/-- Checks if a point lies on the regression line --/
def point_on_line (model : LinearRegression) (x y : ℝ) : Prop :=
  y = model.slope * x + model.intercept

theorem linear_regression_equation 
  (x y : ℝ → ℝ) 
  (h_corr : positively_correlated x y)
  (h_mean_x : sample_mean x = 2)
  (h_mean_y : sample_mean y = 3)
  : ∃ (model : LinearRegression), 
    model.slope = 2 ∧ 
    model.intercept = -1 ∧ 
    point_on_line model 2 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_regression_equation_l2235_223560


namespace NUMINAMATH_CALUDE_roots_of_quadratic_equation_l2235_223580

theorem roots_of_quadratic_equation :
  ∀ x : ℝ, x^2 = 2*x ↔ x = 0 ∨ x = 2 := by sorry

end NUMINAMATH_CALUDE_roots_of_quadratic_equation_l2235_223580


namespace NUMINAMATH_CALUDE_min_value_a_solution_set_l2235_223587

-- Define the function f(x)
def f (x a : ℝ) : ℝ := |x - 4| + |x - a|

-- Theorem for the minimum value of a
theorem min_value_a :
  ∃ (a : ℝ), ∀ (x : ℝ), f x a ≥ a ∧ (∃ (x₀ : ℝ), f x₀ a = a) ∧ a = 2 :=
sorry

-- Theorem for the solution set of f(x) ≤ 5
theorem solution_set :
  ∃ (a : ℝ), a = 2 ∧ {x : ℝ | f x a ≤ 5} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 11/2} :=
sorry

end NUMINAMATH_CALUDE_min_value_a_solution_set_l2235_223587


namespace NUMINAMATH_CALUDE_youtube_video_length_l2235_223556

theorem youtube_video_length (total_time : ℕ) (video1_length : ℕ) (video2_length : ℕ) :
  total_time = 510 ∧
  video1_length = 120 ∧
  video2_length = 270 →
  ∃ (last_video_length : ℕ),
    last_video_length * 2 = total_time - (video1_length + video2_length) ∧
    last_video_length = 60 := by
  sorry

end NUMINAMATH_CALUDE_youtube_video_length_l2235_223556


namespace NUMINAMATH_CALUDE_prime_divides_power_difference_l2235_223551

theorem prime_divides_power_difference (p : ℕ) (n : ℕ) (hp : Nat.Prime p) :
  p ∣ (3^(n+p) - 3^(n+1)) := by
  sorry

end NUMINAMATH_CALUDE_prime_divides_power_difference_l2235_223551


namespace NUMINAMATH_CALUDE_max_area_parabola_triangle_l2235_223541

/-- Given two points on a parabola, prove the maximum area of a triangle formed with a specific third point -/
theorem max_area_parabola_triangle (x₁ x₂ y₁ y₂ : ℝ) : 
  x₁ ≠ x₂ →
  x₁ + x₂ = 4 →
  y₁^2 = 6*x₁ →
  y₂^2 = 6*x₂ →
  let A := (x₁, y₁)
  let B := (x₂, y₂)
  let M := ((x₁ + x₂)/2, (y₁ + y₂)/2)
  let k_AB := (y₂ - y₁)/(x₂ - x₁)
  let C := (5, 0)
  let triangle_area := abs ((x₁ - 5)*(y₂ - 0) + (x₂ - x₁)*(0 - y₁) + (5 - x₂)*(y₁ - y₂)) / 2
  ∃ (max_area : ℝ), max_area = 14 * Real.sqrt 7 / 3 ∧ 
    ∀ (x₁' x₂' y₁' y₂' : ℝ), 
      x₁' ≠ x₂' → 
      x₁' + x₂' = 4 → 
      y₁'^2 = 6*x₁' → 
      y₂'^2 = 6*x₂' → 
      let A' := (x₁', y₁')
      let B' := (x₂', y₂')
      let triangle_area' := abs ((x₁' - 5)*(y₂' - 0) + (x₂' - x₁')*(0 - y₁') + (5 - x₂')*(y₁' - y₂')) / 2
      triangle_area' ≤ max_area := by
  sorry

end NUMINAMATH_CALUDE_max_area_parabola_triangle_l2235_223541


namespace NUMINAMATH_CALUDE_cans_per_person_day2_is_2_5_l2235_223569

/-- Represents the food bank scenario --/
structure FoodBank where
  initial_stock : ℕ
  day1_people : ℕ
  day1_cans_per_person : ℕ
  day1_restock : ℕ
  day2_people : ℕ
  day2_restock : ℕ
  total_cans_given : ℕ

/-- Calculates the number of cans each person took on the second day --/
def cans_per_person_day2 (fb : FoodBank) : ℚ :=
  let day1_remaining := fb.initial_stock - fb.day1_people * fb.day1_cans_per_person
  let after_day1_restock := day1_remaining + fb.day1_restock
  let day2_given := fb.total_cans_given - fb.day1_people * fb.day1_cans_per_person
  day2_given / fb.day2_people

/-- Theorem stating that given the conditions, each person took 2.5 cans on the second day --/
theorem cans_per_person_day2_is_2_5 (fb : FoodBank)
  (h1 : fb.initial_stock = 2000)
  (h2 : fb.day1_people = 500)
  (h3 : fb.day1_cans_per_person = 1)
  (h4 : fb.day1_restock = 1500)
  (h5 : fb.day2_people = 1000)
  (h6 : fb.day2_restock = 3000)
  (h7 : fb.total_cans_given = 2500) :
  cans_per_person_day2 fb = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_cans_per_person_day2_is_2_5_l2235_223569


namespace NUMINAMATH_CALUDE_rabbits_to_add_correct_rabbits_to_add_l2235_223577

theorem rabbits_to_add (initial_rabbits : ℕ) (park_rabbits : ℕ) : ℕ :=
  let final_rabbits := park_rabbits / 3
  final_rabbits - initial_rabbits

theorem correct_rabbits_to_add :
  rabbits_to_add 13 60 = 7 := by sorry

end NUMINAMATH_CALUDE_rabbits_to_add_correct_rabbits_to_add_l2235_223577


namespace NUMINAMATH_CALUDE_rod_length_l2235_223593

/-- Given a rod that can be cut into 40 pieces of 85 cm each, prove that its length is 3400 cm. -/
theorem rod_length (num_pieces : ℕ) (piece_length : ℕ) (h1 : num_pieces = 40) (h2 : piece_length = 85) :
  num_pieces * piece_length = 3400 := by
  sorry

end NUMINAMATH_CALUDE_rod_length_l2235_223593


namespace NUMINAMATH_CALUDE_student_number_problem_l2235_223523

theorem student_number_problem (x : ℝ) : 4 * x - 138 = 102 → x = 60 := by
  sorry

end NUMINAMATH_CALUDE_student_number_problem_l2235_223523


namespace NUMINAMATH_CALUDE_total_taco_ingredients_cost_l2235_223585

def taco_shells_cost : ℝ := 5
def bell_peppers_cost : ℝ := 4 * 1.5
def meat_cost : ℝ := 2 * 3
def tomatoes_cost : ℝ := 3 * 0.75
def cheese_cost : ℝ := 4
def tortillas_cost : ℝ := 2.5
def salsa_cost : ℝ := 3.25

theorem total_taco_ingredients_cost :
  taco_shells_cost + bell_peppers_cost + meat_cost + tomatoes_cost + cheese_cost + tortillas_cost + salsa_cost = 29 := by
  sorry

end NUMINAMATH_CALUDE_total_taco_ingredients_cost_l2235_223585


namespace NUMINAMATH_CALUDE_exists_empty_subsquare_l2235_223597

/-- Represents a point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square in a 2D plane -/
structure Square where
  bottomLeft : Point
  sideLength : ℝ

/-- A function to check if a point is inside a square -/
def isPointInSquare (p : Point) (s : Square) : Prop :=
  s.bottomLeft.x ≤ p.x ∧ p.x < s.bottomLeft.x + s.sideLength ∧
  s.bottomLeft.y ≤ p.y ∧ p.y < s.bottomLeft.y + s.sideLength

/-- The main theorem -/
theorem exists_empty_subsquare 
  (bigSquare : Square) 
  (points : Finset Point) 
  (h1 : bigSquare.sideLength = 4) 
  (h2 : points.card = 15) : 
  ∃ (smallSquare : Square), 
    smallSquare.sideLength = 1 ∧ 
    (∀ (p : Point), p ∈ points → ¬ isPointInSquare p smallSquare) :=
sorry

end NUMINAMATH_CALUDE_exists_empty_subsquare_l2235_223597


namespace NUMINAMATH_CALUDE_smallest_t_is_four_l2235_223504

def is_valid_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

def smallest_valid_t : ℕ → Prop
  | t => is_valid_triangle 7.5 11 (t : ℝ) ∧ 
         ∀ k : ℕ, k < t → ¬is_valid_triangle 7.5 11 (k : ℝ)

theorem smallest_t_is_four : smallest_valid_t 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_t_is_four_l2235_223504


namespace NUMINAMATH_CALUDE_no_meeting_l2235_223579

/-- Represents the positions of Michael and the truck over time -/
structure Positions (t : ℝ) where
  michael : ℝ
  truck : ℝ

/-- The rate at which Michael walks -/
def michael_speed : ℝ := 6

/-- The speed of the garbage truck -/
def truck_speed : ℝ := 12

/-- The distance between trash pails -/
def pail_distance : ℝ := 300

/-- The time the truck stops at each pail -/
def truck_stop_time : ℝ := 40

/-- The total time Michael walks -/
def total_time : ℝ := 900

/-- Calculate the positions of Michael and the truck at time t -/
def calculate_positions (t : ℝ) : Positions t :=
  sorry

/-- Theorem stating that Michael and the truck never meet within the given time -/
theorem no_meeting :
  ∀ t, 0 ≤ t ∧ t ≤ total_time → (calculate_positions t).michael < (calculate_positions t).truck :=
  sorry

end NUMINAMATH_CALUDE_no_meeting_l2235_223579


namespace NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l2235_223533

/-- 
Given an isosceles, obtuse triangle where one angle is 75% larger than a right angle,
prove that the measure of each of the two smallest angles is 45/4 degrees.
-/
theorem isosceles_obtuse_triangle_smallest_angle 
  (α β γ : ℝ) 
  (h_isosceles : α = β)
  (h_obtuse : γ > 90)
  (h_large_angle : γ = 90 * 1.75)
  (h_angle_sum : α + β + γ = 180) : 
  α = 45 / 4 := by
sorry

end NUMINAMATH_CALUDE_isosceles_obtuse_triangle_smallest_angle_l2235_223533


namespace NUMINAMATH_CALUDE_set_A_from_complement_l2235_223529

-- Define the universal set U
def U : Set Nat := {0, 1, 2, 3}

-- Define the complement of A
def complement_A : Set Nat := {2}

-- Theorem to prove
theorem set_A_from_complement : 
  ∀ (A : Set Nat), (U \ A = complement_A) → A = {0, 1, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_A_from_complement_l2235_223529


namespace NUMINAMATH_CALUDE_operation_result_l2235_223526

-- Define a type for the allowed operations
inductive Operation
  | Add
  | Sub
  | Mul
  | Div

def apply_op (op : Operation) (a b : ℚ) : ℚ :=
  match op with
  | Operation.Add => a + b
  | Operation.Sub => a - b
  | Operation.Mul => a * b
  | Operation.Div => a / b

theorem operation_result 
  (diamond circ : Operation) 
  (h : (apply_op diamond 20 4) / (apply_op circ 12 4) = 2) :
  (apply_op diamond 9 3) / (apply_op circ 15 5) = 27 / 20 :=
by sorry

end NUMINAMATH_CALUDE_operation_result_l2235_223526


namespace NUMINAMATH_CALUDE_sector_area_l2235_223554

/-- Given a sector with central angle 2 radians and arc length 4, its area is 4. -/
theorem sector_area (θ : Real) (l : Real) (A : Real) : 
  θ = 2 → l = 4 → A = (1/2) * (l/θ)^2 * θ → A = 4 := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l2235_223554


namespace NUMINAMATH_CALUDE_sum_of_squares_l2235_223513

theorem sum_of_squares (a b : ℝ) : (a^2 + b^2) * (a^2 + b^2 + 4) = 12 → a^2 + b^2 = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l2235_223513


namespace NUMINAMATH_CALUDE_function_nature_l2235_223588

theorem function_nature (n : ℕ) (h : 30 * n = 30 * n) :
  let f : ℝ → ℝ := fun x ↦ x ^ n
  (f 1)^2 + (f (-1))^2 = 2 * ((f 1) + (f (-1)) - 1) →
  ∀ x : ℝ, f (-x) = f x :=
by sorry

end NUMINAMATH_CALUDE_function_nature_l2235_223588


namespace NUMINAMATH_CALUDE_students_with_one_fruit_l2235_223543

theorem students_with_one_fruit (total_apples : Nat) (total_bananas : Nat) (both_fruits : Nat) 
  (h1 : total_apples = 12)
  (h2 : total_bananas = 8)
  (h3 : both_fruits = 5) :
  (total_apples - both_fruits) + (total_bananas - both_fruits) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_with_one_fruit_l2235_223543


namespace NUMINAMATH_CALUDE_pen_count_difference_l2235_223555

theorem pen_count_difference (red : ℕ) (black : ℕ) (blue : ℕ) : 
  red = 8 →
  black = red + 10 →
  red + black + blue = 41 →
  blue > red →
  blue - red = 7 := by
sorry

end NUMINAMATH_CALUDE_pen_count_difference_l2235_223555


namespace NUMINAMATH_CALUDE_driver_net_hourly_rate_l2235_223563

/-- Calculates the driver's net hourly rate after deducting gas expenses -/
theorem driver_net_hourly_rate
  (travel_time : ℝ)
  (speed : ℝ)
  (gasoline_efficiency : ℝ)
  (gasoline_cost : ℝ)
  (driver_compensation : ℝ)
  (h1 : travel_time = 3)
  (h2 : speed = 50)
  (h3 : gasoline_efficiency = 25)
  (h4 : gasoline_cost = 2.5)
  (h5 : driver_compensation = 0.6)
  : (driver_compensation * speed * travel_time - 
     (speed * travel_time / gasoline_efficiency) * gasoline_cost) / travel_time = 25 :=
by sorry

end NUMINAMATH_CALUDE_driver_net_hourly_rate_l2235_223563


namespace NUMINAMATH_CALUDE_daughter_normal_probability_l2235_223519

-- Define the inheritance types
inductive InheritanceType
| XLinked
| Autosomal

-- Define the phenotypes
inductive Phenotype
| Normal
| Affected

-- Define the genotypes
structure Genotype where
  hemophilia : Bool  -- true if carrier
  phenylketonuria : Bool  -- true if carrier

-- Define the parents
structure Parents where
  mother : Genotype
  father : Genotype

-- Define the conditions
def conditions (parents : Parents) : Prop :=
  (InheritanceType.XLinked = InheritanceType.XLinked) ∧  -- Hemophilia is X-linked
  (InheritanceType.Autosomal = InheritanceType.Autosomal) ∧  -- Phenylketonuria is autosomal
  (parents.mother.hemophilia = true) ∧  -- Mother is carrier for hemophilia
  (parents.father.hemophilia = false) ∧  -- Father is not affected by hemophilia
  (parents.mother.phenylketonuria = true) ∧  -- Mother is carrier for phenylketonuria
  (parents.father.phenylketonuria = true)  -- Father is carrier for phenylketonuria

-- Define the probability of a daughter being phenotypically normal
def prob_normal_daughter (parents : Parents) : ℚ :=
  3 / 4

-- The theorem to prove
theorem daughter_normal_probability (parents : Parents) :
  conditions parents → prob_normal_daughter parents = 3 / 4 :=
by sorry

end NUMINAMATH_CALUDE_daughter_normal_probability_l2235_223519


namespace NUMINAMATH_CALUDE_brianne_alex_yard_ratio_l2235_223508

/-- Proves that Brianne's yard is 6 times larger than Alex's yard given the conditions -/
theorem brianne_alex_yard_ratio :
  ∀ (derrick_yard alex_yard brianne_yard : ℝ),
  derrick_yard = 10 →
  alex_yard = derrick_yard / 2 →
  brianne_yard = 30 →
  brianne_yard / alex_yard = 6 := by
sorry

end NUMINAMATH_CALUDE_brianne_alex_yard_ratio_l2235_223508


namespace NUMINAMATH_CALUDE_solution_value_l2235_223544

theorem solution_value (m : ℝ) : 
  (∃ x : ℝ, x = 1 ∧ 2 * x - m = -3) → m = 5 :=
by sorry

end NUMINAMATH_CALUDE_solution_value_l2235_223544


namespace NUMINAMATH_CALUDE_greatest_NPMPP_l2235_223524

/-- A function that checks if a number's square ends with the number itself -/
def endsWithSelf (n : Nat) : Prop :=
  n % 10 = (n * n) % 10

/-- A function that generates a four-digit number with all identical digits -/
def fourIdenticalDigits (d : Nat) : Nat :=
  d * 1000 + d * 100 + d * 10 + d

/-- The theorem stating the greatest possible value of NPMPP -/
theorem greatest_NPMPP : 
  ∃ (M : Nat), 
    M ≤ 9 ∧ 
    endsWithSelf M ∧ 
    ∀ (N : Nat), N ≤ 9 → endsWithSelf N → M ≥ N ∧
    fourIdenticalDigits M * M = 89991 :=
sorry

end NUMINAMATH_CALUDE_greatest_NPMPP_l2235_223524


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l2235_223595

/-- An isosceles triangle with integer side lengths and perimeter 10 --/
structure IsoscelesTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  isIsosceles : (a = b ∧ a ≠ c) ∨ (a = c ∧ a ≠ b) ∨ (b = c ∧ b ≠ a)
  perimeter : a + b + c = 10

/-- The possible side lengths of the isosceles triangle --/
def validSideLengths (t : IsoscelesTriangle) : Prop :=
  (t.a = 3 ∧ t.b = 3 ∧ t.c = 4) ∨ (t.a = 4 ∧ t.b = 4 ∧ t.c = 2)

/-- Theorem stating that the only possible side lengths are (3, 3, 4) or (4, 4, 2) --/
theorem isosceles_triangle_side_lengths (t : IsoscelesTriangle) : validSideLengths t := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_lengths_l2235_223595


namespace NUMINAMATH_CALUDE_min_transportation_cost_min_cost_at_ten_l2235_223590

/-- Represents the total transportation cost function --/
def transportation_cost (x : ℝ) : ℝ := 4 * x + 1980

/-- Theorem stating the minimum transportation cost --/
theorem min_transportation_cost :
  ∀ x : ℝ, 10 ≤ x ∧ x ≤ 50 → transportation_cost x ≥ 2020 :=
by
  sorry

/-- Theorem stating that the minimum cost occurs at x = 10 --/
theorem min_cost_at_ten :
  transportation_cost 10 = 2020 :=
by
  sorry

end NUMINAMATH_CALUDE_min_transportation_cost_min_cost_at_ten_l2235_223590


namespace NUMINAMATH_CALUDE_negation_or_false_implies_and_false_l2235_223528

theorem negation_or_false_implies_and_false (p q : Prop) : 
  ¬(¬(p ∨ q)) → ¬(p ∧ q) := by
  sorry

end NUMINAMATH_CALUDE_negation_or_false_implies_and_false_l2235_223528


namespace NUMINAMATH_CALUDE_solve_equation_l2235_223589

theorem solve_equation : 48 / (7 - 3/4) = 192/25 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2235_223589


namespace NUMINAMATH_CALUDE_triangle_property_l2235_223557

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The condition that a*sin(B) - √3*b*cos(A) = 0 -/
def satisfiesCondition (t : Triangle) : Prop :=
  t.a * Real.sin t.B - Real.sqrt 3 * t.b * Real.cos t.A = 0

theorem triangle_property (t : Triangle) 
  (h : satisfiesCondition t) : 
  t.A = π / 3 ∧ 
  (t.a = Real.sqrt 7 ∧ t.b = 2 → 
    (1/2 : ℝ) * t.b * t.c * Real.sin t.A = (3 * Real.sqrt 3) / 2) :=
sorry


end NUMINAMATH_CALUDE_triangle_property_l2235_223557


namespace NUMINAMATH_CALUDE_quadratic_roots_relation_l2235_223598

theorem quadratic_roots_relation (m n p : ℝ) (hm : m ≠ 0) (hn : n ≠ 0) (hp : p ≠ 0) :
  (∀ x, x^2 + m*x + n = 0 ↔ (x/2)^2 + p*(x/2) + m = 0) →
  n / p = 8 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_relation_l2235_223598


namespace NUMINAMATH_CALUDE_basketball_team_cutoff_l2235_223567

theorem basketball_team_cutoff (girls : ℕ) (boys : ℕ) (called_back : ℕ) 
  (h1 : girls = 9)
  (h2 : boys = 14)
  (h3 : called_back = 2) :
  girls + boys - called_back = 21 := by
  sorry

end NUMINAMATH_CALUDE_basketball_team_cutoff_l2235_223567


namespace NUMINAMATH_CALUDE_license_plate_palindrome_theorem_l2235_223558

/-- The number of possible letters in the alphabet -/
def num_letters : ℕ := 26

/-- The number of possible digits -/
def num_digits : ℕ := 10

/-- The length of the letter sequence in the license plate -/
def letter_length : ℕ := 4

/-- The length of the digit sequence in the license plate -/
def digit_length : ℕ := 4

/-- The probability of a license plate containing at least one palindrome -/
def license_plate_palindrome_probability : ℚ := 655 / 57122

/-- 
Theorem: The probability of a license plate containing at least one palindrome 
(either in the four-letter or four-digit arrangement) is 655/57122.
-/
theorem license_plate_palindrome_theorem : 
  license_plate_palindrome_probability = 655 / 57122 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_palindrome_theorem_l2235_223558


namespace NUMINAMATH_CALUDE_triangle_side_sum_l2235_223596

theorem triangle_side_sum (a b c : ℝ) (h1 : a + b + c = 180) 
  (h2 : a = 60) (h3 : b = 30) (h4 : c = 90) (h5 : 9 = a.sin * 18) :
  18 + 9 * Real.sqrt 3 = 18 + b.sin * 18 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_sum_l2235_223596


namespace NUMINAMATH_CALUDE_painter_workdays_l2235_223540

theorem painter_workdays (initial_painters : ℕ) (initial_days : ℚ) (new_painters : ℕ) :
  initial_painters = 5 →
  initial_days = 4/5 →
  new_painters = 4 →
  (initial_painters : ℚ) * initial_days = new_painters * 1 :=
by sorry

end NUMINAMATH_CALUDE_painter_workdays_l2235_223540


namespace NUMINAMATH_CALUDE_cookies_on_floor_l2235_223516

/-- Calculates the number of cookies thrown on the floor given the initial and additional cookies baked by Alice and Bob, and the final number of edible cookies. -/
theorem cookies_on_floor (alice_initial bob_initial alice_additional bob_additional final_edible : ℕ) :
  alice_initial = 74 →
  bob_initial = 7 →
  alice_additional = 5 →
  bob_additional = 36 →
  final_edible = 93 →
  (alice_initial + bob_initial + alice_additional + bob_additional) - final_edible = 29 := by
  sorry

#check cookies_on_floor

end NUMINAMATH_CALUDE_cookies_on_floor_l2235_223516


namespace NUMINAMATH_CALUDE_clock_hands_opposite_l2235_223574

/-- Represents the number of minutes past 10:00 --/
def x : ℝ := 13

/-- The rate at which the minute hand moves (degrees per minute) --/
def minute_hand_rate : ℝ := 6

/-- The rate at which the hour hand moves (degrees per minute) --/
def hour_hand_rate : ℝ := 0.5

/-- The angle between the minute and hour hands when they are opposite --/
def opposite_angle : ℝ := 180

theorem clock_hands_opposite : 
  0 < x ∧ x < 60 ∧
  minute_hand_rate * (6 + x) + hour_hand_rate * (120 - x + 3) = opposite_angle :=
by sorry

end NUMINAMATH_CALUDE_clock_hands_opposite_l2235_223574


namespace NUMINAMATH_CALUDE_symmetric_point_l2235_223505

/-- The point symmetric to P(2,-3) with respect to the origin is (-2,3). -/
theorem symmetric_point : 
  let P : ℝ × ℝ := (2, -3)
  let symmetric_wrt_origin (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, -p.2)
  symmetric_wrt_origin P = (-2, 3) := by
sorry

end NUMINAMATH_CALUDE_symmetric_point_l2235_223505


namespace NUMINAMATH_CALUDE_gcd_765432_654321_l2235_223550

theorem gcd_765432_654321 : Nat.gcd 765432 654321 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_765432_654321_l2235_223550


namespace NUMINAMATH_CALUDE_b_current_age_b_current_age_proof_l2235_223575

theorem b_current_age : ℕ → ℕ → Prop :=
  fun a b =>
    (a = b + 15) →  -- A is 15 years older than B
    (a - 5 = 2 * (b - 5)) →  -- Five years ago, A's age was twice B's age
    (b = 20)  -- B's current age is 20

-- The proof is omitted
theorem b_current_age_proof : ∃ (a b : ℕ), b_current_age a b :=
  sorry

end NUMINAMATH_CALUDE_b_current_age_b_current_age_proof_l2235_223575


namespace NUMINAMATH_CALUDE_fraction_subtraction_l2235_223536

theorem fraction_subtraction : (18 : ℚ) / 42 - 3 / 8 = 3 / 56 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l2235_223536


namespace NUMINAMATH_CALUDE_min_quadratic_l2235_223509

theorem min_quadratic (x : ℝ) : 
  (∀ y : ℝ, x^2 + 7*x + 3 ≤ y^2 + 7*y + 3) → x = -7/2 := by
  sorry

end NUMINAMATH_CALUDE_min_quadratic_l2235_223509


namespace NUMINAMATH_CALUDE_leonie_cats_l2235_223511

theorem leonie_cats : ∃ n : ℚ, n = (4 / 5) * n + (4 / 5) → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_leonie_cats_l2235_223511


namespace NUMINAMATH_CALUDE_geometry_propositions_l2235_223564

structure Geometry3D where
  Line : Type
  Plane : Type
  parallel : Line → Plane → Prop
  perpendicular : Line → Plane → Prop
  plane_parallel : Plane → Plane → Prop
  plane_perpendicular : Plane → Plane → Prop

variable (G : Geometry3D)

theorem geometry_propositions 
  (l : G.Line) (α β : G.Plane) (h_diff : α ≠ β) :
  (∃ l α β, G.parallel l α ∧ G.parallel l β ∧ ¬ G.plane_parallel α β) ∧
  (∀ l α β, G.perpendicular l α ∧ G.perpendicular l β → G.plane_parallel α β) ∧
  (∃ l α β, G.perpendicular l α ∧ G.parallel l β ∧ ¬ G.plane_parallel α β) ∧
  (∃ l α β, G.plane_perpendicular α β ∧ G.parallel l α ∧ ¬ G.perpendicular l β) :=
by sorry

end NUMINAMATH_CALUDE_geometry_propositions_l2235_223564


namespace NUMINAMATH_CALUDE_red_black_red_probability_l2235_223545

/-- Represents a standard deck of 52 cards -/
def StandardDeck : ℕ := 52

/-- Number of red cards in a standard deck -/
def RedCards : ℕ := 26

/-- Number of black cards in a standard deck -/
def BlackCards : ℕ := 26

/-- Probability of drawing a red card, then a black card, then a red card from a standard deck -/
theorem red_black_red_probability :
  (RedCards : ℚ) * BlackCards * (RedCards - 1) / (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2)) = 13 / 102 := by
  sorry

end NUMINAMATH_CALUDE_red_black_red_probability_l2235_223545


namespace NUMINAMATH_CALUDE_pizza_area_increase_l2235_223530

/-- The increase in area when switching from a circular pizza with diameter 16 inches
    to a square pizza with the same perimeter -/
theorem pizza_area_increase :
  let d : ℝ := 16  -- diameter of circular pizza
  let r : ℝ := d / 2  -- radius of circular pizza
  let circ_perimeter : ℝ := 2 * Real.pi * r  -- perimeter of circular pizza
  let square_side : ℝ := circ_perimeter / 4  -- side length of square pizza
  let circ_area : ℝ := Real.pi * r^2  -- area of circular pizza
  let square_area : ℝ := square_side^2  -- area of square pizza
  (square_area - circ_area) / circ_area = Real.pi / 4 - 1 :=
by sorry

end NUMINAMATH_CALUDE_pizza_area_increase_l2235_223530


namespace NUMINAMATH_CALUDE_division_problem_l2235_223572

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 181 → 
  quotient = 9 → 
  remainder = 1 → 
  dividend = divisor * quotient + remainder →
  divisor = 20 := by sorry

end NUMINAMATH_CALUDE_division_problem_l2235_223572


namespace NUMINAMATH_CALUDE_minimum_value_theorem_l2235_223538

-- Define the variables x, y, and z as positive real numbers
variable (x y z : ℝ)

-- Define the conditions
def positive_conditions : Prop := x > 0 ∧ y > 0 ∧ z > 0
def equation_condition : Prop := x - 2*y + 3*z = 0

-- State the theorem
theorem minimum_value_theorem 
  (h1 : positive_conditions x y z) 
  (h2 : equation_condition x y z) :
  ∃ (f : ℝ → ℝ → ℝ → ℝ), ∀ (a b c : ℝ), 
    positive_conditions a b c → 
    equation_condition a b c → 
    f x y z ≤ f a b c :=
sorry

end NUMINAMATH_CALUDE_minimum_value_theorem_l2235_223538


namespace NUMINAMATH_CALUDE_candy_distribution_l2235_223562

theorem candy_distribution (total_candies : ℕ) 
  (lollipops_per_boy : ℕ) (candy_canes_per_girl : ℕ) : 
  total_candies = 90 →
  lollipops_per_boy = 3 →
  candy_canes_per_girl = 2 →
  ∃ (num_boys num_girls : ℕ),
    num_boys * lollipops_per_boy = total_candies / 3 ∧
    num_girls * candy_canes_per_girl = total_candies * 2 / 3 ∧
    num_boys + num_girls = 40 :=
by sorry

end NUMINAMATH_CALUDE_candy_distribution_l2235_223562


namespace NUMINAMATH_CALUDE_square_plot_area_l2235_223573

/-- Proves that a square plot with given fence costs has an area of 144 square feet -/
theorem square_plot_area (cost_per_foot : ℝ) (total_cost : ℝ) : 
  cost_per_foot = 58 → total_cost = 2784 → 
  (total_cost / (4 * cost_per_foot))^2 = 144 := by
  sorry

end NUMINAMATH_CALUDE_square_plot_area_l2235_223573


namespace NUMINAMATH_CALUDE_contour_area_ratio_l2235_223548

theorem contour_area_ratio (r₁ r₂ : ℝ) (A₁ A₂ : ℝ) (h₁ : 0 < r₁) (h₂ : r₁ < r₂) (h₃ : 0 < A₁) :
  A₂ / A₁ = (r₂ / r₁)^2 :=
sorry

end NUMINAMATH_CALUDE_contour_area_ratio_l2235_223548


namespace NUMINAMATH_CALUDE_corner_cut_length_l2235_223507

/-- Given a rectangular sheet of dimensions 48 m x 36 m, if squares of side length x
    are cut from each corner to form an open box with volume 5120 m³,
    then x = 8 m. -/
theorem corner_cut_length (x : ℝ) : 
  x > 0 ∧ x < 18 ∧ (48 - 2*x) * (36 - 2*x) * x = 5120 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_corner_cut_length_l2235_223507


namespace NUMINAMATH_CALUDE_other_root_of_quadratic_l2235_223501

theorem other_root_of_quadratic (m : ℝ) : 
  (∃ x : ℝ, 7 * x^2 + m * x - 6 = 0 ∧ x = -3) →
  (7 * (2/7)^2 + m * (2/7) - 6 = 0) :=
by sorry

end NUMINAMATH_CALUDE_other_root_of_quadratic_l2235_223501


namespace NUMINAMATH_CALUDE_ellipse_chord_slope_l2235_223535

/-- Given an ellipse with equation 4x^2 + 9y^2 = 144 containing a point P(3, 2),
    the slope of the line containing the chord with P as its midpoint is -2/3. -/
theorem ellipse_chord_slope :
  let ellipse := {(x, y) : ℝ × ℝ | 4 * x^2 + 9 * y^2 = 144}
  let P : ℝ × ℝ := (3, 2)
  P ∈ ellipse →
  ∃ (A B : ℝ × ℝ),
    A ∈ ellipse ∧ B ∈ ellipse ∧
    P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2) ∧
    (B.2 - A.2) / (B.1 - A.1) = -2/3 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_chord_slope_l2235_223535


namespace NUMINAMATH_CALUDE_parabola_vertex_below_x_axis_l2235_223534

/-- A parabola with equation y = x^2 + 2x + a has its vertex below the x-axis -/
def vertex_below_x_axis (a : ℝ) : Prop :=
  ∃ (x y : ℝ), y = x^2 + 2*x + a ∧ y < 0 ∧ ∀ (x' : ℝ), x'^2 + 2*x' + a ≥ y

theorem parabola_vertex_below_x_axis (a : ℝ) :
  vertex_below_x_axis a → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_below_x_axis_l2235_223534


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2235_223527

theorem sqrt_equation_solution :
  ∀ x : ℝ, Real.sqrt (2 + Real.sqrt x) = 3 → x = 49 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2235_223527


namespace NUMINAMATH_CALUDE_school_sample_theorem_l2235_223549

theorem school_sample_theorem (total_students sample_size : ℕ) 
  (h_total : total_students = 1200)
  (h_sample : sample_size = 200)
  (h_stratified : ∃ (boys girls : ℕ), boys + girls = sample_size ∧ boys = girls + 10) :
  ∃ (school_boys : ℕ), 
    school_boys * sample_size = 105 * total_students ∧
    school_boys = 630 := by
sorry

end NUMINAMATH_CALUDE_school_sample_theorem_l2235_223549


namespace NUMINAMATH_CALUDE_hamburgers_for_lunch_l2235_223566

theorem hamburgers_for_lunch (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  initial = 9 → additional = 3 → total = initial + additional → total = 12 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_for_lunch_l2235_223566


namespace NUMINAMATH_CALUDE_sum_perfect_square_l2235_223591

theorem sum_perfect_square (K M : ℕ) : 
  K > 0 → M < 100 → K * (K + 1) = M^2 → (K = 8 ∨ K = 35) := by
  sorry

end NUMINAMATH_CALUDE_sum_perfect_square_l2235_223591


namespace NUMINAMATH_CALUDE_sum_of_max_min_values_l2235_223584

theorem sum_of_max_min_values (f : ℝ → ℝ) (h : f = fun x ↦ 9 * (Real.cos x)^4 + 12 * (Real.sin x)^2 - 4) :
  (⨆ x, f x) + (⨅ x, f x) = 13 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_values_l2235_223584


namespace NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l2235_223503

/-- The maximum y-coordinate of a point on the curve r = sin 3θ is 4√3/9 -/
theorem max_y_coordinate_sin_3theta :
  let r : ℝ → ℝ := λ θ => Real.sin (3 * θ)
  let y : ℝ → ℝ := λ θ => r θ * Real.sin θ
  ∃ (max_y : ℝ), (∀ θ, y θ ≤ max_y) ∧ (max_y = 4 * Real.sqrt 3 / 9) := by
  sorry

end NUMINAMATH_CALUDE_max_y_coordinate_sin_3theta_l2235_223503


namespace NUMINAMATH_CALUDE_angle_C_value_angle_C_range_l2235_223553

noncomputable section

-- Define the triangle ABC
variable (A B C : Real) -- Angles
variable (a b c : Real) -- Sides

-- Define the function f
def f (x : Real) : Real := a^2 * x^2 - (a^2 - b^2) * x - 4 * c^2

-- Theorem 1
theorem angle_C_value (h1 : f 1 = 0) (h2 : B - C = π/3) : C = π/6 := by
  sorry

-- Theorem 2
theorem angle_C_range (h : f 2 = 0) : 0 < C ∧ C ≤ π/3 := by
  sorry

end

end NUMINAMATH_CALUDE_angle_C_value_angle_C_range_l2235_223553


namespace NUMINAMATH_CALUDE_tangent_slope_angle_at_point_one_three_l2235_223571

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 2*x + 4

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 2

-- Theorem statement
theorem tangent_slope_angle_at_point_one_three :
  let x₀ : ℝ := 1
  let y₀ : ℝ := f x₀
  let k : ℝ := f' x₀
  Real.arctan k = π/4 :=
by sorry

end NUMINAMATH_CALUDE_tangent_slope_angle_at_point_one_three_l2235_223571


namespace NUMINAMATH_CALUDE_monomials_like_terms_iff_l2235_223559

/-- Two monomials are like terms if they have the same variables with the same exponents -/
def like_terms (m1 m2 : ℕ → ℕ) : Prop :=
  ∀ v, m1 v = m2 v

/-- The first monomial 4ab^n -/
def monomial1 (n : ℕ) : ℕ → ℕ
| 0 => 1  -- exponent of a
| 1 => n  -- exponent of b
| _ => 0  -- other variables

/-- The second monomial -2a^mb^4 -/
def monomial2 (m : ℕ) : ℕ → ℕ
| 0 => m  -- exponent of a
| 1 => 4  -- exponent of b
| _ => 0  -- other variables

/-- Theorem: The monomials 4ab^n and -2a^mb^4 are like terms if and only if m = 1 and n = 4 -/
theorem monomials_like_terms_iff (m n : ℕ) :
  like_terms (monomial1 n) (monomial2 m) ↔ m = 1 ∧ n = 4 :=
by sorry

end NUMINAMATH_CALUDE_monomials_like_terms_iff_l2235_223559


namespace NUMINAMATH_CALUDE_complex_ratio_theorem_l2235_223502

theorem complex_ratio_theorem (x y : ℂ) 
  (h1 : (x^2 + y^2) / (x + y) = 4)
  (h2 : (x^4 + y^4) / (x^3 + y^3) = 2) :
  (x^6 + y^6) / (x^5 + y^5) = 10 + 2 * Real.sqrt 17 ∨
  (x^6 + y^6) / (x^5 + y^5) = 10 - 2 * Real.sqrt 17 :=
by sorry

end NUMINAMATH_CALUDE_complex_ratio_theorem_l2235_223502


namespace NUMINAMATH_CALUDE_expression_evaluation_l2235_223592

theorem expression_evaluation : 
  let f (x : ℝ) := (x^2 - 4*x + 4) / (2*x) / ((x^2 - 2*x) / x^2) + 1
  f 1 = 1/2 :=
by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l2235_223592


namespace NUMINAMATH_CALUDE_workshop_workers_l2235_223512

/-- The total number of workers in the workshop -/
def total_workers : ℕ := 15

/-- The number of technicians in the workshop -/
def num_technicians : ℕ := 5

/-- The average salary of all workers in the workshop -/
def avg_salary_all : ℚ := 700

/-- The average salary of technicians -/
def avg_salary_technicians : ℚ := 800

/-- The average salary of non-technician workers -/
def avg_salary_rest : ℚ := 650

theorem workshop_workers :
  total_workers = num_technicians + 
    (avg_salary_all * total_workers - avg_salary_technicians * num_technicians) / 
    (avg_salary_rest - avg_salary_all) := by
  sorry

end NUMINAMATH_CALUDE_workshop_workers_l2235_223512


namespace NUMINAMATH_CALUDE_has_unique_prime_divisor_l2235_223583

theorem has_unique_prime_divisor (n m : ℕ) (h1 : n > m) (h2 : m > 0) :
  ∃ p : ℕ, Prime p ∧ (p ∣ (2^n - 1)) ∧ ¬(p ∣ (2^m - 1)) := by
  sorry

end NUMINAMATH_CALUDE_has_unique_prime_divisor_l2235_223583


namespace NUMINAMATH_CALUDE_only_B_on_x_axis_l2235_223532

def point_A : ℝ × ℝ := (-2, -3)
def point_B : ℝ × ℝ := (-3, 0)
def point_C : ℝ × ℝ := (-1, 2)
def point_D : ℝ × ℝ := (0, 3)

def is_on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

theorem only_B_on_x_axis : 
  ¬(is_on_x_axis point_A) ∧
  is_on_x_axis point_B ∧
  ¬(is_on_x_axis point_C) ∧
  ¬(is_on_x_axis point_D) :=
by sorry

end NUMINAMATH_CALUDE_only_B_on_x_axis_l2235_223532


namespace NUMINAMATH_CALUDE_least_common_denominator_l2235_223578

theorem least_common_denominator : 
  let denominators : List Nat := [3, 4, 5, 6, 8, 9, 10]
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 3 4) 5) 6) 8) 9) 10 = 360 := by
  sorry

end NUMINAMATH_CALUDE_least_common_denominator_l2235_223578


namespace NUMINAMATH_CALUDE_min_value_sum_l2235_223510

theorem min_value_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 1/x + 1/y = 1) :
  4*x + 9*y ≥ 25 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 1/x₀ + 1/y₀ = 1 ∧ 4*x₀ + 9*y₀ = 25 :=
by sorry

end NUMINAMATH_CALUDE_min_value_sum_l2235_223510


namespace NUMINAMATH_CALUDE_medication_cost_leap_year_l2235_223581

/-- Calculate the total medication cost for a leap year given the following conditions:
  * 3 types of pills
  * Pill 1 costs $1.5, Pill 2 costs $2.3, Pill 3 costs $3.8
  * Insurance covers 40% of Pill 1, 25% of Pill 2, 10% of Pill 3
  * Discount card provides 15% off Pill 2 and 5% off Pill 3
  * A leap year has 366 days -/
theorem medication_cost_leap_year : 
  let pill1_cost : ℝ := 1.5
  let pill2_cost : ℝ := 2.3
  let pill3_cost : ℝ := 3.8
  let insurance_coverage1 : ℝ := 0.4
  let insurance_coverage2 : ℝ := 0.25
  let insurance_coverage3 : ℝ := 0.1
  let discount_card2 : ℝ := 0.15
  let discount_card3 : ℝ := 0.05
  let days_in_leap_year : ℕ := 366

  let pill1_final_cost := pill1_cost * (1 - insurance_coverage1)
  let pill2_final_cost := pill2_cost * (1 - insurance_coverage2) * (1 - discount_card2)
  let pill3_final_cost := pill3_cost * (1 - insurance_coverage3) * (1 - discount_card3)

  let daily_cost := pill1_final_cost + pill2_final_cost + pill3_final_cost
  let yearly_cost := daily_cost * days_in_leap_year

  yearly_cost = 2055.5835 := by
sorry

end NUMINAMATH_CALUDE_medication_cost_leap_year_l2235_223581


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l2235_223522

theorem sqrt_meaningful_range (x : ℝ) : (∃ y : ℝ, y^2 = x - 1) → x ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l2235_223522


namespace NUMINAMATH_CALUDE_protein_percentage_in_mixture_l2235_223547

/-- Calculates the protein percentage in a mixture of soybean meal and cornmeal. -/
theorem protein_percentage_in_mixture 
  (soybean_protein_percent : ℝ)
  (cornmeal_protein_percent : ℝ)
  (total_mixture_weight : ℝ)
  (soybean_weight : ℝ)
  (cornmeal_weight : ℝ)
  (h1 : soybean_protein_percent = 0.14)
  (h2 : cornmeal_protein_percent = 0.07)
  (h3 : total_mixture_weight = 280)
  (h4 : soybean_weight = 240)
  (h5 : cornmeal_weight = 40)
  (h6 : total_mixture_weight = soybean_weight + cornmeal_weight) :
  (soybean_weight * soybean_protein_percent + cornmeal_weight * cornmeal_protein_percent) / total_mixture_weight = 0.13 := by
  sorry


end NUMINAMATH_CALUDE_protein_percentage_in_mixture_l2235_223547


namespace NUMINAMATH_CALUDE_f_max_value_implies_a_eq_three_l2235_223514

/-- The function f(x) = -4x^3 + ax -/
def f (a : ℝ) (x : ℝ) : ℝ := -4 * x^3 + a * x

/-- The maximum value of f(x) on [-1,1] is 1 -/
def max_value_is_one (a : ℝ) : Prop :=
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → f a x ≤ 1 ∧ ∃ y : ℝ, y ∈ Set.Icc (-1) 1 ∧ f a y = 1

theorem f_max_value_implies_a_eq_three :
  ∀ a : ℝ, max_value_is_one a → a = 3 := by sorry

end NUMINAMATH_CALUDE_f_max_value_implies_a_eq_three_l2235_223514


namespace NUMINAMATH_CALUDE_range_of_a_l2235_223531

theorem range_of_a (a : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + a*x + 1 ≥ 0) ↔ (a < -2 ∨ a > 2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2235_223531


namespace NUMINAMATH_CALUDE_repayment_plan_earnings_l2235_223582

def hourly_rate (hour : ℕ) : ℕ :=
  if hour % 8 = 0 then 8 else hour % 8

def total_earnings (hours : ℕ) : ℕ :=
  (List.range hours).map hourly_rate |>.sum

theorem repayment_plan_earnings :
  total_earnings 50 = 219 :=
by sorry

end NUMINAMATH_CALUDE_repayment_plan_earnings_l2235_223582


namespace NUMINAMATH_CALUDE_max_sock_pairs_l2235_223594

theorem max_sock_pairs (initial_pairs : ℕ) (lost_socks : ℕ) (max_pairs : ℕ) : 
  initial_pairs = 10 →
  lost_socks = 5 →
  max_pairs = 5 →
  max_pairs = initial_pairs - (lost_socks / 2 + lost_socks % 2) :=
by sorry

end NUMINAMATH_CALUDE_max_sock_pairs_l2235_223594


namespace NUMINAMATH_CALUDE_max_value_of_z_l2235_223546

theorem max_value_of_z (x y : ℝ) (h1 : y ≤ 1) (h2 : x + y ≥ 0) (h3 : x - y - 2 ≤ 0) :
  ∃ (z : ℝ), z = x - 2*y ∧ z ≤ 3 ∧ ∀ (w : ℝ), w = x - 2*y → w ≤ z :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_z_l2235_223546


namespace NUMINAMATH_CALUDE_estimate_white_balls_l2235_223552

/-- Represents the contents of the box -/
structure Box where
  black : ℕ
  white : ℕ

/-- Represents the result of the drawing experiment -/
structure DrawResult where
  total : ℕ
  black : ℕ

/-- Calculates the expected number of white balls given the box contents and draw results -/
def expectedWhiteBalls (box : Box) (result : DrawResult) : ℚ :=
  (box.black : ℚ) * (result.total - result.black : ℚ) / result.black

/-- The main theorem statement -/
theorem estimate_white_balls (box : Box) (result : DrawResult) :
  box.black = 4 ∧ result.total = 40 ∧ result.black = 10 →
  expectedWhiteBalls box result = 12 := by
  sorry

end NUMINAMATH_CALUDE_estimate_white_balls_l2235_223552


namespace NUMINAMATH_CALUDE_max_value_polynomial_l2235_223561

theorem max_value_polynomial (x y : ℝ) (h : x + y = 3) :
  ∃ M : ℝ, M = 400 / 11 ∧ 
  ∀ a b : ℝ, a + b = 3 → 
  a^4 * b + a^3 * b + a^2 * b + a * b + a * b^2 + a * b^3 + a * b^4 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_polynomial_l2235_223561


namespace NUMINAMATH_CALUDE_sunzi_wood_problem_l2235_223515

theorem sunzi_wood_problem (x : ℝ) : 
  (∃ rope : ℝ, rope = x + 4.5 ∧ (rope / 2) + 1 = x) → 
  (1/2 * (x + 4.5) = x - 1) := by
sorry

end NUMINAMATH_CALUDE_sunzi_wood_problem_l2235_223515


namespace NUMINAMATH_CALUDE_power_equality_l2235_223520

theorem power_equality (q : ℕ) : 81^10 = 3^q → q = 40 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l2235_223520


namespace NUMINAMATH_CALUDE_principal_amount_correct_l2235_223570

/-- The principal amount borrowed -/
def P : ℝ := 22539.53

/-- The total interest paid after 3 years -/
def total_interest : ℝ := 9692

/-- The interest rate for the first year -/
def r1 : ℝ := 0.12

/-- The interest rate for the second year -/
def r2 : ℝ := 0.14

/-- The interest rate for the third year -/
def r3 : ℝ := 0.17

/-- Theorem stating that the given principal amount results in the specified total interest -/
theorem principal_amount_correct : 
  P * r1 + P * r2 + P * r3 = total_interest := by sorry

end NUMINAMATH_CALUDE_principal_amount_correct_l2235_223570


namespace NUMINAMATH_CALUDE_log_comparison_l2235_223576

theorem log_comparison (a b c : ℝ) : 
  a = (Real.log 6) / (Real.log 3) →
  b = (Real.log 10) / (Real.log 5) →
  c = (Real.log 14) / (Real.log 7) →
  a > b ∧ b > c := by
  sorry

end NUMINAMATH_CALUDE_log_comparison_l2235_223576


namespace NUMINAMATH_CALUDE_hamburgers_leftover_count_l2235_223537

/-- The number of hamburgers made by the restaurant -/
def hamburgers_made : ℕ := 9

/-- The number of hamburgers served during lunch -/
def hamburgers_served : ℕ := 3

/-- The number of hamburgers left over -/
def hamburgers_leftover : ℕ := hamburgers_made - hamburgers_served

theorem hamburgers_leftover_count : hamburgers_leftover = 6 := by
  sorry

end NUMINAMATH_CALUDE_hamburgers_leftover_count_l2235_223537


namespace NUMINAMATH_CALUDE_two_lights_possible_l2235_223539

/-- Represents the state of light bulbs on an infinite integer line -/
def LightState := Int → Bool

/-- Applies the template set S to the light state at position p -/
def applyTemplate (S : Finset Int) (state : LightState) (p : Int) : LightState :=
  fun i => if (i - p) ∈ S then !state i else state i

/-- Counts the number of light bulbs that are on -/
def countOn (state : LightState) : Nat :=
  sorry

theorem two_lights_possible (S : Finset Int) :
  ∃ (ops : List Int), 
    let finalState := ops.foldl (fun st p => applyTemplate S st p) (fun _ => false)
    countOn finalState = 2 :=
  sorry

end NUMINAMATH_CALUDE_two_lights_possible_l2235_223539


namespace NUMINAMATH_CALUDE_abs_2y_minus_7_zero_l2235_223500

theorem abs_2y_minus_7_zero (y : ℚ) : |2 * y - 7| = 0 ↔ y = 7/2 := by
  sorry

end NUMINAMATH_CALUDE_abs_2y_minus_7_zero_l2235_223500


namespace NUMINAMATH_CALUDE_prime_divisibility_l2235_223599

theorem prime_divisibility (p q : Nat) 
  (hp : Prime p) (hq : Prime q) (hpq : p ≠ q) (hp5 : p > 5) (hq5 : q > 5) :
  (p ∣ (5^q - 2^q) → q ∣ (p - 1)) ∧ ¬(p*q ∣ (5^p - 2^p)*(5^q - 2^q)) := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l2235_223599


namespace NUMINAMATH_CALUDE_tunnel_length_l2235_223518

/-- Calculates the length of a tunnel given train and travel parameters -/
theorem tunnel_length
  (train_length : Real)
  (train_speed : Real)
  (exit_time : Real)
  (h1 : train_length = 1.5)
  (h2 : train_speed = 45)
  (h3 : exit_time = 4 / 60) :
  train_speed * exit_time - train_length = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_tunnel_length_l2235_223518
