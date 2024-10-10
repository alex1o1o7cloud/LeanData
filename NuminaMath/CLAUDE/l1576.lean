import Mathlib

namespace cone_slant_height_l1576_157631

/-- Given a cone with lateral surface area of 15π when unfolded and base radius of 3, 
    its slant height is 5. -/
theorem cone_slant_height (lateral_area : ℝ) (base_radius : ℝ) : 
  lateral_area = 15 * Real.pi ∧ base_radius = 3 → 
  (lateral_area / (Real.pi * base_radius) : ℝ) = 5 := by
  sorry

end cone_slant_height_l1576_157631


namespace negation_of_exists_cube_positive_l1576_157655

theorem negation_of_exists_cube_positive :
  (¬ ∃ x : ℝ, x^3 > 0) ↔ (∀ x : ℝ, x^3 ≤ 0) := by
  sorry

end negation_of_exists_cube_positive_l1576_157655


namespace percentage_problem_l1576_157641

theorem percentage_problem (P : ℝ) (x : ℝ) (h1 : x = 264) (h2 : (P / 100) * x = (1 / 3) * x + 110) : P = 75 := by
  sorry

end percentage_problem_l1576_157641


namespace complex_division_result_l1576_157616

theorem complex_division_result : (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I := by
  sorry

end complex_division_result_l1576_157616


namespace point_in_fourth_quadrant_l1576_157657

/-- A point is in the fourth quadrant if its x-coordinate is positive and its y-coordinate is negative -/
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

/-- For any real number x, the point (x^2 + 1, -4) is in the fourth quadrant -/
theorem point_in_fourth_quadrant (x : ℝ) :
  in_fourth_quadrant (x^2 + 1, -4) := by
  sorry


end point_in_fourth_quadrant_l1576_157657


namespace simplify_sqrt_18_l1576_157681

theorem simplify_sqrt_18 : Real.sqrt 18 = 3 * Real.sqrt 2 := by
  sorry

end simplify_sqrt_18_l1576_157681


namespace glycerin_percentage_after_dilution_l1576_157684

def initial_glycerin_percentage : ℝ := 0.9
def initial_volume : ℝ := 4
def added_water : ℝ := 0.8

theorem glycerin_percentage_after_dilution :
  let initial_glycerin := initial_glycerin_percentage * initial_volume
  let final_volume := initial_volume + added_water
  let final_glycerin_percentage := initial_glycerin / final_volume
  final_glycerin_percentage = 0.75 := by sorry

end glycerin_percentage_after_dilution_l1576_157684


namespace king_not_right_mind_queen_indeterminate_l1576_157653

-- Define the mental states
inductive MentalState
| RightMind
| NotRightMind

-- Define the royals
structure Royal where
  name : String
  state : MentalState

-- Define the belief function
def believes (r : Royal) (p : Prop) : Prop := sorry

-- Define the King and Queen of Spades
def King : Royal := ⟨"King of Spades", MentalState.NotRightMind⟩
def Queen : Royal := ⟨"Queen of Spades", MentalState.NotRightMind⟩

-- The main theorem
theorem king_not_right_mind_queen_indeterminate :
  believes Queen (believes King (Queen.state = MentalState.NotRightMind)) →
  (King.state = MentalState.NotRightMind) ∧
  ((Queen.state = MentalState.RightMind) ∨ (Queen.state = MentalState.NotRightMind)) :=
by sorry

end king_not_right_mind_queen_indeterminate_l1576_157653


namespace angle_between_a_and_b_l1576_157622

/-- The angle between two 3D vectors -/
def angle_between_vectors (a b : ℝ × ℝ × ℝ) : ℝ := by sorry

/-- The vector a -/
def a : ℝ × ℝ × ℝ := (1, 1, -4)

/-- The vector b -/
def b : ℝ × ℝ × ℝ := (1, -2, 2)

/-- The theorem stating that the angle between vectors a and b is 135 degrees -/
theorem angle_between_a_and_b : 
  angle_between_vectors a b = 135 * Real.pi / 180 := by sorry

end angle_between_a_and_b_l1576_157622


namespace pair_count_theorem_l1576_157670

def count_pairs (n : ℕ) : ℕ :=
  (n - 50) * (n - 51) / 2 + 1275

theorem pair_count_theorem :
  count_pairs 100 = 2500 :=
sorry

end pair_count_theorem_l1576_157670


namespace min_xy_and_x_plus_y_l1576_157644

/-- Given positive real numbers x and y satisfying x + 8y - xy = 0,
    proves that the minimum value of xy is 32 and
    the minimum value of x + y is 9 + 4√2 -/
theorem min_xy_and_x_plus_y (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x + 8*y - x*y = 0) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 8*y' - x'*y' = 0 → x*y ≤ x'*y') ∧
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' + 8*y' - x'*y' = 0 → x + y ≤ x' + y') ∧
  x*y = 32 ∧ x + y = 9 + 4*Real.sqrt 2 := by
  sorry

end min_xy_and_x_plus_y_l1576_157644


namespace hyperbola_equation_l1576_157654

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    its eccentricity is 2, and the distance from the origin to line AB
    (where A(a, 0) and B(0, -b)) is 3/2, prove that the equation of the
    hyperbola is x²/3 - y²/9 = 1. -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  (∃ c : ℝ, c / a = 2) →
  (∃ d : ℝ, d = 3/2 ∧ d = |(a * b)| / Real.sqrt (a^2 + b^2)) →
  (∀ x y : ℝ, x^2 / 3 - y^2 / 9 = 1) :=
by sorry

end hyperbola_equation_l1576_157654


namespace sum_of_max_min_g_l1576_157692

noncomputable def g (x : ℝ) : ℝ := |x - 3| + |x - 5| - |2*x - 8| + 1

theorem sum_of_max_min_g : 
  ∃ (max_g min_g : ℝ), 
    (∀ x ∈ Set.Icc 3 7, g x ≤ max_g) ∧
    (∃ x ∈ Set.Icc 3 7, g x = max_g) ∧
    (∀ x ∈ Set.Icc 3 7, min_g ≤ g x) ∧
    (∃ x ∈ Set.Icc 3 7, g x = min_g) ∧
    max_g + min_g = 4 :=
by sorry

end sum_of_max_min_g_l1576_157692


namespace range_of_a_for_solution_a_value_for_minimum_l1576_157667

-- Define the function f
def f (a x : ℝ) : ℝ := |2*x - a| + |x - 1|

-- Theorem for part I
theorem range_of_a_for_solution (a : ℝ) :
  (∃ x, f a x ≤ 2 - |x - 1|) ↔ (0 ≤ a ∧ a ≤ 4) :=
sorry

-- Theorem for part II
theorem a_value_for_minimum (a : ℝ) :
  a < 2 → (∀ x, f a x ≥ 3) → (∃ x, f a x = 3) → a = -4 :=
sorry

end range_of_a_for_solution_a_value_for_minimum_l1576_157667


namespace pool_filling_time_l1576_157658

-- Define the rates of the valves
variable (a b c d : ℝ)

-- Define the conditions
def condition1 : Prop := 1/a + 1/b + 1/c = 1/12
def condition2 : Prop := 1/b + 1/c + 1/d = 1/15
def condition3 : Prop := 1/a + 1/d = 1/20

-- Theorem statement
theorem pool_filling_time 
  (h1 : condition1 a b c)
  (h2 : condition2 b c d)
  (h3 : condition3 a d) :
  1/a + 1/b + 1/c + 1/d = 1/10 := by
  sorry


end pool_filling_time_l1576_157658


namespace money_needed_for_perfume_l1576_157648

def perfume_cost : ℕ := 50
def christian_initial : ℕ := 5
def sue_initial : ℕ := 7
def yards_mowed : ℕ := 4
def yard_charge : ℕ := 5
def dogs_walked : ℕ := 6
def dog_charge : ℕ := 2

theorem money_needed_for_perfume :
  perfume_cost - (christian_initial + sue_initial + yards_mowed * yard_charge + dogs_walked * dog_charge) = 6 := by
  sorry

end money_needed_for_perfume_l1576_157648


namespace inequality_system_solution_set_l1576_157642

theorem inequality_system_solution_set
  (x : ℝ) :
  (2 * x ≤ -2 ∧ x + 3 < 4) ↔ x ≤ -1 := by
sorry

end inequality_system_solution_set_l1576_157642


namespace age_difference_l1576_157632

theorem age_difference (A B C : ℕ) (h1 : C = A - 16) : A + B - (B + C) = 16 := by
  sorry

end age_difference_l1576_157632


namespace total_triangles_is_sixteen_l1576_157646

/-- Represents the count of triangles in each size category -/
structure TriangleCounts where
  small : Nat
  medium : Nat
  large : Nat

/-- Calculates the total number of triangles -/
def totalTriangles (counts : TriangleCounts) : Nat :=
  counts.small + counts.medium + counts.large

/-- The given triangle counts for the figure -/
def figureCounts : TriangleCounts :=
  { small := 11, medium := 4, large := 1 }

/-- Theorem stating that the total number of triangles in the figure is 16 -/
theorem total_triangles_is_sixteen :
  totalTriangles figureCounts = 16 := by
  sorry

end total_triangles_is_sixteen_l1576_157646


namespace quadratic_equation_roots_l1576_157695

theorem quadratic_equation_roots (m : ℝ) : 
  (∃ x : ℝ, 3 * x^2 - m * x - 3 = 0 ∧ x = 1) → 
  (∃ y : ℝ, 3 * y^2 - m * y - 3 = 0 ∧ y = -1) :=
by sorry

end quadratic_equation_roots_l1576_157695


namespace gcd_lcm_product_240_l1576_157604

theorem gcd_lcm_product_240 : 
  ∃! (s : Finset Nat), 
    (∀ d ∈ s, ∃ a b : Nat, a > 0 ∧ b > 0 ∧ Nat.gcd a b = d ∧ Nat.gcd a b * Nat.lcm a b = 240) ∧ 
    (∀ d : Nat, (∃ a b : Nat, a > 0 ∧ b > 0 ∧ Nat.gcd a b = d ∧ Nat.gcd a b * Nat.lcm a b = 240) → d ∈ s) ∧
    s.card = 10 :=
by sorry

end gcd_lcm_product_240_l1576_157604


namespace dvd_player_cost_l1576_157672

theorem dvd_player_cost (d m : ℝ) 
  (h1 : d / m = 9 / 2)
  (h2 : d = m + 63) :
  d = 81 := by
sorry

end dvd_player_cost_l1576_157672


namespace quadratic_max_value_l1576_157656

-- Define the quadratic function
def f (x : ℝ) : ℝ := 2 * (x - 1)^2 - 3

-- Define the theorem
theorem quadratic_max_value (a : ℝ) :
  (∀ x, 1 ≤ x ∧ x ≤ a → f x ≤ 15) ∧
  (∃ x, 1 ≤ x ∧ x ≤ a ∧ f x = 15) →
  a = 4 :=
by sorry

end quadratic_max_value_l1576_157656


namespace kite_parabolas_theorem_l1576_157682

/-- Represents a parabola in the form y = ax² + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- Represents the parameters of our problem -/
structure KiteParameters where
  parabola1 : Parabola
  parabola2 : Parabola
  kite_area : ℝ

/-- The main theorem statement -/
theorem kite_parabolas_theorem (params : KiteParameters) : 
  params.parabola1.a + params.parabola2.a = 1.04 :=
by
  sorry

/-- The specific instance of our problem -/
def our_problem : KiteParameters :=
  { parabola1 := { a := 2, b := -3 }
  , parabola2 := { a := -1, b := 5 }
  , kite_area := 20
  }

#check kite_parabolas_theorem our_problem

end kite_parabolas_theorem_l1576_157682


namespace infinite_series_sum_l1576_157683

/-- The sum of the infinite series Σ(n=1 to ∞) [(3n - 2) / (n(n + 1)(n + 3))] is equal to 11/12 -/
theorem infinite_series_sum : 
  ∑' (n : ℕ), (3 * n - 2 : ℝ) / (n * (n + 1) * (n + 3)) = 11 / 12 :=
by sorry

end infinite_series_sum_l1576_157683


namespace line_parametrization_l1576_157620

/-- The line equation --/
def line_equation (x y : ℝ) : Prop := y = (2/3) * x + 3

/-- The parametric equation of the line --/
def parametric_equation (x y s l t : ℝ) : Prop :=
  x = -9 + t * l ∧ y = s + t * (-7)

/-- The theorem stating the values of s and l --/
theorem line_parametrization :
  ∃ (s l : ℝ), (∀ (x y t : ℝ), line_equation x y ↔ parametric_equation x y s l t) ∧ s = -3 ∧ l = -10.5 := by
  sorry

end line_parametrization_l1576_157620


namespace incorrect_inequality_l1576_157652

theorem incorrect_inequality (m n : ℝ) (h : m > n) : ¬(-2 * m > -2 * n) := by
  sorry

end incorrect_inequality_l1576_157652


namespace janice_typing_problem_l1576_157696

theorem janice_typing_problem (typing_speed : ℕ) (initial_typing_time : ℕ) 
  (additional_typing_time : ℕ) (erased_sentences : ℕ) (final_typing_time : ℕ) 
  (total_sentences : ℕ) : 
  typing_speed = 6 →
  initial_typing_time = 20 →
  additional_typing_time = 15 →
  erased_sentences = 40 →
  final_typing_time = 18 →
  total_sentences = 536 →
  total_sentences - (typing_speed * (initial_typing_time + additional_typing_time + final_typing_time) - erased_sentences) = 258 := by
  sorry

end janice_typing_problem_l1576_157696


namespace book_price_range_l1576_157618

-- Define the price of the book
variable (x : ℝ)

-- Define the conditions based on the wrong guesses
def student_A_wrong : Prop := ¬(x ≥ 15)
def student_B_wrong : Prop := ¬(x ≤ 12)
def student_C_wrong : Prop := ¬(x ≤ 10)

-- Theorem statement
theorem book_price_range 
  (hA : student_A_wrong x)
  (hB : student_B_wrong x)
  (hC : student_C_wrong x) :
  12 < x ∧ x < 15 := by
  sorry

end book_price_range_l1576_157618


namespace range_of_a_l1576_157651

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x - 1| < 1 → x ≥ a) ∧ 
  (∃ y : ℝ, y ≥ a ∧ ¬(|y - 1| < 1)) → 
  a ≤ 0 := by
sorry

end range_of_a_l1576_157651


namespace measure_gold_dust_l1576_157611

/-- Represents the available weights for measuring gold dust -/
inductive Weight
  | TwoHundredGram
  | FiftyGram

/-- Represents a case with different available weights -/
inductive Case
  | CaseA
  | CaseB

/-- Represents a weighing operation on a balance scale -/
def Weighing := ℝ → ℝ → Prop

/-- Represents the ability to measure a specific amount of gold dust -/
def CanMeasure (totalGold : ℝ) (targetAmount : ℝ) (weights : List Weight) (case : Case) : Prop :=
  ∃ (w1 w2 w3 : Weighing), 
    (w1 totalGold targetAmount) ∧ 
    (w2 totalGold targetAmount) ∧ 
    (w3 totalGold targetAmount)

/-- The main theorem stating that it's possible to measure 2 kg of gold dust in both cases -/
theorem measure_gold_dust : 
  ∀ (case : Case),
    CanMeasure 9 2 
      (match case with
        | Case.CaseA => [Weight.TwoHundredGram, Weight.FiftyGram]
        | Case.CaseB => [Weight.TwoHundredGram])
      case :=
by
  sorry

end measure_gold_dust_l1576_157611


namespace intersection_of_M_and_N_l1576_157625

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end intersection_of_M_and_N_l1576_157625


namespace ellipse_standard_equation_l1576_157630

def ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

theorem ellipse_standard_equation 
  (major_axis_length : ℝ) 
  (eccentricity : ℝ) :
  major_axis_length = 12 →
  eccentricity = 2/3 →
  ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
    ∀ x y : ℝ, ellipse_equation a b x y ↔ 
      x^2 / 36 + y^2 / 20 = 1 :=
sorry

end ellipse_standard_equation_l1576_157630


namespace twelve_customers_in_line_l1576_157645

/-- The number of customers in a restaurant line -/
def customers_in_line (people_behind_front : ℕ) : ℕ :=
  people_behind_front + 1

/-- Theorem: Given 11 people behind the front person, there are 12 customers in line -/
theorem twelve_customers_in_line :
  customers_in_line 11 = 12 := by
  sorry

end twelve_customers_in_line_l1576_157645


namespace intersection_M_N_l1576_157634

def M : Set ℝ := {x | 0 < x ∧ x < 4}
def N : Set ℝ := {x | 1/3 ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x | 1/3 ≤ x ∧ x < 4} := by
  sorry

end intersection_M_N_l1576_157634


namespace exist_four_digit_square_sum_l1576_157693

/-- A four-digit number that is equal to the square of the sum of its first two digits and last two digits. -/
def IsFourDigitSquareSum (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000 ∧ 
  ∃ (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 0 ≤ b ∧ b < 100 ∧
    n = 100 * a + b ∧ n = (a + b)^2

/-- There exist at least three distinct four-digit numbers that are equal to the square of the sum of their first two digits and last two digits. -/
theorem exist_four_digit_square_sum : 
  ∃ (n₁ n₂ n₃ : ℕ), n₁ ≠ n₂ ∧ n₁ ≠ n₃ ∧ n₂ ≠ n₃ ∧ 
    IsFourDigitSquareSum n₁ ∧ IsFourDigitSquareSum n₂ ∧ IsFourDigitSquareSum n₃ := by
  sorry

end exist_four_digit_square_sum_l1576_157693


namespace root_sum_reciprocal_l1576_157643

theorem root_sum_reciprocal (p q r : ℂ) : 
  (p^3 - 2*p^2 + p - 1 = 0) → 
  (q^3 - 2*q^2 + q - 1 = 0) → 
  (r^3 - 2*r^2 + r - 1 = 0) → 
  (1 / (p + 2) + 1 / (q + 2) + 1 / (r + 2) = 20 / 19) := by
sorry

end root_sum_reciprocal_l1576_157643


namespace number_equation_l1576_157647

theorem number_equation (x : ℝ) : 150 - x = x + 68 ↔ x = 41 := by sorry

end number_equation_l1576_157647


namespace water_depth_relationship_l1576_157626

/-- Represents a cylindrical water tank -/
structure WaterTank where
  height : ℝ
  baseDiameter : ℝ
  horizontalWaterDepth : ℝ

/-- Calculates the water depth when the tank is vertical -/
def verticalWaterDepth (tank : WaterTank) : ℝ :=
  sorry

/-- Theorem stating the relationship between horizontal and vertical water depths -/
theorem water_depth_relationship (tank : WaterTank) 
  (h : tank.height = 20 ∧ tank.baseDiameter = 6 ∧ tank.horizontalWaterDepth = 2) :
  abs (verticalWaterDepth tank - 7.0) < 0.1 := by
  sorry

end water_depth_relationship_l1576_157626


namespace scientific_notation_75500000_l1576_157603

theorem scientific_notation_75500000 :
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ 75500000 = a * (10 : ℝ) ^ n ∧ a = 7.55 ∧ n = 7 := by
  sorry

end scientific_notation_75500000_l1576_157603


namespace parabola_focus_l1576_157627

-- Define the parabola equation
def parabola_equation (x y : ℝ) : Prop := x = -(1/4) * y^2

-- Define the focus of a parabola
def focus (p q : ℝ) : Prop := ∀ x y : ℝ, parabola_equation x y → (x + 1)^2 + y^2 = 1

-- Theorem statement
theorem parabola_focus : focus (-1) 0 := by sorry

end parabola_focus_l1576_157627


namespace triple_root_at_zero_l1576_157666

/-- The polynomial representing the difference between the two functions -/
def P (a b c d m n : ℝ) (x : ℝ) : ℝ :=
  x^7 - 9*x^6 + 27*x^5 + a*x^4 + b*x^3 + c*x^2 + d*x - m*x - n

/-- Theorem stating that the polynomial has a triple root at x = 0 -/
theorem triple_root_at_zero (a b c d m n : ℝ) : 
  ∃ (p q : ℝ), p ≠ q ∧ p ≠ 0 ∧ q ≠ 0 ∧
  ∀ (x : ℝ), P a b c d m n x = (x - p)^2 * (x - q)^2 * x^3 :=
sorry

end triple_root_at_zero_l1576_157666


namespace solve_sticker_problem_l1576_157635

def sticker_problem (initial : ℝ) (bought : ℝ) (birthday : ℝ) (mother : ℝ) (total : ℝ) : Prop :=
  let from_sister := total - (initial + bought + birthday + mother)
  from_sister = 6.0

theorem solve_sticker_problem :
  sticker_problem 20.0 26.0 20.0 58.0 130.0 := by
  sorry

end solve_sticker_problem_l1576_157635


namespace periodic_function_theorem_l1576_157649

/-- A function f: ℝ → ℝ is periodic if there exists a positive real number p such that
    for all x ∈ ℝ, f(x + p) = f(x) -/
def IsPeriodic (f : ℝ → ℝ) : Prop :=
  ∃ p : ℝ, p > 0 ∧ ∀ x : ℝ, f (x + p) = f x

/-- The main theorem: if f satisfies the given functional equation,
    then f is periodic with period 2a -/
theorem periodic_function_theorem (f : ℝ → ℝ) (a : ℝ) 
    (h : a > 0) 
    (eq : ∀ x : ℝ, f (x + a) = 1/2 + Real.sqrt (f x - f x ^ 2)) :
  IsPeriodic f ∧ ∃ p : ℝ, p = 2 * a ∧ ∀ x : ℝ, f (x + p) = f x :=
by sorry

end periodic_function_theorem_l1576_157649


namespace davids_chemistry_marks_l1576_157686

theorem davids_chemistry_marks 
  (english : ℕ) 
  (mathematics : ℕ) 
  (physics : ℕ) 
  (biology : ℕ) 
  (average : ℚ) 
  (num_subjects : ℕ) 
  (h1 : english = 45) 
  (h2 : mathematics = 35) 
  (h3 : physics = 52) 
  (h4 : biology = 55) 
  (h5 : average = 46.8) 
  (h6 : num_subjects = 5) :
  ∃ (chemistry : ℕ), 
    (english + mathematics + physics + biology + chemistry : ℚ) / num_subjects = average ∧ 
    chemistry = 47 := by
  sorry

end davids_chemistry_marks_l1576_157686


namespace ellipse_properties_l1576_157629

-- Define the ellipse C
def ellipse_C (x y a b c : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0 ∧ a = 2*c

-- Define the circle P1
def circle_P1 (x y r : ℝ) : Prop :=
  (x + 4*Real.sqrt 3 / 7)^2 + (y - 3*Real.sqrt 3 / 7)^2 = r^2 ∧ r > 0

-- Define the theorem
theorem ellipse_properties :
  ∀ (a b c : ℝ),
  ellipse_C (Real.sqrt 3) ((Real.sqrt 3) / 2) a b c →
  ellipse_C (-a + 2*c) 0 a b c →
  (∃ (x y r : ℝ), circle_P1 x y r ∧ ellipse_C x y a b c) →
  (∃ (k : ℝ), k > 1 ∧
    (∀ (x y : ℝ), y = k*(x + 1) → 
      (∃ (p q : ℝ), ellipse_C p (k*(p + 1)) a b c ∧ 
                    ellipse_C q (k*(q + 1)) a b c ∧
                    9/4 < (1 + k^2) * (9 / (3 + 4*k^2)) ∧
                    (1 + k^2) * (9 / (3 + 4*k^2)) ≤ 12/5))) →
  c / a = 1/2 ∧ a = 2 ∧ b = Real.sqrt 3 :=
sorry


end ellipse_properties_l1576_157629


namespace zyx_syndrome_ratio_is_one_to_three_l1576_157614

/-- Represents the ratio of patients with ZYX syndrome to those without it -/
structure ZYXRatio where
  with_syndrome : ℕ
  without_syndrome : ℕ

/-- The clinic's patient information -/
structure ClinicInfo where
  total_patients : ℕ
  diagnosed_patients : ℕ

/-- Calculates the ZYX syndrome ratio given clinic information -/
def calculate_zyx_ratio (info : ClinicInfo) : ZYXRatio :=
  { with_syndrome := info.diagnosed_patients,
    without_syndrome := info.total_patients - info.diagnosed_patients }

/-- Simplifies a ZYX ratio by dividing both numbers by their GCD -/
def simplify_ratio (ratio : ZYXRatio) : ZYXRatio :=
  let gcd := Nat.gcd ratio.with_syndrome ratio.without_syndrome
  { with_syndrome := ratio.with_syndrome / gcd,
    without_syndrome := ratio.without_syndrome / gcd }

theorem zyx_syndrome_ratio_is_one_to_three :
  let clinic_info : ClinicInfo := { total_patients := 52, diagnosed_patients := 13 }
  let ratio := simplify_ratio (calculate_zyx_ratio clinic_info)
  ratio.with_syndrome = 1 ∧ ratio.without_syndrome = 3 := by sorry

end zyx_syndrome_ratio_is_one_to_three_l1576_157614


namespace room_length_is_correct_l1576_157617

/-- The length of a rectangular room -/
def room_length : ℝ := 5.5

/-- The width of the room -/
def room_width : ℝ := 3.75

/-- The cost of paving the floor -/
def paving_cost : ℝ := 12375

/-- The rate of paving per square meter -/
def paving_rate : ℝ := 600

/-- Theorem stating that the room length is correct given the conditions -/
theorem room_length_is_correct : 
  room_length * room_width * paving_rate = paving_cost := by sorry

end room_length_is_correct_l1576_157617


namespace larger_number_proof_l1576_157605

theorem larger_number_proof (x y : ℕ) (h1 : x > y) (h2 : x + y = 363) (h3 : x = 16 * y + 6) : x = 342 := by
  sorry

end larger_number_proof_l1576_157605


namespace sqrt_fifth_power_cubed_l1576_157607

theorem sqrt_fifth_power_cubed : (Real.sqrt ((Real.sqrt 5) ^ 4)) ^ 3 = 125 := by
  sorry

end sqrt_fifth_power_cubed_l1576_157607


namespace complex_fraction_equality_l1576_157669

theorem complex_fraction_equality : 
  (1 - Complex.I * Real.sqrt 3) / ((Real.sqrt 3 + Complex.I) ^ 2) = 
  -(1/4) - (Real.sqrt 3 / 4) * Complex.I := by sorry

end complex_fraction_equality_l1576_157669


namespace equation_solution_for_all_y_l1576_157623

theorem equation_solution_for_all_y :
  ∃! x : ℚ, ∀ y : ℚ, 10 * x * y - 15 * y + 4 * x - 6 = 0 :=
by
  -- The proof goes here
  sorry

end equation_solution_for_all_y_l1576_157623


namespace rebecca_earnings_l1576_157606

/-- Rebecca's hair salon earnings calculation -/
theorem rebecca_earnings (haircut_price perm_price dye_job_price dye_cost : ℕ)
  (num_haircuts num_perms num_dye_jobs tips : ℕ) :
  haircut_price = 30 →
  perm_price = 40 →
  dye_job_price = 60 →
  dye_cost = 10 →
  num_haircuts = 4 →
  num_perms = 1 →
  num_dye_jobs = 2 →
  tips = 50 →
  (haircut_price * num_haircuts + 
   perm_price * num_perms + 
   dye_job_price * num_dye_jobs + 
   tips - 
   dye_cost * num_dye_jobs) = 310 :=
by sorry

end rebecca_earnings_l1576_157606


namespace complex_square_simplification_l1576_157638

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem complex_square_simplification :
  (5 - 3 * i)^2 = 16 - 30 * i :=
by
  -- The proof would go here, but we're skipping it as per instructions
  sorry

end complex_square_simplification_l1576_157638


namespace marcias_final_hair_length_l1576_157619

/-- Calculates the final hair length after a series of cuts and growth periods --/
def finalHairLength (initialLength : ℝ) 
                    (firstCutPercentage : ℝ) 
                    (firstGrowthMonths : ℕ) 
                    (firstGrowthRate : ℝ) 
                    (secondCutPercentage : ℝ) 
                    (secondGrowthMonths : ℕ) 
                    (secondGrowthRate : ℝ) 
                    (finalCutLength : ℝ) : ℝ :=
  let afterFirstCut := initialLength * (1 - firstCutPercentage)
  let afterFirstGrowth := afterFirstCut + (firstGrowthMonths : ℝ) * firstGrowthRate
  let afterSecondCut := afterFirstGrowth * (1 - secondCutPercentage)
  let afterSecondGrowth := afterSecondCut + (secondGrowthMonths : ℝ) * secondGrowthRate
  afterSecondGrowth - finalCutLength

/-- Theorem stating that Marcia's final hair length is 22.04 inches --/
theorem marcias_final_hair_length : 
  finalHairLength 24 0.3 3 1.5 0.2 5 1.8 4 = 22.04 := by
  sorry

end marcias_final_hair_length_l1576_157619


namespace chord_of_ellipse_l1576_157624

-- Define the real numbers m, n, s, t
variable (m n s t : ℝ)

-- Define the conditions
def conditions (m n s t : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ s > 0 ∧ t > 0 ∧
  m + n = 3 ∧
  m / s + n / t = 1 ∧
  m < n ∧
  ∀ (s' t' : ℝ), s' > 0 → t' > 0 → m / s' + n / t' = 1 → s + t ≤ s' + t'

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 16 = 1

-- Define the chord equation
def chord_equation (x y : ℝ) : Prop :=
  2 * x + y - 4 = 0

-- Theorem statement
theorem chord_of_ellipse (m n s t : ℝ) :
  conditions m n s t →
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    (x₁ + x₂) / 2 = m ∧ (y₁ + y₂) / 2 = n ∧
    ∀ (x y : ℝ), x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2 → chord_equation x y) :=
by sorry

end chord_of_ellipse_l1576_157624


namespace notebook_duration_example_l1576_157664

/-- The number of days notebooks last given the number of notebooks, pages per notebook, and pages used per day. -/
def notebook_duration (num_notebooks : ℕ) (pages_per_notebook : ℕ) (pages_per_day : ℕ) : ℕ :=
  (num_notebooks * pages_per_notebook) / pages_per_day

/-- Theorem stating that 5 notebooks with 40 pages each, using 4 pages per day, last for 50 days. -/
theorem notebook_duration_example : notebook_duration 5 40 4 = 50 := by
  sorry

end notebook_duration_example_l1576_157664


namespace intersection_implies_a_value_l1576_157698

def A (a : ℝ) : Set ℝ := {-1, a^2 + 1, a^2 - 3}
def B (a : ℝ) : Set ℝ := {a - 3, a - 1, a + 1}

theorem intersection_implies_a_value (a : ℝ) :
  A a ∩ B a = {-2} → a = -1 := by sorry

end intersection_implies_a_value_l1576_157698


namespace square_of_binomial_l1576_157639

theorem square_of_binomial (m : ℝ) : 
  (∃ c : ℝ, ∀ x : ℝ, x^2 - 12*x + m = (x + c)^2) → m = 36 := by
  sorry

end square_of_binomial_l1576_157639


namespace line_equation_of_l_l1576_157610

/-- A line passing through a point with a given slope -/
structure Line where
  point : ℝ × ℝ
  slope : ℝ

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The line l passing through (3,2) with slope -4 -/
def l : Line := { point := (3, 2), slope := -4 }

/-- Theorem: The equation of line l is 4x + y - 14 = 0 -/
theorem line_equation_of_l : 
  ∃ (eq : LineEquation), eq.a = 4 ∧ eq.b = 1 ∧ eq.c = -14 ∧
  ∀ (x y : ℝ), eq.a * x + eq.b * y + eq.c = 0 ↔ y - l.point.2 = l.slope * (x - l.point.1) :=
sorry

end line_equation_of_l_l1576_157610


namespace inverse_proportion_wrench_force_l1576_157663

/-- Proof that for inversely proportional quantities, if F₁ * L₁ = k and F₂ * L₂ = k,
    where F₁ = 300, L₁ = 12, and L₂ = 18, then F₂ = 200. -/
theorem inverse_proportion_wrench_force (k : ℝ) (F₁ F₂ L₁ L₂ : ℝ) 
    (h1 : F₁ * L₁ = k)
    (h2 : F₂ * L₂ = k)
    (h3 : F₁ = 300)
    (h4 : L₁ = 12)
    (h5 : L₂ = 18) :
    F₂ = 200 := by
  sorry

#check inverse_proportion_wrench_force

end inverse_proportion_wrench_force_l1576_157663


namespace congruence_solution_l1576_157687

theorem congruence_solution (x : ℤ) : 
  (10 * x + 3) % 18 = 7 % 18 → 
  ∃ (a m : ℕ), 
    0 < m ∧ 
    0 < a ∧ 
    a < m ∧
    x % m = a % m ∧
    a = 4 ∧ 
    m = 9 ∧
    a + m = 13 := by
  sorry

#check congruence_solution

end congruence_solution_l1576_157687


namespace new_person_weight_l1576_157628

/-- Given a group of 8 people, if replacing one person weighing 65 kg with a new person
    increases the average weight by 3.5 kg, then the weight of the new person is 93 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 3.5 →
  replaced_weight = 65 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 93 :=
by sorry

end new_person_weight_l1576_157628


namespace black_stones_count_l1576_157677

theorem black_stones_count (total : Nat) (white : Nat) : 
  total = 48 → 
  (4 * white) % 37 = 26 → 
  (4 * white) / 37 = 2 → 
  total - white = 23 := by
sorry

end black_stones_count_l1576_157677


namespace f_11_values_l1576_157689

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

axiom coprime_property {f : ℕ → ℕ} {a b : ℕ} (h : is_coprime a b) : 
  f (a * b) = f a * f b

axiom prime_property {f : ℕ → ℕ} {m k : ℕ} (hm : is_prime m) (hk : is_prime k) : 
  f (m + k - 3) = f m + f k - f 3

theorem f_11_values (f : ℕ → ℕ) 
  (h1 : ∀ a b : ℕ, is_coprime a b → f (a * b) = f a * f b)
  (h2 : ∀ m k : ℕ, is_prime m → is_prime k → f (m + k - 3) = f m + f k - f 3) :
  f 11 = 1 ∨ f 11 = 11 :=
sorry

end f_11_values_l1576_157689


namespace collinear_vectors_l1576_157601

/-- Given two vectors a and b in R², prove that if 2a + b is collinear with b,
    then the y-coordinate of a is 1/2. -/
theorem collinear_vectors (l x : ℝ) : 
  let a : ℝ × ℝ := (l, x)
  let b : ℝ × ℝ := (4, 2)
  (∃ (k : ℝ), (2 * a.1 + b.1, 2 * a.2 + b.2) = k • b) → x = 1/2 := by
  sorry

end collinear_vectors_l1576_157601


namespace square_table_correctness_l1576_157662

/-- Converts a base 60 number to base 10 -/
def base60ToBase10 (x : List Nat) : Nat :=
  x.enum.foldl (fun acc (i, digit) => acc + digit * (60 ^ i)) 0

/-- Converts a base 10 number to base 60 -/
def base10ToBase60 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 60) ((m % 60) :: acc)
    aux n []

/-- Checks if a number is a perfect square -/
def isPerfectSquare (n : Nat) : Prop :=
  ∃ m : Nat, m * m = n

/-- Represents the table of squares in base 60 -/
def squareTable : Nat → List Nat := sorry

theorem square_table_correctness :
  ∀ n : Nat, 1 ≤ n ∧ n ≤ 60 →
    base60ToBase10 (squareTable n) = n * n ∧
    isPerfectSquare (base60ToBase10 (squareTable n)) := by
  sorry

end square_table_correctness_l1576_157662


namespace increasing_sequence_with_properties_l1576_157615

theorem increasing_sequence_with_properties :
  ∃ (a : ℕ → ℕ) (C : ℝ), 
    (∀ n, a n < a (n + 1)) ∧ 
    (∀ m : ℕ+, ∃! (i j : ℕ), m = a j - a i) ∧
    (∀ k : ℕ+, (a k : ℝ) ≤ C * (k : ℝ)^3) :=
sorry

end increasing_sequence_with_properties_l1576_157615


namespace gumdrop_cost_l1576_157637

/-- Given 80 cents to buy 20 gumdrops, prove that each gumdrop costs 4 cents. -/
theorem gumdrop_cost (total_money : ℕ) (num_gumdrops : ℕ) (cost_per_gumdrop : ℕ) :
  total_money = 80 ∧ num_gumdrops = 20 ∧ total_money = num_gumdrops * cost_per_gumdrop →
  cost_per_gumdrop = 4 := by
sorry

end gumdrop_cost_l1576_157637


namespace exponent_division_l1576_157690

theorem exponent_division (a : ℝ) : a^8 / a^2 = a^6 :=
by sorry

end exponent_division_l1576_157690


namespace det_dilation_matrix_5_l1576_157691

/-- A dilation matrix with scale factor k -/
def dilationMatrix (k : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  Matrix.diagonal (λ _ => k)

/-- Theorem: The determinant of a 3x3 dilation matrix with scale factor 5 is 125 -/
theorem det_dilation_matrix_5 :
  Matrix.det (dilationMatrix 5) = 125 := by
  sorry

end det_dilation_matrix_5_l1576_157691


namespace total_dress_designs_l1576_157676

/-- The number of fabric colors available. -/
def num_colors : ℕ := 5

/-- The number of patterns available. -/
def num_patterns : ℕ := 4

/-- The number of sleeve designs available. -/
def num_sleeve_designs : ℕ := 3

/-- Each dress design requires exactly one color, one pattern, and one sleeve design. -/
theorem total_dress_designs :
  num_colors * num_patterns * num_sleeve_designs = 60 := by
  sorry

end total_dress_designs_l1576_157676


namespace expected_value_8_sided_die_l1576_157675

def standard_8_sided_die : Finset ℕ := Finset.range 8

theorem expected_value_8_sided_die :
  let outcomes := standard_8_sided_die
  let probability (n : ℕ) := (1 : ℚ) / 8
  let expected_value := (outcomes.sum (λ n => (n + 1 : ℚ) * probability n)) / outcomes.card
  expected_value = 9/2 := by sorry

end expected_value_8_sided_die_l1576_157675


namespace range_of_a_l1576_157613

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |4*x - 3| ≤ 1 → x^2 - (2*a + 1)*x + a*(a + 1) ≤ 0) ∧ 
  (∃ x : ℝ, |4*x - 3| ≤ 1 ∧ x^2 - (2*a + 1)*x + a*(a + 1) > 0) → 
  0 ≤ a ∧ a ≤ 1/2 :=
by sorry

end range_of_a_l1576_157613


namespace toy_car_production_l1576_157640

theorem toy_car_production (yesterday : ℕ) (today : ℕ) : 
  yesterday = 60 → today = 2 * yesterday → yesterday + today = 180 := by
  sorry

end toy_car_production_l1576_157640


namespace jerry_collection_cost_l1576_157621

/-- The amount of money Jerry needs to finish his action figure collection -/
def money_needed (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem: Jerry needs $72 to finish his collection -/
theorem jerry_collection_cost : money_needed 7 16 8 = 72 := by
  sorry

end jerry_collection_cost_l1576_157621


namespace quadratic_equation_solution_l1576_157678

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x + 1 - 9
  ∃ x₁ x₂ : ℝ, x₁ = 4 ∧ x₂ = -2 ∧ f x₁ = 0 ∧ f x₂ = 0 ∧
  ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by
  sorry

end quadratic_equation_solution_l1576_157678


namespace complex_magnitude_sum_reciprocals_l1576_157608

theorem complex_magnitude_sum_reciprocals (z w : ℂ) 
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hzw : Complex.abs (z + w) = 3) :
  Complex.abs (1 / z + 1 / w) = 3 / 8 := by
  sorry

end complex_magnitude_sum_reciprocals_l1576_157608


namespace fraction_evaluation_l1576_157688

theorem fraction_evaluation :
  let x : ℚ := 2/3
  let y : ℚ := 8/10
  (6*x + 10*y) / (60*x*y) = 3/8 := by
sorry

end fraction_evaluation_l1576_157688


namespace every_real_has_cube_root_l1576_157679

theorem every_real_has_cube_root : 
  ∀ y : ℝ, ∃ x : ℝ, x^3 = y := by sorry

end every_real_has_cube_root_l1576_157679


namespace graduation_messages_l1576_157674

theorem graduation_messages (n : ℕ) (h : n = 45) : 
  (n * (n - 1)) / 2 = 990 := by
  sorry

end graduation_messages_l1576_157674


namespace average_speed_theorem_l1576_157685

/-- Proves that the average speed of a trip is 40 mph given specific conditions -/
theorem average_speed_theorem (total_distance : ℝ) (first_half_speed : ℝ) (second_half_time_factor : ℝ) :
  total_distance = 640 →
  first_half_speed = 80 →
  second_half_time_factor = 3 →
  (total_distance / (total_distance / (2 * first_half_speed) * (1 + second_half_time_factor))) = 40 := by
  sorry

#check average_speed_theorem

end average_speed_theorem_l1576_157685


namespace solution_correctness_l1576_157694

variables {α : Type*} [Field α]
variables (x y z x' y' z' x'' y'' z'' u' v' w' : α)

def system_solution (u v w : α) : Prop :=
  x * u + y * v + z * w = u' ∧
  x' * u + y' * v + z' * w = v' ∧
  x'' * u + y'' * v + z'' * w = w'

theorem solution_correctness :
  ∃ (u v w : α),
    system_solution x y z x' y' z' x'' y'' z'' u' v' w' u v w ∧
    u = u' * x + v' * x' + w' * x'' ∧
    v = u' * y + v' * y' + w' * y'' ∧
    w = u' * z + v' * z' + w' * z'' :=
  sorry

end solution_correctness_l1576_157694


namespace coin_game_probabilities_l1576_157668

-- Define the coin probabilities
def p_heads : ℚ := 3/4
def p_tails : ℚ := 1/4

-- Define the games
def game_A : ℕ := 3  -- number of tosses in Game A
def game_C : ℕ := 4  -- number of tosses in Game C

-- Define the winning probability functions
def win_prob (n : ℕ) : ℚ := p_heads^n + p_tails^n

-- Theorem statement
theorem coin_game_probabilities :
  (win_prob game_A = 7/16) ∧ (win_prob game_C = 41/128) :=
sorry

end coin_game_probabilities_l1576_157668


namespace average_of_pqrs_l1576_157633

theorem average_of_pqrs (p q r s : ℝ) (h : (8 / 5) * (p + q + r + s) = 20) :
  (p + q + r + s) / 4 = 3.125 := by
sorry

end average_of_pqrs_l1576_157633


namespace max_min_difference_d_l1576_157673

theorem max_min_difference_d (a b c d : ℝ) 
  (sum_condition : a + b + c + d = 3)
  (sum_squares_condition : a^2 + b^2 + c^2 + d^2 = 20) :
  ∃ (d_min d_max : ℝ), 
    (∀ d', (∃ a' b' c', a' + b' + c' + d' = 3 ∧ a'^2 + b'^2 + c'^2 + d'^2 = 20) → d_min ≤ d' ∧ d' ≤ d_max) ∧
    d_max - d_min = 10 :=
by sorry

end max_min_difference_d_l1576_157673


namespace book_sale_result_l1576_157650

theorem book_sale_result (selling_price : ℝ) (profit_percent : ℝ) (loss_percent : ℝ) :
  selling_price = 4.5 ∧ 
  profit_percent = 25 ∧ 
  loss_percent = 25 →
  (selling_price * 2) - (selling_price / (1 + profit_percent / 100) + selling_price / (1 - loss_percent / 100)) = -0.6 := by
  sorry

end book_sale_result_l1576_157650


namespace geometric_sequence_common_ratio_l1576_157609

/-- Given a geometric sequence with first term -2 and sum of first 3 terms -7/2,
    prove that the common ratio is either 1/2 or -3/2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℚ) 
  (h_geom : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)) 
  (h_a1 : a 1 = -2)
  (h_S3 : (a 0) + (a 1) + (a 2) = -7/2) :
  (a 1) / (a 0) = 1/2 ∨ (a 1) / (a 0) = -3/2 := by
sorry

end geometric_sequence_common_ratio_l1576_157609


namespace squares_to_rectangles_ratio_l1576_157665

/-- The number of rectangles on a 6x6 checkerboard -/
def num_rectangles : ℕ := 441

/-- The number of squares on a 6x6 checkerboard -/
def num_squares : ℕ := 91

/-- Theorem stating that the ratio of squares to rectangles on a 6x6 checkerboard is 13/63 -/
theorem squares_to_rectangles_ratio :
  (num_squares : ℚ) / (num_rectangles : ℚ) = 13 / 63 := by sorry

end squares_to_rectangles_ratio_l1576_157665


namespace green_balls_count_l1576_157699

theorem green_balls_count (total : ℕ) (white yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 →
  white = 10 →
  yellow = 10 →
  red = 47 →
  purple = 3 →
  prob_not_red_purple = 1/2 →
  ∃ green : ℕ, green = 30 ∧ total = white + yellow + red + purple + green :=
by sorry

end green_balls_count_l1576_157699


namespace lauren_jane_equation_l1576_157636

theorem lauren_jane_equation (x : ℝ) :
  (∀ x, |x - 4| = 3 ↔ x^2 + b*x + c = 0) →
  (b : ℝ) = -8 ∧ (c : ℝ) = 7 := by
  sorry

end lauren_jane_equation_l1576_157636


namespace oil_press_statement_is_false_l1576_157697

-- Define the oil press output function
def oil_press_output (num_presses : ℕ) (output : ℕ) : Prop :=
  num_presses > 0 ∧ output > 0 ∧ (num_presses * (output / num_presses) = output)

-- State the theorem
theorem oil_press_statement_is_false :
  oil_press_output 5 260 →
  ¬ (oil_press_output 20 7200) :=
by
  sorry

end oil_press_statement_is_false_l1576_157697


namespace inequality_proof_l1576_157602

theorem inequality_proof (a b c : ℝ) (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h4 : a * b + b * c + c * a = 1) :
  1 / (a + b) + 1 / (b + c) + 1 / (c + a) ≥ 5 / 2 := by
sorry

end inequality_proof_l1576_157602


namespace power_of_81_l1576_157680

theorem power_of_81 : (81 : ℝ) ^ (5/2) = 59049 := by
  sorry

end power_of_81_l1576_157680


namespace crow_votes_l1576_157671

/-- Represents the number of votes for each participant -/
structure Votes where
  rooster : ℕ
  crow : ℕ
  cuckoo : ℕ

/-- Represents Woodpecker's counts -/
structure WoodpeckerCounts where
  total : ℕ
  roosterAndCrow : ℕ
  crowAndCuckoo : ℕ
  cuckooAndRooster : ℕ

/-- The maximum error in Woodpecker's counts -/
def maxError : ℕ := 13

/-- Check if a number is within the error range of another number -/
def withinErrorRange (actual : ℕ) (counted : ℕ) : Prop :=
  (actual ≤ counted + maxError) ∧ (counted ≤ actual + maxError)

/-- The theorem to be proved -/
theorem crow_votes (v : Votes) (w : WoodpeckerCounts) 
  (h1 : withinErrorRange (v.rooster + v.crow + v.cuckoo) w.total)
  (h2 : withinErrorRange (v.rooster + v.crow) w.roosterAndCrow)
  (h3 : withinErrorRange (v.crow + v.cuckoo) w.crowAndCuckoo)
  (h4 : withinErrorRange (v.cuckoo + v.rooster) w.cuckooAndRooster)
  (h5 : w.total = 59)
  (h6 : w.roosterAndCrow = 15)
  (h7 : w.crowAndCuckoo = 18)
  (h8 : w.cuckooAndRooster = 20) :
  v.crow = 13 := by
  sorry

end crow_votes_l1576_157671


namespace parallel_lines_m_value_l1576_157612

/-- Given two lines l₁ and l₂ with equations 3x + 2y - 2 = 0 and (2m-1)x + my + 1 = 0 respectively,
    if l₁ is parallel to l₂, then m = 2. -/
theorem parallel_lines_m_value (m : ℝ) :
  (∀ x y : ℝ, 3 * x + 2 * y - 2 = 0 ↔ (2 * m - 1) * x + m * y + 1 = 0) →
  m = 2 := by
  sorry

end parallel_lines_m_value_l1576_157612


namespace plan_A_cost_per_text_l1576_157659

/-- The cost per text message for Plan A, in dollars -/
def cost_per_text_A : ℝ := 0.25

/-- The monthly fee for Plan A, in dollars -/
def monthly_fee_A : ℝ := 9

/-- The cost per text message for Plan B, in dollars -/
def cost_per_text_B : ℝ := 0.40

/-- The number of text messages at which both plans cost the same -/
def equal_cost_messages : ℕ := 60

theorem plan_A_cost_per_text :
  cost_per_text_A * equal_cost_messages + monthly_fee_A =
  cost_per_text_B * equal_cost_messages :=
by sorry

end plan_A_cost_per_text_l1576_157659


namespace solution_existence_l1576_157661

/-- The set of real solutions (x, y) satisfying both equations -/
def SolutionSet : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 9 = 0 ∧ p.1^2 - 2*p.2 + 6 = 0}

/-- Theorem stating that real solutions exist if and only if y = -5 or y = 3 -/
theorem solution_existence : 
  ∃ (x : ℝ), (x, y) ∈ SolutionSet ↔ y = -5 ∨ y = 3 :=
sorry

end solution_existence_l1576_157661


namespace f_triple_3_l1576_157600

def f (x : ℝ) : ℝ := 3 * x + 2

theorem f_triple_3 : f (f (f 3)) = 107 := by
  sorry

end f_triple_3_l1576_157600


namespace percentage_increase_l1576_157660

theorem percentage_increase (x : ℝ) (h1 : x = 14.4) (h2 : x > 12) :
  (x - 12) / 12 * 100 = 20 := by
  sorry

end percentage_increase_l1576_157660
