import Mathlib

namespace NUMINAMATH_CALUDE_dentist_bill_ratio_l109_10910

def cleaning_cost : ℕ := 70
def filling_cost : ℕ := 120
def extraction_cost : ℕ := 290

def total_bill : ℕ := cleaning_cost + 2 * filling_cost + extraction_cost

theorem dentist_bill_ratio :
  (total_bill : ℚ) / filling_cost = 5 := by sorry

end NUMINAMATH_CALUDE_dentist_bill_ratio_l109_10910


namespace NUMINAMATH_CALUDE_six_digit_number_divisibility_l109_10930

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

end NUMINAMATH_CALUDE_six_digit_number_divisibility_l109_10930


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l109_10967

/-- The function f(x) = x(1+ax)^2 --/
def f (a : ℝ) (x : ℝ) : ℝ := x * (1 + a * x)^2

/-- Proposition stating that a = 2/3 is sufficient but not necessary for f(3) = 27 --/
theorem sufficient_not_necessary (a : ℝ) : 
  (f a 3 = 27 ↔ a = 2/3) ↔ False :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l109_10967


namespace NUMINAMATH_CALUDE_max_red_beads_l109_10995

/-- Represents a string of beads with red, blue, and green colors. -/
structure BeadString where
  total_beads : ℕ
  red_beads : ℕ
  blue_beads : ℕ
  green_beads : ℕ
  sum_constraint : total_beads = red_beads + blue_beads + green_beads
  green_constraint : ∀ n : ℕ, n + 6 ≤ total_beads → ∃ i, n ≤ i ∧ i < n + 6 ∧ green_beads > 0
  blue_constraint : ∀ n : ℕ, n + 11 ≤ total_beads → ∃ i, n ≤ i ∧ i < n + 11 ∧ blue_beads > 0

/-- The maximum number of red beads in a string of 150 beads with given constraints is 112. -/
theorem max_red_beads :
  ∀ bs : BeadString, bs.total_beads = 150 → bs.red_beads ≤ 112 :=
by sorry

end NUMINAMATH_CALUDE_max_red_beads_l109_10995


namespace NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l109_10957

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Two lines are different -/
def Line.different (l1 l2 : Line) : Prop := sorry

/-- A line is in a plane -/
def Line.inPlane (l : Line) (p : Plane) : Prop := sorry

/-- A line is outside a plane -/
def Line.outsidePlane (l : Line) (p : Plane) : Prop := sorry

/-- A line is perpendicular to another line -/
def Line.perpendicular (l1 l2 : Line) : Prop := sorry

/-- A line is perpendicular to a plane -/
def Line.perpendicularToPlane (l : Line) (p : Plane) : Prop := sorry

theorem perpendicular_necessary_not_sufficient
  (α : Plane) (a b l : Line)
  (h1 : a.inPlane α)
  (h2 : b.inPlane α)
  (h3 : l.outsidePlane α)
  (h4 : a.different b) :
  (l.perpendicularToPlane α → (l.perpendicular a ∧ l.perpendicular b)) ∧
  ¬((l.perpendicular a ∧ l.perpendicular b) → l.perpendicularToPlane α) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_necessary_not_sufficient_l109_10957


namespace NUMINAMATH_CALUDE_mencius_view_contradicts_option_a_l109_10948

-- Define the philosophical views
def MenciusView := "Human nature is inherently good"
def OptionAView := "Human nature is evil"

-- Define the passage content
def PassageContent := "Discussion on choices between fish, bear's paws, life, and righteousness"

-- Define Mencius's philosophy
def MenciusPhilosophy := "Advocate for inherent goodness of human nature"

-- Theorem to prove
theorem mencius_view_contradicts_option_a :
  (PassageContent = "Discussion on choices between fish, bear's paws, life, and righteousness") →
  (MenciusPhilosophy = "Advocate for inherent goodness of human nature") →
  (MenciusView ≠ OptionAView) :=
by
  sorry


end NUMINAMATH_CALUDE_mencius_view_contradicts_option_a_l109_10948


namespace NUMINAMATH_CALUDE_female_democrats_count_l109_10975

theorem female_democrats_count (total_participants : ℕ) 
  (female_participants male_participants : ℕ) 
  (female_democrats male_democrats : ℕ) : 
  total_participants = 840 →
  female_participants + male_participants = total_participants →
  female_democrats = female_participants / 2 →
  male_democrats = male_participants / 4 →
  female_democrats + male_democrats = total_participants / 3 →
  female_democrats = 140 := by
  sorry

end NUMINAMATH_CALUDE_female_democrats_count_l109_10975


namespace NUMINAMATH_CALUDE_interest_rate_is_six_percent_l109_10954

/-- Calculates the interest rate given the principal, time, and interest amount -/
def calculate_interest_rate (principal : ℚ) (time : ℚ) (interest : ℚ) : ℚ :=
  (interest * 100) / (principal * time)

theorem interest_rate_is_six_percent 
  (principal : ℚ) 
  (time : ℚ) 
  (interest : ℚ) 
  (h1 : principal = 1050)
  (h2 : time = 6)
  (h3 : interest = principal - 672) :
  calculate_interest_rate principal time interest = 6 := by
  sorry

end NUMINAMATH_CALUDE_interest_rate_is_six_percent_l109_10954


namespace NUMINAMATH_CALUDE_symmetry_implies_m_equals_one_l109_10944

/-- Two points are symmetric about the origin if their coordinates are negations of each other -/
def symmetric_about_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

/-- The theorem stating that if P(2, -1) and Q(-2, m) are symmetric about the origin, then m = 1 -/
theorem symmetry_implies_m_equals_one :
  ∀ m : ℝ, symmetric_about_origin (2, -1) (-2, m) → m = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_symmetry_implies_m_equals_one_l109_10944


namespace NUMINAMATH_CALUDE_problem_equality_l109_10968

/-- The function g as defined in the problem -/
def g (n : ℤ) : ℚ := (1 / 4) * n * (n + 1) * (n + 3)

/-- Theorem stating the equality to be proved -/
theorem problem_equality (s : ℤ) : g s - g (s - 1) + s * (s + 1) = 2 * s^2 + 2 * s := by
  sorry

end NUMINAMATH_CALUDE_problem_equality_l109_10968


namespace NUMINAMATH_CALUDE_system_solution_l109_10934

theorem system_solution :
  ∃ (x y : ℚ), (3 * x - 4 * y = 1) ∧ (5 * x + 2 * y = 6) ∧ (x = 1) ∧ (y = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l109_10934


namespace NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l109_10970

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + x + 1

-- State the theorem
theorem unique_function_satisfying_conditions :
  (∀ x y : ℝ, f (x^2) = (f x)^2 - 2*x*(f x)) ∧
  (∀ x : ℝ, f (-x) = f (x - 1)) ∧
  (∀ x y : ℝ, 1 < x → x < y → f x < f y) ∧
  (∀ x : ℝ, 0 < f x) ∧
  (∀ g : ℝ → ℝ, 
    ((∀ x y : ℝ, g (x^2) = (g x)^2 - 2*x*(g x)) ∧
     (∀ x : ℝ, g (-x) = g (x - 1)) ∧
     (∀ x y : ℝ, 1 < x → x < y → g x < g y) ∧
     (∀ x : ℝ, 0 < g x)) →
    (∀ x : ℝ, g x = f x)) :=
by sorry


end NUMINAMATH_CALUDE_unique_function_satisfying_conditions_l109_10970


namespace NUMINAMATH_CALUDE_swim_team_ratio_l109_10997

theorem swim_team_ratio (total : ℕ) (girls : ℕ) (h1 : total = 96) (h2 : girls = 80) :
  (girls : ℚ) / (total - girls) = 5 / 1 := by
  sorry

end NUMINAMATH_CALUDE_swim_team_ratio_l109_10997


namespace NUMINAMATH_CALUDE_train_distance_problem_l109_10998

theorem train_distance_problem :
  let fast_train_time : ℝ := 5
  let slow_train_time : ℝ := fast_train_time * (1 + 1/5)
  let stop_time : ℝ := 2
  let additional_distance : ℝ := 40
  let distance : ℝ := 150
  let fast_train_speed : ℝ := distance / fast_train_time
  let slow_train_speed : ℝ := distance / slow_train_time
  let fast_train_distance : ℝ := fast_train_speed * stop_time
  let slow_train_distance : ℝ := slow_train_speed * stop_time
  let remaining_distance : ℝ := distance - (fast_train_distance + slow_train_distance)
  remaining_distance = additional_distance :=
by
  sorry

#check train_distance_problem

end NUMINAMATH_CALUDE_train_distance_problem_l109_10998


namespace NUMINAMATH_CALUDE_apple_price_difference_l109_10982

/-- Given the prices of Shimla apples (S), Red Delicious apples (R), and Fuji apples (F) in rupees,
    prove that the difference in price between Shimla and Fuji apples can be expressed as shown,
    given the condition from the problem. -/
theorem apple_price_difference (S R F : ℝ) 
  (h : 1.05 * (S + R) = R + 0.90 * F + 250) :
  S - F = (-0.15 * S - 0.05 * R) / 0.90 + 250 / 0.90 := by
  sorry

end NUMINAMATH_CALUDE_apple_price_difference_l109_10982


namespace NUMINAMATH_CALUDE_middle_term_is_36_l109_10973

/-- An arithmetic sequence with 7 terms -/
structure ArithmeticSequence :=
  (a : Fin 7 → ℝ)
  (is_arithmetic : ∀ i j k : Fin 7, i.val + 1 = j.val ∧ j.val + 1 = k.val →
    a j - a i = a k - a j)

/-- The theorem stating that the middle term of the arithmetic sequence is 36 -/
theorem middle_term_is_36 (seq : ArithmeticSequence)
  (h1 : seq.a 0 = 11)
  (h2 : seq.a 6 = 61) :
  seq.a 3 = 36 := by
  sorry

end NUMINAMATH_CALUDE_middle_term_is_36_l109_10973


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l109_10939

theorem quadratic_inequality_solution_set :
  {x : ℝ | 2 * x^2 - x - 3 > 0} = {x : ℝ | x > 3/2 ∨ x < -1} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l109_10939


namespace NUMINAMATH_CALUDE_sum_consecutive_products_l109_10953

/-- The sum of products of three consecutive integers from 19 to 2001 -/
def S : ℕ → ℕ
  | 0 => 0
  | n + 1 => (18 + n) * (19 + n) * (20 + n) + S n

/-- The main theorem stating the closed form of the sum -/
theorem sum_consecutive_products (n : ℕ) :
  S (1981) = 6 * (Nat.choose 2002 4 - Nat.choose 21 4) :=
by sorry

end NUMINAMATH_CALUDE_sum_consecutive_products_l109_10953


namespace NUMINAMATH_CALUDE_ball_drawing_theorem_l109_10978

/-- Represents the three bags of balls -/
inductive Bag
  | A
  | B
  | C

/-- The number of balls in each bag -/
def ballCount (bag : Bag) : Nat :=
  match bag with
  | Bag.A => 1
  | Bag.B => 2
  | Bag.C => 3

/-- The color of balls in each bag -/
def ballColor (bag : Bag) : String :=
  match bag with
  | Bag.A => "red"
  | Bag.B => "white"
  | Bag.C => "yellow"

/-- The number of ways to draw two balls of different colors -/
def differentColorDraws : Nat := sorry

/-- The number of ways to draw two balls of the same color -/
def sameColorDraws : Nat := sorry

theorem ball_drawing_theorem :
  differentColorDraws = 11 ∧ sameColorDraws = 4 := by sorry

end NUMINAMATH_CALUDE_ball_drawing_theorem_l109_10978


namespace NUMINAMATH_CALUDE_function_composition_equality_l109_10938

/-- Given two functions f and g, where f(x) = Ax^2 - 3B^3 and g(x) = Bx^2,
    if B ≠ 0 and f(g(2)) = 0, then A = 3B/16 -/
theorem function_composition_equality (A B : ℝ) (hB : B ≠ 0) :
  let f := fun x => A * x^2 - 3 * B^3
  let g := fun x => B * x^2
  f (g 2) = 0 → A = 3 * B / 16 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l109_10938


namespace NUMINAMATH_CALUDE_odd_polyhedron_sum_not_nine_l109_10977

/-- Represents a convex polyhedron with odd-sided faces and odd-valence vertices -/
structure OddPolyhedron where
  -- Number of edges
  e : ℕ
  -- Number of faces with i sides (i is odd)
  ℓ : ℕ → ℕ
  -- Number of vertices where i edges meet (i is odd)
  c : ℕ → ℕ
  -- Each face has an odd number of sides
  face_odd : ∀ i, ℓ i > 0 → Odd i
  -- Each vertex has an odd number of edges meeting at it
  vertex_odd : ∀ i, c i > 0 → Odd i
  -- Edge-face relation
  edge_face : 2 * e = ∑' i, i * ℓ i
  -- Edge-vertex relation
  edge_vertex : 2 * e = ∑' i, i * c i
  -- Euler's formula
  euler : e + 2 = (∑' i, ℓ i) + (∑' i, c i)

/-- The sum of triangular faces and vertices where three edges meet cannot be 9 -/
theorem odd_polyhedron_sum_not_nine (P : OddPolyhedron) : ¬(P.ℓ 3 + P.c 3 = 9) := by
  sorry

end NUMINAMATH_CALUDE_odd_polyhedron_sum_not_nine_l109_10977


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l109_10980

theorem cubic_roots_sum (m : ℤ) (a b c : ℤ) :
  (∀ x : ℤ, x^3 - 2015*x + m = 0 ↔ x = a ∨ x = b ∨ x = c) →
  |a| + |b| + |c| = 100 :=
by sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l109_10980


namespace NUMINAMATH_CALUDE_largest_divisible_by_9_after_erasure_l109_10960

def original_number : ℕ := 321321321321

def erase_digits (n : ℕ) (positions : List ℕ) : ℕ :=
  sorry

def is_divisible_by_9 (n : ℕ) : Prop :=
  n % 9 = 0

theorem largest_divisible_by_9_after_erasure :
  ∃ (positions : List ℕ),
    let result := erase_digits original_number positions
    is_divisible_by_9 result ∧
    ∀ (other_positions : List ℕ),
      let other_result := erase_digits original_number other_positions
      is_divisible_by_9 other_result →
      other_result ≤ result ∧
      result = 32132132121 :=
sorry

end NUMINAMATH_CALUDE_largest_divisible_by_9_after_erasure_l109_10960


namespace NUMINAMATH_CALUDE_melany_candy_l109_10922

theorem melany_candy (hugh tommy melany : ℕ) (total_after : ℕ) :
  hugh = 8 →
  tommy = 6 →
  total_after = 7 * 3 →
  hugh + tommy + melany = total_after →
  melany = 7 :=
by sorry

end NUMINAMATH_CALUDE_melany_candy_l109_10922


namespace NUMINAMATH_CALUDE_dentist_age_problem_l109_10999

/-- Proves that given a dentist's current age of 32 years, if one-sixth of his age 8 years ago
    equals one-tenth of his age at a certain time in the future, then that future time is 8 years from now. -/
theorem dentist_age_problem (future_years : ℕ) : 
  (1/6 : ℚ) * (32 - 8) = (1/10 : ℚ) * (32 + future_years) → future_years = 8 := by
  sorry

end NUMINAMATH_CALUDE_dentist_age_problem_l109_10999


namespace NUMINAMATH_CALUDE_isosceles_triangle_side_length_l109_10988

/-- An isosceles triangle with perimeter 16 and one side length 3 has its other side length equal to 6.5 -/
theorem isosceles_triangle_side_length : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a + b + c = 16 →
  (a = b ∧ c = 3) ∨ (a = c ∧ b = 3) ∨ (b = c ∧ a = 3) →
  (a = 6.5 ∧ b = 6.5) ∨ (a = 6.5 ∧ c = 6.5) ∨ (b = 6.5 ∧ c = 6.5) :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_side_length_l109_10988


namespace NUMINAMATH_CALUDE_savings_growth_l109_10951

/-- The amount of money in a savings account after n years -/
def savings_amount (a : ℝ) (n : ℕ) : ℝ :=
  a * (1 + 0.02) ^ n

/-- Theorem: The amount of money in a savings account after n years,
    given an initial deposit of a rubles and a 2% annual interest rate,
    is equal to a × 1.02^n rubles. -/
theorem savings_growth (a : ℝ) (n : ℕ) :
  savings_amount a n = a * 1.02 ^ n :=
by sorry

end NUMINAMATH_CALUDE_savings_growth_l109_10951


namespace NUMINAMATH_CALUDE_work_completion_time_l109_10901

theorem work_completion_time (a_time b_time b_remaining : ℚ) 
  (ha : a_time = 45)
  (hb : b_time = 40)
  (hc : b_remaining = 23) : 
  let x := (b_time * b_remaining * a_time - a_time * b_time) / (a_time * b_time + a_time * b_remaining)
  x = 9 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l109_10901


namespace NUMINAMATH_CALUDE_sons_age_l109_10993

theorem sons_age (son_age father_age : ℕ) : 
  father_age = son_age + 30 →
  father_age + 2 = 2 * (son_age + 2) →
  son_age = 28 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l109_10993


namespace NUMINAMATH_CALUDE_kho_kho_only_count_l109_10959

/-- The number of people who play kabadi -/
def kabadi_players : ℕ := 10

/-- The number of people who play both games -/
def both_players : ℕ := 5

/-- The total number of players -/
def total_players : ℕ := 30

/-- The number of people who play kho kho only -/
def kho_kho_only_players : ℕ := total_players - (kabadi_players + both_players)

theorem kho_kho_only_count : kho_kho_only_players = 20 := by
  sorry

end NUMINAMATH_CALUDE_kho_kho_only_count_l109_10959


namespace NUMINAMATH_CALUDE_sale_discount_proof_l109_10956

theorem sale_discount_proof (original_price : ℝ) : 
  let sale_price := 0.5 * original_price
  let coupon_discount := 0.2
  let final_price := (1 - coupon_discount) * sale_price
  final_price = 0.4 * original_price :=
by sorry

end NUMINAMATH_CALUDE_sale_discount_proof_l109_10956


namespace NUMINAMATH_CALUDE_min_value_of_function_l109_10990

theorem min_value_of_function (x : ℝ) (h : x > 1) :
  ∃ (y_min : ℝ), y_min = 4 * Real.sqrt 2 + 1 ∧
  ∀ (y : ℝ), y = 2 * x + 4 / (x - 1) - 1 → y ≥ y_min := by
sorry

end NUMINAMATH_CALUDE_min_value_of_function_l109_10990


namespace NUMINAMATH_CALUDE_pear_distribution_count_l109_10918

def family_size : ℕ := 7
def elder_count : ℕ := 4

theorem pear_distribution_count : 
  (elder_count : ℕ) * (Nat.factorial (family_size - 2)) = 480 :=
sorry

end NUMINAMATH_CALUDE_pear_distribution_count_l109_10918


namespace NUMINAMATH_CALUDE_cookie_recipe_total_cups_l109_10909

theorem cookie_recipe_total_cups (butter flour sugar : ℕ) (total : ℕ) : 
  (butter : ℚ) / flour = 2 / 5 →
  (sugar : ℚ) / flour = 3 / 5 →
  flour = 15 →
  total = butter + flour + sugar →
  total = 30 := by
sorry

end NUMINAMATH_CALUDE_cookie_recipe_total_cups_l109_10909


namespace NUMINAMATH_CALUDE_molecular_weight_calculation_l109_10963

/-- The molecular weight of AlCl3 in g/mol -/
def molecular_weight_AlCl3 : ℝ := 132

/-- The number of moles given in the problem -/
def given_moles : ℝ := 4

/-- The total weight of the given moles in grams -/
def total_weight : ℝ := 528

theorem molecular_weight_calculation :
  molecular_weight_AlCl3 * given_moles = total_weight :=
by sorry

end NUMINAMATH_CALUDE_molecular_weight_calculation_l109_10963


namespace NUMINAMATH_CALUDE_not_always_same_digit_sum_l109_10964

-- Define a function to calculate the sum of digits
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

-- State the theorem
theorem not_always_same_digit_sum :
  ∃ (N M : ℕ), 
    (sumOfDigits (N + M) = sumOfDigits N) ∧ 
    (∀ k : ℕ, k > 1 → sumOfDigits (N + k * M) ≠ sumOfDigits N) :=
sorry

end NUMINAMATH_CALUDE_not_always_same_digit_sum_l109_10964


namespace NUMINAMATH_CALUDE_range_of_m_for_function_equality_l109_10983

theorem range_of_m_for_function_equality (m : ℝ) : 
  (∀ x₁ ∈ (Set.Icc (-1 : ℝ) 2), ∃ x₀ ∈ (Set.Icc (-1 : ℝ) 2), 
    m * x₁ + 2 = x₀^2 - 2*x₀) → 
  m ∈ Set.Icc (-1 : ℝ) (1/2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_for_function_equality_l109_10983


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l109_10942

theorem inequality_system_solutions :
  let S : Set (ℝ × ℝ) := {(x, y) | 
    x^4 + 8*x^3*y + 16*x^2*y^2 + 16 ≤ 8*x^2 + 32*x*y ∧
    y^4 + 64*x^2*y^2 + 10*y^2 + 25 ≤ 16*x*y^3 + 80*x*y}
  S = {(2/Real.sqrt 11, 5/Real.sqrt 11), 
       (-2/Real.sqrt 11, -5/Real.sqrt 11),
       (2/Real.sqrt 3, 1/Real.sqrt 3), 
       (-2/Real.sqrt 3, -1/Real.sqrt 3)} := by
  sorry


end NUMINAMATH_CALUDE_inequality_system_solutions_l109_10942


namespace NUMINAMATH_CALUDE_constant_function_shifted_l109_10971

-- Define g as a function from real numbers to real numbers
def g : ℝ → ℝ := fun _ ↦ -3

-- Theorem statement
theorem constant_function_shifted (x : ℝ) : g (x - 5) = -3 := by
  sorry

end NUMINAMATH_CALUDE_constant_function_shifted_l109_10971


namespace NUMINAMATH_CALUDE_parallel_necessary_not_sufficient_l109_10961

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

def parallel (a b : V) : Prop := ∃ (k : ℝ), a = k • b ∨ b = k • a

theorem parallel_necessary_not_sufficient :
  (∀ a b : V, a = b → parallel a b) ∧
  (∃ a b : V, parallel a b ∧ a ≠ b) := by sorry

end NUMINAMATH_CALUDE_parallel_necessary_not_sufficient_l109_10961


namespace NUMINAMATH_CALUDE_expression_evaluation_l109_10958

theorem expression_evaluation (a b c : ℝ) 
  (ha : a = 3) (hb : b = 2) (hc : c = 1) : 
  (a^2 + b + c)^2 - (a^2 - b - c)^2 = 108 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l109_10958


namespace NUMINAMATH_CALUDE_sum_odd_500_to_800_l109_10920

def first_odd_after (n : ℕ) : ℕ :=
  if n % 2 = 0 then n + 1 else n + 2

def last_odd_before (n : ℕ) : ℕ :=
  if n % 2 = 0 then n - 1 else n - 2

def sum_odd_between (a b : ℕ) : ℕ :=
  let first := first_odd_after a
  let last := last_odd_before b
  let count := (last - first) / 2 + 1
  count * (first + last) / 2

theorem sum_odd_500_to_800 :
  sum_odd_between 500 800 = 97500 := by
  sorry

end NUMINAMATH_CALUDE_sum_odd_500_to_800_l109_10920


namespace NUMINAMATH_CALUDE_triangle_perimeter_l109_10927

noncomputable def line_through_origin (m : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1}

def vertical_line (x : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 = x}

def sloped_line (m : ℝ) (b : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2 = m * p.1 + b}

def is_equilateral_triangle (t : Set (ℝ × ℝ)) : Prop :=
  ∃ (a b c : ℝ × ℝ), a ∈ t ∧ b ∈ t ∧ c ∈ t ∧
    (a.1 - b.1)^2 + (a.2 - b.2)^2 = (b.1 - c.1)^2 + (b.2 - c.2)^2 ∧
    (b.1 - c.1)^2 + (b.2 - c.2)^2 = (c.1 - a.1)^2 + (c.2 - a.2)^2

def perimeter (t : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem triangle_perimeter :
  ∃ (m : ℝ),
    let l1 := line_through_origin m
    let l2 := vertical_line 1
    let l3 := sloped_line (Real.sqrt 3 / 3) 1
    let t := l1 ∪ l2 ∪ l3
    is_equilateral_triangle t ∧ perimeter t = 3 + 2 * Real.sqrt 3 :=
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l109_10927


namespace NUMINAMATH_CALUDE_scrap_rate_cost_increase_l109_10987

/-- The regression equation for cost of cast iron based on scrap rate -/
def cost_equation (x : ℝ) : ℝ := 56 + 8 * x

/-- Theorem stating the relationship between scrap rate increase and cost increase -/
theorem scrap_rate_cost_increase (x : ℝ) :
  cost_equation (x + 1) - cost_equation x = 8 := by
  sorry

end NUMINAMATH_CALUDE_scrap_rate_cost_increase_l109_10987


namespace NUMINAMATH_CALUDE_simplify_expressions_l109_10932

theorem simplify_expressions :
  (99^2 = 9801) ∧ (2000^2 - 1999 * 2001 = 1) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l109_10932


namespace NUMINAMATH_CALUDE_horse_track_distance_l109_10986

/-- The distance covered by a horse running one turn around a square-shaped track -/
def track_distance (side_length : ℝ) : ℝ := 4 * side_length

/-- Theorem: The distance covered by a horse running one turn around a square-shaped track
    with sides of length 40 meters is equal to 160 meters -/
theorem horse_track_distance :
  track_distance 40 = 160 := by
  sorry

end NUMINAMATH_CALUDE_horse_track_distance_l109_10986


namespace NUMINAMATH_CALUDE_smallest_shift_for_even_function_l109_10902

theorem smallest_shift_for_even_function (f g : ℝ → ℝ) (σ : ℝ) : 
  (∀ x, f x = Real.sin (2 * x + π / 3)) →
  (∀ x, g x = f (x + σ)) →
  (∀ x, g (-x) = g x) →
  σ > 0 →
  (∀ σ' > 0, (∀ x, f (x + σ') = f (-x + σ')) → σ' ≥ σ) →
  σ = π / 12 := by sorry

end NUMINAMATH_CALUDE_smallest_shift_for_even_function_l109_10902


namespace NUMINAMATH_CALUDE_deepak_age_l109_10996

/-- Given the ratio of Arun's age to Deepak's age and Arun's future age, 
    prove Deepak's current age -/
theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 4 / 3 →
  arun_age + 6 = 26 →
  deepak_age = 15 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l109_10996


namespace NUMINAMATH_CALUDE_correct_operation_l109_10937

theorem correct_operation (x y : ℝ) : 2 * x^2 * (3 * x^2 - 5 * y) = 6 * x^4 - 10 * x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l109_10937


namespace NUMINAMATH_CALUDE_sum_of_polynomials_l109_10903

-- Define the polynomials
def f (x : ℝ) : ℝ := -6 * x^3 - 4 * x^2 + 2 * x - 5
def g (x : ℝ) : ℝ := -7 * x^2 + 6 * x - 9
def h (x : ℝ) : ℝ := 6 * x^2 + 7 * x + 3

-- State the theorem
theorem sum_of_polynomials (x : ℝ) : 
  f x + g x + h x = -6 * x^3 - 5 * x^2 + 15 * x - 11 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_polynomials_l109_10903


namespace NUMINAMATH_CALUDE_ending_number_of_range_l109_10919

theorem ending_number_of_range : ∃ n : ℕ, 
  (n ≥ 100) ∧ 
  ((200 + 400) / 2 = ((100 + n) / 2) + 150) ∧ 
  (n = 200) := by
  sorry

end NUMINAMATH_CALUDE_ending_number_of_range_l109_10919


namespace NUMINAMATH_CALUDE_binomial_sum_inequality_l109_10947

theorem binomial_sum_inequality (x : ℝ) (n : ℕ) (h1 : |x| < 1) (h2 : n ≥ 2) :
  (1 - x)^n + (1 + x)^n < 2^n := by
  sorry

end NUMINAMATH_CALUDE_binomial_sum_inequality_l109_10947


namespace NUMINAMATH_CALUDE_solution_set_inequality_l109_10923

theorem solution_set_inequality (a b : ℝ) :
  ({x : ℝ | a * x^2 - 5 * x + b > 0} = {x : ℝ | -3 < x ∧ x < 2}) →
  ({x : ℝ | b * x^2 - 5 * x + a > 0} = {x : ℝ | x < -3 ∨ x > 2}) :=
by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l109_10923


namespace NUMINAMATH_CALUDE_rain_probability_l109_10914

theorem rain_probability (p_friday p_monday : ℝ) 
  (h1 : p_friday = 0.3)
  (h2 : p_monday = 0.6)
  (h3 : 0 ≤ p_friday ∧ p_friday ≤ 1)
  (h4 : 0 ≤ p_monday ∧ p_monday ≤ 1) :
  1 - (1 - p_friday) * (1 - p_monday) = 0.72 := by
  sorry

end NUMINAMATH_CALUDE_rain_probability_l109_10914


namespace NUMINAMATH_CALUDE_cab_journey_time_l109_10908

/-- Given a cab walking at 5/6 of its usual speed and arriving 15 minutes late,
    prove that its usual time to cover the journey is 1.25 hours. -/
theorem cab_journey_time (usual_speed : ℝ) (usual_time : ℝ) 
  (h1 : usual_speed > 0) (h2 : usual_time > 0) : 
  (usual_speed * usual_time = (5/6 * usual_speed) * (usual_time + 1/4)) → 
  usual_time = 5/4 := by
  sorry

end NUMINAMATH_CALUDE_cab_journey_time_l109_10908


namespace NUMINAMATH_CALUDE_alfred_christmas_shopping_goal_l109_10974

def christmas_shopping_goal (initial_amount : ℕ) (monthly_savings : ℕ) (months : ℕ) : ℕ :=
  initial_amount + monthly_savings * months

theorem alfred_christmas_shopping_goal :
  christmas_shopping_goal 100 75 12 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_alfred_christmas_shopping_goal_l109_10974


namespace NUMINAMATH_CALUDE_stratified_sampling_medium_stores_stratified_sampling_medium_stores_correct_l109_10941

theorem stratified_sampling_medium_stores 
  (total_stores medium_stores sample_size : ℕ) 
  (h1 : total_stores > 0) 
  (h2 : medium_stores ≤ total_stores) 
  (h3 : sample_size ≤ total_stores) : 
  ℕ :=
  let medium_stores_to_draw := (medium_stores * sample_size) / total_stores
  medium_stores_to_draw

#check stratified_sampling_medium_stores

theorem stratified_sampling_medium_stores_correct 
  (total_stores medium_stores sample_size : ℕ) 
  (h1 : total_stores > 0) 
  (h2 : medium_stores ≤ total_stores) 
  (h3 : sample_size ≤ total_stores) : 
  stratified_sampling_medium_stores total_stores medium_stores sample_size h1 h2 h3 = 
  (medium_stores * sample_size) / total_stores :=
by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_medium_stores_stratified_sampling_medium_stores_correct_l109_10941


namespace NUMINAMATH_CALUDE_games_lost_l109_10921

theorem games_lost (total_games won_games : ℕ) 
  (h1 : total_games = 18) 
  (h2 : won_games = 15) : 
  total_games - won_games = 3 := by
sorry

end NUMINAMATH_CALUDE_games_lost_l109_10921


namespace NUMINAMATH_CALUDE_ten_object_rotation_l109_10933

/-- Represents a circular arrangement of n objects -/
def CircularArrangement (n : ℕ) := Fin n

/-- The operation of switching two objects in the arrangement -/
def switch (arr : CircularArrangement n) (i j : Fin n) : CircularArrangement n :=
  sorry

/-- Checks if the arrangement is rotated one position clockwise -/
def isRotatedOneStep (original rotated : CircularArrangement n) : Prop :=
  sorry

/-- The minimum number of switches required to rotate the arrangement one step -/
def minSwitches (n : ℕ) : ℕ :=
  sorry

theorem ten_object_rotation (arr : CircularArrangement 10) :
  ∃ (switches : List (Fin 10 × Fin 10)),
    switches.length = 9 ∧
    isRotatedOneStep arr (switches.foldl (λ a (i, j) => switch a i j) arr) :=
  sorry

end NUMINAMATH_CALUDE_ten_object_rotation_l109_10933


namespace NUMINAMATH_CALUDE_unique_distribution_function_decomposition_l109_10913

/-- A distribution function -/
class DistributionFunction (F : ℝ → ℝ) : Prop where
  -- Add necessary axioms for a distribution function

/-- A discrete distribution function -/
class DiscreteDistributionFunction (F : ℝ → ℝ) extends DistributionFunction F : Prop where
  -- Add necessary axioms for a discrete distribution function

/-- An absolutely continuous distribution function -/
class AbsContinuousDistributionFunction (F : ℝ → ℝ) extends DistributionFunction F : Prop where
  -- Add necessary axioms for an absolutely continuous distribution function

/-- A singular distribution function -/
class SingularDistributionFunction (F : ℝ → ℝ) extends DistributionFunction F : Prop where
  -- Add necessary axioms for a singular distribution function

/-- The uniqueness of distribution function decomposition -/
theorem unique_distribution_function_decomposition
  (F : ℝ → ℝ) [DistributionFunction F] :
  ∃! (α₁ α₂ α₃ : ℝ) (Fₐ Fₐbc Fsc : ℝ → ℝ),
    α₁ ≥ 0 ∧ α₂ ≥ 0 ∧ α₃ ≥ 0 ∧
    α₁ + α₂ + α₃ = 1 ∧
    DiscreteDistributionFunction Fₐ ∧
    AbsContinuousDistributionFunction Fₐbc ∧
    SingularDistributionFunction Fsc ∧
    F = λ x => α₁ * Fₐ x + α₂ * Fₐbc x + α₃ * Fsc x :=
by sorry

end NUMINAMATH_CALUDE_unique_distribution_function_decomposition_l109_10913


namespace NUMINAMATH_CALUDE_collinear_points_k_value_l109_10965

/-- Three points are collinear if the slope between any two pairs of points is equal -/
def collinear (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ) : Prop :=
  (y₂ - y₁) * (x₃ - x₂) = (y₃ - y₂) * (x₂ - x₁)

/-- The theorem states that if the points (5, 10), (-3, k), and (-11, 6) are collinear, then k = 8 -/
theorem collinear_points_k_value :
  collinear 5 10 (-3) k (-11) 6 → k = 8 := by
  sorry


end NUMINAMATH_CALUDE_collinear_points_k_value_l109_10965


namespace NUMINAMATH_CALUDE_gcd_problem_l109_10979

theorem gcd_problem :
  ∃! n : ℕ, 80 ≤ n ∧ n ≤ 100 ∧ Nat.gcd n 27 = 9 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l109_10979


namespace NUMINAMATH_CALUDE_vector_dot_product_l109_10972

theorem vector_dot_product (a b : ℝ × ℝ) :
  a + b = (1, -3) ∧ a - b = (3, 7) → a • b = -12 := by sorry

end NUMINAMATH_CALUDE_vector_dot_product_l109_10972


namespace NUMINAMATH_CALUDE_banks_revenue_is_500_l109_10984

/-- Represents the revenue structure for Mr. Banks and Ms. Elizabeth -/
structure RevenueStructure where
  banks_investments : ℕ
  elizabeth_investments : ℕ
  elizabeth_revenue_per_investment : ℕ
  elizabeth_total_revenue_difference : ℕ

/-- Calculates Mr. Banks' revenue per investment given the revenue structure -/
def banks_revenue_per_investment (rs : RevenueStructure) : ℕ :=
  ((rs.elizabeth_investments * rs.elizabeth_revenue_per_investment) - rs.elizabeth_total_revenue_difference) / rs.banks_investments

/-- Theorem stating that Mr. Banks' revenue per investment is $500 given the specific conditions -/
theorem banks_revenue_is_500 (rs : RevenueStructure) 
  (h1 : rs.banks_investments = 8)
  (h2 : rs.elizabeth_investments = 5)
  (h3 : rs.elizabeth_revenue_per_investment = 900)
  (h4 : rs.elizabeth_total_revenue_difference = 500) :
  banks_revenue_per_investment rs = 500 := by
  sorry


end NUMINAMATH_CALUDE_banks_revenue_is_500_l109_10984


namespace NUMINAMATH_CALUDE_midsize_to_fullsize_ratio_l109_10945

/-- Proves that the ratio of the mid-size model's length to the full-size mustang's length is 1:10 -/
theorem midsize_to_fullsize_ratio :
  let full_size : ℝ := 240
  let smallest_size : ℝ := 12
  let mid_size : ℝ := 2 * smallest_size
  (mid_size / full_size) = (1 / 10 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_midsize_to_fullsize_ratio_l109_10945


namespace NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l109_10940

/-- The polynomial z^6 - z^3 + 1 -/
def f (z : ℂ) : ℂ := z^6 - z^3 + 1

/-- The set of roots of f(z) -/
def roots_of_f : Set ℂ := {z : ℂ | f z = 0}

/-- n-th roots of unity -/
def nth_roots_of_unity (n : ℕ) : Set ℂ := {z : ℂ | z^n = 1}

/-- Theorem: 9 is the smallest positive integer n such that all roots of z^6 - z^3 + 1 = 0 are n-th roots of unity -/
theorem smallest_n_for_roots_of_unity :
  ∃ (n : ℕ), n > 0 ∧ roots_of_f ⊆ nth_roots_of_unity n ∧
  ∀ (m : ℕ), m > 0 ∧ m < n → ¬(roots_of_f ⊆ nth_roots_of_unity m) ∧
  n = 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_roots_of_unity_l109_10940


namespace NUMINAMATH_CALUDE_sphere_surface_area_l109_10928

theorem sphere_surface_area (r : ℝ) (h : r = 4) : 
  4 * π * r^2 = 64 * π := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l109_10928


namespace NUMINAMATH_CALUDE_count_special_numbers_l109_10907

/-- Counts the number of four-digit numbers with digit sum 12 that are divisible by 9 -/
def countSpecialNumbers : ℕ :=
  (Finset.range 9).sum fun a =>
    Nat.choose (14 - (a + 1)) 2

/-- The count of four-digit numbers with digit sum 12 that are divisible by 9 is 354 -/
theorem count_special_numbers : countSpecialNumbers = 354 := by
  sorry

end NUMINAMATH_CALUDE_count_special_numbers_l109_10907


namespace NUMINAMATH_CALUDE_power_of_three_mod_five_l109_10936

theorem power_of_three_mod_five : 3^2040 ≡ 1 [ZMOD 5] := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_five_l109_10936


namespace NUMINAMATH_CALUDE_geometric_progression_ratio_l109_10952

theorem geometric_progression_ratio (x y z r : ℂ) : 
  x ≠ 0 ∧ y ≠ 0 ∧ z ≠ 0 ∧ x ≠ y ∧ y ≠ z ∧ x ≠ z →
  ∃ (a : ℂ), x * (y - z) = a ∧ y * (z - x) = a * r ∧ z * (x - y) = a * r^2 →
  x + y + z = 0 →
  r^2 + r + 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_geometric_progression_ratio_l109_10952


namespace NUMINAMATH_CALUDE_tadpoles_kept_l109_10916

theorem tadpoles_kept (total : ℕ) (released_percent : ℚ) (kept : ℕ) : 
  total = 180 → 
  released_percent = 75 / 100 → 
  kept = total - (total * released_percent).floor → 
  kept = 45 := by
sorry

end NUMINAMATH_CALUDE_tadpoles_kept_l109_10916


namespace NUMINAMATH_CALUDE_factorial_expression_equals_100_l109_10950

-- Define factorial
def factorial (n : ℕ) : ℕ := 
  if n = 0 then 1 else n * factorial (n - 1)

-- Theorem statement
theorem factorial_expression_equals_100 : 
  (factorial 11 - factorial 10) / factorial 9 = 100 := by
  sorry

end NUMINAMATH_CALUDE_factorial_expression_equals_100_l109_10950


namespace NUMINAMATH_CALUDE_other_dog_walker_dogs_l109_10992

theorem other_dog_walker_dogs (total_legs : ℕ) (num_walkers : ℕ) (human_legs : ℕ) (dog_legs : ℕ) (mariel_dogs : ℕ) : 
  total_legs = 36 →
  num_walkers = 2 →
  human_legs = 2 →
  dog_legs = 4 →
  mariel_dogs = 5 →
  (total_legs - num_walkers * human_legs) / dog_legs - mariel_dogs = 3 :=
by sorry

end NUMINAMATH_CALUDE_other_dog_walker_dogs_l109_10992


namespace NUMINAMATH_CALUDE_species_x_count_l109_10943

def ant_farm (x y : ℕ) : Prop :=
  -- Initial total number of ants
  x + y = 50 ∧
  -- Total number of ants on Day 4
  81 * x + 16 * y = 2914

theorem species_x_count : ∃ x y : ℕ, ant_farm x y ∧ 81 * x = 2754 := by
  sorry

end NUMINAMATH_CALUDE_species_x_count_l109_10943


namespace NUMINAMATH_CALUDE_sequence_property_characterization_l109_10917

/-- A sequence satisfies the required property if for any k = 1, ..., n, 
    it contains two numbers equal to k with exactly k numbers between them. -/
def satisfies_property (seq : List ℕ) (n : ℕ) : Prop :=
  ∀ k ∈ Finset.range n, ∃ i j, i < j ∧ j - i = k + 1 ∧ 
    seq.nthLe i (by sorry) = k ∧ seq.nthLe j (by sorry) = k

/-- The main theorem stating the necessary and sufficient condition for n -/
theorem sequence_property_characterization (n : ℕ) :
  (∃ seq : List ℕ, seq.length = 2 * n ∧ satisfies_property seq n) ↔ 
  (∃ l : ℕ, n = 4 * l ∨ n = 4 * l - 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_property_characterization_l109_10917


namespace NUMINAMATH_CALUDE_existence_of_numbers_l109_10915

theorem existence_of_numbers : ∃ n : ℕ, 
  70 ≤ n ∧ n ≤ 80 ∧ 
  Nat.gcd 30 n = 10 ∧ 
  200 < Nat.lcm 30 n ∧ Nat.lcm 30 n < 300 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_numbers_l109_10915


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_5_pow_7_plus_10_pow_6_l109_10931

theorem greatest_prime_factor_of_5_pow_7_plus_10_pow_6 :
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (5^7 + 10^6) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (5^7 + 10^6) → q ≤ p :=
by sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_5_pow_7_plus_10_pow_6_l109_10931


namespace NUMINAMATH_CALUDE_sugar_water_concentration_l109_10991

/-- Proves that adding 22.5g of water to 90g of sugar water with 10% concentration results in 8% concentration -/
theorem sugar_water_concentration (initial_mass : ℝ) (initial_concentration : ℝ) 
  (added_water : ℝ) (final_concentration : ℝ) : 
  initial_mass = 90 →
  initial_concentration = 0.1 →
  added_water = 22.5 →
  final_concentration = 0.08 →
  (initial_mass * initial_concentration) / (initial_mass + added_water) = final_concentration :=
by
  sorry

#check sugar_water_concentration

end NUMINAMATH_CALUDE_sugar_water_concentration_l109_10991


namespace NUMINAMATH_CALUDE_opening_weekend_revenue_calculation_l109_10981

/-- Represents the movie's financial data in millions of dollars -/
structure MovieFinancials where
  openingWeekendRevenue : ℝ
  totalRevenue : ℝ
  productionCompanyRevenue : ℝ
  productionCost : ℝ
  profit : ℝ

/-- Theorem stating the opening weekend revenue given the movie's financial conditions -/
theorem opening_weekend_revenue_calculation (m : MovieFinancials) 
  (h1 : m.totalRevenue = 3.5 * m.openingWeekendRevenue)
  (h2 : m.productionCompanyRevenue = 0.6 * m.totalRevenue)
  (h3 : m.profit = m.productionCompanyRevenue - m.productionCost)
  (h4 : m.profit = 192)
  (h5 : m.productionCost = 60) :
  m.openingWeekendRevenue = 120 := by
  sorry

end NUMINAMATH_CALUDE_opening_weekend_revenue_calculation_l109_10981


namespace NUMINAMATH_CALUDE_log_expression_equality_l109_10955

theorem log_expression_equality : Real.log 4 / Real.log 10 + 2 * Real.log 5 / Real.log 10 - (Real.sqrt 3 + 1) ^ 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_log_expression_equality_l109_10955


namespace NUMINAMATH_CALUDE_opposite_of_negative_three_l109_10962

theorem opposite_of_negative_three : -((-3 : ℤ)) = 3 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_three_l109_10962


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l109_10929

/-- Parabola C: y² = 4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Line: y = 2x - 4 -/
def line (x y : ℝ) : Prop := y = 2*x - 4

/-- Point A is on both the parabola and the line -/
def point_A (x y : ℝ) : Prop := parabola_C x y ∧ line x y

/-- Point B is on both the parabola and the line, and is distinct from A -/
def point_B (x y : ℝ) : Prop := parabola_C x y ∧ line x y ∧ (x, y) ≠ (x_A, y_A)
  where
  x_A : ℝ := sorry
  y_A : ℝ := sorry

/-- Point P is on the parabola C -/
def point_P (x y : ℝ) : Prop := parabola_C x y

/-- The area of triangle ABP is 12 -/
def triangle_area (x_A y_A x_B y_B x_P y_P : ℝ) : Prop :=
  abs ((x_A - x_P) * (y_B - y_P) - (x_B - x_P) * (y_A - y_P)) / 2 = 12

/-- The main theorem -/
theorem parabola_line_intersection
  (x_A y_A x_B y_B x_P y_P : ℝ)
  (hA : point_A x_A y_A)
  (hB : point_B x_B y_B)
  (hP : point_P x_P y_P)
  (hArea : triangle_area x_A y_A x_B y_B x_P y_P) :
  (((x_B - x_A)^2 + (y_B - y_A)^2)^(1/2 : ℝ) = 3 * 5^(1/2 : ℝ)) ∧
  ((x_P = 9 ∧ y_P = 6) ∨ (x_P = 4 ∧ y_P = -4)) :=
sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l109_10929


namespace NUMINAMATH_CALUDE_stationary_points_of_f_l109_10911

def f (x : ℝ) : ℝ := x^3 - 3*x + 2

theorem stationary_points_of_f :
  ∀ x : ℝ, (∃ y : ℝ, y ≠ x ∧ (∀ z : ℝ, z ≠ x → |z - x| < |y - x| → |f z - f x| ≤ |f y - f x|)) ↔ x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_stationary_points_of_f_l109_10911


namespace NUMINAMATH_CALUDE_largest_value_l109_10969

def expr_a : ℤ := 2 * 0 * 2006
def expr_b : ℤ := 2 * 0 + 6
def expr_c : ℤ := 2 + 0 * 2006
def expr_d : ℤ := 2 * (0 + 6)
def expr_e : ℤ := 2006 * 0 + 0 * 6

theorem largest_value : 
  expr_d = max expr_a (max expr_b (max expr_c (max expr_d expr_e))) :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l109_10969


namespace NUMINAMATH_CALUDE_smallest_m_for_probability_l109_10946

def probability_condition (m : ℕ) : Prop :=
  (m - 1)^4 > (3/4) * m^4

theorem smallest_m_for_probability : 
  probability_condition 17 ∧ 
  ∀ k : ℕ, k < 17 → ¬ probability_condition k :=
sorry

end NUMINAMATH_CALUDE_smallest_m_for_probability_l109_10946


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l109_10989

theorem sufficient_not_necessary_condition :
  (∀ x : ℝ, 1 < x ∧ x < 2 → x^2 - 4*x + 3 < 0) ∧
  (∃ x : ℝ, x^2 - 4*x + 3 < 0 ∧ (x ≤ 1 ∨ x ≥ 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l109_10989


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l109_10906

/-- The equation of the tangent line to the parabola y = x^2 that is parallel to the line 2x - y + 4 = 0 is 2x - y - 1 = 0 -/
theorem tangent_line_to_parabola : 
  ∀ (x y : ℝ), 
  (y = x^2) →  -- Parabola equation
  (∃ (k : ℝ), k * (2 * x - y + 4) = 0) →  -- Parallel condition
  (2 * x - y - 1 = 0) -- Tangent line equation
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l109_10906


namespace NUMINAMATH_CALUDE_wanda_walking_distance_l109_10924

/-- The distance Wanda walks to school (in miles) -/
def distance_to_school : ℝ := 0.5

/-- The number of times Wanda walks to and from school per day -/
def trips_per_day : ℕ := 2

/-- The number of school days per week -/
def school_days_per_week : ℕ := 5

/-- The number of weeks -/
def num_weeks : ℕ := 4

/-- The total distance Wanda walks in miles after the given number of weeks -/
def total_distance : ℝ := 
  distance_to_school * 2 * trips_per_day * school_days_per_week * num_weeks

theorem wanda_walking_distance : total_distance = 40 := by
  sorry

end NUMINAMATH_CALUDE_wanda_walking_distance_l109_10924


namespace NUMINAMATH_CALUDE_simplify_fraction_l109_10912

theorem simplify_fraction : (5^5 + 5^3) / (5^4 - 5^2) = 1625 / 12 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l109_10912


namespace NUMINAMATH_CALUDE_divisibility_problem_l109_10976

theorem divisibility_problem (p q r s : ℕ+) 
  (h1 : Nat.gcd p q = 30)
  (h2 : Nat.gcd q r = 45)
  (h3 : Nat.gcd r s = 60)
  (h4 : 80 < Nat.gcd s p ∧ Nat.gcd s p < 120) :
  15 ∣ p := by
  sorry

end NUMINAMATH_CALUDE_divisibility_problem_l109_10976


namespace NUMINAMATH_CALUDE_snowflake_four_two_l109_10905

-- Define the snowflake operation
def snowflake (a b : ℕ) : ℕ := a * (b - 1) + a * b

-- Theorem statement
theorem snowflake_four_two : snowflake 4 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_snowflake_four_two_l109_10905


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l109_10900

theorem arithmetic_sequence_problem (a : ℚ) : 
  a > 0 ∧ 
  (∃ d : ℚ, 140 + d = a ∧ a + d = 45/28) → 
  a = 3965/56 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l109_10900


namespace NUMINAMATH_CALUDE_marie_messages_theorem_l109_10966

/-- The number of new messages Marie gets per day -/
def new_messages_per_day : ℕ := 6

/-- The initial number of unread messages -/
def initial_messages : ℕ := 98

/-- The number of messages Marie reads per day -/
def messages_read_per_day : ℕ := 20

/-- The number of days it takes Marie to clear all unread messages -/
def days_to_clear : ℕ := 7

theorem marie_messages_theorem :
  initial_messages + days_to_clear * new_messages_per_day = 
  days_to_clear * messages_read_per_day :=
by sorry

end NUMINAMATH_CALUDE_marie_messages_theorem_l109_10966


namespace NUMINAMATH_CALUDE_locus_of_centers_l109_10904

-- Define the circles C1 and C3
def C1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def C3 (x y : ℝ) : Prop := (x - 3)^2 + y^2 = 9

-- Define the property of being externally tangent to C1 and internally tangent to C3
def is_tangent_to_C1_C3 (a b r : ℝ) : Prop :=
  (a^2 + b^2 = (r + 2)^2) ∧ ((a - 3)^2 + b^2 = (3 - r)^2)

-- State the theorem
theorem locus_of_centers (a b : ℝ) :
  (∃ r, is_tangent_to_C1_C3 a b r) → a^2 - 12*a + 4*b^2 = 0 :=
by sorry

end NUMINAMATH_CALUDE_locus_of_centers_l109_10904


namespace NUMINAMATH_CALUDE_at_least_one_quadratic_has_root_l109_10985

theorem at_least_one_quadratic_has_root (a b c d : ℝ) (h : a * c = 2 * b + 2 * d) :
  (a^2 - 4*b ≥ 0) ∨ (c^2 - 4*d ≥ 0) := by sorry

end NUMINAMATH_CALUDE_at_least_one_quadratic_has_root_l109_10985


namespace NUMINAMATH_CALUDE_car_owners_without_motorcycles_l109_10949

theorem car_owners_without_motorcycles 
  (total_adults : ℕ) 
  (car_owners : ℕ) 
  (motorcycle_owners : ℕ) 
  (h1 : total_adults = 400)
  (h2 : car_owners = 350)
  (h3 : motorcycle_owners = 60)
  (h4 : total_adults ≤ car_owners + motorcycle_owners) :
  car_owners - (car_owners + motorcycle_owners - total_adults) = 340 :=
by sorry

end NUMINAMATH_CALUDE_car_owners_without_motorcycles_l109_10949


namespace NUMINAMATH_CALUDE_line_b_production_l109_10925

/-- Represents the production of a factory with three production lines -/
structure FactoryProduction where
  total : ℕ
  lineA : ℕ
  lineB : ℕ
  lineC : ℕ

/-- 
Given a factory production with three lines where:
1. The total production is 24,000 units
2. The number of units sampled from each line forms an arithmetic sequence
3. The sum of production from all lines equals the total production

Then the production of line B is 8,000 units
-/
theorem line_b_production (prod : FactoryProduction) 
  (h_total : prod.total = 24000)
  (h_arithmetic : prod.lineB * 2 = prod.lineA + prod.lineC)
  (h_sum : prod.lineA + prod.lineB + prod.lineC = prod.total) :
  prod.lineB = 8000 := by
  sorry

end NUMINAMATH_CALUDE_line_b_production_l109_10925


namespace NUMINAMATH_CALUDE_find_k_value_l109_10935

theorem find_k_value (x : ℝ) (k : ℝ) : 
  x = 2 → 
  k / (x - 3) - 1 / (3 - x) = 1 → 
  k = -2 := by
sorry

end NUMINAMATH_CALUDE_find_k_value_l109_10935


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l109_10994

theorem arithmetic_calculation : (2 + 3 * 4 - 5) * 2 + 3 = 21 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l109_10994


namespace NUMINAMATH_CALUDE_score_54_recorded_as_negative_6_l109_10926

/-- Calculates the recorded score based on the base score and actual score -/
def recordedScore (baseScore actualScore : Int) : Int :=
  actualScore - baseScore

/-- Theorem: A score of 54 points is recorded as -6 points when the base score is 60 -/
theorem score_54_recorded_as_negative_6 :
  recordedScore 60 54 = -6 := by
  sorry

end NUMINAMATH_CALUDE_score_54_recorded_as_negative_6_l109_10926
