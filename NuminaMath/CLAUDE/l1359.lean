import Mathlib

namespace reciprocal_of_sum_l1359_135903

theorem reciprocal_of_sum : (1 / ((1 : ℚ) / 4 + (1 : ℚ) / 5)) = 20 / 9 := by
  sorry

end reciprocal_of_sum_l1359_135903


namespace ten_row_triangle_pieces_l1359_135914

/-- Calculates the sum of an arithmetic sequence -/
def arithmeticSum (a : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a + (n - 1) * d) / 2

/-- Calculates the nth triangular number -/
def triangularNumber (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Represents the structure of a triangle made of rods and connectors -/
structure RodTriangle where
  rows : ℕ
  firstRowRods : ℕ
  rodIncrement : ℕ

/-- Calculates the total number of rods in a RodTriangle -/
def totalRods (t : RodTriangle) : ℕ :=
  arithmeticSum t.firstRowRods t.rodIncrement t.rows

/-- Calculates the total number of connectors in a RodTriangle -/
def totalConnectors (t : RodTriangle) : ℕ :=
  triangularNumber (t.rows + 1)

/-- Calculates the total number of pieces (rods and connectors) in a RodTriangle -/
def totalPieces (t : RodTriangle) : ℕ :=
  totalRods t + totalConnectors t

/-- Theorem: The total number of pieces in a ten-row triangle is 231 -/
theorem ten_row_triangle_pieces :
  totalPieces { rows := 10, firstRowRods := 3, rodIncrement := 3 } = 231 := by
  sorry

end ten_row_triangle_pieces_l1359_135914


namespace quadratic_equation_result_l1359_135904

theorem quadratic_equation_result (a : ℝ) (h : a^2 + a - 3 = 0) : a^2 * (a + 4) = 9 := by
  sorry

end quadratic_equation_result_l1359_135904


namespace triangle_theorem_l1359_135994

noncomputable section

variables {a b c : ℝ} {A B C : Real}

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- The main theorem about the triangle -/
theorem triangle_theorem (t : Triangle) 
  (h1 : t.b + t.c = 2 * t.a) 
  (h2 : 3 * t.c * Real.sin t.B = 4 * t.a * Real.sin t.C) : 
  Real.cos t.B = -1/4 ∧ Real.sin (2 * t.B + π/6) = -(3 * Real.sqrt 5 + 7)/16 := by
  sorry

end

end triangle_theorem_l1359_135994


namespace median_to_mean_l1359_135909

theorem median_to_mean (m : ℝ) : 
  let s : Finset ℝ := {m, m + 4, m + 7, m + 10, m + 16}
  m + 7 = 12 →
  (s.sum id) / s.card = 12.4 := by
sorry

end median_to_mean_l1359_135909


namespace inequality_solution_range_l1359_135999

theorem inequality_solution_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * m * x - 8 ≥ 0) ↔ m ≠ 0 := by
sorry

end inequality_solution_range_l1359_135999


namespace sum_of_areas_l1359_135911

/-- A sequence of circles tangent to two half-lines -/
structure TangentCircles where
  d₁ : ℝ
  r₁ : ℝ
  d : ℕ → ℝ
  r : ℕ → ℝ
  h₁ : d₁ > 0
  h₂ : r₁ > 0
  h₃ : ∀ n : ℕ, d n > 0
  h₄ : ∀ n : ℕ, r n > 0
  h₅ : d 1 = d₁
  h₆ : r 1 = r₁
  h₇ : ∀ n : ℕ, n > 1 → d n < d (n-1)
  h₈ : ∀ n : ℕ, r n / d n = r₁ / d₁

theorem sum_of_areas (tc : TangentCircles) :
  (∑' n, π * (tc.r n)^2) = (π/4) * (tc.r₁ * (tc.d₁ + tc.r₁)^2 / tc.d₁) :=
sorry

end sum_of_areas_l1359_135911


namespace divisible_by_seven_l1359_135915

def number (x : ℕ) : ℕ := 
  666666666666666666666666666666666666666666666666666 * 10^51 + 
  x * 10^50 + 
  555555555555555555555555555555555555555555555555555

theorem divisible_by_seven (x : ℕ) : 
  x < 10 → (number x % 7 = 0 ↔ x = 2 ∨ x = 9) := by sorry

end divisible_by_seven_l1359_135915


namespace bianca_extra_flowers_l1359_135946

/-- The number of extra flowers Bianca picked -/
def extra_flowers (tulips roses used : ℕ) : ℕ :=
  tulips + roses - used

/-- Proof that Bianca picked 7 extra flowers -/
theorem bianca_extra_flowers :
  extra_flowers 39 49 81 = 7 := by
  sorry

end bianca_extra_flowers_l1359_135946


namespace valid_pairs_l1359_135920

def is_valid_pair (a b : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧
  (∃ k : ℤ, (k : ℚ) = (a^2 + b : ℚ) / (b^2 - a : ℚ)) ∧
  (∃ m : ℤ, (m : ℚ) = (b^2 + a : ℚ) / (a^2 - b : ℚ))

theorem valid_pairs :
  ∀ a b : ℕ, is_valid_pair a b ↔
    ((a = 2 ∧ b = 2) ∨
     (a = 3 ∧ b = 3) ∨
     (a = 1 ∧ b = 2) ∨
     (a = 2 ∧ b = 1) ∨
     (a = 2 ∧ b = 3) ∨
     (a = 3 ∧ b = 2)) :=
by sorry

end valid_pairs_l1359_135920


namespace people_per_column_l1359_135966

theorem people_per_column (total_people : ℕ) (people_per_column : ℕ) : 
  (total_people = 16 * people_per_column) ∧ 
  (total_people = 15 * 32) → 
  people_per_column = 30 := by
  sorry

end people_per_column_l1359_135966


namespace train_passes_jogger_l1359_135910

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passes_jogger (jogger_speed train_speed : ℝ) (train_length initial_distance : ℝ) :
  jogger_speed = 9 * (1000 / 3600) →
  train_speed = 45 * (1000 / 3600) →
  train_length = 210 →
  initial_distance = 200 →
  (initial_distance + train_length) / (train_speed - jogger_speed) = 41 :=
by sorry

end train_passes_jogger_l1359_135910


namespace total_interest_is_1800_l1359_135980

/-- Calculates the total interest over 10 years when the principal is trebled after 5 years -/
def totalInterest (P R : ℚ) : ℚ :=
  let firstHalfInterest := (P * R * 5) / 100
  let secondHalfInterest := (3 * P * R * 5) / 100
  firstHalfInterest + secondHalfInterest

/-- Theorem stating that the total interest is 1800 given the problem conditions -/
theorem total_interest_is_1800 (P R : ℚ) 
    (h : (P * R * 10) / 100 = 900) : totalInterest P R = 1800 := by
  sorry

#eval totalInterest 1000 9  -- This should evaluate to 1800

end total_interest_is_1800_l1359_135980


namespace trigonometric_problem_l1359_135961

theorem trigonometric_problem (x : ℝ) 
  (h1 : -π/2 < x ∧ x < 0) 
  (h2 : Real.sin x + Real.cos x = 1/5) : 
  (Real.sin x - Real.cos x = -7/5) ∧ 
  (1 / (Real.cos x ^ 2 - Real.sin x ^ 2) = 25/7) := by
sorry

end trigonometric_problem_l1359_135961


namespace extremum_implies_deriv_zero_not_always_converse_l1359_135918

open Set
open Function
open Topology

-- Define a structure for differentiable functions on ℝ
structure DiffFunction where
  f : ℝ → ℝ
  diff : Differentiable ℝ f

variable (f : DiffFunction)

-- Define what it means for a function to have an extremum
def has_extremum (f : DiffFunction) : Prop :=
  ∃ x₀ : ℝ, ∀ x : ℝ, f.f x ≤ f.f x₀ ∨ f.f x ≥ f.f x₀

-- Define what it means for f'(x) = 0 to have a solution
def deriv_has_zero (f : DiffFunction) : Prop :=
  ∃ x : ℝ, deriv f.f x = 0

-- State the theorem
theorem extremum_implies_deriv_zero (f : DiffFunction) : 
  has_extremum f → deriv_has_zero f :=
sorry

-- State that the converse is not always true
theorem not_always_converse : 
  ∃ f : DiffFunction, deriv_has_zero f ∧ ¬has_extremum f :=
sorry

end extremum_implies_deriv_zero_not_always_converse_l1359_135918


namespace vector_coordinates_l1359_135987

/-- A vector in a 2D Cartesian coordinate system -/
structure Vector2D where
  x : ℝ
  y : ℝ

/-- The standard basis vectors -/
def i : Vector2D := ⟨1, 0⟩
def j : Vector2D := ⟨0, 1⟩

/-- Vector addition -/
def add (v w : Vector2D) : Vector2D :=
  ⟨v.x + w.x, v.y + w.y⟩

/-- Scalar multiplication -/
def smul (r : ℝ) (v : Vector2D) : Vector2D :=
  ⟨r * v.x, r * v.y⟩

/-- The main theorem -/
theorem vector_coordinates (x y : ℝ) :
  let a := add (smul x i) (smul y j)
  a = ⟨x, y⟩ := by sorry

end vector_coordinates_l1359_135987


namespace inscribed_circle_radius_l1359_135913

/-- An isosceles triangle with a circle inscribed in it -/
structure IsoscelesTriangleWithInscribedCircle where
  -- Base of the isosceles triangle
  base : ℝ
  -- Height of the isosceles triangle
  height : ℝ
  -- Radius of the inscribed circle
  radius : ℝ
  -- The circle touches the base and both equal sides of the triangle
  touches_sides : True

/-- Theorem stating that for an isosceles triangle with base 20 and height 24, 
    the radius of the inscribed circle is 20/3 -/
theorem inscribed_circle_radius 
  (triangle : IsoscelesTriangleWithInscribedCircle)
  (h_base : triangle.base = 20)
  (h_height : triangle.height = 24) :
  triangle.radius = 20 / 3 := by
  sorry

#check inscribed_circle_radius

end inscribed_circle_radius_l1359_135913


namespace average_monthly_growth_rate_l1359_135998

theorem average_monthly_growth_rate 
  (initial_production : ℕ) 
  (final_production : ℕ) 
  (months : ℕ) 
  (growth_rate : ℝ) :
  initial_production = 100 →
  final_production = 144 →
  months = 2 →
  initial_production * (1 + growth_rate) ^ months = final_production →
  growth_rate = 0.2 := by
sorry

end average_monthly_growth_rate_l1359_135998


namespace math_competition_probabilities_l1359_135923

def number_of_students : ℕ := 5
def number_of_boys : ℕ := 2
def number_of_girls : ℕ := 3
def students_to_select : ℕ := 2

-- Total number of possible selections
def total_selections : ℕ := Nat.choose number_of_students students_to_select

-- Number of ways to select exactly one boy
def exactly_one_boy_selections : ℕ := number_of_boys * number_of_girls

-- Number of ways to select at least one boy
def at_least_one_boy_selections : ℕ := total_selections - Nat.choose number_of_girls students_to_select

theorem math_competition_probabilities :
  (total_selections = 10) ∧
  (exactly_one_boy_selections / total_selections = 3 / 5) ∧
  (at_least_one_boy_selections / total_selections = 7 / 10) := by
  sorry

end math_competition_probabilities_l1359_135923


namespace common_point_of_circumcircles_l1359_135901

-- Define the circle S
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

-- Define a point being outside a circle
def IsOutside (p : ℝ × ℝ) (s : Set (ℝ × ℝ)) : Prop :=
  p ∉ s

-- Define a line passing through a point
def LineThroughPoint (p : ℝ × ℝ) : Set (ℝ × ℝ) :=
  {q : ℝ × ℝ | ∃ (t : ℝ), q = (p.1 + t, p.2 + t)}

-- Define the intersection of a line and a circle
def Intersect (l : Set (ℝ × ℝ)) (s : Set (ℝ × ℝ)) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p ∈ l ∧ p ∈ s}

-- Define the circumcircle of a triangle
def Circumcircle (p1 p2 p3 : ℝ × ℝ) : Set (ℝ × ℝ) :=
  sorry -- Actual definition would be more complex

-- Main theorem
theorem common_point_of_circumcircles
  (S : Set (ℝ × ℝ)) (center : ℝ × ℝ) (radius : ℝ)
  (A B : ℝ × ℝ) :
  S = Circle center radius →
  IsOutside A S →
  IsOutside B S →
  ∃ (C : ℝ × ℝ), C ≠ B ∧
    ∀ (l : Set (ℝ × ℝ)) (M N : ℝ × ℝ),
      l = LineThroughPoint A →
      {M, N} ⊆ Intersect l S →
      C ∈ Circumcircle B M N :=
by sorry

end common_point_of_circumcircles_l1359_135901


namespace largest_divisor_five_consecutive_integers_l1359_135916

theorem largest_divisor_five_consecutive_integers :
  ∀ n : ℤ, ∃ k : ℤ, k > 60 ∧ ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) ∧
  ∀ m : ℤ, m ≤ 60 → (m ∣ (n * (n + 1) * (n + 2) * (n + 3) * (n + 4))) :=
by sorry

end largest_divisor_five_consecutive_integers_l1359_135916


namespace paiges_drawers_l1359_135947

theorem paiges_drawers (clothing_per_drawer : ℕ) (total_clothing : ℕ) (num_drawers : ℕ) :
  clothing_per_drawer = 2 →
  total_clothing = 8 →
  num_drawers * clothing_per_drawer = total_clothing →
  num_drawers = 4 := by
sorry

end paiges_drawers_l1359_135947


namespace cube_root_simplification_l1359_135926

theorem cube_root_simplification :
  (20^3 + 30^3 + 50^3 : ℝ)^(1/3) = 10 * 160^(1/3) := by
  sorry

end cube_root_simplification_l1359_135926


namespace square_area_ratio_l1359_135935

theorem square_area_ratio : 
  ∀ (s₂ : ℝ), s₂ > 0 →
  let s₁ := s₂ * Real.sqrt 2
  let s₃ := s₁ / 2
  let A₂ := s₂ ^ 2
  let A₃ := s₃ ^ 2
  A₃ / A₂ = 1 / 2 := by
    sorry

end square_area_ratio_l1359_135935


namespace queen_middle_school_teachers_l1359_135964

structure School where
  students : ℕ
  classes_per_student : ℕ
  classes_per_teacher : ℕ
  students_per_class : ℕ

def number_of_teachers (school : School) : ℕ :=
  (school.students * school.classes_per_student) / (school.students_per_class * school.classes_per_teacher)

theorem queen_middle_school_teachers :
  let queen_middle : School := {
    students := 1500,
    classes_per_student := 5,
    classes_per_teacher := 5,
    students_per_class := 25
  }
  number_of_teachers queen_middle = 60 := by
  sorry

end queen_middle_school_teachers_l1359_135964


namespace corn_height_after_ten_weeks_l1359_135976

/-- Represents the growth of corn plants over 10 weeks -/
def corn_growth : List ℝ := [
  2,       -- Week 1
  4,       -- Week 2
  16,      -- Week 3
  22,      -- Week 4
  8,       -- Week 5
  16,      -- Week 6
  12.33,   -- Week 7
  7.33,    -- Week 8
  24,      -- Week 9
  36       -- Week 10
]

/-- The total height of the corn plants after 10 weeks -/
def total_height : ℝ := corn_growth.sum

/-- Theorem stating that the total height of the corn plants after 10 weeks is 147.66 inches -/
theorem corn_height_after_ten_weeks : total_height = 147.66 := by
  sorry

end corn_height_after_ten_weeks_l1359_135976


namespace total_cars_l1359_135956

/-- The number of cars owned by each person --/
structure CarOwnership where
  cathy : ℕ
  lindsey : ℕ
  carol : ℕ
  susan : ℕ
  erica : ℕ
  jack : ℕ
  kevin : ℕ

/-- The conditions of car ownership --/
def carOwnershipConditions (c : CarOwnership) : Prop :=
  c.cathy = 5 ∧
  c.lindsey = c.cathy + 4 ∧
  c.susan = c.carol - 2 ∧
  c.carol = 2 * c.cathy ∧
  c.erica = c.lindsey + (c.lindsey / 4) ∧
  c.jack = (c.susan + c.carol) / 2 ∧
  c.kevin = ((c.lindsey + c.cathy) * 9) / 10

/-- The theorem stating the total number of cars --/
theorem total_cars (c : CarOwnership) (h : carOwnershipConditions c) : 
  c.cathy + c.lindsey + c.carol + c.susan + c.erica + c.jack + c.kevin = 65 := by
  sorry


end total_cars_l1359_135956


namespace total_cards_l1359_135941

/-- The number of cards each person has -/
structure CardCounts where
  heike : ℕ
  anton : ℕ
  ann : ℕ
  bertrand : ℕ

/-- The conditions of the card counting problem -/
def card_problem (c : CardCounts) : Prop :=
  c.anton = 3 * c.heike ∧
  c.ann = 6 * c.heike ∧
  c.bertrand = 2 * c.heike ∧
  c.ann = 60

/-- The theorem stating that under the given conditions, 
    the total number of cards is 120 -/
theorem total_cards (c : CardCounts) : 
  card_problem c → c.heike + c.anton + c.ann + c.bertrand = 120 := by
  sorry


end total_cards_l1359_135941


namespace initial_distance_is_54km_l1359_135986

/-- Represents the cycling scenario described in the problem -/
structure CyclingScenario where
  v : ℝ  -- Initial speed in km/h
  t : ℝ  -- Time shown on cycle computer in hours
  d : ℝ  -- Initial distance from home in km

/-- The conditions of the cycling scenario -/
def scenario_conditions (s : CyclingScenario) : Prop :=
  s.d = s.v * s.t ∧  -- Initial condition
  s.d = (2/3 * s.v) + (s.v - 1) * s.t ∧  -- After first speed change
  s.d = (2/3 * s.v) + (3/4 * (s.v - 1)) + (s.v - 2) * s.t  -- After second speed change

/-- The theorem stating that the initial distance is 54 km -/
theorem initial_distance_is_54km (s : CyclingScenario) 
  (h : scenario_conditions s) : s.d = 54 := by
  sorry

#check initial_distance_is_54km

end initial_distance_is_54km_l1359_135986


namespace investment_sum_l1359_135979

theorem investment_sum (P : ℝ) : 
  P * (18 / 100) * 2 - P * (12 / 100) * 2 = 240 → P = 2000 := by sorry

end investment_sum_l1359_135979


namespace factorization_proof_l1359_135955

theorem factorization_proof (x : ℝ) : 
  3 * x^2 * (x - 2) + 4 * x * (x - 2) + 2 * (x - 2) = (x - 2) * (x + 2) * (3 * x + 2) := by
sorry

end factorization_proof_l1359_135955


namespace negative_sum_l1359_135982

theorem negative_sum (a b c : ℝ) 
  (ha : 1 < a ∧ a < 2) 
  (hb : 0 < b ∧ b < 1) 
  (hc : -2 < c ∧ c < -1) : 
  c + b < 0 := by
  sorry

end negative_sum_l1359_135982


namespace interest_rate_calculation_l1359_135954

/-- Calculate the interest rate given simple interest, principal, and time -/
theorem interest_rate_calculation
  (simple_interest : ℝ)
  (principal : ℝ)
  (time : ℝ)
  (h1 : simple_interest = 4016.25)
  (h2 : principal = 10040.625)
  (h3 : time = 5)
  (h4 : simple_interest = principal * (rate / 100) * time) :
  rate = 8 := by
  sorry

end interest_rate_calculation_l1359_135954


namespace find_B_l1359_135952

theorem find_B (A B : ℕ) (h1 : A = 21) (h2 : Nat.gcd A B = 7) (h3 : Nat.lcm A B = 105) :
  B = 35 := by
sorry

end find_B_l1359_135952


namespace quadratic_function_theorem_l1359_135936

/-- A quadratic function with leading coefficient a -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The solution set of f(x) > -2x is (1,3) -/
def solution_set (a b c : ℝ) : Prop :=
  ∀ x, (1 < x ∧ x < 3) ↔ f a b c x > -2 * x

/-- The equation f(x) + 6a = 0 has two equal real roots -/
def equal_roots (a b c : ℝ) : Prop :=
  ∃ r : ℝ, ∀ x, f a b c x + 6 * a = 0 ↔ x = r

theorem quadratic_function_theorem (a b c : ℝ) 
  (h1 : solution_set a b c)
  (h2 : equal_roots a b c)
  (h3 : a < 0) :
  ∀ x, f a b c x = -x^2 - x - 3/5 := by
  sorry

end quadratic_function_theorem_l1359_135936


namespace wire_cutting_l1359_135970

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_piece : ℝ) : 
  total_length = 70 ∧ 
  ratio = 2 / 5 ∧ 
  shorter_piece + (shorter_piece / ratio) = total_length →
  shorter_piece = 20 := by
sorry

end wire_cutting_l1359_135970


namespace fixed_point_existence_l1359_135940

theorem fixed_point_existence (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  ∃ x : ℝ, x = a^(x - 2) - 3 ∧ x = 2 := by
  sorry

end fixed_point_existence_l1359_135940


namespace arithmetic_matrix_middle_value_l1359_135949

/-- Represents a 5x5 matrix where each row and column forms an arithmetic sequence -/
def ArithmeticMatrix := Matrix (Fin 5) (Fin 5) ℝ

/-- Checks if a given row or column of the matrix forms an arithmetic sequence -/
def isArithmeticSequence (seq : Fin 5 → ℝ) : Prop :=
  ∃ d : ℝ, ∀ i : Fin 5, i.val < 4 → seq (i + 1) = seq i + d

/-- The property that all rows and columns of the matrix form arithmetic sequences -/
def allArithmeticSequences (M : ArithmeticMatrix) : Prop :=
  (∀ i : Fin 5, isArithmeticSequence (λ j => M i j)) ∧
  (∀ j : Fin 5, isArithmeticSequence (λ i => M i j))

theorem arithmetic_matrix_middle_value
  (M : ArithmeticMatrix)
  (all_arithmetic : allArithmeticSequences M)
  (first_row_start : M 0 0 = 3)
  (first_row_end : M 0 4 = 15)
  (last_row_start : M 4 0 = 25)
  (last_row_end : M 4 4 = 65) :
  M 2 2 = 27 := by
  sorry

end arithmetic_matrix_middle_value_l1359_135949


namespace optimal_price_reduction_and_profit_l1359_135944

/-- Represents the daily profit function for a flower shop -/
def profit_function (x : ℝ) : ℝ := -2 * x^2 + 60 * x + 800

/-- Represents the constraints on the price reduction -/
def valid_price_reduction (x : ℝ) : Prop := 0 ≤ x ∧ x < 40

/-- Theorem stating the optimal price reduction and maximum profit -/
theorem optimal_price_reduction_and_profit :
  ∃ (x : ℝ), valid_price_reduction x ∧ 
    (∀ y, valid_price_reduction y → profit_function y ≤ profit_function x) ∧
    x = 15 ∧ profit_function x = 1250 :=
sorry

end optimal_price_reduction_and_profit_l1359_135944


namespace inequality_solution_set_l1359_135953

theorem inequality_solution_set (y : ℝ) :
  (2 / (y - 2) + 5 / (y + 3) ≤ 2) ↔ (y ∈ Set.Ioc (-3) (-1) ∪ Set.Ioo 2 4) :=
by sorry

end inequality_solution_set_l1359_135953


namespace f_properties_l1359_135963

noncomputable section

variable (a : ℝ)
variable (h₁ : a > 0)
variable (h₂ : a ≠ 1)

def f (x : ℝ) : ℝ := (a / (a^2 - 1)) * (a^x - a^(-x))

theorem f_properties :
  (∀ x, f a x = -f a (-x)) ∧ 
  (StrictMono (f a)) ∧
  (∀ m, 1 < m → m < Real.sqrt 2 → f a (1 - m) + f a (1 - m^2) < 0) :=
sorry

end f_properties_l1359_135963


namespace travel_expense_fraction_l1359_135942

theorem travel_expense_fraction (initial_amount : ℝ) 
  (clothes_fraction : ℝ) (food_fraction : ℝ) (final_amount : ℝ) :
  initial_amount = 1499.9999999999998 →
  clothes_fraction = 1/3 →
  food_fraction = 1/5 →
  final_amount = 600 →
  let remaining_after_clothes := initial_amount * (1 - clothes_fraction)
  let remaining_after_food := remaining_after_clothes * (1 - food_fraction)
  (remaining_after_food - final_amount) / remaining_after_food = 1/4 := by
sorry

end travel_expense_fraction_l1359_135942


namespace breaking_process_result_l1359_135975

/-- Represents a triangle with its three angles in degrees -/
structure Triangle where
  angle1 : ℝ
  angle2 : ℝ
  angle3 : ℝ

/-- Determines if a triangle is acute-angled -/
def Triangle.isAcute (t : Triangle) : Prop :=
  t.angle1 < 90 ∧ t.angle2 < 90 ∧ t.angle3 < 90

/-- Represents the operation of breaking a triangle -/
def breakTriangle (t : Triangle) : List Triangle :=
  sorry  -- Implementation details omitted

/-- Counts the total number of triangles after breaking process -/
def countTriangles (initial : Triangle) : ℕ :=
  sorry  -- Implementation details omitted

/-- The theorem to be proved -/
theorem breaking_process_result (t : Triangle) 
  (h1 : t.angle1 = 3)
  (h2 : t.angle2 = 88)
  (h3 : t.angle3 = 89) :
  countTriangles t = 11 :=
sorry

end breaking_process_result_l1359_135975


namespace not_divides_power_minus_one_l1359_135934

theorem not_divides_power_minus_one (n : ℕ) (h : n ≥ 2) : ¬(n ∣ 2^n - 1) := by
  sorry

end not_divides_power_minus_one_l1359_135934


namespace roots_greater_than_five_k_range_l1359_135988

theorem roots_greater_than_five_k_range (k : ℝ) : 
  (∀ x : ℝ, x^2 - 11*x + (30 + k) = 0 → x > 5) → 
  0 < k ∧ k ≤ 1/4 := by
sorry

end roots_greater_than_five_k_range_l1359_135988


namespace pyramid_volume_l1359_135938

theorem pyramid_volume (base_side : ℝ) (height : ℝ) (volume : ℝ) :
  base_side = 1 / 2 →
  height = 1 →
  volume = (1 / 3) * (base_side ^ 2) * height →
  volume = 1 / 12 :=
by sorry

end pyramid_volume_l1359_135938


namespace average_coins_per_day_l1359_135965

def coins_collected (day : ℕ) : ℕ :=
  if day = 0 then 0
  else if day < 7 then 10 * day
  else 10 * 7 + 20

def total_coins : ℕ := (List.range 7).map (λ i => coins_collected (i + 1)) |>.sum

theorem average_coins_per_day :
  (total_coins : ℚ) / 7 = 300 / 7 := by sorry

end average_coins_per_day_l1359_135965


namespace largest_of_three_consecutive_odds_l1359_135930

theorem largest_of_three_consecutive_odds (a b c : ℤ) : 
  (∃ n : ℤ, a = 2*n + 1 ∧ b = 2*n + 3 ∧ c = 2*n + 5) →  -- consecutive odd integers
  (a + b + c = -147) →                                  -- sum is -147
  (max a (max b c) = -47) :=                            -- largest is -47
by sorry

end largest_of_three_consecutive_odds_l1359_135930


namespace floor_of_7_9_l1359_135900

theorem floor_of_7_9 : ⌊(7.9 : ℝ)⌋ = 7 := by sorry

end floor_of_7_9_l1359_135900


namespace binomial_sum_36_implies_n_8_l1359_135992

theorem binomial_sum_36_implies_n_8 (n : ℕ+) :
  (Nat.choose n 1 + Nat.choose n 2 = 36) → n = 8 := by
  sorry

end binomial_sum_36_implies_n_8_l1359_135992


namespace sum_of_prime_factors_999973_l1359_135996

theorem sum_of_prime_factors_999973 :
  ∃ (p q r : ℕ), 
    Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
    p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
    999973 = p * q * r ∧
    p + q + r = 171 :=
by
  sorry

end sum_of_prime_factors_999973_l1359_135996


namespace roadway_deck_concrete_amount_l1359_135929

/-- The amount of concrete needed for the roadway deck of a bridge -/
def roadway_deck_concrete (total_concrete : ℕ) (anchor_concrete : ℕ) (pillar_concrete : ℕ) : ℕ :=
  total_concrete - (2 * anchor_concrete + pillar_concrete)

/-- Theorem stating that the roadway deck needs 1600 tons of concrete -/
theorem roadway_deck_concrete_amount :
  roadway_deck_concrete 4800 700 1800 = 1600 := by
  sorry

end roadway_deck_concrete_amount_l1359_135929


namespace absolute_value_equals_opposite_implies_nonpositive_l1359_135921

theorem absolute_value_equals_opposite_implies_nonpositive (a : ℝ) :
  (abs a = -a) → a ≤ 0 := by
  sorry

end absolute_value_equals_opposite_implies_nonpositive_l1359_135921


namespace place_eight_among_twelve_l1359_135928

/-- The number of ways to place black balls among white balls without adjacency. -/
def place_balls (white : ℕ) (black : ℕ) : ℕ :=
  Nat.choose (white + 1) black

/-- Theorem: Placing 8 black balls among 12 white balls without adjacency. -/
theorem place_eight_among_twelve :
  place_balls 12 8 = 1287 := by
  sorry

end place_eight_among_twelve_l1359_135928


namespace incorrect_page_number_l1359_135995

theorem incorrect_page_number (n : ℕ) (x : ℕ) : 
  (n ≥ 1) →
  (x ≤ n) →
  (n * (n + 1) / 2 + x = 2076) →
  x = 60 := by
sorry

end incorrect_page_number_l1359_135995


namespace intersection_equals_A_l1359_135927

def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B : Set ℝ := {x | x ≥ 1}

theorem intersection_equals_A : A ∩ B = A := by
  sorry

end intersection_equals_A_l1359_135927


namespace pentagon_area_sum_l1359_135939

theorem pentagon_area_sum (u v : ℤ) : 
  0 < v → v < u → (u^2 + 3*u*v = 451) → u + v = 21 := by sorry

end pentagon_area_sum_l1359_135939


namespace ellipse_equation_l1359_135948

/-- Proves that an ellipse with given conditions has the equation x^2 + 4y^2 = 1 -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  let e := Real.sqrt 3 / 2
  let ellipse := fun (x y : ℝ) => x^2 / a^2 + y^2 / b^2 = 1
  let line := fun (x y : ℝ) => x - y + 1 = 0
  ∃ (A B C : ℝ × ℝ),
    ellipse A.1 A.2 ∧
    ellipse B.1 B.2 ∧
    line A.1 A.2 ∧
    line B.1 B.2 ∧
    C.1 = 0 ∧
    line C.1 C.2 ∧
    (3 * (B.1 - A.1), 3 * (B.2 - A.2)) = (2 * (C.1 - B.1), 2 * (C.2 - B.2)) →
  e^2 * a^2 = a^2 - b^2 →
  ∀ (x y : ℝ), x^2 + 4*y^2 = 1 ↔ ellipse x y := by sorry

end ellipse_equation_l1359_135948


namespace dog_roaming_area_l1359_135991

/-- The area available for a dog to roam when tied to the corner of an L-shaped garden wall. -/
theorem dog_roaming_area (wall_length : ℝ) (rope_length : ℝ) : wall_length = 16 ∧ rope_length = 8 → 
  (2 * (1/4 * Real.pi * rope_length^2)) = 32 * Real.pi := by
  sorry

end dog_roaming_area_l1359_135991


namespace num_four_digit_numbers_eq_twelve_l1359_135925

/-- The number of different four-digit numbers that can be formed using the cards "2", "0", "0", "9" (where "9" can also be used as "6") -/
def num_four_digit_numbers : ℕ :=
  (Nat.choose 3 2) * 2 * (Nat.factorial 2)

/-- Theorem stating that the number of different four-digit numbers is 12 -/
theorem num_four_digit_numbers_eq_twelve : num_four_digit_numbers = 12 := by
  sorry

#eval num_four_digit_numbers

end num_four_digit_numbers_eq_twelve_l1359_135925


namespace basketball_score_ratio_l1359_135957

theorem basketball_score_ratio : 
  ∀ (marks_two_pointers marks_three_pointers marks_free_throws : ℕ)
    (total_points : ℕ),
  marks_two_pointers = 25 →
  marks_three_pointers = 8 →
  marks_free_throws = 10 →
  total_points = 201 →
  ∃ (ratio : ℚ),
    ratio = 1/2 ∧
    (2 * marks_two_pointers * 2 + ratio * (marks_three_pointers * 3 + marks_free_throws)) +
    (marks_two_pointers * 2 + marks_three_pointers * 3 + marks_free_throws) = total_points :=
by sorry

end basketball_score_ratio_l1359_135957


namespace simplified_expression_equals_sqrt_two_minus_one_l1359_135905

theorem simplified_expression_equals_sqrt_two_minus_one :
  let x : ℝ := Real.sqrt 2 - 1
  (x^2 / (x^2 + 4*x + 4)) / (x / (x + 2)) - (x - 1) / (x + 2) = Real.sqrt 2 - 1 := by
  sorry

end simplified_expression_equals_sqrt_two_minus_one_l1359_135905


namespace equilateral_hyperbola_equation_l1359_135978

/-- An equilateral hyperbola centered at the origin and passing through (0, 3) -/
structure EquilateralHyperbola where
  /-- The equation of the hyperbola in the form y² - x² = a -/
  equation : ℝ → ℝ → ℝ
  /-- The hyperbola passes through the point (0, 3) -/
  passes_through_point : equation 0 3 = equation 0 3
  /-- The hyperbola is centered at the origin -/
  centered_at_origin : ∀ x y, equation x y = equation (-x) (-y)
  /-- The hyperbola is equilateral -/
  equilateral : ∀ x y, equation x y = equation y x

/-- The equation of the equilateral hyperbola is y² - x² = 9 -/
theorem equilateral_hyperbola_equation (h : EquilateralHyperbola) :
  ∀ x y, h.equation x y = y^2 - x^2 - 9 := by sorry

end equilateral_hyperbola_equation_l1359_135978


namespace sine_ratio_equals_one_l1359_135977

theorem sine_ratio_equals_one (c : ℝ) (h : c = 2 * π / 13) :
  (Real.sin (4 * c) * Real.sin (8 * c) * Real.sin (12 * c) * Real.sin (16 * c) * Real.sin (20 * c)) /
  (Real.sin (2 * c) * Real.sin (4 * c) * Real.sin (6 * c) * Real.sin (8 * c) * Real.sin (10 * c)) = 1 :=
by sorry

end sine_ratio_equals_one_l1359_135977


namespace remainder_2017_div_89_l1359_135902

theorem remainder_2017_div_89 : 2017 % 89 = 59 := by
  sorry

end remainder_2017_div_89_l1359_135902


namespace function_value_at_four_l1359_135990

/-- Given a function g: ℝ → ℝ satisfying g(x) + 2g(1 - x) = 6x^2 - 2x for all x,
    prove that g(4) = 32/3 -/
theorem function_value_at_four
  (g : ℝ → ℝ)
  (h : ∀ x, g x + 2 * g (1 - x) = 6 * x^2 - 2 * x) :
  g 4 = 32/3 := by
  sorry

end function_value_at_four_l1359_135990


namespace carlos_total_earnings_l1359_135908

-- Define the problem parameters
def hours_week1 : ℕ := 18
def hours_week2 : ℕ := 30
def extra_earnings : ℕ := 54

-- Define Carlos's hourly wage as a rational number
def hourly_wage : ℚ := 54 / 12

-- Theorem statement
theorem carlos_total_earnings :
  (hours_week1 : ℚ) * hourly_wage + (hours_week2 : ℚ) * hourly_wage = 216 := by
  sorry

#eval (hours_week1 : ℚ) * hourly_wage + (hours_week2 : ℚ) * hourly_wage

end carlos_total_earnings_l1359_135908


namespace sum_gcd_lcm_18_30_45_l1359_135967

def A : ℕ := Nat.gcd 18 (Nat.gcd 30 45)
def B : ℕ := Nat.lcm 18 (Nat.lcm 30 45)

theorem sum_gcd_lcm_18_30_45 : A + B = 93 := by
  sorry

end sum_gcd_lcm_18_30_45_l1359_135967


namespace product_of_roots_l1359_135931

theorem product_of_roots (x₁ x₂ : ℝ) 
  (h_distinct : x₁ ≠ x₂)
  (h₁ : x₁^2 - 2*x₁ = 1)
  (h₂ : x₂^2 - 2*x₂ = 1) : 
  x₁ * x₂ = -1 := by sorry

end product_of_roots_l1359_135931


namespace garage_wheels_eq_22_l1359_135951

/-- The number of wheels in Timmy's parents' garage -/
def garage_wheels : ℕ :=
  let num_cars : ℕ := 2
  let num_lawnmowers : ℕ := 1
  let num_bicycles : ℕ := 3
  let num_tricycles : ℕ := 1
  let num_unicycles : ℕ := 1
  let wheels_per_car : ℕ := 4
  let wheels_per_lawnmower : ℕ := 4
  let wheels_per_bicycle : ℕ := 2
  let wheels_per_tricycle : ℕ := 3
  let wheels_per_unicycle : ℕ := 1
  num_cars * wheels_per_car +
  num_lawnmowers * wheels_per_lawnmower +
  num_bicycles * wheels_per_bicycle +
  num_tricycles * wheels_per_tricycle +
  num_unicycles * wheels_per_unicycle

theorem garage_wheels_eq_22 : garage_wheels = 22 := by
  sorry

end garage_wheels_eq_22_l1359_135951


namespace nine_digit_multiply_six_property_l1359_135960

/-- A function that checks if a natural number contains each digit from 1 to 9 exactly once --/
def containsAllDigitsOnce (n : ℕ) : Prop :=
  ∀ d : Fin 9, ∃! p : ℕ, n / 10^p % 10 = d.val + 1

/-- A function that represents the multiplication of a 9-digit number by 6 --/
def multiplyBySix (n : ℕ) : ℕ := n * 6

/-- Theorem stating the existence of 9-digit numbers with the required property --/
theorem nine_digit_multiply_six_property :
  ∃ n : ℕ, 
    100000000 ≤ n ∧ n < 1000000000 ∧
    containsAllDigitsOnce n ∧
    containsAllDigitsOnce (multiplyBySix n) :=
sorry

end nine_digit_multiply_six_property_l1359_135960


namespace ice_cream_combinations_l1359_135943

theorem ice_cream_combinations (n_flavors m_toppings : ℕ) 
  (h_flavors : n_flavors = 5) 
  (h_toppings : m_toppings = 7) : 
  n_flavors * Nat.choose m_toppings 3 = 175 := by
  sorry

#check ice_cream_combinations

end ice_cream_combinations_l1359_135943


namespace exactly_one_tail_in_three_flips_l1359_135922

-- Define a fair coin
def fair_coin_prob : ℝ := 0.5

-- Define the number of flips
def num_flips : ℕ := 3

-- Define the number of tails we want
def num_tails : ℕ := 1

-- Define the binomial coefficient function
def choose (n k : ℕ) : ℕ := 
  Nat.choose n k

-- Define the probability of exactly k successes in n trials
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (choose n k : ℝ) * p^k * (1 - p)^(n - k)

-- Theorem statement
theorem exactly_one_tail_in_three_flips :
  binomial_probability num_flips num_tails fair_coin_prob = 0.375 := by
  sorry

end exactly_one_tail_in_three_flips_l1359_135922


namespace grade12_selection_l1359_135985

/-- Represents the number of students selected from each grade -/
structure GradeSelection where
  grade10 : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Represents the ratio of students in grades 10, 11, and 12 -/
structure GradeRatio where
  k : ℕ
  grade11 : ℕ
  grade12 : ℕ

/-- Theorem: Given the conditions, prove that 360 students were selected from grade 12 -/
theorem grade12_selection
  (total_sample : ℕ)
  (ratio : GradeRatio)
  (selection : GradeSelection)
  (h1 : total_sample = 1200)
  (h2 : ratio = { k := 2, grade11 := 5, grade12 := 3 })
  (h3 : selection.grade10 = 240)
  (h4 : selection.grade10 + selection.grade11 + selection.grade12 = total_sample)
  (h5 : selection.grade10 * (ratio.k + ratio.grade11 + ratio.grade12) = 
        total_sample * ratio.k) :
  selection.grade12 = 360 := by
  sorry


end grade12_selection_l1359_135985


namespace sphere_volume_from_surface_area_l1359_135945

/-- Given a sphere with surface area 256π cm², its volume is (2048/3)π cm³. -/
theorem sphere_volume_from_surface_area :
  ∀ (r : ℝ), 
  (4 * Real.pi * r^2 = 256 * Real.pi) → 
  ((4/3) * Real.pi * r^3 = (2048/3) * Real.pi) :=
by sorry

end sphere_volume_from_surface_area_l1359_135945


namespace minimum_value_of_F_l1359_135907

/-- A function is odd if f(-x) = -f(x) for all x -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem minimum_value_of_F (m n : ℝ) (f g : ℝ → ℝ) :
  (∀ x > 0, f x + n * g x + x + 2 ≤ 8) →
  OddFunction f →
  OddFunction g →
  ∃ c, c = -4 ∧ ∀ x < 0, m * f x + n * g x + x + 2 ≥ c :=
sorry

end minimum_value_of_F_l1359_135907


namespace floor_of_three_point_six_l1359_135932

theorem floor_of_three_point_six : ⌊(3.6 : ℝ)⌋ = 3 := by sorry

end floor_of_three_point_six_l1359_135932


namespace rectangle_with_hole_to_square_l1359_135950

/-- Represents a rectangle with a hole -/
structure RectangleWithHole where
  width : ℝ
  height : ℝ
  hole_width : ℝ
  hole_height : ℝ

/-- Calculates the usable area of a rectangle with a hole -/
def usable_area (r : RectangleWithHole) : ℝ :=
  r.width * r.height - r.hole_width * r.hole_height

/-- Theorem: A 9x12 rectangle with a 1x8 hole can be cut into two equal parts that form a 10x10 square -/
theorem rectangle_with_hole_to_square :
  ∃ (r : RectangleWithHole),
    r.width = 9 ∧
    r.height = 12 ∧
    r.hole_width = 1 ∧
    r.hole_height = 8 ∧
    usable_area r = 100 ∧
    ∃ (side_length : ℝ),
      side_length * side_length = usable_area r ∧
      side_length = 10 :=
by sorry

end rectangle_with_hole_to_square_l1359_135950


namespace circular_cross_section_shapes_l1359_135968

-- Define the geometric shapes
inductive GeometricShape
  | Cube
  | Sphere
  | Cylinder
  | PentagonalPrism

-- Define a function to check if a shape can have a circular cross-section
def canHaveCircularCrossSection (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.Sphere => true
  | GeometricShape.Cylinder => true
  | _ => false

-- Theorem stating that only sphere and cylinder can have circular cross-sections
theorem circular_cross_section_shapes :
  ∀ (shape : GeometricShape),
    canHaveCircularCrossSection shape ↔ (shape = GeometricShape.Sphere ∨ shape = GeometricShape.Cylinder) :=
by sorry

end circular_cross_section_shapes_l1359_135968


namespace bake_sale_group_composition_l1359_135933

theorem bake_sale_group_composition (total : ℕ) (boys : ℕ) : 
  (boys : ℚ) / total = 35 / 100 →
  ((boys - 3 : ℚ) / total) = 40 / 100 →
  boys = 21 := by
sorry

end bake_sale_group_composition_l1359_135933


namespace intersection_nonempty_implies_a_greater_than_one_l1359_135906

-- Define the sets A and B
def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*x + a ≥ 0}

-- Theorem statement
theorem intersection_nonempty_implies_a_greater_than_one (a : ℝ) :
  (∃ x, x ∈ A ∩ B a) → a > 1 := by
  sorry

end intersection_nonempty_implies_a_greater_than_one_l1359_135906


namespace min_diagonals_regular_2017_gon_l1359_135989

/-- The number of vertices in the regular polygon -/
def n : ℕ := 2017

/-- The number of different diagonal lengths in a regular n-gon -/
def num_different_lengths (n : ℕ) : ℕ := (n - 3) / 2

/-- The minimum number of diagonals to select to guarantee two of the same length -/
def min_diagonals_same_length (n : ℕ) : ℕ := num_different_lengths n + 1

theorem min_diagonals_regular_2017_gon :
  min_diagonals_same_length n = 1008 :=
sorry

end min_diagonals_regular_2017_gon_l1359_135989


namespace complex_number_location_l1359_135984

theorem complex_number_location (z : ℂ) (h : z + z * Complex.I = 3 + 2 * Complex.I) : 
  0 < z.re ∧ z.im < 0 := by
  sorry

end complex_number_location_l1359_135984


namespace right_triangle_area_l1359_135973

theorem right_triangle_area (a b c : ℝ) (h1 : a = 5) (h2 : c = 13) (h3 : a^2 + b^2 = c^2) :
  (1/2) * a * b = 30 := by
  sorry

end right_triangle_area_l1359_135973


namespace sqrt_equation_solution_l1359_135962

theorem sqrt_equation_solution (x : ℝ) (h : x > 1) :
  (Real.sqrt (5 * x) / Real.sqrt (3 * (x - 1)) = 2) → x = 12 / 7 := by
  sorry

end sqrt_equation_solution_l1359_135962


namespace cost_price_calculation_l1359_135919

/-- The selling price of the computer table -/
def selling_price : ℝ := 5750

/-- The markup percentage applied by the shop owner -/
def markup_percentage : ℝ := 15

/-- The cost price of the computer table -/
def cost_price : ℝ := 5000

/-- Theorem stating that the given cost price is correct based on the selling price and markup -/
theorem cost_price_calculation : 
  selling_price = cost_price * (1 + markup_percentage / 100) := by
  sorry

end cost_price_calculation_l1359_135919


namespace missing_number_in_proportion_l1359_135958

theorem missing_number_in_proportion : 
  ∃ x : ℚ, (2 : ℚ) / x = (4 : ℚ) / 3 / (10 : ℚ) / 3 ∧ x = 5 := by
  sorry

end missing_number_in_proportion_l1359_135958


namespace solution_l1359_135974

/-- Represents the amount of gold coins each merchant has -/
structure MerchantWealth where
  foma : ℕ
  ierema : ℕ
  yuliy : ℕ

/-- The conditions of the problem -/
def problem_conditions (w : MerchantWealth) : Prop :=
  (w.foma - 70 = w.ierema + 70) ∧ 
  (w.foma - 40 = w.yuliy)

/-- The amount of gold coins Foma should give to Ierema to equalize their wealth -/
def coins_to_equalize (w : MerchantWealth) : ℕ :=
  (w.foma - w.ierema) / 2

/-- Theorem stating the solution to the problem -/
theorem solution (w : MerchantWealth) 
  (h : problem_conditions w) : 
  coins_to_equalize w = 55 := by
  sorry

end solution_l1359_135974


namespace equation_equivalence_l1359_135981

theorem equation_equivalence (a b c : ℕ) 
  (ha : 0 < a ∧ a ≤ 10) 
  (hb : 0 < b ∧ b ≤ 10) 
  (hc : 0 < c ∧ c ≤ 10) : 
  (10 * a + b) * (10 * a + c) = 100 * a^2 + 100 * a + 11 * b * c ↔ b + 11 * c = 10 * a :=
by sorry

end equation_equivalence_l1359_135981


namespace polynomial_independence_l1359_135924

-- Define the polynomials A and B
def A (x a : ℝ) : ℝ := x^2 + a*x
def B (x b : ℝ) : ℝ := 2*b*x^2 - 4*x - 1

-- Define the combined polynomial 2A + B
def combined_polynomial (x a b : ℝ) : ℝ := 2 * A x a + B x b

-- Theorem statement
theorem polynomial_independence (a b : ℝ) : 
  (∀ x : ℝ, ∃ c : ℝ, combined_polynomial x a b = c) ↔ (a = 2 ∧ b = -1) := by
  sorry

end polynomial_independence_l1359_135924


namespace local_minima_dense_of_continuous_nowhere_monotone_l1359_135972

open Set
open Topology
open Function

/-- A function is nowhere monotone if it is not monotone on any subinterval -/
def NowhereMonotone (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y z, a ≤ x ∧ x < y ∧ y < z ∧ z ≤ b →
    (f x < f y ∧ f y > f z) ∨ (f x > f y ∧ f y < f z)

/-- The set of local minima of a function -/
def LocalMinima (f : ℝ → ℝ) : Set ℝ :=
  {x | ∃ ε > 0, ∀ y, |y - x| < ε → f y ≥ f x}

theorem local_minima_dense_of_continuous_nowhere_monotone
  (f : ℝ → ℝ)
  (hf_cont : ContinuousOn f (Icc 0 1))
  (hf_nm : NowhereMonotone f 0 1) :
  Dense (LocalMinima f ∩ Icc 0 1) :=
sorry

end local_minima_dense_of_continuous_nowhere_monotone_l1359_135972


namespace diamond_equation_solution_l1359_135993

-- Define the diamond operation
def diamond (A B : ℝ) : ℝ := 4 * A - 3 * B + 7

-- State the theorem
theorem diamond_equation_solution :
  ∃! A : ℝ, diamond A 10 = 57 ∧ A = 20 := by sorry

end diamond_equation_solution_l1359_135993


namespace joe_fruit_probability_l1359_135971

def num_fruits : ℕ := 4
def num_meals : ℕ := 3

theorem joe_fruit_probability :
  let p_same := (1 / num_fruits : ℚ) ^ num_meals * num_fruits
  1 - p_same = 15 / 16 := by sorry

end joe_fruit_probability_l1359_135971


namespace music_program_band_members_l1359_135983

theorem music_program_band_members :
  ∀ (total_students : ℕ) 
    (band_percentage : ℚ) 
    (chorus_percentage : ℚ) 
    (band_members : ℕ) 
    (chorus_members : ℕ),
  total_students = 36 →
  band_percentage = 1/5 →
  chorus_percentage = 1/4 →
  band_members + chorus_members = total_students →
  (band_percentage * band_members : ℚ) = (chorus_percentage * chorus_members : ℚ) →
  band_members = 16 := by
sorry

end music_program_band_members_l1359_135983


namespace problem_solution_l1359_135997

def p (m : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ a ≠ b ∧ 
  ∀ x y : ℝ, x^2 / (4 - m) + y^2 / m = 1 ↔ (x / a)^2 + (y / b)^2 = 1

def q (m : ℝ) : Prop := ∀ x : ℝ, x^2 + 2*m*x + 1 > 0

def S (m : ℝ) : Prop := ∃ x : ℝ, m*x^2 + 2*m*x + 2 - m = 0

theorem problem_solution :
  (∀ m : ℝ, S m → (m < 0 ∨ m ≥ 1)) ∧
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(q m) → 1 ≤ m ∧ m < 2) :=
sorry

end problem_solution_l1359_135997


namespace light_toggle_theorem_l1359_135937

/-- Represents the state of a light (on or off) -/
inductive LightState
| Off
| On

/-- Represents a position in the 5x5 grid -/
structure Position where
  row : Fin 5
  col : Fin 5

/-- The type of the 5x5 grid of lights -/
def Grid := Fin 5 → Fin 5 → LightState

/-- Toggles a light and its adjacent lights in the same row and column -/
def toggle (grid : Grid) (pos : Position) : Grid := sorry

/-- Checks if exactly one light is on in the grid -/
def exactlyOneOn (grid : Grid) : Prop := sorry

/-- The set of possible positions for the single on light -/
def possiblePositions : Set Position :=
  {⟨2, 2⟩, ⟨2, 4⟩, ⟨3, 3⟩, ⟨4, 2⟩, ⟨4, 4⟩}

/-- The initial grid with all lights off -/
def initialGrid : Grid := fun _ _ => LightState.Off

theorem light_toggle_theorem :
  ∀ (finalGrid : Grid),
    (∃ (toggleSequence : List Position),
      finalGrid = toggleSequence.foldl toggle initialGrid) →
    exactlyOneOn finalGrid →
    ∃ (pos : Position), finalGrid pos.row pos.col = LightState.On ∧ pos ∈ possiblePositions :=
sorry

end light_toggle_theorem_l1359_135937


namespace base_5_103_eq_28_l1359_135917

/-- Converts a list of digits in base b to its decimal representation -/
def to_decimal (digits : List Nat) (b : Nat) : Nat :=
  digits.enum.foldr (fun (i, d) acc => acc + d * b^i) 0

/-- The decimal representation of 103 in base 5 -/
def base_5_103 : Nat := to_decimal [3, 0, 1] 5

theorem base_5_103_eq_28 : base_5_103 = 28 := by sorry

end base_5_103_eq_28_l1359_135917


namespace lawn_width_proof_l1359_135969

theorem lawn_width_proof (length width : ℝ) (road_width cost_per_sqm total_cost : ℝ) : 
  length = 80 →
  road_width = 15 →
  cost_per_sqm = 3 →
  total_cost = 5625 →
  (road_width * width + road_width * length - road_width * road_width) * cost_per_sqm = total_cost →
  width = 60 := by
sorry

end lawn_width_proof_l1359_135969


namespace largest_common_divisor_36_60_l1359_135912

theorem largest_common_divisor_36_60 : 
  ∃ (n : ℕ), n > 0 ∧ n ∣ 36 ∧ n ∣ 60 ∧ ∀ (m : ℕ), m > 0 ∧ m ∣ 36 ∧ m ∣ 60 → m ≤ n :=
by
  sorry

end largest_common_divisor_36_60_l1359_135912


namespace solve_maple_tree_price_l1359_135959

/-- Represents the problem of calculating the price per maple tree --/
def maple_tree_price_problem (initial_cash : ℕ) (cypress_trees : ℕ) (pine_trees : ℕ) (maple_trees : ℕ)
  (cypress_price : ℕ) (pine_price : ℕ) (cabin_price : ℕ) (remaining_cash : ℕ) : Prop :=
  let total_after_sale := cabin_price + remaining_cash
  let total_from_trees := total_after_sale - initial_cash
  let cypress_revenue := cypress_trees * cypress_price
  let pine_revenue := pine_trees * pine_price
  let maple_revenue := total_from_trees - cypress_revenue - pine_revenue
  maple_revenue / maple_trees = 300

/-- The main theorem stating the solution to the problem --/
theorem solve_maple_tree_price :
  maple_tree_price_problem 150 20 600 24 100 200 129000 350 := by
  sorry

#check solve_maple_tree_price

end solve_maple_tree_price_l1359_135959
