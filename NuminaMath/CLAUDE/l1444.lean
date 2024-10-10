import Mathlib

namespace smallest_multiple_thirty_two_works_smallest_x_is_32_l1444_144443

theorem smallest_multiple (x : ℕ) : x > 0 ∧ 800 ∣ (450 * x) → x ≥ 32 :=
by sorry

theorem thirty_two_works : 800 ∣ (450 * 32) :=
by sorry

theorem smallest_x_is_32 : ∃ x : ℕ, x > 0 ∧ 800 ∣ (450 * x) ∧ ∀ y : ℕ, (y > 0 ∧ 800 ∣ (450 * y)) → x ≤ y :=
by sorry

end smallest_multiple_thirty_two_works_smallest_x_is_32_l1444_144443


namespace amanda_remaining_money_l1444_144464

/-- Calculates the remaining money after purchases -/
def remaining_money (initial_amount : ℕ) (item1_cost : ℕ) (item1_quantity : ℕ) (item2_cost : ℕ) : ℕ :=
  initial_amount - (item1_cost * item1_quantity + item2_cost)

/-- Theorem: Amanda's remaining money after purchases -/
theorem amanda_remaining_money :
  remaining_money 50 9 2 25 = 7 := by
  sorry

end amanda_remaining_money_l1444_144464


namespace consecutive_integers_with_prime_factors_l1444_144475

theorem consecutive_integers_with_prime_factors 
  (n s m : ℕ+) : 
  ∃ (x : ℕ), ∀ (j : ℕ), j ∈ Finset.range m → 
    (∃ (p : Finset ℕ), p.card = n ∧ 
      (∀ q ∈ p, Nat.Prime q ∧ 
        (∃ (k : ℕ), k ≥ s ∧ (q^k : ℕ) ∣ (x + j)))) :=
sorry

end consecutive_integers_with_prime_factors_l1444_144475


namespace f_neg_two_eq_five_l1444_144421

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the property of being an even function
def is_even (g : ℝ → ℝ) : Prop :=
  ∀ x, g x = g (-x)

-- State the theorem
theorem f_neg_two_eq_five
  (h1 : is_even (λ x => f x + x))
  (h2 : f 2 = 1) :
  f (-2) = 5 :=
sorry

end f_neg_two_eq_five_l1444_144421


namespace horse_cow_price_system_l1444_144455

/-- Represents the price of a horse in yuan -/
def horse_price : ℝ := sorry

/-- Represents the price of a cow in yuan -/
def cow_price : ℝ := sorry

/-- The system of equations correctly represents the given conditions about horse and cow prices -/
theorem horse_cow_price_system :
  (2 * horse_price + cow_price - 10000 = (1/2) * horse_price) ∧
  (10000 - (horse_price + 2 * cow_price) = (1/2) * cow_price) := by
  sorry

end horse_cow_price_system_l1444_144455


namespace root_of_log_equation_l1444_144416

theorem root_of_log_equation :
  ∃! x : ℝ, x > 1 ∧ Real.log x = x - 5 ∧ 5 < x ∧ x < 6 := by sorry

end root_of_log_equation_l1444_144416


namespace sum_inequality_and_equality_condition_l1444_144480

theorem sum_inequality_and_equality_condition (a b c : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (h : (a + 1) * (b + 1) * (c + 1) = 8) : 
  a + b + c ≥ 3 ∧ (a + b + c = 3 ↔ a = 1 ∧ b = 1 ∧ c = 1) := by
  sorry

end sum_inequality_and_equality_condition_l1444_144480


namespace difference_of_squares_factorization_l1444_144423

theorem difference_of_squares_factorization (y : ℝ) : 64 - 16 * y^2 = 16 * (2 - y) * (2 + y) := by
  sorry

end difference_of_squares_factorization_l1444_144423


namespace first_divisor_problem_l1444_144483

theorem first_divisor_problem (x : ℕ) : x = 7 ↔ 
  x > 0 ∧ 
  x ≠ 15 ∧ 
  184 % x = 2 ∧ 
  184 % 15 = 4 ∧ 
  ∀ y : ℕ, y > 0 ∧ y < x ∧ y ≠ 15 → 184 % y ≠ 2 := by
sorry

end first_divisor_problem_l1444_144483


namespace light_glow_start_time_l1444_144470

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat

/-- Converts a Time to total seconds -/
def Time.toSeconds (t : Time) : Nat :=
  t.hours * 3600 + t.minutes * 60 + t.seconds

/-- Converts total seconds to a Time -/
def Time.fromSeconds (s : Nat) : Time :=
  { hours := s / 3600
  , minutes := (s % 3600) / 60
  , seconds := s % 60 }

/-- Subtracts two Times, assuming t1 ≥ t2 -/
def Time.sub (t1 t2 : Time) : Time :=
  Time.fromSeconds (t1.toSeconds - t2.toSeconds)

theorem light_glow_start_time 
  (glow_interval : Nat) 
  (glow_count : Nat) 
  (end_time : Time) : 
  glow_interval = 21 →
  glow_count = 236 →
  end_time = { hours := 3, minutes := 20, seconds := 47 } →
  Time.sub end_time (Time.fromSeconds (glow_interval * glow_count)) = 
    { hours := 1, minutes := 58, seconds := 11 } :=
by sorry

end light_glow_start_time_l1444_144470


namespace stella_annual_income_l1444_144448

def monthly_income : ℕ := 4919
def unpaid_leave_months : ℕ := 2
def months_in_year : ℕ := 12

theorem stella_annual_income :
  (monthly_income * (months_in_year - unpaid_leave_months)) = 49190 := by
  sorry

end stella_annual_income_l1444_144448


namespace math_contest_participants_l1444_144466

theorem math_contest_participants : ∃ n : ℕ, 
  n > 0 ∧ 
  n = n / 3 + n / 4 + n / 5 + 26 ∧ 
  n = 120 := by
sorry

end math_contest_participants_l1444_144466


namespace bamboo_volume_sum_l1444_144447

/-- Given a sequence of 9 terms forming an arithmetic progression,
    where the sum of the first 4 terms is 3 and the sum of the last 3 terms is 4,
    prove that the sum of the 2nd, 3rd, and 8th terms is 17/6. -/
theorem bamboo_volume_sum (a : Fin 9 → ℚ) 
  (arithmetic_seq : ∀ i j k : Fin 9, a (i + 1) - a i = a (j + 1) - a j)
  (sum_first_four : a 0 + a 1 + a 2 + a 3 = 3)
  (sum_last_three : a 6 + a 7 + a 8 = 4) :
  a 1 + a 2 + a 7 = 17/6 := by
  sorry

end bamboo_volume_sum_l1444_144447


namespace apple_count_l1444_144484

theorem apple_count (blue_apples : ℕ) (yellow_apples : ℕ) : 
  blue_apples = 5 →
  yellow_apples = 2 * blue_apples →
  (blue_apples + yellow_apples) - ((blue_apples + yellow_apples) / 5) = 12 := by
sorry

end apple_count_l1444_144484


namespace replaced_person_weight_l1444_144417

/-- Given a group of 10 persons, if replacing one person with a new person
    weighing 110 kg increases the average weight by 5 kg, then the weight
    of the replaced person is 60 kg. -/
theorem replaced_person_weight
  (initial_count : ℕ)
  (new_person_weight : ℝ)
  (average_increase : ℝ)
  (h_initial_count : initial_count = 10)
  (h_new_person_weight : new_person_weight = 110)
  (h_average_increase : average_increase = 5)
  : ∃ (initial_average : ℝ) (replaced_weight : ℝ),
    initial_count * (initial_average + average_increase) =
    initial_count * initial_average + new_person_weight - replaced_weight ∧
    replaced_weight = 60 := by
  sorry

end replaced_person_weight_l1444_144417


namespace distinct_hands_count_l1444_144465

def special_deck_size : ℕ := 60
def hand_size : ℕ := 13

theorem distinct_hands_count : (special_deck_size.choose hand_size) = 75287520 := by
  sorry

end distinct_hands_count_l1444_144465


namespace salary_reduction_percentage_l1444_144439

theorem salary_reduction_percentage 
  (original : ℝ) 
  (reduced : ℝ) 
  (increase_percentage : ℝ) 
  (h1 : increase_percentage = 38.88888888888889)
  (h2 : reduced * (1 + increase_percentage / 100) = original) :
  ∃ (reduction_percentage : ℝ), 
    reduction_percentage = 28 ∧ 
    reduced = original * (1 - reduction_percentage / 100) := by
  sorry

end salary_reduction_percentage_l1444_144439


namespace qadi_advice_leads_to_winner_l1444_144432

/-- Represents a son in the problem -/
structure Son where
  camel : Nat  -- Each son has a camel, represented by a natural number

/-- Represents the state of the race -/
structure RaceState where
  son1 : Son
  son2 : Son
  winner : Option Son

/-- The function that determines the winner based on arrival times -/
def determineWinner (arrivalTime1 arrivalTime2 : Nat) : Option Son :=
  if arrivalTime1 > arrivalTime2 then some { camel := 1 }
  else if arrivalTime2 > arrivalTime1 then some { camel := 2 }
  else none

/-- The function that simulates the race -/
def race (initialState : RaceState) : RaceState :=
  let arrivalTime1 := initialState.son1.camel
  let arrivalTime2 := initialState.son2.camel
  { initialState with winner := determineWinner arrivalTime1 arrivalTime2 }

/-- The function that swaps the camels -/
def swapCamels (state : RaceState) : RaceState :=
  { state with
    son1 := { camel := state.son2.camel }
    son2 := { camel := state.son1.camel } }

/-- The main theorem to prove -/
theorem qadi_advice_leads_to_winner (initialState : RaceState) :
  (race (swapCamels initialState)).winner.isSome :=
sorry


end qadi_advice_leads_to_winner_l1444_144432


namespace logarithm_expression_equality_l1444_144408

theorem logarithm_expression_equality : 
  2 * Real.log 2 / Real.log 3 - Real.log (32 / 9) / Real.log 3 + Real.log 8 / Real.log 3 - 5 ^ (Real.log 3 / Real.log 5) = -1 := by
  sorry

end logarithm_expression_equality_l1444_144408


namespace absolute_value_inequality_l1444_144488

theorem absolute_value_inequality (a : ℝ) :
  (∃ x : ℝ, |x + 1| + |x - 3| ≤ a) → a ≥ 4 := by
  sorry

end absolute_value_inequality_l1444_144488


namespace smallest_enclosing_sphere_radius_l1444_144429

-- Define a sphere with center and radius
structure Sphere where
  center : ℝ × ℝ × ℝ
  radius : ℝ

-- Define the eight spheres
def octantSpheres : List Sphere :=
  [⟨(2, 2, 2), 2⟩, ⟨(-2, 2, 2), 2⟩, ⟨(2, -2, 2), 2⟩, ⟨(2, 2, -2), 2⟩,
   ⟨(-2, -2, 2), 2⟩, ⟨(-2, 2, -2), 2⟩, ⟨(2, -2, -2), 2⟩, ⟨(-2, -2, -2), 2⟩]

-- Function to check if a sphere is tangent to coordinate planes
def isTangentToCoordinatePlanes (s : Sphere) : Prop :=
  let (x, y, z) := s.center
  (|x| = s.radius ∨ |y| = s.radius ∨ |z| = s.radius)

-- Function to check if a sphere contains another sphere
def containsSphere (outer : Sphere) (inner : Sphere) : Prop :=
  let (x₁, y₁, z₁) := outer.center
  let (x₂, y₂, z₂) := inner.center
  ((x₁ - x₂)^2 + (y₁ - y₂)^2 + (z₁ - z₂)^2)^(1/2) + inner.radius ≤ outer.radius

-- Theorem statement
theorem smallest_enclosing_sphere_radius :
  ∃ (r : ℝ), r = 2 + 2 * Real.sqrt 3 ∧
  (∀ s ∈ octantSpheres, isTangentToCoordinatePlanes s) ∧
  (∀ r' : ℝ, r' < r →
    ∃ s ∈ octantSpheres, ¬containsSphere ⟨(0, 0, 0), r'⟩ s) ∧
  (∀ s ∈ octantSpheres, containsSphere ⟨(0, 0, 0), r⟩ s) := by
  sorry

end smallest_enclosing_sphere_radius_l1444_144429


namespace scientific_notation_of_3790000_l1444_144469

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  property : 1 ≤ coefficient ∧ coefficient < 10

/-- Function to convert a positive real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

/-- Theorem stating that 3,790,000 in scientific notation is 3.79 × 10^6 -/
theorem scientific_notation_of_3790000 :
  toScientificNotation 3790000 = ScientificNotation.mk 3.79 6 (by norm_num) :=
sorry

end scientific_notation_of_3790000_l1444_144469


namespace akiras_weight_l1444_144457

/-- Given the weights of pairs of people, determine Akira's weight -/
theorem akiras_weight (akira jamie rabia : ℕ) 
  (h1 : akira + jamie = 101)
  (h2 : akira + rabia = 91)
  (h3 : rabia + jamie = 88) :
  akira = 52 := by
  sorry

end akiras_weight_l1444_144457


namespace circle_trajectory_l1444_144411

/-- A circle with equation x^2 + y^2 - ax + 2y + 1 = 0 -/
def circle1 (a : ℝ) (x y : ℝ) : Prop :=
  x^2 + y^2 - a*x + 2*y + 1 = 0

/-- The unit circle with equation x^2 + y^2 = 1 -/
def circle2 (x y : ℝ) : Prop :=
  x^2 + y^2 = 1

/-- The line y = x - l -/
def symmetry_line (l : ℝ) (x y : ℝ) : Prop :=
  y = x - l

/-- Circle P passes through the point C(-a, a) -/
def circle_p_passes_through (a : ℝ) (x y : ℝ) : Prop :=
  (x + a)^2 + (y - a)^2 = x^2 + y^2

/-- Circle P is tangent to the y-axis -/
def circle_p_tangent_y_axis (x y : ℝ) : Prop :=
  x^2 + y^2 = x^2

/-- The trajectory equation of the center P -/
def trajectory_equation (x y : ℝ) : Prop :=
  y^2 + 4*x - 4*y + 8 = 0

theorem circle_trajectory :
  ∀ (a l : ℝ) (x y : ℝ),
  (∃ (x₁ y₁ : ℝ), circle1 a x₁ y₁) →
  (∃ (x₂ y₂ : ℝ), circle2 x₂ y₂) →
  (∀ (x₁ y₁ x₂ y₂ : ℝ), circle1 a x₁ y₁ → circle2 x₂ y₂ → 
    ∃ (x₃ y₃ : ℝ), symmetry_line l x₃ y₃ ∧ 
    (x₃ = (x₁ + x₂) / 2 ∧ y₃ = (y₁ + y₂) / 2)) →
  circle_p_passes_through a x y →
  circle_p_tangent_y_axis x y →
  trajectory_equation x y :=
by sorry

end circle_trajectory_l1444_144411


namespace sum_even_odd_is_odd_l1444_144412

def P : Set Int := {x | ∃ k, x = 2 * k}
def Q : Set Int := {x | ∃ k, x = 2 * k + 1}
def R : Set Int := {x | ∃ k, x = 4 * k + 1}

theorem sum_even_odd_is_odd (a b : Int) (ha : a ∈ P) (hb : b ∈ Q) : a + b ∈ Q := by
  sorry

end sum_even_odd_is_odd_l1444_144412


namespace sequence_first_term_l1444_144477

/-- Given a sequence {a_n} defined by a_n = (√2)^(n-2), prove that a_1 = √2/2 -/
theorem sequence_first_term (a : ℕ → ℝ) (h : ∀ n, a n = (Real.sqrt 2) ^ (n - 2)) :
  a 1 = Real.sqrt 2 / 2 := by
  sorry

end sequence_first_term_l1444_144477


namespace meaningful_fraction_l1444_144458

theorem meaningful_fraction (x : ℝ) :
  (2 * x - 1 ≠ 0) ↔ (x ≠ 1/2) := by sorry

end meaningful_fraction_l1444_144458


namespace john_money_left_l1444_144450

/-- Calculates the amount of money John has left after giving some to his parents -/
def money_left (initial : ℚ) (mother_fraction : ℚ) (father_fraction : ℚ) : ℚ :=
  initial - (initial * mother_fraction) - (initial * father_fraction)

/-- Theorem stating that John has $65 left after giving money to his parents -/
theorem john_money_left :
  money_left 200 (3/8) (3/10) = 65 := by
  sorry

#eval money_left 200 (3/8) (3/10)

end john_money_left_l1444_144450


namespace inner_square_side_length_l1444_144436

/-- A square with side length 2 -/
structure Square :=
  (A B C D : ℝ × ℝ)
  (is_square : A = (0, 2) ∧ B = (2, 2) ∧ C = (2, 0) ∧ D = (0, 0))

/-- A smaller square inside the main square -/
structure InnerSquare (outer : Square) :=
  (P Q R S : ℝ × ℝ)
  (P_midpoint : P = (1, 2))
  (S_on_BC : S.1 = 2)
  (is_square : (P.1 - S.1)^2 + (P.2 - S.2)^2 = (Q.1 - R.1)^2 + (Q.2 - R.2)^2)

/-- The theorem to be proved -/
theorem inner_square_side_length (outer : Square) (inner : InnerSquare outer) :
  Real.sqrt ((inner.P.1 - inner.S.1)^2 + (inner.P.2 - inner.S.2)^2) = 1 := by
  sorry

end inner_square_side_length_l1444_144436


namespace equation_one_real_root_l1444_144463

/-- The equation x + √(x-4) = 6 has exactly one real root. -/
theorem equation_one_real_root :
  ∃! x : ℝ, x + Real.sqrt (x - 4) = 6 := by
  sorry

end equation_one_real_root_l1444_144463


namespace monochromatic_unit_area_triangle_exists_l1444_144427

-- Define a color type
inductive Color
| Red
| Green
| Blue

-- Define a point in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring of the plane
def Coloring := Point → Color

-- Define a triangle
structure Triangle where
  a : Point
  b : Point
  c : Point

-- Function to calculate the area of a triangle
def triangleArea (t : Triangle) : ℝ := sorry

-- Define what it means for a triangle to be monochromatic
def isMonochromatic (t : Triangle) (coloring : Coloring) : Prop :=
  coloring t.a = coloring t.b ∧ coloring t.b = coloring t.c

-- The main theorem
theorem monochromatic_unit_area_triangle_exists (coloring : Coloring) :
  ∃ t : Triangle, triangleArea t = 1 ∧ isMonochromatic t coloring := by sorry

end monochromatic_unit_area_triangle_exists_l1444_144427


namespace largest_common_remainder_l1444_144482

theorem largest_common_remainder :
  ∀ n : ℕ, 2013 ≤ n ∧ n ≤ 2156 →
  (∃ r : ℕ, n % 5 = r ∧ n % 11 = r ∧ n % 13 = r) →
  (∀ s : ℕ, (∃ m : ℕ, 2013 ≤ m ∧ m ≤ 2156 ∧ m % 5 = s ∧ m % 11 = s ∧ m % 13 = s) → s ≤ 4) :=
by sorry

#check largest_common_remainder

end largest_common_remainder_l1444_144482


namespace max_k_for_f_geq_kx_l1444_144449

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.sin x

theorem max_k_for_f_geq_kx :
  ∀ k : ℝ, (∀ x : ℝ, x ∈ Set.Icc 0 (Real.pi / 2) → f x ≥ k * x) ↔ k ≤ 1 := by
  sorry

end max_k_for_f_geq_kx_l1444_144449


namespace continuous_thin_stripe_probability_l1444_144489

-- Define the cube and its properties
def Cube := Fin 6

-- Define stripe properties
inductive StripeThickness
| thin
| thick

def StripeOrientation := Fin 4

structure Stripe :=
  (thickness : StripeThickness)
  (orientation : StripeOrientation)

def CubeConfiguration := Cube → Stripe

-- Define a function to check if a configuration has a continuous thin stripe
def hasContinuousThinStripe (config : CubeConfiguration) : Prop :=
  sorry -- Implementation details omitted

-- Define the probability space
def totalConfigurations : ℕ := 8^6

-- Define the number of favorable configurations
def favorableConfigurations : ℕ := 6144

-- Theorem statement
theorem continuous_thin_stripe_probability :
  (favorableConfigurations : ℚ) / totalConfigurations = 3 / 128 :=
sorry

end continuous_thin_stripe_probability_l1444_144489


namespace problem_solution_l1444_144445

theorem problem_solution : (3358 / 46) - 27 = 46 := by
  sorry

end problem_solution_l1444_144445


namespace addition_commutative_example_l1444_144428

theorem addition_commutative_example : 73 + 93 + 27 = 73 + 27 + 93 := by
  sorry

end addition_commutative_example_l1444_144428


namespace deer_count_l1444_144418

theorem deer_count (total : ℕ) 
  (h1 : (total : ℚ) * (1/10) = (total : ℚ) * (1/10))  -- 10% of deer have 8 antlers
  (h2 : (total : ℚ) * (1/10) * (1/4) = (total : ℚ) * (1/10) * (1/4))  -- 25% of 8-antlered deer have albino fur
  (h3 : (total : ℚ) * (1/10) * (1/4) = 23)  -- There are 23 albino 8-antlered deer
  : total = 920 :=
by sorry

end deer_count_l1444_144418


namespace zero_exponent_l1444_144452

theorem zero_exponent (n : ℤ) (h : n ≠ 0) : n^0 = 1 := by
  sorry

end zero_exponent_l1444_144452


namespace product_scaling_l1444_144430

theorem product_scaling (a b c : ℝ) (h : 14.97 * 46 = 688.62) :
  1.497 * 4.6 = 6.8862 := by
  sorry

end product_scaling_l1444_144430


namespace max_intersections_three_circles_one_line_l1444_144491

/-- The maximum number of intersection points between three circles -/
def max_circle_intersections : ℕ := 6

/-- The maximum number of intersection points between three circles and a line -/
def max_circle_line_intersections : ℕ := 6

/-- The maximum number of intersection points between 3 different circles and 1 straight line -/
theorem max_intersections_three_circles_one_line :
  max_circle_intersections + max_circle_line_intersections = 12 := by sorry

end max_intersections_three_circles_one_line_l1444_144491


namespace park_area_l1444_144460

/-- A rectangular park with specific length-width relationship and perimeter --/
structure RectangularPark where
  width : ℝ
  length : ℝ
  length_eq : length = 3 * width + 30
  perimeter_eq : 2 * (length + width) = 780

/-- The area of the rectangular park is 27000 square meters --/
theorem park_area (park : RectangularPark) : park.length * park.width = 27000 := by
  sorry

end park_area_l1444_144460


namespace binary_addition_subtraction_l1444_144486

/-- Represents a binary number as a list of bits (0 or 1) -/
def BinaryNumber := List Bool

/-- Converts a binary number to its decimal representation -/
def binaryToDecimal (b : BinaryNumber) : ℕ :=
  b.foldl (fun acc bit => 2 * acc + if bit then 1 else 0) 0

/-- The binary number 10110₂ -/
def b1 : BinaryNumber := [true, false, true, true, false]

/-- The binary number 1101₂ -/
def b2 : BinaryNumber := [true, true, false, true]

/-- The binary number 1010₂ -/
def b3 : BinaryNumber := [true, false, true, false]

/-- The binary number 1110₂ -/
def b4 : BinaryNumber := [true, true, true, false]

/-- The binary number 11111₂ (the expected result) -/
def result : BinaryNumber := [true, true, true, true, true]

theorem binary_addition_subtraction :
  binaryToDecimal b1 + binaryToDecimal b2 - binaryToDecimal b3 + binaryToDecimal b4 =
  binaryToDecimal result := by
  sorry

end binary_addition_subtraction_l1444_144486


namespace external_angle_ninety_degrees_l1444_144441

theorem external_angle_ninety_degrees (a b c : ℝ) (h1 : a = 40) (h2 : b = 50) 
  (h3 : a + b + c = 180) (x : ℝ) (h4 : x + c = 180) : x = 90 := by
  sorry

end external_angle_ninety_degrees_l1444_144441


namespace pigeon_percentage_among_non_sparrows_l1444_144405

def bird_distribution (pigeon sparrow crow dove : ℝ) : Prop :=
  pigeon + sparrow + crow + dove = 100 ∧
  pigeon = 40 ∧
  sparrow = 20 ∧
  crow = 15 ∧
  dove = 25

theorem pigeon_percentage_among_non_sparrows 
  (pigeon sparrow crow dove : ℝ) 
  (h : bird_distribution pigeon sparrow crow dove) : 
  (pigeon / (pigeon + crow + dove)) * 100 = 50 := by
  sorry

end pigeon_percentage_among_non_sparrows_l1444_144405


namespace least_tiles_cover_room_l1444_144415

def room_length : ℕ := 624
def room_width : ℕ := 432

theorem least_tiles_cover_room (length : ℕ) (width : ℕ) 
  (h1 : length = room_length) (h2 : width = room_width) : 
  ∃ (tile_size : ℕ), 
    tile_size > 0 ∧ 
    length % tile_size = 0 ∧ 
    width % tile_size = 0 ∧ 
    (length / tile_size) * (width / tile_size) = 117 ∧
    ∀ (other_size : ℕ), 
      other_size > 0 → 
      length % other_size = 0 → 
      width % other_size = 0 → 
      other_size ≤ tile_size :=
by sorry

end least_tiles_cover_room_l1444_144415


namespace adlai_animal_legs_l1444_144494

/-- The number of legs for each animal type --/
def dogsLegs : ℕ := 4
def chickenLegs : ℕ := 2
def catsLegs : ℕ := 4
def spidersLegs : ℕ := 8
def octopusLegs : ℕ := 0

/-- Adlai's animal collection --/
def numDogs : ℕ := 2
def numChickens : ℕ := 1
def numCats : ℕ := 3
def numSpiders : ℕ := 4
def numOctopuses : ℕ := 5

/-- The total number of animal legs in Adlai's collection --/
def totalLegs : ℕ := 
  numDogs * dogsLegs + 
  numChickens * chickenLegs + 
  numCats * catsLegs + 
  numSpiders * spidersLegs + 
  numOctopuses * octopusLegs

theorem adlai_animal_legs : totalLegs = 54 := by
  sorry

end adlai_animal_legs_l1444_144494


namespace root_product_expression_l1444_144442

theorem root_product_expression (p q : ℝ) (α β γ δ : ℂ) : 
  (α^2 + p*α + 2 = 0) → 
  (β^2 + p*β + 2 = 0) → 
  (γ^2 + q*γ + 3 = 0) → 
  (δ^2 + q*δ + 3 = 0) → 
  (α - γ)*(β - γ)*(α + δ)*(β + δ) = 3*(q^2 - p^2) := by
sorry

end root_product_expression_l1444_144442


namespace chosen_number_l1444_144496

theorem chosen_number (x : ℝ) : x / 5 - 154 = 6 → x = 800 := by
  sorry

end chosen_number_l1444_144496


namespace only_prime_three_satisfies_l1444_144431

def set_A (p : ℕ) : Set ℕ :=
  {x | ∃ k : ℕ, 1 ≤ k ∧ k ≤ (p - 1) / 2 ∧ x = (k^2 + 1) % p}

def set_B (p g : ℕ) : Set ℕ :=
  {x | ∃ k : ℕ, 1 ≤ k ∧ k ≤ (p - 1) / 2 ∧ x = (g^k) % p}

theorem only_prime_three_satisfies (p : ℕ) :
  (Nat.Prime p ∧ Odd p ∧ (∃ g : ℕ, set_A p = set_B p g)) ↔ p = 3 :=
sorry

end only_prime_three_satisfies_l1444_144431


namespace cubic_roots_sum_l1444_144410

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - 7*a^2 + 5*a + 2 = 0) →
  (b^3 - 7*b^2 + 5*b + 2 = 0) →
  (c^3 - 7*c^2 + 5*c + 2 = 0) →
  (a / (b*c + 1) + b / (a*c + 1) + c / (a*b + 1) = 15/2) := by
sorry

end cubic_roots_sum_l1444_144410


namespace pinecone_problem_l1444_144407

theorem pinecone_problem :
  ∃! n : ℕ, n < 350 ∧ 
  2 ∣ n ∧ 3 ∣ n ∧ 4 ∣ n ∧ 5 ∣ n ∧ 6 ∣ n ∧ 9 ∣ n ∧ 
  ¬(7 ∣ n) ∧ ¬(8 ∣ n) ∧
  n = 180 :=
by
  sorry

end pinecone_problem_l1444_144407


namespace room_tiles_theorem_l1444_144472

/-- Given a room with length and width in centimeters, 
    calculate the least number of square tiles required to cover the floor. -/
def leastNumberOfTiles (length width : ℕ) : ℕ :=
  let tileSize := Nat.gcd length width
  (length / tileSize) * (width / tileSize)

/-- Theorem stating that for a room of 624 cm by 432 cm, 
    the least number of square tiles required is 117. -/
theorem room_tiles_theorem :
  leastNumberOfTiles 624 432 = 117 := by
  sorry

#eval leastNumberOfTiles 624 432

end room_tiles_theorem_l1444_144472


namespace square_sum_constant_l1444_144476

theorem square_sum_constant (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(5 - x) + (5 - x)^2 = 49 := by
  sorry

end square_sum_constant_l1444_144476


namespace horner_evaluation_f_5_l1444_144493

def f (x : ℝ) : ℝ := 2*x^7 - 9*x^6 + 5*x^5 - 49*x^4 - 5*x^3 + 2*x^2 + x + 1

theorem horner_evaluation_f_5 : f 5 = 56 := by sorry

end horner_evaluation_f_5_l1444_144493


namespace z_in_first_quadrant_iff_m_gt_two_l1444_144468

-- Define the complex number z as a function of m
def z (m : ℝ) : ℂ := (1 + Complex.I) * (m - 2 * Complex.I)

-- Define the condition for z to be in the first quadrant
def is_in_first_quadrant (z : ℂ) : Prop :=
  z.re > 0 ∧ z.im > 0

-- Theorem statement
theorem z_in_first_quadrant_iff_m_gt_two (m : ℝ) :
  is_in_first_quadrant (z m) ↔ m > 2 := by
  sorry


end z_in_first_quadrant_iff_m_gt_two_l1444_144468


namespace product_of_algebraic_expressions_l1444_144422

theorem product_of_algebraic_expressions (a b : ℝ) :
  (-8 * a * b) * ((3 / 4) * a^2 * b) = -6 * a^3 * b^2 := by sorry

end product_of_algebraic_expressions_l1444_144422


namespace seat_arrangement_count_l1444_144413

/-- The number of ways to select and arrange 3 people from a group of 7 --/
def seatArrangements : ℕ := 70

/-- The number of people in the class --/
def totalPeople : ℕ := 7

/-- The number of people to be rearranged --/
def peopleToRearrange : ℕ := 3

/-- The number of ways to arrange 3 people in a circle (considering rotations as identical) --/
def circularArrangements : ℕ := 2

theorem seat_arrangement_count :
  seatArrangements = circularArrangements * (Nat.choose totalPeople peopleToRearrange) := by
  sorry

end seat_arrangement_count_l1444_144413


namespace exam_students_count_l1444_144453

theorem exam_students_count :
  let first_division_percent : ℚ := 27/100
  let second_division_percent : ℚ := 54/100
  let just_passed_count : ℕ := 57
  let total_students : ℕ := 300
  (first_division_percent + second_division_percent < 1) →
  (1 - first_division_percent - second_division_percent) * total_students = just_passed_count :=
by sorry

end exam_students_count_l1444_144453


namespace interest_rate_calculation_l1444_144406

theorem interest_rate_calculation (total_sum second_part : ℚ) 
  (h1 : total_sum = 2704)
  (h2 : second_part = 1664)
  (h3 : total_sum > second_part) :
  let first_part := total_sum - second_part
  let interest_first := first_part * (3/100) * 8
  let interest_second := second_part * (5/100) * 3
  interest_first = interest_second := by sorry

end interest_rate_calculation_l1444_144406


namespace calculation_proof_no_solution_proof_l1444_144498

-- Part 1
theorem calculation_proof : Real.sqrt 3 ^ 2 - (2023 + Real.pi / 2) ^ 0 - (-1) ^ (-1 : Int) = 3 := by sorry

-- Part 2
theorem no_solution_proof :
  ¬∃ x : ℝ, (5 * x - 4 > 3 * x) ∧ ((2 * x - 1) / 3 < x / 2) := by sorry

end calculation_proof_no_solution_proof_l1444_144498


namespace product_one_sum_lower_bound_l1444_144446

theorem product_one_sum_lower_bound (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) 
  (h_prod : a * b * c * d = 1) : 
  a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d ≥ 10 := by
  sorry

end product_one_sum_lower_bound_l1444_144446


namespace inverse_of_B_cubed_l1444_144454

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = ![![3, -2], ![1, 4]] →
  (B^3)⁻¹ = ![![7, -70], ![35, 42]] := by
sorry

end inverse_of_B_cubed_l1444_144454


namespace man_walking_running_time_l1444_144435

/-- Given a man who walks at 5 km/h for 5 hours, prove that the time taken to cover the same distance when running at 15 km/h is 1.6667 hours. -/
theorem man_walking_running_time (walking_speed : ℝ) (walking_time : ℝ) (running_speed : ℝ) :
  walking_speed = 5 →
  walking_time = 5 →
  running_speed = 15 →
  (walking_speed * walking_time) / running_speed = 1.6667 := by
  sorry

#eval (5 * 5) / 15

end man_walking_running_time_l1444_144435


namespace inequality_solution_l1444_144424

theorem inequality_solution (p q r : ℝ) 
  (h1 : ∀ x : ℝ, (x - p) * (x - q) / (x - r) ≥ 0 ↔ x ≤ -5 ∨ (20 ≤ x ∧ x ≤ 30))
  (h2 : p < q) : 
  p + 2*q + 3*r = 65 := by
  sorry

end inequality_solution_l1444_144424


namespace inequality_solution_set_l1444_144490

theorem inequality_solution_set : 
  {x : ℝ | |x|^3 - 2*x^2 - 4*|x| + 3 < 0} = 
  {x : ℝ | -3 < x ∧ x < -1} ∪ {x : ℝ | 1 < x ∧ x < 3} := by
  sorry

end inequality_solution_set_l1444_144490


namespace min_value_at_two_l1444_144400

/-- The function f(c) = 2c^2 - 8c + 1 attains its minimum value at c = 2 -/
theorem min_value_at_two (c : ℝ) : 
  IsMinOn (fun c => 2 * c^2 - 8 * c + 1) univ 2 := by
  sorry

end min_value_at_two_l1444_144400


namespace hyperbola_eccentricity_l1444_144409

/-- Given a hyperbola C: x²/a² - y²/b² = 1 (a > 0, b > 0), 
    if a focus F and its symmetric point with respect to one asymptote 
    lies on the other asymptote, then the eccentricity e of the hyperbola is 2 -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {p : ℝ × ℝ | p.1^2 / a^2 - p.2^2 / b^2 = 1}
  let asymptote₁ := {p : ℝ × ℝ | p.2 = (b / a) * p.1}
  let asymptote₂ := {p : ℝ × ℝ | p.2 = -(b / a) * p.1}
  ∃ (F : ℝ × ℝ), F ∈ C ∧ 
    (∃ (S : ℝ × ℝ), S ∈ asymptote₂ ∧ 
      (∀ (p : ℝ × ℝ), p ∈ asymptote₁ → 
        ((F.1 + S.1) / 2 = p.1 ∧ (F.2 + S.2) / 2 = p.2))) →
  let e := Real.sqrt ((a^2 + b^2) / a^2)
  e = 2 := by
sorry

end hyperbola_eccentricity_l1444_144409


namespace symmetric_probability_l1444_144401

/-- Represents a standard die with faces labeled 1 to 6 -/
def StandardDie : Type := Fin 6

/-- The number of dice being rolled -/
def numDice : Nat := 9

/-- The sum we are comparing to -/
def targetSum : Nat := 14

/-- The sum we want to prove has the same probability as the target sum -/
def symmetricSum : Nat := 49

/-- Function to calculate the probability of a specific sum occurring when rolling n dice -/
noncomputable def probabilityOfSum (n : Nat) (sum : Nat) : ℚ := sorry

theorem symmetric_probability :
  probabilityOfSum numDice targetSum = probabilityOfSum numDice symmetricSum := by sorry

end symmetric_probability_l1444_144401


namespace more_birch_than_fir_l1444_144485

/-- Represents a forest with fir and birch trees -/
structure Forest where
  fir_trees : ℕ
  birch_trees : ℕ

/-- A forest satisfies the Baron's condition if each fir tree has exactly 10 birch trees at 1 km distance -/
def satisfies_baron_condition (f : Forest) : Prop :=
  f.birch_trees = 10 * f.fir_trees

/-- Theorem: In a forest satisfying the Baron's condition, there are more birch trees than fir trees -/
theorem more_birch_than_fir (f : Forest) (h : satisfies_baron_condition f) : 
  f.birch_trees > f.fir_trees :=
sorry


end more_birch_than_fir_l1444_144485


namespace red_jellybeans_count_l1444_144479

/-- Proves that the number of red jellybeans is 120 given the specified conditions -/
theorem red_jellybeans_count (total : ℕ) (blue : ℕ) (purple : ℕ) (orange : ℕ)
  (h_total : total = 200)
  (h_blue : blue = 14)
  (h_purple : purple = 26)
  (h_orange : orange = 40) :
  total - (blue + purple + orange) = 120 := by
  sorry

end red_jellybeans_count_l1444_144479


namespace problem_1_problem_2_l1444_144438

theorem problem_1 : 3 * Real.sqrt 3 - Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 27 = - Real.sqrt 2 := by
  sorry

theorem problem_2 : (Real.sqrt 5 - Real.sqrt 3) * (Real.sqrt 5 + Real.sqrt 3) = 2 := by
  sorry

end problem_1_problem_2_l1444_144438


namespace cell_population_after_9_days_l1444_144433

/-- Calculates the number of cells after a given number of days, 
    given an initial population and a tripling rate every 3 days -/
def cell_population (initial_cells : ℕ) (days : ℕ) : ℕ :=
  initial_cells * (3 ^ (days / 3))

/-- Theorem stating that the cell population after 9 days is 36, 
    given an initial population of 4 cells -/
theorem cell_population_after_9_days :
  cell_population 4 9 = 36 := by
  sorry

#eval cell_population 4 9

end cell_population_after_9_days_l1444_144433


namespace quotient_relation_l1444_144495

theorem quotient_relation : ∃ (k l : ℝ), k ≠ l ∧ (64 / k = 4 * (64 / l)) := by
  sorry

end quotient_relation_l1444_144495


namespace emily_garden_problem_l1444_144461

/-- The number of small gardens Emily has -/
def num_small_gardens (total_seeds : ℕ) (big_garden_seeds : ℕ) (seeds_per_small_garden : ℕ) : ℕ :=
  (total_seeds - big_garden_seeds) / seeds_per_small_garden

/-- Emily's gardening problem -/
theorem emily_garden_problem :
  num_small_gardens 41 29 4 = 3 := by
  sorry

end emily_garden_problem_l1444_144461


namespace smallest_four_digit_solution_l1444_144404

theorem smallest_four_digit_solution : ∃ (x : ℕ), 
  (x ≥ 1000 ∧ x < 10000) ∧
  (5 * x ≡ 25 [ZMOD 20]) ∧
  (3 * x + 4 ≡ 10 [ZMOD 7]) ∧
  (-x + 3 ≡ 2 * x [ZMOD 15]) ∧
  (∀ y : ℕ, y ≥ 1000 ∧ y < x →
    ¬((5 * y ≡ 25 [ZMOD 20]) ∧
      (3 * y + 4 ≡ 10 [ZMOD 7]) ∧
      (-y + 3 ≡ 2 * y [ZMOD 15]))) ∧
  x = 1021 :=
by sorry

end smallest_four_digit_solution_l1444_144404


namespace total_spent_on_games_l1444_144440

def batman_game_cost : ℝ := 13.6
def superman_game_cost : ℝ := 5.06

theorem total_spent_on_games :
  batman_game_cost + superman_game_cost = 18.66 := by
  sorry

end total_spent_on_games_l1444_144440


namespace quadratic_properties_l1444_144499

def quadratic_function (m : ℝ) (x : ℝ) : ℝ := m * x^2 - 4 * m * x + 3 * m

theorem quadratic_properties :
  ∀ m : ℝ,
  (∀ x : ℝ, quadratic_function m x = 0 ↔ x = 1 ∨ x = 3) ∧
  (m < 0 → 
    (∃ x₀ : ℝ, 1 ≤ x₀ ∧ x₀ ≤ 4 ∧ quadratic_function m x₀ = 2 ∧
      ∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → quadratic_function m x ≤ 2) →
    (∀ x : ℝ, 1 ≤ x ∧ x ≤ 4 → quadratic_function m x ≥ -6) ∧
    (∃ x₁ : ℝ, 1 ≤ x₁ ∧ x₁ ≤ 4 ∧ quadratic_function m x₁ = -6)) ∧
  (m ≤ -4/3 ∨ m ≥ 4/5 ↔
    ∃ x : ℝ, 2 ≤ x ∧ x ≤ 4 ∧ quadratic_function m x = (m + 4) / 2) :=
by sorry

end quadratic_properties_l1444_144499


namespace expression_value_l1444_144403

theorem expression_value : (2^200 + 5^201)^2 - (2^200 - 5^201)^2 = 20 * 10^201 := by
  sorry

end expression_value_l1444_144403


namespace line_parallel_plane_relationship_l1444_144437

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Defines when a line is parallel to a plane -/
def parallel_line_plane (l : Line) (α : Plane) : Prop := sorry

/-- Defines when a line is contained within a plane -/
def line_in_plane (a : Line) (α : Plane) : Prop := sorry

/-- Defines when two lines are parallel -/
def parallel_lines (l1 l2 : Line) : Prop := sorry

/-- Defines when two lines are skew -/
def skew_lines (l1 l2 : Line) : Prop := sorry

/-- Theorem: If a line is parallel to a plane, and another line is contained within that plane,
    then the two lines are either parallel or skew -/
theorem line_parallel_plane_relationship (l a : Line) (α : Plane) 
  (h1 : parallel_line_plane l α) (h2 : line_in_plane a α) :
  parallel_lines l a ∨ skew_lines l a := by sorry

end line_parallel_plane_relationship_l1444_144437


namespace walkway_problem_l1444_144425

/-- Represents the walkway scenario -/
structure Walkway where
  length : ℝ
  time_with : ℝ
  time_against : ℝ

/-- Calculates the time to walk when the walkway is not moving -/
noncomputable def time_stationary (w : Walkway) : ℝ :=
  w.length * 2 * w.time_with * w.time_against / (w.time_against + w.time_with) / w.time_with

/-- Theorem statement for the walkway problem -/
theorem walkway_problem (w : Walkway) 
  (h1 : w.length = 100)
  (h2 : w.time_with = 25)
  (h3 : w.time_against = 150) :
  abs (time_stationary w - 300 / 7) < 0.001 := by
  sorry

end walkway_problem_l1444_144425


namespace infinitely_many_very_good_pairs_l1444_144473

/-- A pair of natural numbers is "good" if they consist of the same prime divisors, possibly in different powers. -/
def isGoodPair (m n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → (p ∣ m ↔ p ∣ n)

/-- A pair of natural numbers is "very good" if both the pair and their successors form "good" pairs. -/
def isVeryGoodPair (m n : ℕ) : Prop :=
  isGoodPair m n ∧ isGoodPair (m + 1) (n + 1) ∧ m ≠ n

/-- There exist infinitely many "very good" pairs of natural numbers. -/
theorem infinitely_many_very_good_pairs :
  ∀ k : ℕ, ∃ m n : ℕ, m > k ∧ n > k ∧ isVeryGoodPair m n :=
sorry

end infinitely_many_very_good_pairs_l1444_144473


namespace can_lids_per_box_l1444_144481

theorem can_lids_per_box (initial_lids : ℕ) (final_lids : ℕ) (num_boxes : ℕ) :
  initial_lids = 14 →
  final_lids = 53 →
  num_boxes = 3 →
  (final_lids - initial_lids) / num_boxes = 13 :=
by sorry

end can_lids_per_box_l1444_144481


namespace sqrt_6_between_2_and_3_l1444_144402

theorem sqrt_6_between_2_and_3 : 2 < Real.sqrt 6 ∧ Real.sqrt 6 < 3 := by
  sorry

end sqrt_6_between_2_and_3_l1444_144402


namespace circles_internally_tangent_l1444_144419

theorem circles_internally_tangent : ∃ (C₁ C₂ : ℝ × ℝ) (r₁ r₂ : ℝ),
  (∀ (x y : ℝ), x^2 + y^2 - 6*x + 4*y + 12 = 0 ↔ (x - C₁.1)^2 + (y - C₁.2)^2 = r₁^2) ∧
  (∀ (x y : ℝ), x^2 + y^2 - 14*x - 2*y + 14 = 0 ↔ (x - C₂.1)^2 + (y - C₂.2)^2 = r₂^2) ∧
  (C₂.1 - C₁.1)^2 + (C₂.2 - C₁.2)^2 = (r₂ - r₁)^2 ∧
  r₂ > r₁ := by
  sorry

end circles_internally_tangent_l1444_144419


namespace arithmetic_evaluation_l1444_144462

theorem arithmetic_evaluation : 6 + 18 / 3 - 3^2 - 4 * 2 = -5 := by
  sorry

end arithmetic_evaluation_l1444_144462


namespace total_coins_l1444_144474

/-- Represents a 3x3 grid of cells --/
def Grid := Fin 3 → Fin 3 → ℕ

/-- The sum of coins in the corner cells --/
def corner_sum (g : Grid) : ℕ :=
  g 0 0 + g 0 2 + g 2 0 + g 2 2

/-- The number of coins in the center cell --/
def center_value (g : Grid) : ℕ :=
  g 1 1

/-- Theorem stating the total number of coins in the grid --/
theorem total_coins (g : Grid) 
  (h_corner : corner_sum g = 8) 
  (h_center : center_value g = 3) : 
  ∃ (total : ℕ), total = 8 :=
sorry

end total_coins_l1444_144474


namespace rationalize_denominator_l1444_144444

theorem rationalize_denominator : 
  3 / (Real.sqrt 5 - 2) = 3 * Real.sqrt 5 + 6 := by
  sorry

end rationalize_denominator_l1444_144444


namespace min_value_of_b_l1444_144467

noncomputable def f (x a : ℝ) : ℝ := (x - a)^2 + (Real.log (x^2 - 2*a))^2

theorem min_value_of_b :
  ∃ (b : ℝ), (∀ (a : ℝ), ∃ (x₀ : ℝ), x₀ > 0 ∧ f x₀ a ≤ b) ∧
  (∀ (b' : ℝ), (∀ (a : ℝ), ∃ (x₀ : ℝ), x₀ > 0 ∧ f x₀ a ≤ b') → b ≤ b') ∧
  b = 4/5 :=
sorry

end min_value_of_b_l1444_144467


namespace female_democrats_count_l1444_144414

theorem female_democrats_count (total : ℕ) (female : ℕ) (male : ℕ) : 
  total = 780 →
  female + male = total →
  (female / 2 + male / 4 : ℚ) = total / 3 →
  female / 2 = 130 :=
by sorry

end female_democrats_count_l1444_144414


namespace inequality_proof_l1444_144478

theorem inequality_proof (x y z t : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0) (non_neg_t : t ≥ 0)
  (sum_condition : x + y + z + t = 7) : 
  Real.sqrt (x^2 + y^2) + Real.sqrt (x^2 + 1) + Real.sqrt (z^2 + y^2) + 
  Real.sqrt (t^2 + 64) + Real.sqrt (z^2 + t^2) ≥ 17 := by
  sorry

end inequality_proof_l1444_144478


namespace lcm_problem_l1444_144497

theorem lcm_problem (a b : ℕ+) (h1 : b = 852) (h2 : Nat.lcm a b = 5964) : a = 852 := by
  sorry

end lcm_problem_l1444_144497


namespace total_stars_is_116_l1444_144451

/-- The number of people in the Young Pioneers group -/
def n : ℕ := sorry

/-- The total number of lucky stars planned to be made -/
def total_stars : ℕ := sorry

/-- Condition 1: If each person makes 10 stars, they will be 6 stars short of completing the plan -/
axiom condition1 : 10 * n + 6 = total_stars

/-- Condition 2: If 4 of them each make 8 stars and the rest each make 12 stars, they will just complete the plan -/
axiom condition2 : 4 * 8 + (n - 4) * 12 = total_stars

/-- Theorem: The total number of lucky stars planned to be made is 116 -/
theorem total_stars_is_116 : total_stars = 116 := by sorry

end total_stars_is_116_l1444_144451


namespace square_cut_perimeter_l1444_144420

/-- Given a square with side length 2a and a line y = 2x/3 cutting through it,
    the perimeter of one piece divided by a is 6 + (2√13 + 3√2)/3 -/
theorem square_cut_perimeter (a : ℝ) (a_pos : a > 0) :
  let square := {(x, y) | -a ≤ x ∧ x ≤ a ∧ -a ≤ y ∧ y ≤ a}
  let line := {(x, y) | y = (2/3) * x}
  let piece := {p ∈ square | p.2 ≤ (2/3) * p.1 ∨ (p.1 = a ∧ p.2 ≥ (2/3) * a) ∨ (p.1 = -a ∧ p.2 ≤ -(2/3) * a)}
  let perimeter := Real.sqrt ((2*a)^2 + ((4*a)/3)^2) + (4*a)/3 + 2*a + a * Real.sqrt 2
  perimeter / a = 6 + (2 * Real.sqrt 13 + 3 * Real.sqrt 2) / 3 := by
  sorry

end square_cut_perimeter_l1444_144420


namespace fruit_stand_average_price_l1444_144471

theorem fruit_stand_average_price (apple_price orange_price : ℚ)
  (total_fruits : ℕ) (oranges_removed : ℕ) (kept_avg_price : ℚ)
  (h1 : apple_price = 40/100)
  (h2 : orange_price = 60/100)
  (h3 : total_fruits = 10)
  (h4 : oranges_removed = 4)
  (h5 : kept_avg_price = 50/100) :
  ∃ (apples oranges : ℕ),
    apples + oranges = total_fruits ∧
    (apple_price * apples + orange_price * oranges) / total_fruits = 54/100 :=
by sorry

end fruit_stand_average_price_l1444_144471


namespace second_train_length_l1444_144487

/-- The length of a train given crossing time and speeds -/
def train_length (l1 : ℝ) (v1 v2 : ℝ) (t : ℝ) : ℝ :=
  (v1 + v2) * t - l1

/-- Theorem: Given the conditions, the length of the second train is 210 meters -/
theorem second_train_length :
  let l1 : ℝ := 290  -- Length of first train in meters
  let v1 : ℝ := 120 * 1000 / 3600  -- Speed of first train in m/s
  let v2 : ℝ := 80 * 1000 / 3600   -- Speed of second train in m/s
  let t : ℝ := 9    -- Crossing time in seconds
  train_length l1 v1 v2 t = 210 := by
sorry


end second_train_length_l1444_144487


namespace history_not_statistics_l1444_144492

theorem history_not_statistics (total : ℕ) (history : ℕ) (statistics : ℕ) (history_or_statistics : ℕ) :
  total = 89 →
  history = 36 →
  statistics = 32 →
  history_or_statistics = 59 →
  history - (history + statistics - history_or_statistics) = 27 := by
  sorry

end history_not_statistics_l1444_144492


namespace celine_change_l1444_144459

def laptop_price : ℕ := 600
def smartphone_price : ℕ := 400
def laptops_bought : ℕ := 2
def smartphones_bought : ℕ := 4
def total_money : ℕ := 3000

theorem celine_change : 
  total_money - (laptop_price * laptops_bought + smartphone_price * smartphones_bought) = 200 := by
  sorry

end celine_change_l1444_144459


namespace distance_to_x_axis_l1444_144426

def line (x y : ℝ) : Prop := y = 2 * x + 1

theorem distance_to_x_axis (k : ℝ) (h : line (-2) k) : 
  |k| = 3 := by sorry

end distance_to_x_axis_l1444_144426


namespace square_rotation_around_hexagon_l1444_144456

theorem square_rotation_around_hexagon :
  let hexagon_angle : ℝ := 120
  let square_angle : ℝ := 90
  let rotation_per_movement : ℝ := 360 - (hexagon_angle + square_angle)
  let total_rotation : ℝ := 3 * rotation_per_movement
  total_rotation % 360 = 90 := by sorry

end square_rotation_around_hexagon_l1444_144456


namespace circle_tangency_l1444_144434

theorem circle_tangency (m : ℝ) : 
  (∃ x y : ℝ, (x - m)^2 + (y + 2)^2 = 9 ∧ (x + 1)^2 + (y - m)^2 = 4) →
  (∃ x y : ℝ, (x - m)^2 + (y + 2)^2 = 9) →
  (∃ x y : ℝ, (x + 1)^2 + (y - m)^2 = 4) →
  (m = -2 ∨ m = -1) := by
sorry

end circle_tangency_l1444_144434
