import Mathlib

namespace NUMINAMATH_CALUDE_merry_go_round_diameter_l3507_350775

/-- The diameter of a circular platform with area 3.14 square yards is 2 yards. -/
theorem merry_go_round_diameter : 
  ∀ (r : ℝ), r > 0 → π * r^2 = 3.14 → 2 * r = 2 := by sorry

end NUMINAMATH_CALUDE_merry_go_round_diameter_l3507_350775


namespace NUMINAMATH_CALUDE_power_equation_solution_l3507_350766

theorem power_equation_solution (m n : ℕ) : 
  (1/5 : ℚ)^m * (1/4 : ℚ)^n = 1/(10^4 : ℚ) → m = 4 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_power_equation_solution_l3507_350766


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l3507_350740

theorem elevator_weight_problem (adults_avg_weight : ℝ) (elevator_max_weight : ℝ) (next_person_max_weight : ℝ) :
  adults_avg_weight = 140 →
  elevator_max_weight = 600 →
  next_person_max_weight = 52 →
  (elevator_max_weight - 3 * adults_avg_weight - next_person_max_weight) = 128 :=
by sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l3507_350740


namespace NUMINAMATH_CALUDE_tournament_matches_divisible_by_seven_l3507_350719

/-- Represents a single elimination tournament --/
structure Tournament :=
  (total_players : ℕ)
  (bye_players : ℕ)

/-- Calculates the total number of matches in a tournament --/
def total_matches (t : Tournament) : ℕ := t.total_players - 1

/-- Theorem: In a tournament with 120 players and 32 byes, the total matches is divisible by 7 --/
theorem tournament_matches_divisible_by_seven :
  ∀ (t : Tournament), t.total_players = 120 → t.bye_players = 32 →
  ∃ (k : ℕ), total_matches t = 7 * k := by
  sorry

end NUMINAMATH_CALUDE_tournament_matches_divisible_by_seven_l3507_350719


namespace NUMINAMATH_CALUDE_tangent_line_b_value_l3507_350750

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 3*x^2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

/-- The tangent line function -/
def g (x b : ℝ) : ℝ := -3*x + b

theorem tangent_line_b_value :
  ∀ b : ℝ, (∃ x : ℝ, f x = g x b ∧ f' x = -3) → b = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_b_value_l3507_350750


namespace NUMINAMATH_CALUDE_quadratic_inequality_always_true_l3507_350774

theorem quadratic_inequality_always_true :
  ∀ x : ℝ, 3 * x^2 + 9 * x ≥ -12 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_always_true_l3507_350774


namespace NUMINAMATH_CALUDE_angle_is_90_degrees_l3507_350723

/-- Represents a point on or above the Earth's surface -/
structure EarthPoint where
  latitude : Real
  longitude : Real
  elevation : Real

/-- Calculate the angle between three points on or above the Earth's surface -/
def angleBAC (earthRadius : Real) (a b c : EarthPoint) : Real :=
  sorry

theorem angle_is_90_degrees (earthRadius : Real) :
  let a : EarthPoint := { latitude := 0, longitude := 100, elevation := 0 }
  let b : EarthPoint := { latitude := 30, longitude := -90, elevation := 0 }
  let c : EarthPoint := { latitude := 90, longitude := 0, elevation := 2 }
  angleBAC earthRadius a b c = 90 := by
  sorry

end NUMINAMATH_CALUDE_angle_is_90_degrees_l3507_350723


namespace NUMINAMATH_CALUDE_max_daily_sales_l3507_350721

def price (t : ℕ) : ℝ :=
  if 0 < t ∧ t < 25 then t + 20
  else if 25 ≤ t ∧ t ≤ 30 then -t + 100
  else 0

def sales_volume (t : ℕ) : ℝ :=
  if 0 < t ∧ t ≤ 30 then -t + 40
  else 0

def daily_sales (t : ℕ) : ℝ := price t * sales_volume t

theorem max_daily_sales :
  ∃ (max_sales : ℝ) (max_day : ℕ),
    max_sales = 1125 ∧
    max_day = 25 ∧
    ∀ t : ℕ, 0 < t ∧ t ≤ 30 → daily_sales t ≤ max_sales ∧
    daily_sales max_day = max_sales :=
  sorry

end NUMINAMATH_CALUDE_max_daily_sales_l3507_350721


namespace NUMINAMATH_CALUDE_playground_children_count_l3507_350720

theorem playground_children_count (boys girls : ℕ) 
  (h1 : boys = 44) 
  (h2 : girls = 53) : 
  boys + girls = 97 := by
sorry

end NUMINAMATH_CALUDE_playground_children_count_l3507_350720


namespace NUMINAMATH_CALUDE_exists_line_through_P_intersecting_hyperbola_l3507_350780

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the point P
def P : ℝ × ℝ := (2, 1)

-- Define a line passing through P with slope k
def line (k : ℝ) (x y : ℝ) : Prop := y - P.2 = k * (x - P.1)

-- Define the midpoint condition
def is_midpoint (p a b : ℝ × ℝ) : Prop :=
  p.1 = (a.1 + b.1) / 2 ∧ p.2 = (a.2 + b.2) / 2

-- Theorem statement
theorem exists_line_through_P_intersecting_hyperbola :
  ∃ (k : ℝ) (A B : ℝ × ℝ),
    hyperbola A.1 A.2 ∧
    hyperbola B.1 B.2 ∧
    line k A.1 A.2 ∧
    line k B.1 B.2 ∧
    is_midpoint P A B :=
  sorry

end NUMINAMATH_CALUDE_exists_line_through_P_intersecting_hyperbola_l3507_350780


namespace NUMINAMATH_CALUDE_range_of_a_l3507_350700

def A (a : ℝ) : Set ℝ := {x | (x - 6) * (x - (2 * a + 5)) > 0}

def B (a : ℝ) : Set ℝ := {x | ((a^2 + 2) - x) * (2 * a - x) < 0}

theorem range_of_a :
  ∀ a : ℝ, 
    a > 1/2 →
    (B a ⊆ A a) →
    (B a ≠ A a) →
    a ∈ Set.Ioo (1/2 : ℝ) 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l3507_350700


namespace NUMINAMATH_CALUDE_rose_garden_problem_l3507_350731

/-- Rose garden problem -/
theorem rose_garden_problem (total_rows : ℕ) (roses_per_row : ℕ) (total_pink : ℕ) :
  total_rows = 10 →
  roses_per_row = 20 →
  total_pink = 40 →
  ∃ (red_fraction : ℚ),
    red_fraction = 1/2 ∧
    ∀ (row : ℕ),
      row ≤ total_rows →
      ∃ (red white pink : ℕ),
        red + white + pink = roses_per_row ∧
        white = (3/5 : ℚ) * (roses_per_row - red) ∧
        pink = roses_per_row - red - white ∧
        red = (red_fraction * roses_per_row : ℚ) :=
by sorry

end NUMINAMATH_CALUDE_rose_garden_problem_l3507_350731


namespace NUMINAMATH_CALUDE_set_operation_equality_l3507_350733

def U : Set ℝ := Set.univ

def A : Set ℝ := {x : ℝ | x > 0}

def B : Set ℝ := {x : ℝ | x ≤ -1}

theorem set_operation_equality : 
  (A ∩ (U \ B)) ∪ (B ∩ (U \ A)) = {x : ℝ | x > 0 ∨ x ≤ -1} := by sorry

end NUMINAMATH_CALUDE_set_operation_equality_l3507_350733


namespace NUMINAMATH_CALUDE_product_of_roots_cubic_equation_l3507_350730

theorem product_of_roots_cubic_equation :
  let f : ℝ → ℝ := λ x => 2 * x^3 - 7 * x^2 - 6
  let roots := {r : ℝ | f r = 0}
  ∀ r s t : ℝ, r ∈ roots → s ∈ roots → t ∈ roots → r * s * t = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_cubic_equation_l3507_350730


namespace NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l3507_350781

open Real

theorem least_positive_t_for_geometric_progression :
  ∃ (t : ℝ) (α : ℝ),
    0 < α ∧ α < π / 2 ∧
    (∃ (r : ℝ),
      arcsin (sin α) * r = arcsin (sin (3 * α)) ∧
      arcsin (sin (3 * α)) * r = arcsin (sin (5 * α)) ∧
      arcsin (sin (5 * α)) * r = arcsin (sin (t * α))) ∧
    (∀ (t' : ℝ) (α' : ℝ),
      0 < α' ∧ α' < π / 2 ∧
      (∃ (r' : ℝ),
        arcsin (sin α') * r' = arcsin (sin (3 * α')) ∧
        arcsin (sin (3 * α')) * r' = arcsin (sin (5 * α')) ∧
        arcsin (sin (5 * α')) * r' = arcsin (sin (t' * α'))) →
      t ≤ t') ∧
    t = 9 + 4 * Real.sqrt 5 :=
by sorry

end NUMINAMATH_CALUDE_least_positive_t_for_geometric_progression_l3507_350781


namespace NUMINAMATH_CALUDE_alpha_value_l3507_350746

-- Define complex numbers α and β
variable (α β : ℂ)

-- Define the conditions
variable (h1 : (α + β).re > 0)
variable (h2 : (Complex.I * (α - 3 * β)).re > 0)
variable (h3 : β = 4 + 3 * Complex.I)

-- Theorem to prove
theorem alpha_value : α = 12 - 3 * Complex.I := by
  sorry

end NUMINAMATH_CALUDE_alpha_value_l3507_350746


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_a_range_l3507_350711

theorem quadratic_roots_imply_a_range (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ > 0 ∧ x₂ < 0 ∧ 
   x₁^2 + a*x₁ + a^2 - 1 = 0 ∧ x₂^2 + a*x₂ + a^2 - 1 = 0) →
  -1 < a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_a_range_l3507_350711


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l3507_350765

theorem absolute_value_inequality (x y : ℝ) :
  (∀ x y : ℝ, x > y ∧ y > 0 → abs x > abs y) ∧
  (∃ x y : ℝ, abs x > abs y ∧ ¬(x > y ∧ y > 0)) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l3507_350765


namespace NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_sum_l3507_350706

theorem odd_sum_of_squares_implies_odd_sum (n m : ℤ) 
  (h : Odd (n^2 + m^2)) : Odd (n + m) := by
  sorry

end NUMINAMATH_CALUDE_odd_sum_of_squares_implies_odd_sum_l3507_350706


namespace NUMINAMATH_CALUDE_cafeteria_lasagnas_l3507_350722

/-- The number of lasagnas made by the school cafeteria -/
def num_lasagnas : ℕ := sorry

/-- The amount of ground mince used for each lasagna (in pounds) -/
def mince_per_lasagna : ℕ := 2

/-- The amount of ground mince used for each cottage pie (in pounds) -/
def mince_per_cottage_pie : ℕ := 3

/-- The total amount of ground mince used (in pounds) -/
def total_mince_used : ℕ := 500

/-- The number of cottage pies made -/
def num_cottage_pies : ℕ := 100

/-- Theorem stating that the number of lasagnas made is 100 -/
theorem cafeteria_lasagnas : num_lasagnas = 100 := by sorry

end NUMINAMATH_CALUDE_cafeteria_lasagnas_l3507_350722


namespace NUMINAMATH_CALUDE_x_value_l3507_350783

theorem x_value (x y : ℝ) 
  (h1 : x - y = 8)
  (h2 : x + y = 16)
  (h3 : x * y = 48) : x = 12 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l3507_350783


namespace NUMINAMATH_CALUDE_bill_division_l3507_350777

/-- Proves that when three people divide a 99-dollar bill evenly, each person pays 33 dollars. -/
theorem bill_division (total_bill : ℕ) (num_people : ℕ) (each_share : ℕ) :
  total_bill = 99 → num_people = 3 → each_share = total_bill / num_people → each_share = 33 := by
  sorry

#check bill_division

end NUMINAMATH_CALUDE_bill_division_l3507_350777


namespace NUMINAMATH_CALUDE_negation_equivalence_l3507_350708

-- Define a triangle
structure Triangle where
  -- Add necessary fields for a triangle

-- Define an obtuse angle
def isObtuseAngle (angle : Real) : Prop := angle > Real.pi / 2

-- Define the property of having at most one obtuse angle
def atMostOneObtuseAngle (t : Triangle) : Prop :=
  ∃ (a b c : Real), isObtuseAngle a → ¬(isObtuseAngle b ∨ isObtuseAngle c)

-- Define the property of having at least two obtuse angles
def atLeastTwoObtuseAngles (t : Triangle) : Prop :=
  ∃ (a b : Real), isObtuseAngle a ∧ isObtuseAngle b

-- Theorem stating that the negation of "at most one obtuse angle" 
-- is equivalent to "at least two obtuse angles"
theorem negation_equivalence (t : Triangle) : 
  ¬(atMostOneObtuseAngle t) ↔ atLeastTwoObtuseAngles t := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3507_350708


namespace NUMINAMATH_CALUDE_fraction_sum_l3507_350702

theorem fraction_sum (a b : ℕ) (h1 : a.Coprime b) (h2 : a > 0) (h3 : b > 0)
  (h4 : (5 : ℚ) / 6 * (a^2 : ℚ) / (b^2 : ℚ) = 2 * (a : ℚ) / (b : ℚ)) :
  a + b = 17 := by
sorry

end NUMINAMATH_CALUDE_fraction_sum_l3507_350702


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l3507_350767

theorem polynomial_evaluation :
  ∃ x : ℝ, x > 0 ∧ x^2 - 3*x - 18 = 0 ∧ x^3 - 3*x^2 - 9*x + 5 = 59 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l3507_350767


namespace NUMINAMATH_CALUDE_turtle_initial_coins_l3507_350762

def bridge_crossing (initial_coins : ℕ) : Prop :=
  let after_first_crossing := 3 * initial_coins - 30
  let after_second_crossing := 3 * after_first_crossing - 30
  after_second_crossing = 0

theorem turtle_initial_coins : 
  ∃ (x : ℕ), bridge_crossing x ∧ x = 15 :=
sorry

end NUMINAMATH_CALUDE_turtle_initial_coins_l3507_350762


namespace NUMINAMATH_CALUDE_sum_negative_condition_l3507_350754

theorem sum_negative_condition (x y : ℝ) :
  (∃ (x y : ℝ), (x < 0 ∨ y < 0) ∧ x + y ≥ 0) ∧
  (∀ (x y : ℝ), x + y < 0 → (x < 0 ∨ y < 0)) :=
sorry

end NUMINAMATH_CALUDE_sum_negative_condition_l3507_350754


namespace NUMINAMATH_CALUDE_smallest_perfect_square_div_by_5_and_6_l3507_350709

theorem smallest_perfect_square_div_by_5_and_6 : 
  ∃ n : ℕ, n > 0 ∧ 
  (∃ m : ℕ, n = m^2) ∧ 
  n % 5 = 0 ∧ 
  n % 6 = 0 ∧ 
  (∀ k : ℕ, k > 0 → (∃ m : ℕ, k = m^2) → k % 5 = 0 → k % 6 = 0 → k ≥ n) ∧
  n = 900 := by
sorry

end NUMINAMATH_CALUDE_smallest_perfect_square_div_by_5_and_6_l3507_350709


namespace NUMINAMATH_CALUDE_peanuts_in_box_l3507_350716

theorem peanuts_in_box (initial_peanuts : ℕ) (added_peanuts : ℕ) :
  initial_peanuts = 4 → added_peanuts = 12 → initial_peanuts + added_peanuts = 16 := by
  sorry

end NUMINAMATH_CALUDE_peanuts_in_box_l3507_350716


namespace NUMINAMATH_CALUDE_range_of_a_l3507_350705

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, |x + 1| + |x - a| ≥ 5) → 
  a ∈ Set.Ici 4 ∪ Set.Iic (-6) := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l3507_350705


namespace NUMINAMATH_CALUDE_store_marbles_proof_l3507_350737

/-- The number of marbles initially in the store, given the number of customers,
    marbles bought per customer, and remaining marbles after sales. -/
def initial_marbles (customers : ℕ) (marbles_per_customer : ℕ) (remaining_marbles : ℕ) : ℕ :=
  customers * marbles_per_customer + remaining_marbles

theorem store_marbles_proof :
  initial_marbles 20 15 100 = 400 :=
by sorry

end NUMINAMATH_CALUDE_store_marbles_proof_l3507_350737


namespace NUMINAMATH_CALUDE_comic_arrangement_count_l3507_350736

def arrange_comics (spiderman : Nat) (archie : Nat) (garfield : Nat) : Nat :=
  Nat.factorial spiderman * (Nat.factorial archie * Nat.factorial garfield * Nat.factorial 2)

theorem comic_arrangement_count :
  arrange_comics 7 6 5 = 871219200 := by
  sorry

end NUMINAMATH_CALUDE_comic_arrangement_count_l3507_350736


namespace NUMINAMATH_CALUDE_max_value_of_product_l3507_350778

/-- The function f(x) = 6x^3 - ax^2 - 2bx + 2 -/
def f (a b x : ℝ) : ℝ := 6 * x^3 - a * x^2 - 2 * b * x + 2

/-- The derivative of f(x) -/
def f' (a b x : ℝ) : ℝ := 18 * x^2 - 2 * a * x - 2 * b

theorem max_value_of_product (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_extremum : f' a b 1 = 0) : 
  a * b ≤ (81 : ℝ) / 4 ∧ ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ f' a₀ b₀ 1 = 0 ∧ a₀ * b₀ = (81 : ℝ) / 4 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_product_l3507_350778


namespace NUMINAMATH_CALUDE_smallest_possible_value_l3507_350796

theorem smallest_possible_value (x : ℕ+) (m n : ℕ+) : 
  m = 60 →
  Nat.gcd m n = x + 5 →
  Nat.lcm m n = x * (x + 5)^2 →
  n ≥ 2000 :=
sorry

end NUMINAMATH_CALUDE_smallest_possible_value_l3507_350796


namespace NUMINAMATH_CALUDE_smallest_non_negative_solution_l3507_350743

theorem smallest_non_negative_solution (x : ℕ) : x = 2 ↔ 
  (∀ y : ℕ, (42 * y + 10) % 15 = 5 → y ≥ x) ∧ (42 * x + 10) % 15 = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_non_negative_solution_l3507_350743


namespace NUMINAMATH_CALUDE_first_player_advantage_l3507_350735

/-- A game board configuration -/
structure BoardConfig where
  spaces : ℕ
  s₁ : ℕ
  s₂ : ℕ

/-- The probability of a player winning -/
def winProbability (player : ℕ) (config : BoardConfig) : ℝ :=
  sorry

/-- The theorem stating that the first player has a higher probability of winning -/
theorem first_player_advantage (config : BoardConfig) 
    (h : config.spaces ≥ 12) 
    (h_start : config.s₁ = config.s₂) : 
  winProbability 1 config > 1/2 :=
sorry

end NUMINAMATH_CALUDE_first_player_advantage_l3507_350735


namespace NUMINAMATH_CALUDE_womens_average_age_l3507_350752

theorem womens_average_age 
  (n : ℕ) 
  (initial_avg : ℝ) 
  (age1 age2 : ℕ) 
  (new_avg_increase : ℝ) :
  n = 8 →
  age1 = 20 →
  age2 = 28 →
  new_avg_increase = 2 →
  (n * initial_avg - (age1 + age2 : ℝ) + 2 * ((n * initial_avg + n * new_avg_increase - n * initial_avg + age1 + age2) / 2)) / n = initial_avg + new_avg_increase →
  ((n * initial_avg + n * new_avg_increase - n * initial_avg + age1 + age2) / 2) / 2 = 32 :=
by sorry

end NUMINAMATH_CALUDE_womens_average_age_l3507_350752


namespace NUMINAMATH_CALUDE_total_cost_for_two_rides_l3507_350786

def base_fare : ℚ := 2
def per_mile_charge : ℚ := (3 : ℚ) / 10
def first_ride_distance : ℕ := 8
def second_ride_distance : ℕ := 5

def ride_cost (distance : ℕ) : ℚ :=
  base_fare + per_mile_charge * distance

theorem total_cost_for_two_rides :
  ride_cost first_ride_distance + ride_cost second_ride_distance = (79 : ℚ) / 10 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_for_two_rides_l3507_350786


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l3507_350727

/-- The eccentricity of a hyperbola with equation x²/4 - y²/3 = 1 is √7/2 -/
theorem hyperbola_eccentricity : 
  let a : ℝ := 2
  let b : ℝ := Real.sqrt 3
  let c : ℝ := Real.sqrt (a^2 + b^2)
  let e : ℝ := c / a
  e = Real.sqrt 7 / 2 := by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l3507_350727


namespace NUMINAMATH_CALUDE_probability_of_winning_combination_l3507_350714

/-- A function that returns true if a list of integers has a product that is a power of 6 -/
def isPowerOfSixProduct (list : List Int) : Bool :=
  sorry

/-- A function that returns all valid combinations of 5 different integers from 1 to 30
    whose product is a power of 6 -/
def validCombinations : List (List Int) :=
  sorry

/-- The total number of ways to choose 5 different integers from 1 to 30 -/
def totalCombinations : Int :=
  sorry

theorem probability_of_winning_combination :
  (List.length validCombinations) / totalCombinations = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_winning_combination_l3507_350714


namespace NUMINAMATH_CALUDE_anita_gave_five_apples_l3507_350739

/-- Represents the number of apples Tessa initially had -/
def initial_apples : ℕ := 4

/-- Represents the number of apples Tessa now has -/
def current_apples : ℕ := 9

/-- Represents the number of apples Anita gave Tessa -/
def apples_from_anita : ℕ := current_apples - initial_apples

theorem anita_gave_five_apples : apples_from_anita = 5 := by
  sorry

end NUMINAMATH_CALUDE_anita_gave_five_apples_l3507_350739


namespace NUMINAMATH_CALUDE_travel_time_calculation_l3507_350712

/-- Given a person traveling at a constant speed for a certain distance,
    calculate the time taken for the journey. -/
theorem travel_time_calculation (speed : ℝ) (distance : ℝ) (h1 : speed = 75) (h2 : distance = 300) :
  distance / speed = 4 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l3507_350712


namespace NUMINAMATH_CALUDE_ruble_payment_l3507_350798

theorem ruble_payment (x : ℤ) (h : x > 7) : ∃ (a b : ℕ), x = 3 * a + 5 * b := by
  sorry

end NUMINAMATH_CALUDE_ruble_payment_l3507_350798


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3507_350724

theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x : ℝ, x > a → x^2 - 5*x + 6 ≥ 0) ∧ 
  (∃ x : ℝ, x^2 - 5*x + 6 ≥ 0 ∧ x ≤ a) ↔ 
  a ≥ 3 := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l3507_350724


namespace NUMINAMATH_CALUDE_carpet_transformation_l3507_350744

/-- Represents a rectangle with width and height -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents a piece cut from a rectangle -/
structure CutPiece where
  width : ℕ
  height : ℕ

/-- Represents the original carpet -/
def original_carpet : Rectangle := { width := 9, height := 12 }

/-- Represents the piece cut off by the dragon -/
def dragon_cut : CutPiece := { width := 1, height := 8 }

/-- Represents the final square carpet -/
def final_carpet : Rectangle := { width := 10, height := 10 }

/-- Function to calculate the area of a rectangle -/
def area (r : Rectangle) : ℕ := r.width * r.height

/-- Function to calculate the area of a cut piece -/
def cut_area (c : CutPiece) : ℕ := c.width * c.height

/-- Theorem stating that it's possible to transform the damaged carpet into a square -/
theorem carpet_transformation :
  ∃ (part1 part2 part3 : Rectangle),
    area original_carpet - cut_area dragon_cut =
    area part1 + area part2 + area part3 ∧
    area final_carpet = area part1 + area part2 + area part3 := by
  sorry

end NUMINAMATH_CALUDE_carpet_transformation_l3507_350744


namespace NUMINAMATH_CALUDE_correct_water_ratio_l3507_350763

/-- Represents the time in minutes to fill the bathtub with hot water -/
def hot_water_fill_time : ℝ := 23

/-- Represents the time in minutes to fill the bathtub with cold water -/
def cold_water_fill_time : ℝ := 17

/-- Represents the ratio of hot water to cold water when the bathtub is full -/
def hot_to_cold_ratio : ℝ := 1.5

/-- Represents the delay in minutes before opening the cold water tap -/
def cold_water_delay : ℝ := 7

/-- Proves that opening the cold water tap after the specified delay results in the correct ratio of hot to cold water -/
theorem correct_water_ratio : 
  let hot_water_volume := (hot_water_fill_time - cold_water_delay) / hot_water_fill_time
  let cold_water_volume := cold_water_delay / cold_water_fill_time
  hot_water_volume = hot_to_cold_ratio * cold_water_volume := by
  sorry

end NUMINAMATH_CALUDE_correct_water_ratio_l3507_350763


namespace NUMINAMATH_CALUDE_books_bought_l3507_350749

theorem books_bought (initial_books final_books : ℕ) 
  (h1 : initial_books = 50)
  (h2 : final_books = 151) :
  final_books - initial_books = 101 := by
  sorry

end NUMINAMATH_CALUDE_books_bought_l3507_350749


namespace NUMINAMATH_CALUDE_ahsme_score_uniqueness_l3507_350768

/-- AHSME scoring system and uniqueness of solution for score 85 -/
theorem ahsme_score_uniqueness :
  ∃! (c w : ℕ),
    c + w ≤ 30 ∧
    85 = 30 + 4 * c - w ∧
    ∀ (s : ℕ), s ≥ 85 →
      (∃! (c' w' : ℕ), c' + w' ≤ 30 ∧ s = 30 + 4 * c' - w') →
      s = 85 :=
by sorry

end NUMINAMATH_CALUDE_ahsme_score_uniqueness_l3507_350768


namespace NUMINAMATH_CALUDE_average_difference_l3507_350745

theorem average_difference (a b c : ℝ) : 
  (a + b) / 2 = 45 → (b + c) / 2 = 50 → c - a = 10 := by
  sorry

end NUMINAMATH_CALUDE_average_difference_l3507_350745


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3507_350707

theorem purely_imaginary_complex_number (a : ℝ) :
  let z := (a + 2 * Complex.I) / (3 - 4 * Complex.I)
  (∃ (b : ℝ), z = b * Complex.I) → a = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3507_350707


namespace NUMINAMATH_CALUDE_repeating_decimal_35_equals_fraction_l3507_350758

/-- The repeating decimal 0.3535... is equal to 35/99 -/
theorem repeating_decimal_35_equals_fraction : ∃ (x : ℚ), x = 35 / 99 ∧ 100 * x - x = 35 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_35_equals_fraction_l3507_350758


namespace NUMINAMATH_CALUDE_equation_equivalence_l3507_350734

theorem equation_equivalence (x y : ℝ) : 
  (2 * x - y = 3) ↔ (y = 2 * x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_equivalence_l3507_350734


namespace NUMINAMATH_CALUDE_percent_relation_l3507_350788

theorem percent_relation (a b : ℝ) (h : a = 1.2 * b) : 
  (4 * b) / a = 10/3 := by sorry

end NUMINAMATH_CALUDE_percent_relation_l3507_350788


namespace NUMINAMATH_CALUDE_sphere_radius_from_segment_l3507_350703

/-- A spherical segment is a portion of a sphere cut off by a plane. -/
structure SphericalSegment where
  base_diameter : ℝ
  height : ℝ

/-- The radius of a sphere given a spherical segment. -/
def sphere_radius (segment : SphericalSegment) : ℝ :=
  sorry

theorem sphere_radius_from_segment (segment : SphericalSegment) 
  (h1 : segment.base_diameter = 24)
  (h2 : segment.height = 8) :
  sphere_radius segment = 13 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_segment_l3507_350703


namespace NUMINAMATH_CALUDE_marble_selection_combinations_l3507_350756

def total_marbles : ℕ := 15
def special_marbles : ℕ := 6
def marbles_to_choose : ℕ := 5
def special_marbles_to_choose : ℕ := 2

theorem marble_selection_combinations :
  (Nat.choose special_marbles special_marbles_to_choose) *
  (Nat.choose (total_marbles - special_marbles) (marbles_to_choose - special_marbles_to_choose)) = 1260 := by
  sorry

end NUMINAMATH_CALUDE_marble_selection_combinations_l3507_350756


namespace NUMINAMATH_CALUDE_water_usage_per_person_l3507_350769

/-- Given a family's water usage, prove the amount of water needed per person per day. -/
theorem water_usage_per_person
  (cost_per_gallon : ℝ)
  (family_size : ℕ)
  (daily_cost : ℝ)
  (h1 : cost_per_gallon = 1)
  (h2 : family_size = 6)
  (h3 : daily_cost = 3) :
  daily_cost / (cost_per_gallon * family_size) = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_water_usage_per_person_l3507_350769


namespace NUMINAMATH_CALUDE_opposite_silver_is_yellow_l3507_350757

-- Define the colors
inductive Color
| Blue | Yellow | Orange | Black | Silver | Violet

-- Define the faces of the cube
inductive Face
| Top | Bottom | Front | Back | Left | Right

-- Define the cube as a function from Face to Color
def Cube := Face → Color

-- Define the three views
def view1 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Yellow ∧ c Face.Right = Color.Orange

def view2 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Black ∧ c Face.Right = Color.Orange

def view3 (c : Cube) : Prop :=
  c Face.Top = Color.Blue ∧ c Face.Front = Color.Violet ∧ c Face.Right = Color.Orange

-- Define the theorem
theorem opposite_silver_is_yellow (c : Cube) :
  view1 c → view2 c → view3 c →
  (∃ f : Face, c f = Color.Silver) →
  (∃ f : Face, c f = Color.Yellow) →
  c Face.Front = Color.Yellow :=
by sorry

end NUMINAMATH_CALUDE_opposite_silver_is_yellow_l3507_350757


namespace NUMINAMATH_CALUDE_intersection_M_naturals_l3507_350761

def M : Set ℤ := {-1, 0, 1}

theorem intersection_M_naturals :
  M ∩ Set.range (Nat.cast : ℕ → ℤ) = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_naturals_l3507_350761


namespace NUMINAMATH_CALUDE_work_on_different_days_probability_l3507_350772

/-- The number of members in the group -/
def num_members : ℕ := 3

/-- The number of days in a week -/
def num_days : ℕ := 7

/-- The probability that the members work on different days -/
def prob_different_days : ℚ := 30 / 49

theorem work_on_different_days_probability :
  (num_members.factorial * (num_days - num_members).choose num_members) / num_days ^ num_members = prob_different_days := by
  sorry

end NUMINAMATH_CALUDE_work_on_different_days_probability_l3507_350772


namespace NUMINAMATH_CALUDE_one_thirds_in_nine_halves_l3507_350747

theorem one_thirds_in_nine_halves : (9 / 2) / (1 / 3) = 27 / 2 := by
  sorry

end NUMINAMATH_CALUDE_one_thirds_in_nine_halves_l3507_350747


namespace NUMINAMATH_CALUDE_congruence_solution_l3507_350771

theorem congruence_solution (y : ℤ) : 
  (10 * y + 3) % 18 = 7 % 18 → y % 9 = 4 % 9 := by
sorry

end NUMINAMATH_CALUDE_congruence_solution_l3507_350771


namespace NUMINAMATH_CALUDE_cos_pi_plus_2alpha_l3507_350790

theorem cos_pi_plus_2alpha (α : Real) 
  (h : Real.sin (Real.pi / 2 - α) = 1 / 3) : 
  Real.cos (Real.pi + 2 * α) = 7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cos_pi_plus_2alpha_l3507_350790


namespace NUMINAMATH_CALUDE_fraction_meaningful_l3507_350728

theorem fraction_meaningful (x : ℝ) : 
  (∃ y : ℝ, y = 3 / (x - 2)) ↔ x ≠ 2 := by sorry

end NUMINAMATH_CALUDE_fraction_meaningful_l3507_350728


namespace NUMINAMATH_CALUDE_only_D_cannot_form_triangle_l3507_350773

/-- A set of three line segments that may or may not form a triangle -/
structure SegmentSet where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if a set of segments can form a triangle using the triangle inequality theorem -/
def can_form_triangle (s : SegmentSet) : Prop :=
  s.a + s.b > s.c ∧ s.b + s.c > s.a ∧ s.c + s.a > s.b

/-- The given sets of line segments -/
def set_A : SegmentSet := ⟨2, 3, 4⟩
def set_B : SegmentSet := ⟨3, 6, 7⟩
def set_C : SegmentSet := ⟨5, 6, 7⟩
def set_D : SegmentSet := ⟨2, 2, 6⟩

/-- Theorem stating that set D is the only set that cannot form a triangle -/
theorem only_D_cannot_form_triangle : 
  can_form_triangle set_A ∧ 
  can_form_triangle set_B ∧ 
  can_form_triangle set_C ∧ 
  ¬can_form_triangle set_D := by
  sorry

end NUMINAMATH_CALUDE_only_D_cannot_form_triangle_l3507_350773


namespace NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l3507_350782

/-- The ratio of the volume of a sphere to the volume of a hemisphere -/
theorem sphere_to_hemisphere_volume_ratio (p : ℝ) (p_pos : p > 0) :
  (4 / 3 * Real.pi * p ^ 3) / (1 / 2 * 4 / 3 * Real.pi * (3 * p) ^ 3) = 2 / 27 := by
  sorry

end NUMINAMATH_CALUDE_sphere_to_hemisphere_volume_ratio_l3507_350782


namespace NUMINAMATH_CALUDE_cube_roots_of_unity_l3507_350742

theorem cube_roots_of_unity (α β : ℂ) 
  (h1 : Complex.abs α = 1) 
  (h2 : Complex.abs β = 1) 
  (h3 : α + β + 1 = 0) : 
  α^3 = 1 ∧ β^3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cube_roots_of_unity_l3507_350742


namespace NUMINAMATH_CALUDE_f_value_at_pi_24_max_monotone_interval_exists_max_monotone_interval_l3507_350787

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sqrt 3 * Real.sin x * Real.cos x + 2 * (Real.cos x)^2

theorem f_value_at_pi_24 : f (Real.pi / 24) = Real.sqrt 2 + 1 := by sorry

theorem max_monotone_interval : 
  ∀ m : ℝ, (∀ x y : ℝ, -m ≤ x ∧ x < y ∧ y ≤ m → f x < f y) → m ≤ Real.pi / 6 := by sorry

theorem exists_max_monotone_interval : 
  ∃ m : ℝ, m = Real.pi / 6 ∧ 
    (∀ x y : ℝ, -m ≤ x ∧ x < y ∧ y ≤ m → f x < f y) ∧
    (∀ m' : ℝ, m' > Real.pi / 6 → ¬(∀ x y : ℝ, -m' ≤ x ∧ x < y ∧ y ≤ m' → f x < f y)) := by sorry

end NUMINAMATH_CALUDE_f_value_at_pi_24_max_monotone_interval_exists_max_monotone_interval_l3507_350787


namespace NUMINAMATH_CALUDE_F_equality_implies_a_half_l3507_350797

/-- Definition of function F -/
def F (a b c : ℝ) : ℝ := a * (b^2 + c^2) + b * c

/-- Theorem: If F(a, 3, 4) = F(a, 2, 5), then a = 1/2 -/
theorem F_equality_implies_a_half :
  ∀ a : ℝ, F a 3 4 = F a 2 5 → a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_F_equality_implies_a_half_l3507_350797


namespace NUMINAMATH_CALUDE_course_selection_methods_l3507_350704

theorem course_selection_methods (n : ℕ) (k : ℕ) : 
  n = 3 → k = 4 → n ^ k = 81 := by sorry

end NUMINAMATH_CALUDE_course_selection_methods_l3507_350704


namespace NUMINAMATH_CALUDE_equation_solution_l3507_350729

theorem equation_solution : 
  ∃! x : ℚ, (x - 100) / 3 = (5 - 3 * x) / 7 ∧ x = 715 / 16 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l3507_350729


namespace NUMINAMATH_CALUDE_white_ball_mutually_exclusive_l3507_350725

-- Define the set of balls
inductive Ball : Type
  | Red : Ball
  | Black : Ball
  | White : Ball

-- Define the set of people
inductive Person : Type
  | A : Person
  | B : Person
  | C : Person

-- Define a distribution as a function from Person to Ball
def Distribution := Person → Ball

-- Define the event "person receives the white ball"
def receives_white_ball (p : Person) (d : Distribution) : Prop :=
  d p = Ball.White

-- State the theorem
theorem white_ball_mutually_exclusive :
  ∀ (d : Distribution),
    (∀ (p1 p2 : Person), p1 ≠ p2 → d p1 ≠ d p2) →
    ¬(receives_white_ball Person.A d ∧ receives_white_ball Person.B d) :=
by sorry

end NUMINAMATH_CALUDE_white_ball_mutually_exclusive_l3507_350725


namespace NUMINAMATH_CALUDE_monic_cubic_polynomial_value_l3507_350770

/-- A monic cubic polynomial is a polynomial of the form x^3 + ax^2 + bx + c -/
def MonicCubicPolynomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ x^3 + a*x^2 + b*x + c

theorem monic_cubic_polynomial_value (a b c : ℝ) :
  let p := MonicCubicPolynomial a b c
  (p 2 = 3) → (p 4 = 9) → (p 6 = 19) → (p 8 = -9) := by
  sorry

end NUMINAMATH_CALUDE_monic_cubic_polynomial_value_l3507_350770


namespace NUMINAMATH_CALUDE_base_6_representation_of_1729_base_6_to_decimal_1729_l3507_350793

/-- Converts a natural number to its base-6 representation as a list of digits -/
def toBase6 (n : ℕ) : List ℕ :=
  if n < 6 then [n]
  else (n % 6) :: toBase6 (n / 6)

/-- Converts a list of base-6 digits to a natural number -/
def fromBase6 (digits : List ℕ) : ℕ :=
  digits.foldr (fun d acc => d + 6 * acc) 0

theorem base_6_representation_of_1729 :
  toBase6 1729 = [1, 0, 0, 0, 2, 1] :=
sorry

theorem base_6_to_decimal_1729 :
  fromBase6 [1, 0, 0, 0, 2, 1] = 1729 :=
sorry

end NUMINAMATH_CALUDE_base_6_representation_of_1729_base_6_to_decimal_1729_l3507_350793


namespace NUMINAMATH_CALUDE_input_is_input_command_l3507_350748

-- Define the type for programming commands
inductive ProgrammingCommand
  | PRINT
  | INPUT
  | THEN
  | END

-- Define a function to check if a command is used for input
def isInputCommand (cmd : ProgrammingCommand) : Prop :=
  match cmd with
  | ProgrammingCommand.INPUT => True
  | _ => False

-- Theorem: INPUT is the only command used for receiving user input
theorem input_is_input_command :
  ∀ (cmd : ProgrammingCommand),
    isInputCommand cmd ↔ cmd = ProgrammingCommand.INPUT :=
  sorry

end NUMINAMATH_CALUDE_input_is_input_command_l3507_350748


namespace NUMINAMATH_CALUDE_both_p_and_q_false_l3507_350784

-- Define proposition p
def p : Prop := ∀ x : ℝ, 2^x > x^2

-- Define proposition q
def q : Prop := (∀ a b : ℝ, a*b > 4 → (a > 2 ∧ b > 2)) ∧ 
                ¬(∀ a b : ℝ, (a > 2 ∧ b > 2) → a*b > 4)

-- Theorem stating that both p and q are false
theorem both_p_and_q_false : ¬p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_both_p_and_q_false_l3507_350784


namespace NUMINAMATH_CALUDE_square_rectangle_area_relation_l3507_350732

theorem square_rectangle_area_relation : 
  ∀ x : ℝ,
  let square_side : ℝ := x - 5
  let rect_length : ℝ := x - 4
  let rect_width : ℝ := x + 3
  let square_area : ℝ := square_side ^ 2
  let rect_area : ℝ := rect_length * rect_width
  (rect_area = 3 * square_area) →
  (∃ y : ℝ, y ≠ x ∧ 
    let square_side' : ℝ := y - 5
    let rect_length' : ℝ := y - 4
    let rect_width' : ℝ := y + 3
    let square_area' : ℝ := square_side' ^ 2
    let rect_area' : ℝ := rect_length' * rect_width'
    (rect_area' = 3 * square_area')) →
  x + y = 33/2 :=
by sorry

end NUMINAMATH_CALUDE_square_rectangle_area_relation_l3507_350732


namespace NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_simplify_expression_4_l3507_350718

-- Problem 1
theorem simplify_expression_1 (a b : ℝ) :
  4 * a^3 + 2 * b - 2 * a^3 + b = 2 * a^3 + 3 * b := by sorry

-- Problem 2
theorem simplify_expression_2 (x : ℝ) :
  2 * x^2 + 6 * x - 6 - (-2 * x^2 + 4 * x + 1) = 4 * x^2 + 2 * x - 7 := by sorry

-- Problem 3
theorem simplify_expression_3 (a b : ℝ) :
  3 * (3 * a^2 - 2 * a * b) - 2 * (4 * a^2 - a * b) = a^2 - 4 * a * b := by sorry

-- Problem 4
theorem simplify_expression_4 (x y : ℝ) :
  6 * x * y^2 - (2 * x - (1/2) * (2 * x - 4 * x * y^2) - x * y^2) = 5 * x * y^2 - x := by sorry

end NUMINAMATH_CALUDE_simplify_expression_1_simplify_expression_2_simplify_expression_3_simplify_expression_4_l3507_350718


namespace NUMINAMATH_CALUDE_percent_to_decimal_five_percent_to_decimal_l3507_350759

theorem percent_to_decimal (p : ℚ) : p / 100 = p * (1 / 100) := by sorry

theorem five_percent_to_decimal : (5 : ℚ) / 100 = 0.05 := by sorry

end NUMINAMATH_CALUDE_percent_to_decimal_five_percent_to_decimal_l3507_350759


namespace NUMINAMATH_CALUDE_line_perp_to_plane_l3507_350799

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation for lines and planes
variable (perp : Line → Line → Prop)
variable (perp_plane : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the parallel relation for lines
variable (parallel : Line → Line → Prop)

-- Define the intersection operation for planes
variable (intersect : Plane → Plane → Line)

-- Theorem statement
theorem line_perp_to_plane 
  (m n : Line) 
  (α β : Plane) 
  (h_diff_lines : m ≠ n) 
  (h_diff_planes : α ≠ β) 
  (h1 : perp_plane α β) 
  (h2 : intersect α β = m) 
  (h3 : perp m n) : 
  perp_line_plane n β :=
sorry

end NUMINAMATH_CALUDE_line_perp_to_plane_l3507_350799


namespace NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l3507_350760

theorem unique_solution_lcm_gcd_equation : 
  ∃! n : ℕ+, Nat.lcm n 120 = Nat.gcd n 120 + 300 ∧ n = 180 := by sorry

end NUMINAMATH_CALUDE_unique_solution_lcm_gcd_equation_l3507_350760


namespace NUMINAMATH_CALUDE_gcf_18_30_l3507_350789

theorem gcf_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcf_18_30_l3507_350789


namespace NUMINAMATH_CALUDE_functional_equation_problem_l3507_350755

/-- The functional equation problem -/
theorem functional_equation_problem (α : ℝ) (hα : α ≠ 0) :
  (∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x^2 + y + f y) = (f x)^2 + α * y) ↔ α = 2 :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_problem_l3507_350755


namespace NUMINAMATH_CALUDE_circle_radius_half_l3507_350792

theorem circle_radius_half (x y : ℝ) : 
  (π * x^2 = π * y^2) →  -- Circles x and y have the same area
  (2 * π * x = 14 * π) →  -- Circle x has a circumference of 14π
  y / 2 = 3.5 :=  -- Half of the radius of circle y is 3.5
by
  sorry

end NUMINAMATH_CALUDE_circle_radius_half_l3507_350792


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l3507_350710

def is_valid_number (n : ℕ) : Prop :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  n = 10 * 23 ∧
  a + b + c + d = 26 ∧
  (b * d / 10) % 10 = a + c ∧
  ∃ m : ℕ, b * d - c^2 = 2^m

theorem unique_four_digit_number : ∃! n : ℕ, n ≥ 1000 ∧ n < 10000 ∧ is_valid_number n :=
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l3507_350710


namespace NUMINAMATH_CALUDE_at_least_one_goes_probability_l3507_350776

def prob_at_least_one_goes (prob_A prob_B : ℚ) : Prop :=
  1 - (1 - prob_A) * (1 - prob_B) = 2/5

theorem at_least_one_goes_probability :
  prob_at_least_one_goes (1/4 : ℚ) (1/5 : ℚ) :=
by
  sorry

end NUMINAMATH_CALUDE_at_least_one_goes_probability_l3507_350776


namespace NUMINAMATH_CALUDE_playground_area_l3507_350726

/-- 
A rectangular playground has a perimeter of 100 meters and its length is twice its width. 
This theorem proves that the area of such a playground is 5000/9 square meters.
-/
theorem playground_area (width : ℝ) (length : ℝ) : 
  (2 * length + 2 * width = 100) →  -- Perimeter condition
  (length = 2 * width) →            -- Length-width relation
  (length * width = 5000 / 9) :=    -- Area calculation
by sorry

end NUMINAMATH_CALUDE_playground_area_l3507_350726


namespace NUMINAMATH_CALUDE_popsicle_sticks_per_group_l3507_350713

theorem popsicle_sticks_per_group 
  (total_sticks : ℕ) 
  (num_groups : ℕ) 
  (sticks_left : ℕ) 
  (h1 : total_sticks = 170) 
  (h2 : num_groups = 10) 
  (h3 : sticks_left = 20) :
  (total_sticks - sticks_left) / num_groups = 15 := by
  sorry

end NUMINAMATH_CALUDE_popsicle_sticks_per_group_l3507_350713


namespace NUMINAMATH_CALUDE_inequality_equivalence_l3507_350717

theorem inequality_equivalence (x : ℝ) : 
  (3 * x - 2 < (x + 2)^2 ∧ (x + 2)^2 < 9 * x - 6) ↔ (2 < x ∧ x < 3) := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l3507_350717


namespace NUMINAMATH_CALUDE_soccer_match_players_l3507_350785

/-- The number of players in a soccer match -/
def num_players : ℕ := 11

/-- The total number of socks in the washing machine -/
def total_socks : ℕ := 22

/-- Each player wears exactly two socks -/
def socks_per_player : ℕ := 2

/-- Theorem: The number of players is 11 given the conditions -/
theorem soccer_match_players :
  num_players = total_socks / socks_per_player :=
by sorry

end NUMINAMATH_CALUDE_soccer_match_players_l3507_350785


namespace NUMINAMATH_CALUDE_calculate_income_l3507_350738

/-- Represents a person's monthly income and expenses -/
structure MonthlyFinances where
  income : ℝ
  household_percent : ℝ
  clothes_percent : ℝ
  medicines_percent : ℝ
  savings : ℝ

/-- Theorem stating the conditions and the result to be proved -/
theorem calculate_income (finances : MonthlyFinances)
  (household_cond : finances.household_percent = 35)
  (clothes_cond : finances.clothes_percent = 20)
  (medicines_cond : finances.medicines_percent = 5)
  (savings_cond : finances.savings = 15000)
  (total_cond : finances.household_percent + finances.clothes_percent + finances.medicines_percent + (finances.savings / finances.income * 100) = 100) :
  finances.income = 37500 := by
  sorry

end NUMINAMATH_CALUDE_calculate_income_l3507_350738


namespace NUMINAMATH_CALUDE_weight_of_new_person_l3507_350791

theorem weight_of_new_person (initial_count : ℕ) (avg_increase : ℝ) (old_weight : ℝ) :
  initial_count = 15 →
  avg_increase = 2.3 →
  old_weight = 80 →
  let total_increase := initial_count * avg_increase
  let new_weight := old_weight + total_increase
  new_weight = 114.5 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_new_person_l3507_350791


namespace NUMINAMATH_CALUDE_equation_solutions_l3507_350779

theorem equation_solutions (x : ℚ) :
  (x = 2/9 ∧ 81 * x^2 + 220 = 196 * x - 15) →
  (5/9 : ℚ)^2 * 81 + 220 = 196 * (5/9 : ℚ) - 15 := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l3507_350779


namespace NUMINAMATH_CALUDE_parabola_y_intercept_l3507_350794

/-- A parabola passing through two given points has a specific y-intercept -/
theorem parabola_y_intercept (b c : ℝ) : 
  (∀ x y, y = 2 * x^2 + b * x + c → 
    ((x = -2 ∧ y = -20) ∨ (x = 2 ∧ y = 24))) → 
  c = -6 := by sorry

end NUMINAMATH_CALUDE_parabola_y_intercept_l3507_350794


namespace NUMINAMATH_CALUDE_product_digit_sum_l3507_350741

theorem product_digit_sum (k : ℕ) : k = 222 ↔ 9 * k = 2000 ∧ ∃! (n : ℕ), 9 * n = 2000 := by
  sorry

end NUMINAMATH_CALUDE_product_digit_sum_l3507_350741


namespace NUMINAMATH_CALUDE_cost_calculation_l3507_350764

/-- The cost of mangos per kg -/
def mango_cost : ℝ := sorry

/-- The cost of rice per kg -/
def rice_cost : ℝ := sorry

/-- The cost of flour per kg -/
def flour_cost : ℝ := 21

/-- The total cost of 4 kg of mangos, 3 kg of rice, and 5 kg of flour -/
def total_cost : ℝ := 4 * mango_cost + 3 * rice_cost + 5 * flour_cost

theorem cost_calculation :
  (10 * mango_cost = 24 * rice_cost) →
  (6 * flour_cost = 2 * rice_cost) →
  total_cost = 898.8 := by sorry

end NUMINAMATH_CALUDE_cost_calculation_l3507_350764


namespace NUMINAMATH_CALUDE_cyrus_remaining_pages_l3507_350751

/-- Represents the number of pages written on each day --/
structure DailyPages where
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day4 : ℕ

/-- Calculates the remaining pages to be written --/
def remainingPages (total : ℕ) (daily : DailyPages) : ℕ :=
  total - (daily.day1 + daily.day2 + daily.day3 + daily.day4)

/-- Theorem stating the number of remaining pages Cyrus needs to write --/
theorem cyrus_remaining_pages :
  let total := 500
  let daily := DailyPages.mk 25 (25 * 2) ((25 * 2) * 2) 10
  remainingPages total daily = 315 := by
  sorry

end NUMINAMATH_CALUDE_cyrus_remaining_pages_l3507_350751


namespace NUMINAMATH_CALUDE_owen_burger_purchases_l3507_350753

/-- The number of burgers Owen purchased each day in June -/
def burgers_per_day (days_in_june : ℕ) (burger_cost : ℕ) (total_spent : ℕ) : ℕ :=
  (total_spent / burger_cost) / days_in_june

/-- Proof that Owen purchased 2 burgers each day in June -/
theorem owen_burger_purchases :
  burgers_per_day 30 12 720 = 2 := by
  sorry

end NUMINAMATH_CALUDE_owen_burger_purchases_l3507_350753


namespace NUMINAMATH_CALUDE_missing_number_proof_l3507_350715

theorem missing_number_proof (a b x : ℕ) (h1 : a = 105) (h2 : b = 147) 
  (h3 : a^3 = 21 * x * 15 * b) : x = 25 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l3507_350715


namespace NUMINAMATH_CALUDE_yogurt_milk_calculation_l3507_350795

/-- The cost of milk per liter in dollars -/
def milk_cost : ℚ := 3/2

/-- The cost of fruit per kilogram in dollars -/
def fruit_cost : ℚ := 2

/-- The amount of fruit needed for one batch of yogurt in kilograms -/
def fruit_per_batch : ℚ := 3

/-- The total cost to produce three batches of yogurt in dollars -/
def total_cost_three_batches : ℚ := 63

/-- The number of liters of milk needed for one batch of yogurt -/
def milk_per_batch : ℚ := 10

theorem yogurt_milk_calculation :
  milk_per_batch * milk_cost * 3 + fruit_per_batch * fruit_cost * 3 = total_cost_three_batches :=
sorry

end NUMINAMATH_CALUDE_yogurt_milk_calculation_l3507_350795


namespace NUMINAMATH_CALUDE_quadratic_solution_sum_l3507_350701

theorem quadratic_solution_sum (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) :
  (∃ x : ℝ, x^2 + 14*x = 65 ∧ x > 0 ∧ x = Real.sqrt a - b) →
  a + b = 121 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_sum_l3507_350701
