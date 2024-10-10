import Mathlib

namespace f_negative_2014_l2017_201727

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem f_negative_2014 (h1 : ∀ x, f x = -f (-x))  -- f is odd
                        (h2 : ∀ x, f (x + 3) = f x)  -- f has period 3
                        (h3 : ∀ x ∈ Set.Icc 0 1, f x = x^2 - x + 2)  -- f on [0,1]
                        : f (-2014) = -2 := by
  sorry

end f_negative_2014_l2017_201727


namespace inverse_proportion_problem_l2017_201784

/-- Two real numbers are inversely proportional -/
def InverselyProportional (a b : ℝ) : Prop :=
  ∃ k : ℝ, k ≠ 0 ∧ a * b = k

theorem inverse_proportion_problem (a₁ a₂ b₁ b₂ : ℝ) 
  (h_inverse : InverselyProportional a₁ b₁) 
  (h_initial : a₁ = 40 ∧ b₁ = 8) 
  (h_final : b₂ = 10) : 
  a₂ = 32 ∧ InverselyProportional a₂ b₂ :=
sorry

end inverse_proportion_problem_l2017_201784


namespace exterior_angle_measure_l2017_201768

theorem exterior_angle_measure (a b : ℝ) (ha : a = 70) (hb : b = 40) :
  180 - a = 110 :=
by
  sorry

end exterior_angle_measure_l2017_201768


namespace number_of_bs_l2017_201763

/-- Represents the number of students earning each grade in a philosophy class. -/
structure GradeDistribution where
  total : ℕ
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if the grade distribution satisfies the given conditions. -/
def isValidDistribution (g : GradeDistribution) : Prop :=
  g.total = 40 ∧
  g.a = 0.5 * g.b ∧
  g.c = 2 * g.b ∧
  g.a + g.b + g.c = g.total

/-- Theorem stating the number of B's in the class. -/
theorem number_of_bs (g : GradeDistribution) 
  (h : isValidDistribution g) : g.b = 40 / 3.5 := by
  sorry

end number_of_bs_l2017_201763


namespace equation_solution_l2017_201733

theorem equation_solution : 
  ∀ x : ℝ, (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) ↔ (x = 4 + Real.sqrt 2 ∨ x = 4 - Real.sqrt 2) :=
by sorry

end equation_solution_l2017_201733


namespace min_value_of_f_l2017_201713

noncomputable def f (x : ℝ) : ℝ := (x - 3) * Real.exp x

theorem min_value_of_f :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x ≥ f x_min ∧ f x_min = -Real.exp 2 :=
sorry

end min_value_of_f_l2017_201713


namespace count_seating_arrangements_l2017_201759

/-- Represents a seating arrangement in a 5x5 classroom -/
def SeatingArrangement := Fin 5 → Fin 5 → Bool

/-- A seating arrangement is valid if for each occupied desk, either its row or column is full -/
def is_valid (arrangement : SeatingArrangement) : Prop :=
  ∀ i j, arrangement i j → 
    (∀ k, arrangement i k) ∨ (∀ k, arrangement k j)

/-- The total number of valid seating arrangements -/
def total_arrangements : ℕ := sorry

theorem count_seating_arrangements :
  total_arrangements = 962 := by sorry

end count_seating_arrangements_l2017_201759


namespace ball_weights_l2017_201794

/-- The weight of a red ball in grams -/
def red_weight : ℝ := sorry

/-- The weight of a yellow ball in grams -/
def yellow_weight : ℝ := sorry

/-- The total weight of 5 red balls and 3 yellow balls in grams -/
def total_weight_1 : ℝ := 5 * red_weight + 3 * yellow_weight

/-- The total weight of 5 yellow balls and 3 red balls in grams -/
def total_weight_2 : ℝ := 5 * yellow_weight + 3 * red_weight

theorem ball_weights :
  total_weight_1 = 42 ∧ total_weight_2 = 38 → red_weight = 6 ∧ yellow_weight = 4 := by
  sorry

end ball_weights_l2017_201794


namespace vector_equality_l2017_201715

def a : ℝ × ℝ := (4, 2)
def b (k : ℝ) : ℝ × ℝ := (2 - k, k - 1)

theorem vector_equality (k : ℝ) :
  ‖a + b k‖ = ‖a - b k‖ → k = 3 := by sorry

end vector_equality_l2017_201715


namespace polynomial_independent_of_x_l2017_201749

-- Define the polynomial
def polynomial (x y a b : ℝ) : ℝ := 9*x^3 + y^2 + a*x - b*x^3 + x + 5

-- State the theorem
theorem polynomial_independent_of_x (y a b : ℝ) :
  (∀ x₁ x₂ : ℝ, polynomial x₁ y a b = polynomial x₂ y a b) →
  a - b = -10 := by
  sorry

end polynomial_independent_of_x_l2017_201749


namespace first_column_is_seven_l2017_201739

/-- Represents a 5x2 grid with one empty cell -/
def Grid := Fin 9 → Fin 9

/-- The sum of a column in the grid -/
def column_sum (g : Grid) (col : Fin 5) : ℕ :=
  if col = 0 then g 0
  else if col = 1 then g 1 + g 2
  else if col = 2 then g 3 + g 4
  else if col = 3 then g 5 + g 6
  else g 7 + g 8

/-- Predicate for a valid grid arrangement -/
def is_valid_grid (g : Grid) : Prop :=
  (∀ i j : Fin 9, i ≠ j → g i ≠ g j) ∧
  (∀ col : Fin 4, column_sum g (col + 1) = column_sum g col + 1)

theorem first_column_is_seven (g : Grid) (h : is_valid_grid g) : g 0 = 7 := by
  sorry

end first_column_is_seven_l2017_201739


namespace failing_marks_difference_l2017_201796

/-- The number of marks needed to pass the exam -/
def passing_marks : ℝ := 199.99999999999997

/-- The percentage of marks obtained by the failing candidate -/
def failing_percentage : ℝ := 0.30

/-- The percentage of marks obtained by the passing candidate -/
def passing_percentage : ℝ := 0.45

/-- The number of marks the passing candidate gets above the passing mark -/
def marks_above_passing : ℝ := 25

/-- Theorem stating the number of marks by which the failing candidate fails -/
theorem failing_marks_difference : 
  let total_marks := (passing_marks + marks_above_passing) / passing_percentage
  passing_marks - (failing_percentage * total_marks) = 50 := by
sorry

end failing_marks_difference_l2017_201796


namespace odd_prime_and_odd_natural_not_divide_l2017_201769

theorem odd_prime_and_odd_natural_not_divide (p n : ℕ) : 
  Nat.Prime p → Odd p → Odd n → ¬(p * n + 1 ∣ p^p - 1) := by
  sorry

end odd_prime_and_odd_natural_not_divide_l2017_201769


namespace polynomial_division_l2017_201764

theorem polynomial_division (x : ℤ) : 
  ∃ (p : ℤ → ℤ), x^13 + 2*x + 180 = (x^2 - x + 3) * p x := by
sorry

end polynomial_division_l2017_201764


namespace shyne_eggplant_packets_l2017_201786

/-- The number of eggplants that can be grown from one seed packet -/
def eggplants_per_packet : ℕ := 14

/-- The number of sunflowers that can be grown from one seed packet -/
def sunflowers_per_packet : ℕ := 10

/-- The number of sunflower seed packets Shyne bought -/
def sunflower_packets : ℕ := 6

/-- The total number of plants Shyne can grow in her backyard -/
def total_plants : ℕ := 116

/-- The number of eggplant seed packets Shyne bought -/
def eggplant_packets : ℕ := 4

theorem shyne_eggplant_packets : 
  eggplant_packets * eggplants_per_packet + sunflower_packets * sunflowers_per_packet = total_plants :=
by sorry

end shyne_eggplant_packets_l2017_201786


namespace line_intersects_segment_m_range_l2017_201714

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- A line in 2D space defined by the equation x + my + m = 0 -/
structure Line where
  m : ℝ

def intersectsSegment (l : Line) (a b : Point) : Prop :=
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧
    (1 - t) * a.x + t * b.x + l.m * ((1 - t) * a.y + t * b.y) + l.m = 0

theorem line_intersects_segment_m_range (l : Line) :
  let a : Point := ⟨-1, 1⟩
  let b : Point := ⟨2, -2⟩
  intersectsSegment l a b → 1/2 ≤ l.m ∧ l.m ≤ 2 := by sorry

end line_intersects_segment_m_range_l2017_201714


namespace quadratic_equation_unique_solution_l2017_201741

theorem quadratic_equation_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 12 * x + c = 0) →  -- exactly one solution
  (a + c = 14) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 7 - Real.sqrt 31 ∧ c = 7 + Real.sqrt 31) := by
sorry

end quadratic_equation_unique_solution_l2017_201741


namespace jake_weight_loss_l2017_201760

theorem jake_weight_loss (total_weight jake_weight : ℝ) 
  (h1 : total_weight = 290)
  (h2 : jake_weight = 196) : 
  jake_weight - 2 * (total_weight - jake_weight) = 8 :=
sorry

end jake_weight_loss_l2017_201760


namespace least_k_for_inequality_l2017_201716

theorem least_k_for_inequality (k : ℤ) : 
  (∀ m : ℤ, m < k → (0.000010101 * (10 : ℝ)^m ≤ 10000)) →
  (0.000010101 * (10 : ℝ)^k > 10000) →
  k = 9 := by
sorry

end least_k_for_inequality_l2017_201716


namespace fraction_power_equality_l2017_201746

theorem fraction_power_equality : (72000 ^ 5) / (18000 ^ 5) = 1024 := by sorry

end fraction_power_equality_l2017_201746


namespace janes_age_l2017_201791

/-- Jane's babysitting problem -/
theorem janes_age :
  ∀ (jane_start_age : ℕ) 
    (years_since_stopped : ℕ) 
    (oldest_babysat_age : ℕ),
  jane_start_age = 16 →
  years_since_stopped = 10 →
  oldest_babysat_age = 24 →
  ∃ (jane_current_age : ℕ),
    jane_current_age = 38 ∧
    (∀ (child_age : ℕ),
      child_age ≤ oldest_babysat_age →
      child_age ≤ (jane_current_age - years_since_stopped) / 2) :=
by sorry

end janes_age_l2017_201791


namespace kitchen_renovation_rate_l2017_201711

/-- The hourly rate for professionals renovating Kamil's kitchen -/
def hourly_rate (professionals : ℕ) (hours_per_day : ℕ) (days : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (professionals * hours_per_day * days)

/-- Theorem stating the hourly rate for the kitchen renovation professionals -/
theorem kitchen_renovation_rate : 
  hourly_rate 2 6 7 1260 = 15 := by
  sorry

end kitchen_renovation_rate_l2017_201711


namespace geometric_sequence_range_l2017_201761

theorem geometric_sequence_range (a₁ a₂ a₃ a₄ : ℝ) :
  (∃ q : ℝ, a₂ = a₁ * q ∧ a₃ = a₂ * q ∧ a₄ = a₃ * q) →
  (0 < a₁ ∧ a₁ < 1) →
  (1 < a₂ ∧ a₂ < 2) →
  (2 < a₃ ∧ a₃ < 4) →
  (2 * Real.sqrt 2 < a₄ ∧ a₄ < 16) :=
by sorry

end geometric_sequence_range_l2017_201761


namespace bake_sale_donation_percentage_l2017_201772

/-- Proves that the percentage of bake sale proceeds donated to the shelter is 75% --/
theorem bake_sale_donation_percentage :
  ∀ (carwash_earnings bake_sale_earnings lawn_mowing_earnings total_donation : ℚ),
  carwash_earnings = 100 →
  bake_sale_earnings = 80 →
  lawn_mowing_earnings = 50 →
  total_donation = 200 →
  0.9 * carwash_earnings + 1 * lawn_mowing_earnings + 
    (total_donation - (0.9 * carwash_earnings + 1 * lawn_mowing_earnings)) = total_donation →
  (total_donation - (0.9 * carwash_earnings + 1 * lawn_mowing_earnings)) / bake_sale_earnings = 0.75 :=
by sorry

end bake_sale_donation_percentage_l2017_201772


namespace correct_dot_counts_l2017_201766

/-- Represents a single die face -/
inductive DieFace
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the four visible faces of the dice configuration -/
structure VisibleFaces :=
  (A : DieFace)
  (B : DieFace)
  (C : DieFace)
  (D : DieFace)

/-- Counts the number of dots on a die face -/
def dotCount (face : DieFace) : Nat :=
  match face with
  | DieFace.one => 1
  | DieFace.two => 2
  | DieFace.three => 3
  | DieFace.four => 4
  | DieFace.five => 5
  | DieFace.six => 6

/-- The configuration of dice as described in the problem -/
def diceConfiguration : VisibleFaces :=
  { A := DieFace.three
  , B := DieFace.five
  , C := DieFace.six
  , D := DieFace.five }

/-- Theorem stating the correct number of dots on each visible face -/
theorem correct_dot_counts :
  dotCount diceConfiguration.A = 3 ∧
  dotCount diceConfiguration.B = 5 ∧
  dotCount diceConfiguration.C = 6 ∧
  dotCount diceConfiguration.D = 5 :=
sorry

end correct_dot_counts_l2017_201766


namespace ahn_max_number_l2017_201790

theorem ahn_max_number : ∃ (max : ℕ), max = 650 ∧ 
  ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 → 3 * (300 - n) + 50 ≤ max :=
by sorry

end ahn_max_number_l2017_201790


namespace bee_speed_difference_l2017_201775

/-- Proves the difference in bee's speed between two flight segments -/
theorem bee_speed_difference (time_daisy_rose time_rose_poppy : ℝ)
  (distance_difference : ℝ) (speed_daisy_rose : ℝ)
  (h1 : time_daisy_rose = 10)
  (h2 : time_rose_poppy = 6)
  (h3 : distance_difference = 8)
  (h4 : speed_daisy_rose = 2.6) :
  speed_daisy_rose * time_daisy_rose - distance_difference = 
  (speed_daisy_rose + 0.4) * time_rose_poppy := by
  sorry

end bee_speed_difference_l2017_201775


namespace log_equality_implies_ratio_l2017_201797

theorem log_equality_implies_ratio (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : (Real.log a / Real.log 9) = (Real.log b / Real.log 12) ∧ 
       (Real.log a / Real.log 9) = (Real.log (2 * (a + b)) / Real.log 16)) : 
  b / a = Real.sqrt 3 + 1 := by
sorry

end log_equality_implies_ratio_l2017_201797


namespace sasha_train_journey_l2017_201747

/-- Represents a day of the week -/
inductive DayOfWeek
  | Saturday
  | Sunday
  | Monday

/-- Represents the train journey -/
structure TrainJourney where
  departureDay : DayOfWeek
  arrivalDay : DayOfWeek
  journeyDuration : Nat
  departureDateNumber : Nat
  arrivalDateNumber : Nat
  trainCarNumber : Nat
  seatNumber : Nat

/-- The conditions of Sasha's train journey -/
def sashasJourney : TrainJourney :=
  { departureDay := DayOfWeek.Saturday
  , arrivalDay := DayOfWeek.Monday
  , journeyDuration := 50
  , departureDateNumber := 31  -- Assuming end of month
  , arrivalDateNumber := 2     -- Assuming start of next month
  , trainCarNumber := 2
  , seatNumber := 1
  }

theorem sasha_train_journey :
  ∀ (journey : TrainJourney),
    journey.departureDay = DayOfWeek.Saturday →
    journey.arrivalDay = DayOfWeek.Monday →
    journey.journeyDuration = 50 →
    journey.arrivalDateNumber = journey.trainCarNumber →
    journey.seatNumber < journey.trainCarNumber →
    journey.departureDateNumber > journey.trainCarNumber →
    journey.trainCarNumber = 2 ∧ journey.seatNumber = 1 := by
  sorry

#check sasha_train_journey

end sasha_train_journey_l2017_201747


namespace rectangle_from_triangles_l2017_201770

/-- Represents a right-angled triangle tile with integer side lengths -/
structure Triangle :=
  (a b c : ℕ)

/-- Represents a rectangle with integer side lengths -/
structure Rectangle :=
  (width height : ℕ)

/-- Checks if a triangle is valid (right-angled and positive sides) -/
def isValidTriangle (t : Triangle) : Prop :=
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧ t.a^2 + t.b^2 = t.c^2

/-- Checks if a rectangle can be formed from a given number of triangles -/
def canFormRectangle (r : Rectangle) (t : Triangle) : Prop :=
  ∃ (n : ℕ), n > 0 ∧ 2 * n * t.a * t.b = r.width * r.height

theorem rectangle_from_triangles 
  (jackTile : Triangle)
  (targetRect : Rectangle)
  (h1 : isValidTriangle jackTile)
  (h2 : jackTile.a = 3 ∧ jackTile.b = 4 ∧ jackTile.c = 5)
  (h3 : targetRect.width = 2016 ∧ targetRect.height = 2021) :
  canFormRectangle targetRect jackTile :=
sorry

end rectangle_from_triangles_l2017_201770


namespace tims_garden_fence_length_l2017_201731

/-- The perimeter of an irregular pentagon with given side lengths -/
def pentagon_perimeter (a b c d e : ℝ) : ℝ := a + b + c + d + e

/-- Theorem: The perimeter of Tim's garden fence -/
theorem tims_garden_fence_length :
  pentagon_perimeter 28 32 25 35 39 = 159 := by
  sorry

end tims_garden_fence_length_l2017_201731


namespace ellipse_sum_bounds_l2017_201744

theorem ellipse_sum_bounds (x y : ℝ) : 
  x^2 / 2 + y^2 / 3 = 1 → 
  ∃ (S : ℝ), S = x + y ∧ -Real.sqrt 5 ≤ S ∧ S ≤ Real.sqrt 5 ∧
  (∃ (x₁ y₁ : ℝ), x₁^2 / 2 + y₁^2 / 3 = 1 ∧ x₁ + y₁ = -Real.sqrt 5) ∧
  (∃ (x₂ y₂ : ℝ), x₂^2 / 2 + y₂^2 / 3 = 1 ∧ x₂ + y₂ = Real.sqrt 5) :=
by sorry

end ellipse_sum_bounds_l2017_201744


namespace perpendicular_vectors_implies_m_equals_three_l2017_201736

/-- Given vectors a and b in ℝ², if a is perpendicular to (a - b), then the second component of b is -1 and m = 3. -/
theorem perpendicular_vectors_implies_m_equals_three (a b : ℝ × ℝ) (m : ℝ) 
    (h1 : a = (2, 1))
    (h2 : b = (m, -1))
    (h3 : a • (a - b) = 0) :
  m = 3 := by
  sorry

end perpendicular_vectors_implies_m_equals_three_l2017_201736


namespace roots_nature_l2017_201774

/-- The quadratic equation x^2 + 2x + m = 0 -/
def quadratic_equation (x m : ℝ) : Prop :=
  x^2 + 2*x + m = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  4 - 4*m

/-- The nature of the roots is determined by the value of m -/
theorem roots_nature (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ quadratic_equation x m ∧ quadratic_equation y m) ∨
  (∃ x : ℝ, quadratic_equation x m ∧ ∀ y : ℝ, quadratic_equation y m → y = x) ∨
  (∀ x : ℝ, ¬quadratic_equation x m) :=
sorry

end roots_nature_l2017_201774


namespace initial_value_proof_l2017_201729

-- Define the property tax rate
def tax_rate : ℝ := 0.10

-- Define the new assessed value
def new_value : ℝ := 28000

-- Define the property tax increase
def tax_increase : ℝ := 800

-- Theorem statement
theorem initial_value_proof :
  ∃ (initial_value : ℝ),
    initial_value * tax_rate + tax_increase = new_value * tax_rate ∧
    initial_value = 20000 :=
by sorry

end initial_value_proof_l2017_201729


namespace divisibility_problem_l2017_201706

theorem divisibility_problem (m n : ℕ) (hm : m > 0) (hn : n > 0) 
  (h_div : (5 * m + n) ∣ (5 * n + m)) : m ∣ n := by
  sorry

end divisibility_problem_l2017_201706


namespace stretches_per_meter_l2017_201776

/-- Given the following conversions between paces, stretches, leaps, and meters:
    p paces equals q stretches,
    r leaps equals s stretches,
    t leaps equals u meters,
    prove that the number of stretches in one meter is ts/ur. -/
theorem stretches_per_meter
  (p q r s t u : ℝ)
  (h1 : p * q⁻¹ = 1)  -- p paces equals q stretches
  (h2 : r * s⁻¹ = 1)  -- r leaps equals s stretches
  (h3 : t * u⁻¹ = 1)  -- t leaps equals u meters
  : 1 = t * s * (u * r)⁻¹ :=
sorry

end stretches_per_meter_l2017_201776


namespace circle_center_and_radius_l2017_201783

-- Define the circle equation
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 - 4*x = 0

-- Theorem statement
theorem circle_center_and_radius :
  ∃ (center_x center_y radius : ℝ),
    (center_x = 2 ∧ center_y = 0 ∧ radius = 2) ∧
    (∀ (x y : ℝ), circle_equation x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2) :=
by sorry

end circle_center_and_radius_l2017_201783


namespace shopping_spree_cost_equalization_l2017_201771

/-- Given the spending amounts and agreement to equally share costs, 
    prove that the difference between what Charlie gives to Bob and 
    what Alice gives to Bob is 30. -/
theorem shopping_spree_cost_equalization 
  (charlie_spent : ℝ) 
  (alice_spent : ℝ) 
  (bob_spent : ℝ) 
  (h1 : charlie_spent = 150)
  (h2 : alice_spent = 180)
  (h3 : bob_spent = 210)
  (c : ℝ)  -- amount Charlie gives to Bob
  (a : ℝ)  -- amount Alice gives to Bob
  (h4 : c = (charlie_spent + alice_spent + bob_spent) / 3 - charlie_spent)
  (h5 : a = (charlie_spent + alice_spent + bob_spent) / 3 - alice_spent) :
  c - a = 30 := by
sorry


end shopping_spree_cost_equalization_l2017_201771


namespace sqrt_calculations_l2017_201779

theorem sqrt_calculations :
  (∃ (x y : ℝ), x = Real.sqrt 3 ∧ y = Real.sqrt 2 ∧
    x * y - Real.sqrt 12 / Real.sqrt 8 = Real.sqrt 6 / 2) ∧
  ((Real.sqrt 2 - 3)^2 - Real.sqrt 2^2 - Real.sqrt (2^2) - Real.sqrt 2 = 7 - 7 * Real.sqrt 2) :=
by sorry

end sqrt_calculations_l2017_201779


namespace belt_cost_calculation_l2017_201717

def initial_budget : ℕ := 200
def shirt_cost : ℕ := 30
def pants_cost : ℕ := 46
def coat_cost : ℕ := 38
def socks_cost : ℕ := 11
def shoes_cost : ℕ := 41
def amount_left : ℕ := 16

theorem belt_cost_calculation : 
  initial_budget - (shirt_cost + pants_cost + coat_cost + socks_cost + shoes_cost + amount_left) = 18 := by
  sorry

end belt_cost_calculation_l2017_201717


namespace ball_cost_price_l2017_201730

/-- The cost price of a single ball -/
def cost_price : ℕ := sorry

/-- The selling price of 20 balls -/
def selling_price : ℕ := 720

/-- The number of balls sold -/
def balls_sold : ℕ := 20

/-- The number of balls whose cost equals the loss -/
def balls_loss : ℕ := 5

theorem ball_cost_price : 
  cost_price = 48 ∧ 
  selling_price = balls_sold * cost_price - balls_loss * cost_price :=
sorry

end ball_cost_price_l2017_201730


namespace triangle_angle_measure_l2017_201732

theorem triangle_angle_measure (A B C : Real) (h1 : 3 * Real.sin A + 4 * Real.cos B = 6)
  (h2 : 4 * Real.sin B + 3 * Real.cos A = 1) (h3 : A + B + C = Real.pi) :
  C = Real.pi / 6 := by
  sorry

end triangle_angle_measure_l2017_201732


namespace hotel_floors_l2017_201785

theorem hotel_floors (available_rooms : ℕ) (rooms_per_floor : ℕ) (unavailable_floors : ℕ) : 
  available_rooms = 90 → rooms_per_floor = 10 → unavailable_floors = 1 →
  (available_rooms / rooms_per_floor + unavailable_floors = 10) := by
sorry

end hotel_floors_l2017_201785


namespace special_sum_of_squares_l2017_201705

theorem special_sum_of_squares (n : ℕ) (a b : ℕ) : 
  n ≥ 2 →
  n = a^2 + b^2 →
  (∀ d : ℕ, d > 1 ∧ d ∣ n → a ≤ d) →
  a ∣ n →
  b ∣ n →
  n = 8 ∨ n = 20 :=
by sorry

end special_sum_of_squares_l2017_201705


namespace factorial_minus_one_mod_930_l2017_201710

theorem factorial_minus_one_mod_930 : (Nat.factorial 30 - 1) % 930 = 29 := by
  sorry

end factorial_minus_one_mod_930_l2017_201710


namespace log_sum_equals_six_l2017_201758

theorem log_sum_equals_six :
  2 * (Real.log 10 / Real.log 5) + (Real.log 0.25 / Real.log 5) + 8^(2/3) = 6 := by
  sorry

end log_sum_equals_six_l2017_201758


namespace time_to_school_building_l2017_201778

/-- Proves that the time to get from the school gate to the school building is 6 minutes -/
theorem time_to_school_building 
  (total_time : ℕ) 
  (time_to_gate : ℕ) 
  (time_to_room : ℕ) 
  (h1 : total_time = 30) 
  (h2 : time_to_gate = 15) 
  (h3 : time_to_room = 9) : 
  total_time - time_to_gate - time_to_room = 6 := by
  sorry

#check time_to_school_building

end time_to_school_building_l2017_201778


namespace sufficient_not_necessary_l2017_201793

theorem sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 1 ∧ b > 2 → a + b > 3 ∧ a * b > 2) ∧
  (∃ a b : ℝ, a + b > 3 ∧ a * b > 2 ∧ ¬(a > 1 ∧ b > 2)) :=
by sorry

end sufficient_not_necessary_l2017_201793


namespace grocer_purchase_price_l2017_201708

/-- Represents the price at which the grocer purchased 3 pounds of bananas -/
def purchase_price : ℝ := sorry

/-- Represents the total quantity of bananas purchased in pounds -/
def total_quantity : ℝ := 72

/-- Represents the profit made by the grocer -/
def profit : ℝ := 6

/-- Represents the selling price of 4 pounds of bananas -/
def selling_price : ℝ := 1

/-- Theorem stating that the purchase price for 3 pounds of bananas is $0.50 -/
theorem grocer_purchase_price : purchase_price = 0.50 := by
  sorry

end grocer_purchase_price_l2017_201708


namespace sapling_planting_equation_l2017_201752

theorem sapling_planting_equation (x : ℤ) : 
  (∀ (total : ℤ), (5 * x + 3 = total) ↔ (6 * x = total + 4)) :=
by sorry

end sapling_planting_equation_l2017_201752


namespace tens_digit_of_9_to_1503_l2017_201753

theorem tens_digit_of_9_to_1503 : ∃ n : ℕ, n ≥ 0 ∧ n < 10 ∧ 9^1503 ≡ 20 + n [ZMOD 100] :=
sorry

end tens_digit_of_9_to_1503_l2017_201753


namespace arithmetic_progression_of_primes_l2017_201748

theorem arithmetic_progression_of_primes (p d : ℕ) : 
  p ≠ 3 →
  Prime (p - d) →
  Prime p →
  Prime (p + d) →
  ∃ k : ℕ, d = 6 * k :=
sorry

end arithmetic_progression_of_primes_l2017_201748


namespace geometric_sequence_sum_l2017_201755

/-- Given a geometric sequence {a_n}, prove that if a_1 + a_3 = 20 and a_2 + a_4 = 40, then a_3 + a_5 = 80 -/
theorem geometric_sequence_sum (a : ℕ → ℝ) (h_geom : ∀ n, a (n + 1) / a n = a (n + 2) / a (n + 1)) 
  (h_sum1 : a 1 + a 3 = 20) (h_sum2 : a 2 + a 4 = 40) : 
  a 3 + a 5 = 80 := by
sorry

end geometric_sequence_sum_l2017_201755


namespace flu_virus_diameter_scientific_notation_l2017_201702

theorem flu_virus_diameter_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.000000823 = a * 10^n ∧ 1 ≤ a ∧ a < 10 ∧ a = 8.23 ∧ n = -7 := by
  sorry

end flu_virus_diameter_scientific_notation_l2017_201702


namespace no_zero_points_when_k_is_one_exactly_one_zero_point_when_k_is_negative_exists_k_with_two_zero_points_l2017_201718

-- Define the piecewise function f(x)
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then Real.exp x - k * x else k * x^2 - x + 1

-- Statement 1: When k = 1, f(x) has no zero points
theorem no_zero_points_when_k_is_one :
  ∀ x : ℝ, f 1 x ≠ 0 := by sorry

-- Statement 2: When k < 0, f(x) has exactly one zero point
theorem exactly_one_zero_point_when_k_is_negative :
  ∀ k : ℝ, k < 0 → ∃! x : ℝ, f k x = 0 := by sorry

-- Statement 3: There exists a k such that f(x) has two zero points
theorem exists_k_with_two_zero_points :
  ∃ k : ℝ, ∃ x y : ℝ, x ≠ y ∧ f k x = 0 ∧ f k y = 0 := by sorry

end no_zero_points_when_k_is_one_exactly_one_zero_point_when_k_is_negative_exists_k_with_two_zero_points_l2017_201718


namespace least_integer_satisfying_inequality_l2017_201703

theorem least_integer_satisfying_inequality :
  ∃ (x : ℤ), (∀ (y : ℤ), 3 * |2 * y - 1| + 6 < 24 → x ≤ y) ∧ (3 * |2 * x - 1| + 6 < 24) :=
by
  -- The proof would go here
  sorry

end least_integer_satisfying_inequality_l2017_201703


namespace regular_17gon_symmetry_sum_l2017_201725

/-- The number of sides in the regular polygon -/
def n : ℕ := 17

/-- The number of lines of symmetry in a regular n-gon -/
def L (n : ℕ) : ℕ := n

/-- The smallest positive angle of rotational symmetry in degrees for a regular n-gon -/
def R (n : ℕ) : ℚ := 360 / n

/-- Theorem: For a regular 17-gon, the sum of its number of lines of symmetry
    and its smallest positive angle of rotational symmetry (in degrees)
    is equal to 17 + (360 / 17) -/
theorem regular_17gon_symmetry_sum :
  (L n : ℚ) + R n = 17 + 360 / 17 := by sorry

end regular_17gon_symmetry_sum_l2017_201725


namespace y_divisibility_l2017_201707

def y : ℕ := 64 + 96 + 192 + 256 + 352 + 480 + 4096 + 8192

theorem y_divisibility : 
  (∃ k : ℕ, y = 32 * k) ∧ ¬(∃ m : ℕ, y = 64 * m) :=
by sorry

end y_divisibility_l2017_201707


namespace prime_between_squares_l2017_201719

/-- A number is a perfect square if it's the square of some integer. -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

/-- A number is prime if it's greater than 1 and its only divisors are 1 and itself. -/
def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m ∣ p → m = 1 ∨ m = p

theorem prime_between_squares : 
  ∃! p : ℕ, is_prime p ∧ 
    (∃ n : ℕ, is_perfect_square n ∧ p = n + 12) ∧
    (∃ m : ℕ, is_perfect_square m ∧ p + 9 = m) :=
sorry

end prime_between_squares_l2017_201719


namespace smallest_self_repeating_square_end_l2017_201743

/-- A function that returns the last n digits of a natural number in base 10 -/
def lastNDigits (n : ℕ) (digits : ℕ) : ℕ :=
  n % (10 ^ digits)

/-- The theorem stating that 40625 is the smallest positive integer N such that
    N and N^2 end in the same sequence of five digits in base 10,
    with the first of these five digits being non-zero -/
theorem smallest_self_repeating_square_end : ∀ N : ℕ,
  N > 0 ∧ 
  lastNDigits N 5 = lastNDigits (N^2) 5 ∧
  N ≥ 10000 →
  N ≥ 40625 := by
  sorry

end smallest_self_repeating_square_end_l2017_201743


namespace multiply_by_hundred_l2017_201780

theorem multiply_by_hundred (x : ℝ) : x = 15.46 → x * 100 = 1546 := by
  sorry

end multiply_by_hundred_l2017_201780


namespace quadrilateral_classification_l2017_201740

/-
  Definitions:
  - a, b, c, d: vectors representing sides AB, BC, CD, DA of quadrilateral ABCD
  - m, n: real numbers
-/

variable (a b c d : ℝ × ℝ)
variable (m n : ℝ)

/-- A quadrilateral is a rectangle if its adjacent sides are perpendicular and opposite sides are equal -/
def is_rectangle (a b c d : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 = 0 ∧
  b.1 * c.1 + b.2 * c.2 = 0 ∧
  c.1 * d.1 + c.2 * d.2 = 0 ∧
  d.1 * a.1 + d.2 * a.2 = 0 ∧
  a.1^2 + a.2^2 = c.1^2 + c.2^2 ∧
  b.1^2 + b.2^2 = d.1^2 + d.2^2

/-- A quadrilateral is an isosceles trapezoid if it has one pair of parallel sides and the other pair of equal length -/
def is_isosceles_trapezoid (a b c d : ℝ × ℝ) : Prop :=
  (a.1 * d.2 - a.2 * d.1 = b.1 * c.2 - b.2 * c.1) ∧
  (a.1^2 + a.2^2 = c.1^2 + c.2^2) ∧
  (a.1 * d.2 - a.2 * d.1 ≠ 0 ∨ b.1 * c.2 - b.2 * c.1 ≠ 0)

theorem quadrilateral_classification (h1 : a.1 * b.1 + a.2 * b.2 = m) 
                                     (h2 : b.1 * c.1 + b.2 * c.2 = m)
                                     (h3 : c.1 * d.1 + c.2 * d.2 = n)
                                     (h4 : d.1 * a.1 + d.2 * a.2 = n) :
  (m = n → is_rectangle a b c d) ∧
  (m ≠ n → is_isosceles_trapezoid a b c d) :=
sorry

end quadrilateral_classification_l2017_201740


namespace unique_coprime_solution_l2017_201767

theorem unique_coprime_solution (n : ℕ+) :
  ∀ p q : ℤ,
  p > 0 ∧ q > 0 ∧
  Int.gcd p q = 1 ∧
  p + q^2 = (n.val^2 + 1) * p^2 + q →
  p = n.val + 1 ∧ q = n.val^2 + n.val + 1 := by
  sorry

end unique_coprime_solution_l2017_201767


namespace carls_dad_contribution_l2017_201723

def weekly_savings : ℕ := 25
def weeks_saved : ℕ := 6
def coat_cost : ℕ := 170
def bill_fraction : ℚ := 1/3

theorem carls_dad_contribution :
  let total_savings := weekly_savings * weeks_saved
  let remaining_savings := total_savings - (bill_fraction * total_savings).floor
  coat_cost - remaining_savings = 70 := by
  sorry

end carls_dad_contribution_l2017_201723


namespace conjunction_false_implies_one_false_l2017_201787

theorem conjunction_false_implies_one_false (p q : Prop) :
  (p ∧ q) = False → (p = False ∨ q = False) :=
by sorry

end conjunction_false_implies_one_false_l2017_201787


namespace pencils_taken_l2017_201754

theorem pencils_taken (initial_pencils remaining_pencils : ℕ) 
  (h1 : initial_pencils = 79)
  (h2 : remaining_pencils = 75) :
  initial_pencils - remaining_pencils = 4 := by
  sorry

end pencils_taken_l2017_201754


namespace isosceles_triangle_not_unique_l2017_201762

/-- An isosceles triangle -/
structure IsoscelesTriangle where
  /-- The base angle of the isosceles triangle -/
  baseAngle : ℝ
  /-- The altitude to the base of the isosceles triangle -/
  altitude : ℝ
  /-- The base of the isosceles triangle -/
  base : ℝ

/-- Theorem stating that an isosceles triangle is not uniquely determined by one angle and the altitude to one of its sides -/
theorem isosceles_triangle_not_unique (α : ℝ) (h : ℝ) : 
  ∃ t1 t2 : IsoscelesTriangle, t1.baseAngle = α ∧ t1.altitude = h ∧ 
  t2.baseAngle = α ∧ t2.altitude = h ∧ t1 ≠ t2 := by
  sorry

end isosceles_triangle_not_unique_l2017_201762


namespace max_garden_area_l2017_201704

/-- Represents the dimensions of a rectangular garden. -/
structure GardenDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangular garden given its dimensions. -/
def gardenArea (d : GardenDimensions) : ℝ :=
  d.length * d.width

/-- Calculates the perimeter of a rectangular garden given its dimensions. -/
def gardenPerimeter (d : GardenDimensions) : ℝ :=
  2 * (d.length + d.width)

/-- Theorem stating the maximum area of a garden with given constraints. -/
theorem max_garden_area :
  ∀ d : GardenDimensions,
    d.length ≥ 100 →
    d.width ≥ 60 →
    gardenPerimeter d = 360 →
    gardenArea d ≤ 8000 :=
by sorry

end max_garden_area_l2017_201704


namespace quadratic_roots_problem_l2017_201750

theorem quadratic_roots_problem (a b p q : ℝ) : 
  p ≠ q ∧ p ≠ 0 ∧ q ≠ 0 ∧
  a ≠ b ∧ a ≠ 0 ∧ b ≠ 0 ∧
  p^2 - a*p + b = 0 ∧
  q^2 - a*q + b = 0 ∧
  a^2 - p*a - q = 0 ∧
  b^2 - p*b - q = 0 →
  a = 1 ∧ b = -2 ∧ p = -1 ∧ q = 2 := by
sorry

end quadratic_roots_problem_l2017_201750


namespace set_operations_l2017_201781

-- Define the universal set U
def U : Set Int := {-3, -1, 0, 1, 2, 3, 4, 6}

-- Define set A
def A : Set Int := {0, 2, 4, 6}

-- Define the complement of A in U
def C_UA : Set Int := {-1, -3, 1, 3}

-- Define the complement of B in U
def C_UB : Set Int := {-1, 0, 2}

-- Define set B
def B : Set Int := U \ C_UB

-- Theorem to prove
theorem set_operations :
  (A ∩ B = {4, 6}) ∧ (A ∪ B = {-3, 0, 1, 2, 3, 4, 6}) := by
  sorry

end set_operations_l2017_201781


namespace train_length_l2017_201735

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed : ℝ) (time : ℝ) : 
  speed = 60 → time = 5 → ∃ (length : ℝ), abs (length - 83.35) < 0.01 := by
  sorry

end train_length_l2017_201735


namespace electricity_billing_theorem_l2017_201788

/-- Represents the tariff rates for different zones --/
structure TariffRates where
  peak : ℝ
  night : ℝ
  half_peak : ℝ

/-- Represents the meter readings --/
structure MeterReadings where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  e : ℝ
  f : ℝ

/-- Calculates the maximum possible additional payment --/
def max_additional_payment (rates : TariffRates) (readings : MeterReadings) (paid_amount : ℝ) : ℝ :=
  sorry

/-- Calculates the expected difference between company's calculation and customer's payment --/
def expected_difference (rates : TariffRates) (readings : MeterReadings) (paid_amount : ℝ) : ℝ :=
  sorry

/-- Main theorem stating the correct results for the given problem --/
theorem electricity_billing_theorem (rates : TariffRates) (readings : MeterReadings) (paid_amount : ℝ) :
  rates.peak = 4.03 ∧ rates.night = 1.01 ∧ rates.half_peak = 3.39 ∧
  readings.a = 1214 ∧ readings.b = 1270 ∧ readings.c = 1298 ∧
  readings.d = 1337 ∧ readings.e = 1347 ∧ readings.f = 1402 ∧
  paid_amount = 660.72 →
  max_additional_payment rates readings paid_amount = 397.34 ∧
  expected_difference rates readings paid_amount = 19.30 :=
by sorry

end electricity_billing_theorem_l2017_201788


namespace circle_condition_l2017_201728

/-- A circle in the xy-plane can be represented by the equation x^2 + y^2 - x + y + m = 0,
    where m is a real number. This theorem states that for the equation to represent a circle,
    the value of m must be less than 1/4. -/
theorem circle_condition (m : ℝ) :
  (∃ (x y : ℝ), x^2 + y^2 - x + y + m = 0 ∧ 
   ∀ (a b : ℝ), (a - (1/2))^2 + (b - (1/2))^2 = ((1/2)^2 + (1/2)^2 - m)) →
  m < 1/4 := by
  sorry

end circle_condition_l2017_201728


namespace sin_increasing_in_interval_l2017_201709

theorem sin_increasing_in_interval :
  let f : ℝ → ℝ := λ x ↦ Real.sin (2 * x - π / 6)
  ∀ x y, -π/6 < x ∧ x < y ∧ y < π/3 → f x < f y :=
by sorry

end sin_increasing_in_interval_l2017_201709


namespace arctan_sum_property_l2017_201721

theorem arctan_sum_property (a b c : ℝ) 
  (h : Real.arctan a + Real.arctan b + Real.arctan c + π / 2 = 0) : 
  (a * b + b * c + c * a = 1) ∧ (a + b + c < a * b * c) := by
  sorry

end arctan_sum_property_l2017_201721


namespace mikes_remaining_cards_l2017_201757

/-- Given Mike's initial number of baseball cards and the number of cards Sam bought,
    prove that Mike's remaining number of cards is the difference between his initial number
    and the number Sam bought. -/
theorem mikes_remaining_cards (initial_cards sam_bought : ℕ) :
  initial_cards - sam_bought = initial_cards - sam_bought :=
by sorry

/-- Mike's initial number of baseball cards -/
def mike_initial_cards : ℕ := 87

/-- Number of cards Sam bought from Mike -/
def sam_bought_cards : ℕ := 13

/-- Mike's remaining number of cards -/
def mike_remaining_cards : ℕ := mike_initial_cards - sam_bought_cards

#eval mike_remaining_cards  -- Should output 74

end mikes_remaining_cards_l2017_201757


namespace simplify_expression_l2017_201745

theorem simplify_expression (x : ℚ) : 
  ((3 * x + 6) - 5 * x) / 3 = -2 * x / 3 + 2 := by sorry

end simplify_expression_l2017_201745


namespace binomial_coefficient_equation_unique_solution_l2017_201751

theorem binomial_coefficient_equation_unique_solution :
  ∃! n : ℕ, (Nat.choose 25 n + Nat.choose 25 12 = Nat.choose 26 13) ∧ n = 13 := by
  sorry

end binomial_coefficient_equation_unique_solution_l2017_201751


namespace weight_of_calcium_hydride_l2017_201720

/-- The atomic weight of calcium in g/mol -/
def Ca_weight : ℝ := 40.08

/-- The atomic weight of hydrogen in g/mol -/
def H_weight : ℝ := 1.008

/-- The molecular weight of calcium hydride (CaH2) in g/mol -/
def CaH2_weight : ℝ := Ca_weight + 2 * H_weight

/-- The number of moles of calcium hydride -/
def moles : ℝ := 6

/-- Theorem: The weight of 6 moles of calcium hydride (CaH2) is 252.576 grams -/
theorem weight_of_calcium_hydride : moles * CaH2_weight = 252.576 := by
  sorry

end weight_of_calcium_hydride_l2017_201720


namespace odd_periodic_monotone_increasing_l2017_201724

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def is_periodic (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y < b → f x < f y

theorem odd_periodic_monotone_increasing (f : ℝ → ℝ) 
  (h_odd : is_odd f)
  (h_periodic : is_periodic f 4)
  (h_monotone : monotone_increasing_on f 0 2) :
  f 3 < 0 ∧ 0 < f 1 := by sorry

end odd_periodic_monotone_increasing_l2017_201724


namespace max_abs_sum_on_circle_l2017_201700

theorem max_abs_sum_on_circle (x y : ℝ) (h : x^2 + y^2 = 4) :
  |x| + |y| ≤ 2 * Real.sqrt 2 ∧ ∃ x y : ℝ, x^2 + y^2 = 4 ∧ |x| + |y| = 2 * Real.sqrt 2 :=
by sorry

end max_abs_sum_on_circle_l2017_201700


namespace system_solution_l2017_201798

theorem system_solution (a b c m n k : ℚ) :
  (∃ x y : ℚ, a * x + b * y = c ∧ m * x - n * y = k ∧ x = -3 ∧ y = 4) →
  (∃ x y : ℚ, a * (x + y) + b * (x - y) = c ∧ m * (x + y) - n * (x - y) = k ∧ x = 1/2 ∧ y = -7/2) :=
by sorry

end system_solution_l2017_201798


namespace equation_solutions_l2017_201737

theorem equation_solutions : 
  -- Equation 1
  (∃ x : ℝ, 4 * (x - 1)^2 - 8 = 0) ∧
  (∀ x : ℝ, 4 * (x - 1)^2 - 8 = 0 → (x = 1 + Real.sqrt 2 ∨ x = 1 - Real.sqrt 2)) ∧
  -- Equation 2
  (∃ x : ℝ, 2 * x * (x - 3) = x - 3) ∧
  (∀ x : ℝ, 2 * x * (x - 3) = x - 3 → (x = 3 ∨ x = 1/2)) ∧
  -- Equation 3
  (∃ x : ℝ, x^2 - 10*x + 16 = 0) ∧
  (∀ x : ℝ, x^2 - 10*x + 16 = 0 → (x = 8 ∨ x = 2)) ∧
  -- Equation 4
  (∃ x : ℝ, 2*x^2 + 3*x - 1 = 0) ∧
  (∀ x : ℝ, 2*x^2 + 3*x - 1 = 0 → (x = (Real.sqrt 17 - 3) / 4 ∨ x = -(Real.sqrt 17 + 3) / 4)) := by
  sorry


end equation_solutions_l2017_201737


namespace remainder_divisibility_l2017_201765

theorem remainder_divisibility (N : ℤ) : 
  (∃ k : ℤ, N = 39 * k + 17) → (∃ m : ℤ, N = 13 * m + 4) :=
by
  sorry

end remainder_divisibility_l2017_201765


namespace sandwich_non_filler_percentage_l2017_201722

/-- Given a sandwich weighing 180 grams with 45 grams of fillers,
    prove that the percentage of the sandwich that is not filler is 75%. -/
theorem sandwich_non_filler_percentage
  (total_weight : ℝ)
  (filler_weight : ℝ)
  (h1 : total_weight = 180)
  (h2 : filler_weight = 45) :
  (total_weight - filler_weight) / total_weight * 100 = 75 := by
  sorry

end sandwich_non_filler_percentage_l2017_201722


namespace train_speed_problem_l2017_201777

theorem train_speed_problem (length_train1 length_train2 distance_between speed_train2 time_to_cross : ℝ)
  (h1 : length_train1 = 100)
  (h2 : length_train2 = 150)
  (h3 : distance_between = 50)
  (h4 : speed_train2 = 15)
  (h5 : time_to_cross = 60)
  : ∃ speed_train1 : ℝ,
    speed_train1 = 10 ∧
    (length_train1 + length_train2 + distance_between) / time_to_cross = speed_train2 - speed_train1 :=
by sorry

end train_speed_problem_l2017_201777


namespace total_interest_is_1380_l2017_201795

def total_investment : ℝ := 17000
def low_rate_investment : ℝ := 12000
def low_rate : ℝ := 0.04
def high_rate : ℝ := 0.18

def calculate_total_interest : ℝ := 
  let high_rate_investment := total_investment - low_rate_investment
  let low_rate_interest := low_rate_investment * low_rate
  let high_rate_interest := high_rate_investment * high_rate
  low_rate_interest + high_rate_interest

theorem total_interest_is_1380 : 
  calculate_total_interest = 1380 := by sorry

end total_interest_is_1380_l2017_201795


namespace work_earnings_equality_l2017_201712

theorem work_earnings_equality (t : ℝ) : 
  (t + 2) * (4 * t - 2) = (4 * t - 7) * (t + 3) + 3 → t = 14 := by
  sorry

end work_earnings_equality_l2017_201712


namespace mary_saw_90_snakes_l2017_201773

/-- The number of breeding balls -/
def num_breeding_balls : ℕ := 5

/-- The number of snakes in each breeding ball -/
def snakes_per_ball : ℕ := 12

/-- The number of additional pairs of snakes -/
def num_additional_pairs : ℕ := 15

/-- The total number of snakes Mary saw -/
def total_snakes : ℕ := num_breeding_balls * snakes_per_ball + 2 * num_additional_pairs

theorem mary_saw_90_snakes : total_snakes = 90 := by
  sorry

end mary_saw_90_snakes_l2017_201773


namespace sum_range_l2017_201792

theorem sum_range (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x^2 + 2*x*y + 4*y^2 = 1) :
  1/2 < x + y ∧ x + y < 1 := by
sorry

end sum_range_l2017_201792


namespace hound_catches_hare_l2017_201742

/-- The number of jumps required for a hound to catch a hare -/
def catchHare (initialSeparation : ℕ) (hareJump : ℕ) (houndJump : ℕ) : ℕ :=
  initialSeparation / (houndJump - hareJump)

/-- Theorem stating that given the specific conditions, the hound catches the hare in 75 jumps -/
theorem hound_catches_hare :
  catchHare 150 7 9 = 75 := by
  sorry

#eval catchHare 150 7 9

end hound_catches_hare_l2017_201742


namespace emily_songs_l2017_201789

theorem emily_songs (x : ℕ) : x + 7 = 13 → x = 6 := by
  sorry

end emily_songs_l2017_201789


namespace total_protest_days_l2017_201734

theorem total_protest_days (first_protest : ℕ) (second_protest_percentage : ℚ) : 
  first_protest = 4 →
  second_protest_percentage = 25 / 100 →
  first_protest + (first_protest + first_protest * second_protest_percentage) = 9 := by
sorry

end total_protest_days_l2017_201734


namespace calculation_part1_calculation_part2_l2017_201701

-- Part 1
theorem calculation_part1 : 
  (1/8)^(-(2/3)) - 4*(-3)^4 + (2 + 1/4)^(1/2) - (1.5)^2 = -320.75 := by sorry

-- Part 2
theorem calculation_part2 : 
  (Real.log 5 / Real.log 10)^2 + (Real.log 2 / Real.log 10) * (Real.log 50 / Real.log 10) - 
  (Real.log 8 / Real.log (1/2)) + (Real.log (427/3) / Real.log 3) = 
  1 - (Real.log 5 / Real.log 10)^2 + (Real.log 5 / Real.log 10) + (Real.log 2 / Real.log 10) + 2 := by sorry

end calculation_part1_calculation_part2_l2017_201701


namespace winnie_lollipops_l2017_201738

/-- The number of lollipops Winnie keeps for herself -/
def lollipops_kept (total : ℕ) (friends : ℕ) : ℕ :=
  total % friends

theorem winnie_lollipops :
  lollipops_kept 432 14 = 12 := by sorry

end winnie_lollipops_l2017_201738


namespace earth_total_area_l2017_201782

/-- The ocean area on Earth's surface in million square kilometers -/
def ocean_area : ℝ := 361

/-- The difference between ocean and land area in million square kilometers -/
def area_difference : ℝ := 2.12

/-- The total area of the Earth in million square kilometers -/
def total_area : ℝ := ocean_area + (ocean_area - area_difference)

theorem earth_total_area :
  total_area = 5.10 := by
  sorry

end earth_total_area_l2017_201782


namespace place_balls_in_boxes_theorem_l2017_201726

/-- The number of ways to place 4 distinct balls into 4 distinct boxes such that exactly two boxes remain empty -/
def place_balls_in_boxes : ℕ :=
  let n_balls : ℕ := 4
  let n_boxes : ℕ := 4
  let n_empty_boxes : ℕ := 2
  -- The actual calculation is not implemented here
  84

/-- Theorem stating that the number of ways to place 4 distinct balls into 4 distinct boxes
    such that exactly two boxes remain empty is 84 -/
theorem place_balls_in_boxes_theorem :
  place_balls_in_boxes = 84 := by
  sorry

end place_balls_in_boxes_theorem_l2017_201726


namespace shortest_distance_parabola_to_line_l2017_201799

/-- The shortest distance between a point on the parabola y = x^2 - 9x + 25 
    and a point on the line y = x - 8 is 4√2. -/
theorem shortest_distance_parabola_to_line :
  let parabola := {(x, y) : ℝ × ℝ | y = x^2 - 9*x + 25}
  let line := {(x, y) : ℝ × ℝ | y = x - 8}
  ∃ (d : ℝ), d = 4 * Real.sqrt 2 ∧ 
    ∀ (A : ℝ × ℝ) (B : ℝ × ℝ), A ∈ parabola → B ∈ line → 
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≥ d :=
by sorry

end shortest_distance_parabola_to_line_l2017_201799


namespace polynomial_equation_sum_l2017_201756

theorem polynomial_equation_sum (a b c d : ℤ) :
  (∀ x : ℤ, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 + x^2 + 8*x - 12) →
  a + b + c + d = 1 := by
  sorry

end polynomial_equation_sum_l2017_201756
