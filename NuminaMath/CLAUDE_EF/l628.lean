import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_pair4_opposite_l628_62825

-- Define the concept of opposite numbers
def are_opposite (a b : ℚ) : Prop := a = -b

-- Define the pairs of numbers
def pair1 : ℚ × ℚ := (-2, 1/2)
def pair2 : ℚ × ℚ := (1, 1)  -- |(-1)| = 1
def pair3 : ℚ × ℚ := (9, 9)  -- (-3)^2 = 3^2 = 9
def pair4 : ℚ × ℚ := (-5, 5)  -- -(-5) = 5

-- Theorem stating that only pair4 consists of opposite numbers
theorem only_pair4_opposite :
  ¬(are_opposite pair1.1 pair1.2) ∧
  ¬(are_opposite pair2.1 pair2.2) ∧
  ¬(are_opposite pair3.1 pair3.2) ∧
  (are_opposite pair4.1 pair4.2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_pair4_opposite_l628_62825


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_b_l628_62876

noncomputable def angle_between_vectors (a b : ℝ × ℝ) : ℝ := sorry

theorem angle_between_a_and_b
  (a : ℝ × ℝ)
  (b : ℝ × ℝ := (-1, 1))
  (c : ℝ × ℝ := (2, -2))
  (h1 : ‖a‖ = Real.sqrt 2)
  (h2 : a • (b + c) = 1) :
  angle_between_vectors a b = 2 * Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_b_l628_62876


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2pi_minus_alpha_l628_62824

theorem cos_2pi_minus_alpha (α : Real) 
  (h1 : α ∈ Set.Ioo (π / 2) (3 * π / 2)) 
  (h2 : Real.tan α = -12 / 5) : 
  Real.cos (2 * π - α) = -5 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2pi_minus_alpha_l628_62824


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l628_62847

/-- A circle centered at the origin with radius 1 -/
def unit_circle (x y : ℝ) : Prop := x^2 + y^2 = 1

/-- A line with slope 1 and y-intercept a -/
def line (x y a : ℝ) : Prop := y = x + a

/-- The ratio of arc lengths when the line intersects the circle -/
noncomputable def arc_length_ratio (a : ℝ) : Prop := 
  ∃ (r : ℝ), r = 1/3 ∧ r > 0 ∧ r < 1

/-- The theorem stating the possible values of a -/
theorem line_circle_intersection (a : ℝ) :
  (∃ x y : ℝ, unit_circle x y ∧ line x y a) →
  arc_length_ratio a →
  a = 1 ∨ a = -1 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_circle_intersection_l628_62847


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_approx_l628_62868

/-- A geometric sequence with sixth term 6! and ninth term 9! -/
structure GeometricSequence where
  a : ℝ  -- first term
  r : ℝ  -- common ratio
  sixth_term : a * r^5 = 720  -- 6! = 720
  ninth_term : a * r^8 = 362880  -- 9! = 362880

/-- The first term of the geometric sequence is approximately 0.66 -/
theorem first_term_approx (seq : GeometricSequence) : 
  ∃ ε > 0, |seq.a - 0.66| < ε := by
  sorry

#eval Nat.factorial 6
#eval Nat.factorial 9

end NUMINAMATH_CALUDE_ERRORFEEDBACK_first_term_approx_l628_62868


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_angle_theorem_l628_62899

/-- Represents a pyramid with all lateral faces forming the same angle with the base -/
structure Pyramid where
  k : ℝ  -- ratio of total surface area to base area
  α : ℝ  -- angle between lateral faces and base plane

/-- The angle between lateral faces and base plane in a pyramid with given surface area ratio -/
noncomputable def angle_for_ratio (k : ℝ) : ℝ := Real.arccos (1 / (k - 1))

theorem pyramid_angle_theorem (p : Pyramid) (h_k : p.k > 2) :
  p.α = angle_for_ratio p.k ∧ 
  0 < p.α ∧ 
  p.α < π / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_angle_theorem_l628_62899


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_holes_l628_62805

def holes_problem (total_holes : ℕ) (fill_percentage : ℚ) : ℕ :=
  total_holes - (↑total_holes * fill_percentage).floor.toNat

theorem remaining_holes :
  holes_problem 8 (3/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_holes_l628_62805


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_u_l628_62802

theorem min_value_of_u :
  ∃ (min_u : ℝ), min_u = 12/5 ∧
  ∀ (x y : ℝ), -2 < x → x < 2 → -2 < y → y < 2 → x * y = -1 →
    4 / (4 - x^2) + 9 / (9 - y^2) ≥ min_u :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_u_l628_62802


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_min_value_negative_3x_fx_l628_62854

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (3 : ℝ)^x - (3 : ℝ)^(-x)

-- Theorem for part 1
theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂ := by sorry

-- Theorem for part 2
theorem min_value_negative_3x_fx : ∀ x : ℝ, x ∈ Set.Icc 1 2 → -(3 : ℝ)^x * f x ≥ -8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_min_value_negative_3x_fx_l628_62854


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_cycle_l628_62821

theorem divisibility_cycle (a b c n : ℕ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) (hn : n > 0)
  (h1 : b ∣ a^n) 
  (h2 : c ∣ b^n) 
  (h3 : a ∣ c^n) : 
  a * b * c ∣ (a + b + c)^(n^2 + n + 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_cycle_l628_62821


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_trips_theorem_l628_62811

/-- Represents the water needs for different animals on Carla's farm -/
structure FarmWaterNeeds where
  pig_water : ℝ
  horse_water : ℝ
  chicken_tank : ℝ
  cow_water : ℝ
  goat_water : ℝ

/-- Represents the number of animals on Carla's farm -/
structure FarmAnimals where
  pigs : ℕ
  horses : ℕ
  chickens : ℕ
  cows : ℕ
  goats : ℕ

/-- Calculates the total water needed for the farm -/
def total_water_needed (needs : FarmWaterNeeds) (animals : FarmAnimals) : ℝ :=
  needs.pig_water * animals.pigs +
  needs.horse_water * animals.horses +
  needs.chicken_tank +
  needs.cow_water * animals.cows +
  needs.goat_water * animals.goats

/-- Represents Carla's water carrying capacity -/
def carla_capacity : ℝ := 5

/-- Calculates the number of trips needed to bring water -/
noncomputable def trips_needed (total_water : ℝ) (capacity : ℝ) : ℕ :=
  Int.toNat (Int.ceil (total_water / capacity))

/-- Theorem stating the number of trips Carla needs to make -/
theorem carla_trips_theorem (needs : FarmWaterNeeds) (animals : FarmAnimals) :
  needs.pig_water = 3 ∧
  needs.horse_water = 2 * needs.pig_water ∧
  needs.chicken_tank = 30 ∧
  needs.cow_water = 1.5 * needs.horse_water ∧
  needs.goat_water = 0.75 * needs.pig_water ∧
  animals.pigs = 8 ∧
  animals.horses = 10 ∧
  animals.chickens = 12 ∧
  animals.cows = 6 ∧
  animals.goats = 15 →
  trips_needed (total_water_needed needs animals) carla_capacity = 41 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_carla_trips_theorem_l628_62811


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_knights_tour_7x7_l628_62848

/-- Represents a chess square with its color -/
inductive Square
  | White
  | Black
deriving Inhabited

/-- Represents a chessboard -/
def Chessboard := List (List Square)

/-- A knight's move changes the color of the square -/
def knightMove (s : Square) : Square :=
  match s with
  | Square.White => Square.Black
  | Square.Black => Square.White

/-- Creates a 7x7 chessboard with alternating colors -/
def create7x7Chessboard (startWithWhite : Bool) : Chessboard :=
  sorry

/-- Counts the number of white and black squares on a chessboard -/
def countSquares (board : Chessboard) : (Nat × Nat) :=
  sorry

/-- Theorem: It's impossible to complete a knight's tour on a 7x7 chessboard -/
theorem no_knights_tour_7x7 (startSquare : Square) :
  ¬∃ (tour : List Square), 
    tour.length = 49 ∧ 
    tour.head? = some startSquare ∧
    (∀ i, i < tour.length - 1 → tour[i+1]? = some (knightMove tour[i]!)) :=
by
  sorry

#check no_knights_tour_7x7

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_knights_tour_7x7_l628_62848


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_same_terminal_side_l628_62844

/-- Given a point P(3, √3) on the terminal side of angle α, 
    the set of all angles with the same terminal side as α is {x | x = 2kπ + π/6, k ∈ ℤ} -/
theorem angles_same_terminal_side 
  (α : ℝ) 
  (h : ∃ (t : ℝ), t * 3 = Real.sqrt 3 ∧ Real.tan α = t) : 
  {x : ℝ | ∃ (k : ℤ), x = 2 * Real.pi * ↑k + Real.pi / 6} = 
  {x : ℝ | ∃ (k : ℤ), x = 2 * Real.pi * ↑k + α} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angles_same_terminal_side_l628_62844


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_for_two_l628_62882

/-- A random variable X following a binomial distribution B(n, p) -/
def binomial_distribution (n : ℕ) (p : ℝ) : Type :=
  {X : ℕ → ℝ // ∀ k, 0 ≤ k ∧ k ≤ n → X k = Nat.choose n k * p^k * (1-p)^(n-k)}

/-- The expected value of a binomial distribution B(n, p) is n * p -/
axiom binomial_expectation {n : ℕ} {p : ℝ} (X : binomial_distribution n p) :
  ∃ EX : ℝ, EX = n * p

/-- The probability mass function of a binomial distribution -/
def binomial_pmf (n : ℕ) (p : ℝ) (k : ℕ) : ℝ :=
  Nat.choose n k * p^k * (1-p)^(n-k)

/-- The main theorem -/
theorem binomial_probability_for_two (X : binomial_distribution 6 (1/3)) 
  (h : ∃ EX : ℝ, EX = 2) :
  binomial_pmf 6 (1/3) 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_probability_for_two_l628_62882


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l628_62832

def a : ℕ → ℚ
  | 0 => 1/4  -- Define for n = 0 to cover all natural numbers
  | 1 => 1/4
  | n+2 => 1/2 * a (n+1) + 1/(2^(n+2))

theorem a_formula (n : ℕ) : n ≥ 1 → a n = (2*n - 1) / 2^(n+1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_formula_l628_62832


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_girls_at_event_l628_62872

/-- Represents a school with a given number of students and boy-to-girl ratio --/
structure School where
  students : ℕ
  boy_ratio : ℕ
  girl_ratio : ℕ

/-- Calculates the number of girls in a school --/
def girls_in_school (s : School) : ℕ :=
  s.students * s.girl_ratio / (s.boy_ratio + s.girl_ratio)

/-- Theorem stating the fraction of girls at the joint event --/
theorem fraction_of_girls_at_event (school_a school_b : School)
  (ha : school_a.students = 300 ∧ school_a.boy_ratio = 3 ∧ school_a.girl_ratio = 2)
  (hb : school_b.students = 240 ∧ school_b.boy_ratio = 2 ∧ school_b.girl_ratio = 3) :
  (girls_in_school school_a + girls_in_school school_b) * 45 =
  (school_a.students + school_b.students) * 22 := by
  sorry

#eval girls_in_school ⟨300, 3, 2⟩
#eval girls_in_school ⟨240, 2, 3⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fraction_of_girls_at_event_l628_62872


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_campaign_is_correct_l628_62816

def campaign_definition : String := "campaign, activity"

theorem campaign_is_correct : campaign_definition = "campaign, activity" := by
  rfl

#check campaign_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_campaign_is_correct_l628_62816


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_divisible_by_n_l628_62850

theorem subset_sum_divisible_by_n (n : ℕ) (S : Finset ℕ) 
  (h1 : n ≥ 3)
  (h2 : S.card = n - 1)
  (h3 : ∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ ¬(n ∣ (a - b))) :
  ∃ T : Finset ℕ, T.Nonempty ∧ T ⊆ S ∧ n ∣ (T.sum id) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_subset_sum_divisible_by_n_l628_62850


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_ab_max_l628_62833

/-- Represents a hyperbola in the Cartesian coordinate system -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : 0 < a
  h_pos_b : 0 < b

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- The right focus of the hyperbola -/
def rightFocus : Point := ⟨2, 0⟩

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := 2 / h.a

/-- The length of the vector AB -/
noncomputable def vectorABLength (a b : Point) : ℝ :=
  Real.sqrt ((a.x - b.x)^2 + (a.y - b.y)^2)

/-- Theorem: The eccentricity of the hyperbola is √2 when |AB| is maximum -/
theorem hyperbola_eccentricity_when_ab_max (h : Hyperbola) :
  ∃ (A B : Point),
    (A ≠ B) ∧
    (∀ (A' B' : Point), vectorABLength A' B' ≤ vectorABLength A B) →
    eccentricity h = Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_when_ab_max_l628_62833


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_box_cost_proof_l628_62862

/-- Represents the dimensions of a box or item -/
structure Dimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of a given Dimensions -/
def volume (d : Dimensions) : ℝ := d.length * d.width * d.height

/-- Represents the pricing structure for boxes -/
structure BoxPricing where
  first_hundred_price : ℝ
  additional_price : ℝ

/-- Calculates the total cost for a given number of boxes -/
def calculate_box_cost (pricing : BoxPricing) (num_boxes : ℕ) : ℝ :=
  if num_boxes ≤ 100 then
    pricing.first_hundred_price * num_boxes
  else
    100 * pricing.first_hundred_price + (num_boxes - 100) * pricing.additional_price

theorem minimum_box_cost_proof (box_dim : Dimensions) (empty_space_ratio : ℝ)
    (large_item1 : Dimensions) (large_item2 : Dimensions) (large_item3 : Dimensions)
    (remaining_items : ℕ) (pricing : BoxPricing) :
    box_dim.length = 18 ∧ box_dim.width = 22 ∧ box_dim.height = 15 ∧
    empty_space_ratio = 0.2 ∧
    large_item1.length = 72 ∧ large_item1.width = 66 ∧ large_item1.height = 45 ∧
    large_item2.length = 54 ∧ large_item2.width = 48 ∧ large_item2.height = 40 ∧
    large_item3.length = 36 ∧ large_item3.width = 77 ∧ large_item3.height = 60 ∧
    remaining_items = 127 ∧
    pricing.first_hundred_price = 0.6 ∧ pricing.additional_price = 0.55 →
    calculate_box_cost pricing (Nat.ceil ((volume large_item1 + volume large_item2 + volume large_item3) /
      (volume box_dim * (1 - empty_space_ratio)))) = 61.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_box_cost_proof_l628_62862


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_l628_62853

/-- The time taken for a, b, and c to complete a work together given their individual completion times -/
theorem combined_work_time (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (1 / (1/a + 1/b + 1/c)) = 36 / 11 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_combined_work_time_l628_62853


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_fourth_quadrant_l628_62838

theorem sin_double_angle_fourth_quadrant (α : ℝ) 
  (h : -π/2 < α ∧ α < 0) : Real.sin (2 * α) < 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_double_angle_fourth_quadrant_l628_62838


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_fractions_l628_62875

def f1 : ℚ := 7 / 10
def f2 : ℚ := 8 / 9
def f3 : ℚ := 3 / 8
def f4 : ℚ := 5 / 12

def lcm_fractions (a b c d : ℚ) : ℚ := 
  lcm (a.num) (lcm (b.num) (lcm (c.num) (d.num))) /
  gcd (a.den) (gcd (b.den) (gcd (c.den) (d.den)))

theorem lcm_of_fractions : lcm_fractions f1 f2 f3 f4 = 840 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lcm_of_fractions_l628_62875


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_completion_theorem_l628_62831

/-- Represents a round-trip journey with four equal segments in each direction -/
structure Journey where
  segment_length : ℝ
  total_length : ℝ := 8 * segment_length

/-- Calculates the percentage of the journey completed -/
noncomputable def journey_completion_percentage (j : Journey) : ℝ :=
  let completed_distance := j.segment_length + 0.75 * j.segment_length + 0.5 * j.segment_length + 0.25 * j.segment_length
  (completed_distance / j.total_length) * 100

/-- Theorem stating that the journey completion percentage is 31.25% -/
theorem journey_completion_theorem (j : Journey) :
  journey_completion_percentage j = 31.25 := by
  sorry

#eval "Lean code compiled successfully!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_completion_theorem_l628_62831


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l628_62836

-- Define the floor function as noncomputable
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Theorem statement
theorem floor_expression_evaluation :
  (floor 6.5) * (floor (2 / 3 : ℝ)) + (floor 2) * (7.2 : ℝ) + (floor 8.3) - (6.6 : ℝ) = 9.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_floor_expression_evaluation_l628_62836


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_function_satisfying_conditions_l628_62891

/-- A function that satisfies the given conditions -/
def satisfies_conditions (f : ℝ → ℝ) : Prop :=
  (∀ x, x ∈ Set.Ioo 0 2 → f x > f 0) ∧
  ¬(∀ x y, x ∈ Set.Icc 0 2 → y ∈ Set.Icc 0 2 → x ≤ y → f x ≤ f y)

/-- There exists a continuous function that satisfies the conditions -/
theorem exists_function_satisfying_conditions :
  ∃ f : ℝ → ℝ, Continuous f ∧ satisfies_conditions f := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_function_satisfying_conditions_l628_62891


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kyungsoo_second_furthest_l628_62819

-- Define the people and their jump distances
noncomputable def kyungsoo_jump : ℝ := 2.3
noncomputable def younghee_jump : ℝ := 9/10
noncomputable def jinju_jump : ℝ := 1.8
noncomputable def chanho_jump : ℝ := 2.5

-- Define a function to check if a number is the second largest among four numbers
def is_second_largest (a b c d x : ℝ) : Prop :=
  (x < max a (max b (max c d))) ∧
  (x > min (max a b) (max c d))

-- Theorem statement
theorem kyungsoo_second_furthest :
  is_second_largest chanho_jump kyungsoo_jump jinju_jump younghee_jump kyungsoo_jump :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_kyungsoo_second_furthest_l628_62819


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_olivia_boxes_l628_62810

/-- Proves that Olivia collected 3 boxes of cookies given the conditions of the problem. -/
theorem olivia_boxes (cookies_per_box : ℕ) (abigail_boxes : ℕ) (grayson_fraction : ℚ) (total_cookies : ℕ)
  (h1 : cookies_per_box = 48)
  (h2 : abigail_boxes = 2)
  (h3 : grayson_fraction = 3/4)
  (h4 : total_cookies = 276)
  : (total_cookies - (abigail_boxes * cookies_per_box + (grayson_fraction * ↑cookies_per_box).floor)) / cookies_per_box = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_olivia_boxes_l628_62810


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomials_root_difference_l628_62870

theorem cubic_polynomials_root_difference (f g : ℝ → ℝ) (s : ℝ) : 
  (∀ x, f x - g x = s^2) →
  (∃ a, f = λ x ↦ (x - (s + 2)) * (x - (s + 6)) * (x - a)) →
  (∃ b, g = λ x ↦ (x - (s + 4)) * (x - (s + 8)) * (x - b)) →
  (∃ p q r, ∀ x, f x = x^3 + p * x^2 + q * x + r) →
  (∃ u v w, ∀ x, g x = x^3 + u * x^2 + v * x + w) →
  s = 8 := by
  sorry

#check cubic_polynomials_root_difference

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomials_root_difference_l628_62870


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l628_62839

/-- The perimeter of a triangle with sides 5, 20, and 30 is 55 -/
theorem triangle_perimeter : 5 + 20 + 30 = 55 := by
  ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_perimeter_l628_62839


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ninthNinthDigitIsCorrect_l628_62849

/-- The 99th digit after the decimal point in the decimal expansion of 2/9 + 3/11 -/
def ninthNinthDigit : ℕ := 4

/-- The decimal expansion of 2/9 -/
def twoNinths : ℚ := 2 / 9

/-- The decimal expansion of 3/11 -/
def threeElevenths : ℚ := 3 / 11

/-- The sum of 2/9 and 3/11 -/
def sum : ℚ := twoNinths + threeElevenths

/-- Returns the nth digit after the decimal point in the decimal expansion of a rational number -/
def nthDigitAfterDecimal (q : ℚ) (n : ℕ) : ℕ := sorry

theorem ninthNinthDigitIsCorrect : 
  (nthDigitAfterDecimal sum 99 = ninthNinthDigit) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ninthNinthDigitIsCorrect_l628_62849


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_property_taxes_l628_62812

/-- Bill's financial situation --/
structure BillFinances where
  grossSalary : ℕ
  takeHomeSalary : ℕ
  salesTaxes : ℕ
  incomeTaxRate : ℚ

/-- Calculate Bill's property taxes --/
def propertyTaxes (b : BillFinances) : ℕ :=
  b.grossSalary - (b.takeHomeSalary + b.salesTaxes + Nat.floor (b.incomeTaxRate * ↑b.grossSalary))

/-- Theorem: Bill's property taxes are $2000 --/
theorem bill_property_taxes :
  let b : BillFinances := {
    grossSalary := 50000,
    takeHomeSalary := 40000,
    salesTaxes := 3000,
    incomeTaxRate := 1/10
  }
  propertyTaxes b = 2000 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_property_taxes_l628_62812


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l628_62869

/-- Calculates the market value of a stock given its face value, dividend rate, and yield. -/
noncomputable def market_value (face_value : ℝ) (dividend_rate : ℝ) (yield : ℝ) : ℝ :=
  (face_value * dividend_rate) / yield

/-- Theorem: The market value of a stock with face value $100, 9% dividend rate, and 25% yield is $36. -/
theorem stock_market_value :
  let face_value : ℝ := 100
  let dividend_rate : ℝ := 0.09
  let yield : ℝ := 0.25
  market_value face_value dividend_rate yield = 36 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval market_value 100 0.09 0.25

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stock_market_value_l628_62869


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cube_side_correct_largest_parallelepiped_dims_correct_l628_62806

/-- Represents a tetrahedron with mutually perpendicular edges from vertex S -/
structure Tetrahedron where
  a : ℝ
  b : ℝ
  c : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  c_pos : 0 < c

/-- The side length of the largest cube with vertex S inside the tetrahedron -/
noncomputable def largest_cube_side (t : Tetrahedron) : ℝ :=
  (t.a * t.b * t.c) / (t.a * t.b + t.b * t.c + t.a * t.c)

/-- The dimensions of the largest rectangular parallelepiped with vertex S inside the tetrahedron -/
noncomputable def largest_parallelepiped_dims (t : Tetrahedron) : ℝ × ℝ × ℝ :=
  (t.a / 3, t.b / 3, t.c / 3)

theorem largest_cube_side_correct (t : Tetrahedron) :
  largest_cube_side t = (t.a * t.b * t.c) / (t.a * t.b + t.b * t.c + t.a * t.c) := by
  -- The proof is skipped for now
  sorry

theorem largest_parallelepiped_dims_correct (t : Tetrahedron) :
  largest_parallelepiped_dims t = (t.a / 3, t.b / 3, t.c / 3) := by
  -- The proof is skipped for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_cube_side_correct_largest_parallelepiped_dims_correct_l628_62806


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_volume_equality_l628_62809

/-- Given a triangle ABC with base a and height h, and a line DE parallel to AB at distance x from AB, 
    this theorem states the conditions for which the volume of the cylinder formed by rotating 
    parallelogram DE D'E' equals the volume of a sphere with radius R. -/
theorem cylinder_sphere_volume_equality 
  (a h R : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ x < h ∧
    (2 * Real.pi * x^2 * ((a * (h - x)) / h) = (4/3) * Real.pi * R^3)) ↔ 
  (R^2 < h^2 / 2 ∨ (h < a / 2 ∧ h^2 / 2 ≤ R^2 ∧ R^2 ≤ (a^2 * h) / (8 * (a - h)))) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_cylinder_sphere_volume_equality_l628_62809


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_three_l628_62800

theorem largest_of_three (a b c : ℝ) 
  (sum_eq : a + b + c = 2)
  (sum_prod_eq : a * b + a * c + b * c = -7)
  (prod_eq : a * b * c = -14) :
  max a (max b c) = Real.sqrt 7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_of_three_l628_62800


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_subset_complement_N_l628_62877

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M as the domain of y = ln(x-1)
def M : Set ℝ := {x | x > 1}

-- Define set N
def N : Set ℝ := {x | x^2 - x < 0}

-- State the theorem
theorem domain_subset_complement_N : M ⊆ (U \ N) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_subset_complement_N_l628_62877


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_seven_half_l628_62834

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.c / t.a * Real.cos t.C = Real.cos t.A ∧
  t.b + t.c = 2 + Real.sqrt 2 ∧
  Real.cos t.B = 3/4

-- Define the area function
noncomputable def triangle_area (t : Triangle) : ℝ :=
  1/2 * t.a * t.c * Real.sin t.B

-- Theorem statement
theorem triangle_area_is_sqrt_seven_half (t : Triangle) 
  (h : triangle_conditions t) : triangle_area t = Real.sqrt 7 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_is_sqrt_seven_half_l628_62834


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_fencing_cost_per_meter_l628_62851

/-- Represents the dimensions and cost of fencing a rectangular farm. -/
structure Farm where
  area : ℝ
  short_side : ℝ
  total_cost : ℝ

/-- Calculates the cost per meter of fencing for a given farm. -/
noncomputable def cost_per_meter (f : Farm) : ℝ :=
  let long_side := f.area / f.short_side
  let diagonal := Real.sqrt (long_side ^ 2 + f.short_side ^ 2)
  let total_length := long_side + f.short_side + diagonal
  f.total_cost / total_length

/-- Theorem stating that for a farm with given dimensions and total cost,
    the cost per meter of fencing is 12. -/
theorem farm_fencing_cost_per_meter (f : Farm)
    (h_area : f.area = 1200)
    (h_short : f.short_side = 30)
    (h_cost : f.total_cost = 1440) :
    cost_per_meter f = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_fencing_cost_per_meter_l628_62851


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_value_l628_62889

/-- The shaded area of a square with side length 40 inches and 5 inscribed circles -/
noncomputable def shaded_area_square_with_circles : ℝ := by
  -- Define the side length of the square
  let side_length : ℝ := 40

  -- Define the number of circles
  let num_circles : ℕ := 5

  -- Define the area of the square
  let square_area : ℝ := side_length ^ 2

  -- Define the radius of each circle (side_length / 4)
  let circle_radius : ℝ := side_length / 4

  -- Define the area of one circle
  let circle_area : ℝ := Real.pi * circle_radius ^ 2

  -- Define the total area of all circles
  let total_circles_area : ℝ := num_circles * circle_area

  -- Define the shaded area
  exact square_area - total_circles_area

/-- The shaded area is equal to 1600 - 125π square inches -/
theorem shaded_area_value : shaded_area_square_with_circles = 1600 - 125 * Real.pi := by
  -- Unfold the definition of shaded_area_square_with_circles
  unfold shaded_area_square_with_circles

  -- Simplify the expression
  simp [Real.pi]

  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_value_l628_62889


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_intersection_l628_62895

/-- The distance from the pole to the common point of two curves in polar coordinates -/
noncomputable def distance_to_common_point : ℝ :=
  (Real.sqrt 5 + 1) / 2

/-- First curve equation -/
noncomputable def curve1 (θ : ℝ) : ℝ := Real.cos θ + 1

/-- Second curve equation -/
def curve2 (ρ : ℝ → ℝ) (θ : ℝ) : Prop := ρ θ * Real.cos θ = 1

theorem distance_to_intersection (ρ : ℝ → ℝ) (θ : ℝ) :
  (∀ θ, ρ θ = curve1 θ) →
  (∀ θ, curve2 ρ θ) →
  ρ θ = distance_to_common_point := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_intersection_l628_62895


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chip_game_theorem_l628_62884

/-- The chip game between players A and B -/
structure ChipGame where
  initial_chips : ℕ := 3
  prob_a_win : ℝ := 0.3
  prob_b_win : ℝ := 0.2
  prob_draw : ℝ := 0.5

/-- The number of chips player A has after the first round -/
def chips_after_first_round (game : ChipGame) : ℕ → ℝ
  | 2 => game.prob_b_win
  | 3 => game.prob_draw
  | 4 => game.prob_a_win
  | _ => 0

/-- The expected number of chips player A has after the first round -/
def expected_chips_after_first_round (game : ChipGame) : ℝ :=
  2 * game.prob_b_win + 3 * game.prob_draw + 4 * game.prob_a_win

/-- The probability of the game ending after exactly four rounds -/
def prob_game_ends_four_rounds (game : ChipGame) : ℝ :=
  let c3_2 := 3
  c3_2 * game.prob_a_win^2 * game.prob_draw * game.prob_a_win +
  c3_2 * game.prob_b_win^2 * game.prob_draw * game.prob_b_win

/-- The probability of player A winning when they have i chips -/
noncomputable def prob_a_wins (game : ChipGame) : ℕ → ℝ := 
  fun i => sorry  -- We define this as a function, but leave the implementation as sorry

/-- The main theorem about the chip game -/
theorem chip_game_theorem (game : ChipGame) :
  (expected_chips_after_first_round game = 3.1) ∧
  (prob_game_ends_four_rounds game = 0.0525) ∧
  (∀ i : ℕ, i < 5 → (prob_a_wins game (i+1) - prob_a_wins game i) = 
    (2/3) * (prob_a_wins game i - prob_a_wins game (i-1))) :=
by
  sorry  -- We use sorry to skip the proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chip_game_theorem_l628_62884


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_right_angle_point_l628_62897

/-- A hyperbola with equation x^2 - y^2 = 4 -/
def Hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 4

/-- The foci of the hyperbola -/
noncomputable def Foci : ℝ × ℝ × ℝ × ℝ := (-2 * Real.sqrt 2, 0, 2 * Real.sqrt 2, 0)

/-- Point P lies on the right branch of the hyperbola -/
def RightBranch (x : ℝ) : Prop := x > 0

/-- The angle F₁PF₂ is 90° -/
def RightAngle (x y : ℝ) : Prop :=
  let (x1, _, x2, _) := Foci
  (y / (x - x1)) * (y / (x - x2)) = -1

theorem hyperbola_right_angle_point (x y : ℝ) :
  Hyperbola x y → RightBranch x → RightAngle x y → x = Real.sqrt 6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_right_angle_point_l628_62897


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_positive_l628_62815

def f (m n : ℤ) (x : ℝ) : ℝ := x^2 + m*x + n

theorem quadratic_positive (m n : ℤ) :
  (f m n 2014 > 0) → (f m n 2015 > 0) →
  ∀ x : ℝ, x ∈ Set.Icc 2014 2015 → f m n x > 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_positive_l628_62815


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jenna_earnings_400_miles_l628_62865

/-- Calculates Jenna's earnings for a round trip as a truck driver -/
noncomputable def jennasEarnings (oneWayDistance : ℝ) : ℝ :=
  let baseRate1 := 0.40
  let baseRate2 := 0.50
  let baseRate3 := 0.60
  let bonusAmount := 100
  let bonusDistance := 500
  let maintenanceCost := 50
  let maintenanceDistance := 500
  let fuelCostRate := 0.15

  let earnings1 := min oneWayDistance 100 * baseRate1
  let earnings2 := max (min oneWayDistance 300 - 100) 0 * baseRate2
  let earnings3 := max (oneWayDistance - 300) 0 * baseRate3
  
  let totalEarnings := 2 * (earnings1 + earnings2 + earnings3)
  let bonuses := ⌊(2 * oneWayDistance) / bonusDistance⌋ * bonusAmount
  let maintenanceCosts := ⌊(2 * oneWayDistance) / maintenanceDistance⌋ * maintenanceCost
  let fuelCosts := totalEarnings * fuelCostRate

  totalEarnings + bonuses - maintenanceCosts - fuelCosts

/-- Theorem stating that Jenna's earnings for a 400-mile one-way round trip is $390 -/
theorem jenna_earnings_400_miles :
  jennasEarnings 400 = 390 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jenna_earnings_400_miles_l628_62865


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_calculation_l628_62813

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 + 4*x + 9*y^2 + 18*y + 20 = 0

-- Define the area of an ellipse
noncomputable def ellipse_area (a b : ℝ) : ℝ := Real.pi * a * b

-- Theorem statement
theorem ellipse_area_calculation :
  ∃ (a b : ℝ), (∀ x y : ℝ, ellipse_equation x y ↔ (x+2)^2 / a^2 + (y+1)^2 / b^2 = 1) ∧
  ellipse_area a b = 7 * Real.pi / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_area_calculation_l628_62813


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_car_washing_earnings_l628_62818

/-- Mary's monthly earnings from washing cars -/
noncomputable def car_washing_earnings : ℚ := 20

/-- Mary's monthly earnings from walking dogs -/
noncomputable def dog_walking_earnings : ℚ := 40

/-- The number of months Mary saves -/
def saving_months : ℕ := 5

/-- The total amount Mary saves -/
noncomputable def total_savings : ℚ := 150

/-- Mary's total monthly earnings -/
noncomputable def total_earnings : ℚ := car_washing_earnings + dog_walking_earnings

/-- Mary's monthly savings -/
noncomputable def monthly_savings : ℚ := total_earnings / 2

theorem mary_car_washing_earnings :
  car_washing_earnings = 20 ∧
  dog_walking_earnings = 40 ∧
  saving_months = 5 ∧
  total_savings = 150 ∧
  monthly_savings * saving_months = total_savings := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mary_car_washing_earnings_l628_62818


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_bread_slices_l628_62885

/-- Calculates the remaining bread slices after breakfast and lunch -/
theorem remaining_bread_slices (total_slices : ℕ) (breakfast_fraction : ℚ) (lunch_slices : ℕ) : 
  total_slices = 12 → 
  breakfast_fraction = 1/3 → 
  lunch_slices = 2 → 
  total_slices - (Nat.floor (breakfast_fraction * ↑total_slices) + lunch_slices) = 6 := by
  sorry

#check remaining_bread_slices

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_bread_slices_l628_62885


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l628_62890

theorem inequality_solution_set : 
  {x : ℝ | (2 * x) / (x + 2) ≤ 3} = Set.Iic (-6) ∪ Set.Ioi (-2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_l628_62890


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_inequality_range_l628_62837

theorem sine_cosine_inequality_range (a : ℝ) :
  (∀ x : ℝ, (Real.sin x)^6 + (Real.cos x)^6 + a * Real.sin x * Real.cos x ≥ 0) ↔ 
  a ≥ -5/4 ∧ a ≤ 11/4 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sine_cosine_inequality_range_l628_62837


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_deductive_reasoning_syllogism_l628_62867

-- Define deductive reasoning
def DeductiveReasoning : Type := Unit

-- Define the concept of a syllogism
def Syllogism : Type := Unit

-- Define the general pattern of deductive reasoning
def GeneralPattern (dr : DeductiveReasoning) : Prop := True

-- Theorem stating that the general pattern of deductive reasoning is in the form of a syllogism
theorem deductive_reasoning_syllogism :
  ∀ (dr : DeductiveReasoning),
  ∃ (s : Syllogism), GeneralPattern dr :=
by
  intro dr
  use ()
  trivial

#check deductive_reasoning_syllogism

end NUMINAMATH_CALUDE_ERRORFEEDBACK_deductive_reasoning_syllogism_l628_62867


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_on_circle_l628_62857

/-- The circle C -/
def C (x y : ℝ) : Prop := (x - 1/2)^2 + (y + 1/2)^2 = 1/2

/-- The line L -/
def L (x y : ℝ) : Prop := y = -3/4 * x + 1/4

/-- The chord length -/
noncomputable def chord_length : ℝ := Real.sqrt 2

theorem chord_length_on_circle :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    C x₁ y₁ ∧ C x₂ y₂ ∧
    L x₁ y₁ ∧ L x₂ y₂ ∧
    Real.sqrt ((x₁ - x₂)^2 + (y₁ - y₂)^2) = chord_length := by
  sorry

#check chord_length_on_circle

end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_on_circle_l628_62857


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_after_reflection_l628_62814

noncomputable def point (x y : ℝ) : ℝ × ℝ := (x, y)

def reflect_over_x_axis (p : ℝ × ℝ) : ℝ × ℝ :=
  (p.1, -p.2)

noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)

theorem segment_length_after_reflection :
  let F := point (-4) 3
  let F' := reflect_over_x_axis F
  distance F F' = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_segment_length_after_reflection_l628_62814


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_sum_150_l628_62887

theorem consecutive_integers_sum_150 : 
  ∃! k : ℕ, k > 0 ∧ 
  k = (Finset.filter 
    (λ p : ℕ × ℕ ↦ 
      let (a, n) := p
      n ≥ 2 ∧ 
      a > 0 ∧ 
      n * (2 * a + n - 1) = 300)
    (Finset.product (Finset.range 151) (Finset.range 151))).card :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_consecutive_integers_sum_150_l628_62887


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l628_62807

-- Define the integrand function
noncomputable def f (x : ℝ) : ℝ := (x^3 - 6*x^2 + 13*x - 6) / ((x+2)*(x-2)^3)

-- Define the antiderivative function
noncomputable def F (x : ℝ) : ℝ := Real.log (abs (x+2)) - 1 / (2*(x-2)^2)

theorem integral_equality (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -2) :
  deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equality_l628_62807


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_value_imply_sin_l628_62826

/-- Given a function f(x) = a*sin(x) + b*cos(x) where a ≠ 0 and b ≠ 0,
    if the graph of f(x) is symmetric about x = π/6 and f(x₀) = 8/5 * a,
    then sin(2x₀ + π/6) = 7/25 -/
theorem symmetry_and_value_imply_sin (a b x₀ : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let f := λ x => a * Real.sin x + b * Real.cos x
  (∀ x, f (π/3 - x) = f (π/3 + x)) →  -- Symmetry about x = π/6
  f x₀ = 8/5 * a →
  Real.sin (2 * x₀ + π/6) = 7/25 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_and_value_imply_sin_l628_62826


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_to_fraction_l628_62804

-- Define the infinite geometric series sum
noncomputable def infinite_geometric_sum (a₁ : ℝ) (q : ℝ) : ℝ := a₁ / (1 - q)

-- Define the recurring decimal 5.2̇5̇
noncomputable def recurring_decimal : ℝ := 5 + infinite_geometric_sum 0.25 0.01

-- Theorem statement
theorem recurring_decimal_to_fraction : 
  recurring_decimal = 520 / 99 := by
  -- Expand the definition of recurring_decimal
  unfold recurring_decimal
  unfold infinite_geometric_sum
  -- Perform algebraic manipulations
  -- This is where we would normally prove the equality step by step
  -- For now, we'll use sorry to skip the proof
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_recurring_decimal_to_fraction_l628_62804


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_product_triple_l628_62820

def T (n : ℕ) : Set ℕ := {i | 4 ≤ i ∧ i ≤ n}

def has_product_triple (S : Set ℕ) : Prop :=
  ∃ x y z, x ∈ S ∧ y ∈ S ∧ z ∈ S ∧ x * y = z

theorem smallest_n_with_product_triple : 
  (∀ n < 1024, ∃ A B : Set ℕ, A ∪ B = T n ∧ A ∩ B = ∅ ∧ 
    ¬(has_product_triple A ∨ has_product_triple B)) ∧
  (∀ A B : Set ℕ, A ∪ B = T 1024 ∧ A ∩ B = ∅ → 
    has_product_triple A ∨ has_product_triple B) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_with_product_triple_l628_62820


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairings_l628_62845

/-- A pairing is a list of 5 pairs of natural numbers -/
def Pairing : Type := List (Nat × Nat)

/-- Check if a pairing is valid according to our conditions -/
def isValidPairing (p : Pairing) : Prop :=
  p.length = 5 ∧
  p.all (fun (a, b) => a ≥ 1 ∧ a ≤ 10 ∧ b ≥ 1 ∧ b ≤ 10) ∧
  p.all (fun (a, b) => a < b) ∧
  (List.range 10).all (fun n => p.any (fun (a, b) => a = n + 1 ∨ b = n + 1)) ∧
  p.all (fun (a, b) => (a + b) ∈ [9, 10, 11, 12, 13])

/-- The set of all valid pairings -/
def validPairings : Set Pairing :=
  {p | isValidPairing p}

/-- We assume that validPairings is finite -/
instance : Fintype validPairings := sorry

theorem count_valid_pairings :
  Fintype.card validPairings = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_pairings_l628_62845


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l628_62886

/-- The ellipse on which point M moves --/
def ellipse (x y : ℝ) : Prop := x^2/25 + y^2/9 = 1

/-- Point A --/
def A : ℝ × ℝ := (4, 0)

/-- Point B --/
def B : ℝ × ℝ := (2, 2)

/-- Distance between two points --/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- The theorem to be proved --/
theorem max_distance_sum :
  ∃ max_sum : ℝ, (∀ M : ℝ × ℝ, ellipse M.1 M.2 →
    ∀ N : ℝ × ℝ, ellipse N.1 N.2 →
      distance M A + distance M B ≤ max_sum) ∧
    max_sum = 10 + 2 * Real.sqrt 10 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_sum_l628_62886


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_is_blue_l628_62808

/-- Represents the colors of the cube's faces -/
inductive CubeColor
  | Blue
  | Orange
  | Yellow
  | Silver
  | Violet
  | Black

/-- Represents a cube with colored faces -/
structure Cube where
  faces : Fin 6 → CubeColor
  distinct_colors : ∀ i j, i ≠ j → faces i ≠ faces j

/-- Represents an orientation of the cube -/
structure CubeOrientation where
  top : CubeColor
  front : CubeColor
  right : CubeColor

/-- The theorem statement -/
theorem opposite_face_is_blue (c : Cube) 
  (o1 o2 o3 : CubeOrientation)
  (h1 : o1.top = CubeColor.Blue ∧ o2.top = CubeColor.Blue ∧ o3.top = CubeColor.Blue)
  (h2 : o1.right = CubeColor.Silver ∧ o2.right = CubeColor.Silver ∧ o3.right = CubeColor.Silver)
  (h3 : o1.front ≠ CubeColor.Orange ∧ o2.front ≠ CubeColor.Orange ∧ o3.front ≠ CubeColor.Orange)
  (h4 : o1.top ≠ CubeColor.Orange ∧ o2.top ≠ CubeColor.Orange ∧ o3.top ≠ CubeColor.Orange)
  (h5 : o1.right ≠ CubeColor.Orange ∧ o2.right ≠ CubeColor.Orange ∧ o3.right ≠ CubeColor.Orange)
  : ∃ (i j : Fin 6), c.faces i = CubeColor.Orange ∧ c.faces j = CubeColor.Blue ∧ 
    (i.val + j.val = 5 ∨ (i.val = 0 ∧ j.val = 5) ∨ (i.val = 5 ∧ j.val = 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_opposite_face_is_blue_l628_62808


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_stats_l628_62840

noncomputable def reading_hours : List ℝ := [4, 5, 5, 6, 10]

noncomputable def mean (xs : List ℝ) : ℝ := (xs.sum) / xs.length

noncomputable def variance (xs : List ℝ) : ℝ :=
  let μ := mean xs
  (xs.map (fun x => (x - μ)^2)).sum / xs.length

theorem reading_stats :
  mean reading_hours = 6 ∧ variance reading_hours = 4.4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reading_stats_l628_62840


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l628_62879

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the problem conditions
def triangle_condition (t : Triangle) : Prop :=
  t.b * Real.tan t.A = (2 * t.c - t.b) * Real.tan t.B

-- Theorem statement
theorem triangle_theorem (t : Triangle) 
  (h : triangle_condition t) : 
  t.A = π/3 ∧ 
  (∀ B C : Real, (Real.cos B)^2 + (Real.cos C)^2 ≥ 1/2) :=
by
  sorry

#check triangle_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l628_62879


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fits_l628_62864

structure Building where
  length : ℝ
  width : ℝ

structure Tank where
  diameter : ℝ

structure Yard where
  side_length : ℝ
  buildings : List Building
  tanks : List Tank

noncomputable def distance_from_boundary (yard : Yard) (point : ℝ × ℝ) : ℝ :=
  min (min point.1 (yard.side_length - point.1)) (min point.2 (yard.side_length - point.2))

noncomputable def distance_from_building (building : Building) (point : ℝ × ℝ) : ℝ :=
  sorry

noncomputable def distance_from_tank (tank : Tank) (point : ℝ × ℝ) : ℝ :=
  sorry

def is_valid_point (yard : Yard) (point : ℝ × ℝ) : Prop :=
  (distance_from_boundary yard point ≥ 5) ∧
  (∀ b ∈ yard.buildings, distance_from_building b point ≥ 5) ∧
  (∀ t ∈ yard.tanks, distance_from_tank t point ≥ 5)

theorem flower_bed_fits (yard : Yard) : 
  yard.side_length = 70 ∧
  yard.buildings = [⟨20, 10⟩, ⟨25, 15⟩, ⟨30, 30⟩] ∧
  yard.tanks = [⟨10⟩, ⟨10⟩] →
  ∃ point : ℝ × ℝ, is_valid_point yard point :=
by
  sorry

#check flower_bed_fits

end NUMINAMATH_CALUDE_ERRORFEEDBACK_flower_bed_fits_l628_62864


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l628_62801

open Real

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 2 * sin (2 * x + π / 6)

-- Define the transformation
noncomputable def transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2 * (x + π / 6))

-- State the theorem
theorem transformation_result :
  ∀ x, transform f x = 2 * cos (4 * x) := by
  intro x
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transformation_result_l628_62801


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l628_62817

-- Define the function
def f (x : ℝ) := x^2 - 2*x + 2

-- Define the domain
def domain : Set ℝ := Set.Icc (-1) 2

-- Theorem statement
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = Set.Icc 1 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l628_62817


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_seven_l628_62883

theorem last_digit_seven (x : ℝ) (h : x^2 - 3*x + 1 = 0) :
  ∀ n : ℕ, ∃ k : ℤ, x^(2^n) + (1/x)^(2^n) = 10*k + 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_digit_seven_l628_62883


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l628_62894

/-- The equation of a parabola -/
noncomputable def parabola_equation (x : ℝ) : ℝ := (1 / 8) * (x^2 - 8*x + 16)

/-- The directrix of a parabola -/
noncomputable def directrix (parabola : ℝ → ℝ) : ℝ := 
  -- We'll define this later, for now it's just a placeholder
  0

/-- Theorem stating that the directrix of the given parabola is y = -2 -/
theorem parabola_directrix :
  directrix parabola_equation = -2 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_directrix_l628_62894


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_decreasing_l628_62846

-- Define the quadratic equation
noncomputable def quadratic (x m : ℝ) : ℝ := x^2 - 3*x + m

-- Define the inverse proportion function
noncomputable def inverse_prop (x m : ℝ) : ℝ := m / x

theorem inverse_prop_decreasing (m : ℝ) (x₁ x₂ y₁ y₂ : ℝ) :
  (∃ r : ℝ, quadratic r m = 0 ∧ (∀ s : ℝ, quadratic s m = 0 → s = r)) →  -- Two equal real roots
  inverse_prop x₁ m = y₁ →  -- Point A on the inverse proportion function
  inverse_prop x₂ m = y₂ →  -- Point B on the inverse proportion function
  0 < x₁ →
  x₁ < x₂ →
  y₁ > y₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_prop_decreasing_l628_62846


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_negative_one_l628_62803

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 
  (((x - 1) / 2) ^ 2) - ((x - 1) / 2)

-- State the theorem
theorem f_of_three_equals_negative_one : f 3 = -1 := by
  -- Expand the definition of f
  unfold f
  -- Simplify the expression
  simp
  -- Complete the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_of_three_equals_negative_one_l628_62803


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l628_62842

theorem triangle_properties (A B C : Real) (BC : Real) :
  (0 < A) → (A < π) →
  (0 < B) → (B < π) →
  (0 < C) → (C < π) →
  A + B + C = π →
  Real.sin A ^ 2 - Real.sin B ^ 2 - Real.sin C ^ 2 = Real.sin B * Real.sin C →
  BC = 3 →
  (A = 2 * π / 3 ∧
   ∃ (perimeter : Real),
     perimeter ≤ BC + 2 * Real.sqrt 3 ∧
     ∀ (other_perimeter : Real),
       other_perimeter = BC + 2 * Real.sqrt 3 * Real.cos ((π / 6 - B) / 2) →
       other_perimeter ≤ perimeter) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l628_62842


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_size_third_winner_tournament_size_fourth_winner_l628_62863

/-- Represents a volleyball tournament --/
structure Tournament where
  teams : Finset Nat
  won : Nat → Nat → Prop
  played_all : ∀ i j, i ∈ teams → j ∈ teams → i ≠ j → (won i j ∨ won j i)
  no_self_play : ∀ i, i ∈ teams → ¬(won i i)

/-- For every pair of teams, there exists a third team that won against both --/
def has_third_winner (t : Tournament) : Prop :=
  ∀ i j, i ∈ t.teams → j ∈ t.teams → i ≠ j → 
    ∃ k, k ∈ t.teams ∧ k ≠ i ∧ k ≠ j ∧ t.won k i ∧ t.won k j

/-- For any three teams, there exists another team that won against all three --/
def has_fourth_winner (t : Tournament) : Prop :=
  ∀ i j k, i ∈ t.teams → j ∈ t.teams → k ∈ t.teams → 
    i ≠ j ∧ j ≠ k ∧ i ≠ k → 
    ∃ l, l ∈ t.teams ∧ l ≠ i ∧ l ≠ j ∧ l ≠ k ∧ t.won l i ∧ t.won l j ∧ t.won l k

theorem tournament_size_third_winner (t : Tournament) (h : has_third_winner t) : 
  Finset.card t.teams ≥ 7 := by
  sorry

theorem tournament_size_fourth_winner (t : Tournament) (h : has_fourth_winner t) : 
  Finset.card t.teams ≥ 15 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tournament_size_third_winner_tournament_size_fourth_winner_l628_62863


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_theorem_l628_62852

/-- The hexagon vertices -/
def hexagonVertices : List (ℝ × ℝ) := [(0,0), (1,2), (2,3), (3,2), (4,0), (2,1)]

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Calculate the perimeter of the hexagon -/
noncomputable def hexagonPerimeter (vertices : List (ℝ × ℝ)) : ℝ :=
  let pairs := List.zip vertices (List.rotateLeft vertices 1)
  List.sum (List.map (fun (p1, p2) => distance p1 p2) pairs)

/-- Express a real number in the form a + b√2 + c√5 + d√10 -/
noncomputable def expressAsSum (x : ℝ) : ℤ × ℤ × ℤ × ℤ :=
  sorry  -- This function would decompose x into the required form

theorem hexagon_perimeter_theorem :
  let perimeter := hexagonPerimeter hexagonVertices
  let (a, b, c, d) := expressAsSum perimeter
  perimeter = 3 * Real.sqrt 5 + 2 * Real.sqrt 2 ∧ a + b + c + d = 5 := by
    sorry

#check hexagon_perimeter_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_perimeter_theorem_l628_62852


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equation_solution_l628_62827

open Real MeasureTheory

noncomputable def integrand (x t : ℝ) (φ : ℝ → ℝ) : ℝ :=
  (Real.exp (-x * t / (1 - x)) / (1 - x)) * Real.exp (-t) * φ t

theorem integral_equation_solution (x : ℝ) (hx : |x| < 1) :
  ∫ t in Set.Ici 0, integrand x t (λ t => t) = 1 - x := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_equation_solution_l628_62827


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l628_62861

noncomputable def f (x : ℝ) : ℝ := (x^3 - 8) / (x + 8)

theorem domain_of_f :
  ∀ x : ℝ, x ≠ -8 ↔ ∃ y : ℝ, f x = y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l628_62861


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_sum_l628_62858

-- Define the dilation transformation
def dilation (a b : ℝ) (x y : ℝ) : ℝ × ℝ :=
  (a * x, b * y)

-- Define the original line
def original_line (x y : ℝ) : Prop :=
  x - 2 * y = 2

-- Define the transformed line
def transformed_line (x' y' : ℝ) : Prop :=
  2 * x' - y' = 4

-- State the theorem
theorem dilation_sum (a b : ℝ) :
  (∀ x y, original_line x y → transformed_line (dilation a b x y).1 (dilation a b x y).2) →
  a + b = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_dilation_sum_l628_62858


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l628_62822

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 3 * x - 7 + Real.log x

-- State the theorem
theorem zero_in_interval :
  ∃ x : ℝ, x > 2 ∧ x < 3 ∧ f x = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l628_62822


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_is_correct_l628_62835

-- Define the functions f, g, and h
noncomputable def f (x : ℝ) : ℝ := 4 * x + 5
noncomputable def g (x : ℝ) : ℝ := x^2 - 2*x - 1
noncomputable def h (x : ℝ) : ℝ := f (g x)

-- Define the inverse function of h
noncomputable def h_inverse (x : ℝ) : ℝ := 1 + (Real.sqrt (3*x + 12)) / 4

-- Theorem statement
theorem h_inverse_is_correct :
  ∀ x : ℝ, x ≥ -3 → (h (h_inverse x) = x ∧ h_inverse (h x) = x) :=
by
  sorry

#check h_inverse_is_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_h_inverse_is_correct_l628_62835


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l628_62855

noncomputable def a : ℕ → ℝ
  | 0 => 2  -- Adding case for 0 to cover all natural numbers
  | 1 => 2
  | n + 2 => (a (n + 1))^2 / (a (n + 1) + 2)

theorem sequence_properties :
  (∀ n : ℕ, n ≥ 1 → a n ≤ 1 / 2^(n - 2)) ∧
  (∑' n : ℕ, 2 * ↑n * a n / (a n + 2)) < 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l628_62855


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_operation_result_l628_62860

-- Define the binary operations as noncomputable
noncomputable def otimes (a b : ℝ) : ℝ := (a + b) / (a - b)
noncomputable def oplus (b a : ℝ) : ℝ := (b - a) / (b + a)

-- State the theorem
theorem special_operation_result :
  oplus (otimes 8 6) 2 = 5 / 9 := by
  -- Proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_operation_result_l628_62860


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_N_is_120_l628_62859

/-- A 6 × N table where each entry is an integer between 1 and 6 -/
def Table (N : ℕ) := Fin 6 → Fin N → Fin 6

/-- Predicate to check if a column is a permutation of 1 to 6 -/
def IsPermutation (col : Fin 6 → Fin 6) : Prop :=
  ∀ i : Fin 6, ∃ j : Fin 6, col j = i

/-- Predicate to check if two columns have at least one equal entry -/
def HasEqualEntry (T : Table N) (i j : Fin N) : Prop :=
  ∃ r : Fin 6, T r i = T r j

/-- Predicate to check if two columns have at least one different entry -/
def HasDifferentEntry (T : Table N) (i j : Fin N) : Prop :=
  ∃ s : Fin 6, T s i ≠ T s j

/-- Main theorem: The largest N satisfying all conditions is 120 -/
theorem largest_N_is_120 :
  (∃ N : ℕ, N > 0 ∧
    ∃ T : Table N,
      (∀ j : Fin N, IsPermutation (λ i => T i j)) ∧
      (∀ i j : Fin N, i ≠ j → HasEqualEntry T i j) ∧
      (∀ i j : Fin N, i ≠ j → HasDifferentEntry T i j)) ∧
  (∀ M : ℕ, M > 120 →
    ¬∃ T : Table M,
      (∀ j : Fin M, IsPermutation (λ i => T i j)) ∧
      (∀ i j : Fin M, i ≠ j → HasEqualEntry T i j) ∧
      (∀ i j : Fin M, i ≠ j → HasDifferentEntry T i j)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_N_is_120_l628_62859


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_750_l628_62856

/-- Represents a rectangle with given width and perimeter. -/
structure Rectangle where
  width : ℝ
  perimeter : ℝ

/-- Calculates the length of a rectangle given its width and perimeter. -/
noncomputable def Rectangle.length (r : Rectangle) : ℝ :=
  (r.perimeter - 2 * r.width) / 2

/-- Calculates the area of a rectangle. -/
noncomputable def Rectangle.area (r : Rectangle) : ℝ :=
  r.width * r.length

/-- Theorem: A rectangle with perimeter 110 meters and width 25 meters has an area of 750 square meters. -/
theorem rectangle_area_is_750 (r : Rectangle) (h1 : r.perimeter = 110) (h2 : r.width = 25) :
  r.area = 750 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_is_750_l628_62856


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_sum_l628_62888

-- Define the function g
def g : ℝ → ℝ := sorry

-- Define the inverse of g
def g_inv : ℝ → ℝ := sorry

-- Axioms for g being invertible
axiom g_inv_left : ∀ x, g_inv (g x) = x
axiom g_inv_right : ∀ x, g (g_inv x) = x

-- Define the given values of g
axiom g_1 : g 1 = 3
axiom g_2 : g 2 = 4
axiom g_3 : g 3 = 6
axiom g_4 : g 4 = 8
axiom g_5 : g 5 = 9
axiom g_6 : g 6 = 10

-- Theorem to prove
theorem g_composition_sum : 
  g (g 2) + g (g_inv 9) + g_inv (g_inv 6) = 19 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_composition_sum_l628_62888


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_l628_62871

theorem triangle_side_calculation (a b c : ℝ) (θ : ℝ) : 
  c = 2 * Real.sqrt 7 →
  θ = π / 6 →
  a / b = 1 / (2 * Real.sqrt 3) →
  c^2 = a^2 + b^2 - 2*a*b*(Real.cos θ) →
  (a = 2 ∧ b = 4 * Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_side_calculation_l628_62871


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_usb_capacity_is_16_l628_62878

/-- Represents the capacity of a USB drive in gigabytes -/
def USBCapacity : ℝ → Prop := sorry

/-- The total capacity of the USB drive -/
def total_capacity : ℝ := sorry

/-- The used capacity of the USB drive -/
def used_capacity : ℝ := sorry

/-- The available (unused) capacity of the USB drive -/
def available_capacity : ℝ := sorry

/-- Theorem stating the total capacity of the USB drive is 16 gigabytes -/
theorem usb_capacity_is_16 :
  USBCapacity total_capacity ∧
  used_capacity = 0.5 * total_capacity ∧
  available_capacity = 8 ∧
  total_capacity = used_capacity + available_capacity →
  total_capacity = 16 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_usb_capacity_is_16_l628_62878


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_20_terms_with_constraint_l628_62830

/-- Represents an arithmetic progression -/
structure ArithmeticProgression where
  a : ℝ  -- First term
  d : ℝ  -- Common difference

/-- The nth term of an arithmetic progression -/
noncomputable def nthTerm (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  ap.a + (n - 1 : ℝ) * ap.d

/-- The sum of the first n terms of an arithmetic progression -/
noncomputable def sumOfTerms (ap : ArithmeticProgression) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * ap.a + (n - 1 : ℝ) * ap.d)

/-- Theorem about the sum of first 20 terms in a specific arithmetic progression -/
theorem sum_of_20_terms_with_constraint (ap : ArithmeticProgression) 
    (h : nthTerm ap 4 + nthTerm ap 12 = 20) :
    sumOfTerms ap 20 = 200 + 50 * ap.d := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_20_terms_with_constraint_l628_62830


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_for_negative_sqrt3_over_3_slope_l628_62829

/-- The angle of inclination of a line with slope -√3/3 is 150° -/
theorem angle_of_inclination_for_negative_sqrt3_over_3_slope :
  ∀ (l : Real → Real),
  (∀ x y, y - l x = (-Real.sqrt 3 / 3) * (y - x)) →
  Real.arctan (-Real.sqrt 3 / 3) = 150 * π / 180 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_of_inclination_for_negative_sqrt3_over_3_slope_l628_62829


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_max_value_condition_l628_62828

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (4 - x) + Real.sqrt (x - 1)

-- Theorem for the domain of f
theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Icc 1 4 := by sorry

-- Theorem for the necessary and sufficient condition
theorem max_value_condition (a : ℝ) :
  (∃ M, ∀ x ∈ Set.Icc a (a + 1), f x ≤ M) ↔ (3/2 < a ∧ a ≤ 3) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_max_value_condition_l628_62828


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_half_in_sequence_l628_62898

open Set
open BigOperators
open Finset

theorem exists_half_in_sequence (n : ℕ) (a : Fin n → ℝ) 
  (h_range : ∀ i, 0 < a i ∧ a i < 1) 
  (f : Finset (Fin n) → ℝ) 
  (h_f : ∀ I : Finset (Fin n), f I = ∏ i in I, a i * ∏ i in Iᶜ, (1 - a i)) 
  (h_sum : ∑ I in filter (λ I => Odd I.card) (powerset (univ : Finset (Fin n))), f I = (1/2 : ℝ)) : 
  ∃ i, a i = (1/2 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_half_in_sequence_l628_62898


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_calculation_l628_62896

/-- Calculates the percentage gain/loss given the item price and total loss -/
noncomputable def calculatePercentage (itemPrice : ℝ) (totalLoss : ℝ) : ℝ :=
  totalLoss / (2 * itemPrice)

theorem percentage_calculation (itemPrice totalLoss : ℝ) 
  (h1 : itemPrice > 0) 
  (h2 : totalLoss > 0) :
  let percentage := calculatePercentage itemPrice totalLoss
  ∃ (gain loss : ℝ),
    gain = itemPrice + itemPrice * percentage ∧
    loss = itemPrice - itemPrice * percentage ∧
    gain - loss = totalLoss := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_calculation_l628_62896


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lava_lamp_probability_l628_62892

/-- The number of red lamps --/
def num_red : ℕ := 4

/-- The number of blue lamps --/
def num_blue : ℕ := 2

/-- The total number of lamps --/
def total_lamps : ℕ := num_red + num_blue

/-- The number of lamps to be turned on --/
def lamps_on : ℕ := 3

/-- The probability of the specific arrangement and state --/
def specific_arrangement_probability : ℚ := 3 / 100

/-- Theorem stating the probability of the specific arrangement and state --/
theorem lava_lamp_probability :
  (↑(Nat.choose total_lamps lamps_on) : ℚ)⁻¹ *
  (↑(Nat.choose num_blue 1) : ℚ)⁻¹ *
  (↑(Nat.choose (num_red - 1) 1) : ℚ)⁻¹ *
  (↑(Nat.choose (num_red - 2) 1) : ℚ)⁻¹ *
  ↑(Nat.choose (lamps_on - 1) 2) = specific_arrangement_probability := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lava_lamp_probability_l628_62892


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_not_pentagon_l628_62893

/-- A unit cube in 3D space -/
def UnitCube : Set (Fin 3 → ℝ) :=
  {p | ∀ i, 0 ≤ p i ∧ p i ≤ 1}

/-- A projection from 3D to 2D -/
noncomputable def Projection : (Fin 3 → ℝ) → (Fin 2 → ℝ) := sorry

/-- A simple convex pentagon in 2D -/
def SimpleConvexPentagon : Set (Fin 2 → ℝ) := sorry

/-- The theorem stating that the projection of a unit cube cannot be a simple convex pentagon -/
theorem projection_not_pentagon : 
  ¬∃ (proj : (Fin 3 → ℝ) → (Fin 2 → ℝ)), 
    (Projection = proj) ∧ (proj '' UnitCube = SimpleConvexPentagon) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_not_pentagon_l628_62893


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_tangent_line_slope_tangent_is_tangent_l628_62841

/-- The main function f(x) = x + 1 + ln x -/
noncomputable def f (x : ℝ) : ℝ := x + 1 + Real.log x

/-- The quadratic function g(x) = ax² + (a+2)x + 1 -/
def g (a x : ℝ) : ℝ := a * x^2 + (a + 2) * x + 1

/-- The derivative of f -/
noncomputable def f' (x : ℝ) : ℝ := 1 + 1/x

/-- The tangent line of f at x = 1 -/
def tangent_line (x : ℝ) : ℝ := 2 * x

theorem tangent_line_value : tangent_line 1 = f 1 := by sorry

theorem tangent_line_slope : (f' 1 : ℝ) = 2 := by sorry

theorem tangent_is_tangent (a : ℝ) : 
  (∃ x, x ≠ 1 ∧ tangent_line x = g a x) → 
  (∃! x, tangent_line x = g a x) →
  a = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_value_tangent_line_slope_tangent_is_tangent_l628_62841


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_with_cutout_area_l628_62880

-- Define constants
noncomputable def π : ℝ := Real.pi
def r : ℝ := 10  -- radius of hemisphere
def r_cutout : ℝ := 3  -- radius of cutout

-- Define the surface area of a hemisphere
noncomputable def hemisphere_surface_area (radius : ℝ) : ℝ := 2 * π * radius^2

-- Define the area of a circle
noncomputable def circle_area (radius : ℝ) : ℝ := π * radius^2

-- Theorem statement
theorem hemisphere_with_cutout_area :
  hemisphere_surface_area r + circle_area r - circle_area r_cutout = 291 * π :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hemisphere_with_cutout_area_l628_62880


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l628_62843

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the properties of the triangle
noncomputable def circumradius (t : Triangle) : ℝ := 1
noncomputable def area (t : Triangle) : ℝ := 1/4

-- State the theorem
theorem triangle_inequality (t : Triangle) : 
  circumradius t = 1 → area t = 1/4 → 
  Real.sqrt t.a + Real.sqrt t.b + Real.sqrt t.c < 1/t.a + 1/t.b + 1/t.c :=
by
  intro h_circumradius h_area
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_inequality_l628_62843


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_odd_and_increasing_l628_62874

-- Define the function f(x) = sin x
noncomputable def f (x : ℝ) : ℝ := Real.sin x

-- State the theorem
theorem sin_odd_and_increasing :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x < f y) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_odd_and_increasing_l628_62874


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l628_62866

noncomputable def f (a b x : ℝ) : ℝ := a * (1/4)^(abs x) + b

theorem function_properties (a b : ℝ) 
  (h1 : f a b 0 = 0)  -- passes through origin
  (h2 : ∀ ε > 0, ∃ δ > 0, ∀ x, abs x > δ → abs (f a b x - 2) < ε)  -- approaches y = 2
  : (a = -2 ∧ b = 2) ∧ 
    (∀ x y, f a b x = f a b y ∧ x ≠ y → x + y = 0) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_function_properties_l628_62866


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volumes_of_divided_prism_l628_62823

/-- Regular triangular prism with given properties -/
structure RegularTriangularPrism where
  -- Base side length
  base_side : ℝ
  -- Area of the cross-section by the dividing plane
  cross_section_area : ℝ
  -- Assumption that the base side is 2√14
  base_side_eq : base_side = 2 * Real.sqrt 14
  -- Assumption that the cross-section area is 21
  cross_section_area_eq : cross_section_area = 21

/-- Volumes of the parts after division -/
noncomputable def volumes (prism : RegularTriangularPrism) : (ℝ × ℝ) :=
  (112 / 3, 154 / 3)

/-- Theorem stating that the volumes of the parts are as calculated -/
theorem volumes_of_divided_prism (prism : RegularTriangularPrism) :
  volumes prism = (112 / 3, 154 / 3) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volumes_of_divided_prism_l628_62823


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_face_area_l628_62873

/-- The total area of the four triangular faces of a right, square-based pyramid -/
noncomputable def totalArea (baseEdge : ℝ) (lateralEdge : ℝ) : ℝ :=
  4 * (1/2 * baseEdge * Real.sqrt (lateralEdge^2 - (baseEdge/2)^2))

/-- Theorem: The total area of the four triangular faces of a right, square-based pyramid
    with base edges of 8 units and lateral edges of 7 units is equal to 16√33 square units -/
theorem pyramid_face_area : totalArea 8 7 = 16 * Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_face_area_l628_62873


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_black_cells_l628_62881

/-- Represents the state of a cell on the board -/
inductive CellState
| White
| Black

/-- Represents a move by Player A (coloring four cells black) -/
structure MoveA where
  cells : Finset (Fin 8 × Fin 8)
  card_eq_four : cells.card = 4

/-- Represents a move by Player B (coloring a row or column white) -/
inductive MoveB
| Row (i : Fin 8)
| Col (j : Fin 8)

/-- The game state -/
structure GameState where
  board : Fin 8 → Fin 8 → CellState

/-- Applies Player A's move to the game state -/
def applyMoveA (state : GameState) (move : MoveA) : GameState :=
  sorry

/-- Applies Player B's move to the game state -/
def applyMoveB (state : GameState) (move : MoveB) : GameState :=
  sorry

/-- Counts the number of black cells on the board -/
def GameState.countBlackCells (state : GameState) : Nat :=
  sorry

/-- Theorem stating the maximum number of black cells possible -/
theorem max_black_cells :
  ∀ (strategy : GameState → MoveB),
  ∃ (moves : List MoveA),
  ∀ (state : GameState),
  (moves.foldl (λ s m => applyMoveB (applyMoveA s m) (strategy (applyMoveA s m))) state).countBlackCells ≤ 25 :=
by
  sorry

#eval "Compilation successful!"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_black_cells_l628_62881
