import Mathlib

namespace NUMINAMATH_CALUDE_circular_ring_area_l959_95966

/-- Given a regular n-gon with area t, the area of the circular ring formed by
    its inscribed and circumscribed circles is (π * t * tan(180°/n)) / n. -/
theorem circular_ring_area (n : ℕ) (t : ℝ) (h1 : n ≥ 3) (h2 : t > 0) :
  let T := (Real.pi * t * Real.tan (Real.pi / n)) / n
  ∃ (r R : ℝ), r > 0 ∧ R > r ∧
    t = n * r^2 * Real.sin (Real.pi / n) * Real.cos (Real.pi / n) ∧
    R = r / Real.cos (Real.pi / n) ∧
    T = Real.pi * (R^2 - r^2) :=
by sorry

end NUMINAMATH_CALUDE_circular_ring_area_l959_95966


namespace NUMINAMATH_CALUDE_transformed_standard_deviation_l959_95928

variable (x : Fin 10 → ℝ)

def standardDeviation (y : Fin 10 → ℝ) : ℝ := sorry

theorem transformed_standard_deviation 
  (h : standardDeviation x = 2) : 
  standardDeviation (fun i => 2 * x i - 1) = 4 := by sorry

end NUMINAMATH_CALUDE_transformed_standard_deviation_l959_95928


namespace NUMINAMATH_CALUDE_sector_area_l959_95937

theorem sector_area (perimeter : ℝ) (central_angle : ℝ) (h1 : perimeter = 4) (h2 : central_angle = 2) :
  let radius := perimeter / (2 + central_angle)
  let arc_length := radius * central_angle
  let area := (1 / 2) * radius * arc_length
  area = 1 := by sorry

end NUMINAMATH_CALUDE_sector_area_l959_95937


namespace NUMINAMATH_CALUDE_equation_solutions_l959_95961

theorem equation_solutions :
  (∀ x : ℝ, x^2 - 4 = 0 ↔ x = 2 ∨ x = -2) ∧
  (∀ x : ℝ, x^2 - 2*x = 3 ↔ x = -1 ∨ x = 3) := by
sorry

end NUMINAMATH_CALUDE_equation_solutions_l959_95961


namespace NUMINAMATH_CALUDE_platform_length_l959_95917

/-- Given a train with speed 72 km/h and length 290.04 m, crossing a platform in 26 seconds,
    prove that the length of the platform is 229.96 m. -/
theorem platform_length (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) :
  train_speed = 72 →
  train_length = 290.04 →
  crossing_time = 26 →
  ∃ platform_length : ℝ,
    platform_length = 229.96 ∧
    platform_length = train_speed * (1000 / 3600) * crossing_time - train_length :=
by sorry

end NUMINAMATH_CALUDE_platform_length_l959_95917


namespace NUMINAMATH_CALUDE_total_wax_needed_l959_95904

def wax_already_has : ℕ := 28
def wax_still_needs : ℕ := 260

theorem total_wax_needed : wax_already_has + wax_still_needs = 288 := by
  sorry

end NUMINAMATH_CALUDE_total_wax_needed_l959_95904


namespace NUMINAMATH_CALUDE_f_2011_equals_6_l959_95915

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop := ∀ x, f (a - x) = f (a + x)

theorem f_2011_equals_6 (f : ℝ → ℝ) 
    (h_even : is_even_function f)
    (h_sym : symmetric_about f 2)
    (h_sum : f 2011 + 2 * f 1 = 18) :
  f 2011 = 6 := by
sorry

end NUMINAMATH_CALUDE_f_2011_equals_6_l959_95915


namespace NUMINAMATH_CALUDE_credit_card_interest_rate_l959_95920

theorem credit_card_interest_rate 
  (initial_balance : ℝ) 
  (payment : ℝ) 
  (new_balance : ℝ) 
  (h1 : initial_balance = 150)
  (h2 : payment = 50)
  (h3 : new_balance = 120) :
  (new_balance - (initial_balance - payment)) / initial_balance * 100 = 13.33 := by
sorry

end NUMINAMATH_CALUDE_credit_card_interest_rate_l959_95920


namespace NUMINAMATH_CALUDE_fruit_box_composition_l959_95991

/-- Represents the contents of the fruit box -/
structure FruitBox where
  apples : ℕ
  pears : ℕ

/-- The total number of fruits in the box -/
def FruitBox.total (box : FruitBox) : ℕ := box.apples + box.pears

/-- Predicate to check if selecting n fruits always includes at least one apple -/
def always_includes_apple (box : FruitBox) (n : ℕ) : Prop :=
  box.pears < n

/-- Predicate to check if selecting n fruits always includes at least one pear -/
def always_includes_pear (box : FruitBox) (n : ℕ) : Prop :=
  box.apples < n

/-- The main theorem stating the unique composition of the fruit box -/
theorem fruit_box_composition :
  ∃! (box : FruitBox),
    box.total ≥ 5 ∧
    always_includes_apple box 3 ∧
    always_includes_pear box 4 :=
  sorry

end NUMINAMATH_CALUDE_fruit_box_composition_l959_95991


namespace NUMINAMATH_CALUDE_percentage_problem_l959_95925

theorem percentage_problem (P : ℝ) : (P / 100 * 1265) / 6 = 543.95 ↔ P = 258 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l959_95925


namespace NUMINAMATH_CALUDE_speed_conversion_l959_95985

theorem speed_conversion (speed_ms : ℝ) (speed_kmh : ℝ) : 
  speed_ms = 0.2790697674418605 ∧ speed_kmh = 1.0046511627906978 → 
  speed_ms = speed_kmh / 3.6 :=
by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l959_95985


namespace NUMINAMATH_CALUDE_equation_solution_l959_95943

theorem equation_solution : ∃ (x₁ x₂ : ℝ), 
  (x₁ = 1 + Real.sqrt 3 ∧ x₂ = 1 - Real.sqrt 3) ∧
  (x₁^2 = (4*x₁ - 2)/(x₁ - 2) ∧ x₂^2 = (4*x₂ - 2)/(x₂ - 2)) :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l959_95943


namespace NUMINAMATH_CALUDE_quadrilateral_exists_for_four_lines_l959_95990

/-- A line in a plane --/
structure Line :=
  (a b c : ℝ)

/-- A point in a plane --/
structure Point :=
  (x y : ℝ)

/-- A region in a plane --/
structure Region :=
  (vertices : List Point)

/-- Checks if a region is a quadrilateral --/
def isQuadrilateral (r : Region) : Prop :=
  r.vertices.length = 4

/-- The set of all regions formed by the intersections of the given lines --/
def regionsFormedByLines (lines : List Line) : Set Region :=
  sorry

/-- The theorem stating that among the regions formed by 4 intersecting lines, 
    there exists at least one quadrilateral --/
theorem quadrilateral_exists_for_four_lines 
  (lines : List Line) 
  (h : lines.length = 4) : 
  ∃ r ∈ regionsFormedByLines lines, isQuadrilateral r :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_exists_for_four_lines_l959_95990


namespace NUMINAMATH_CALUDE_tan_two_beta_l959_95982

theorem tan_two_beta (α β : ℝ) 
  (h1 : Real.tan (α + β) = 2) 
  (h2 : Real.tan (α - β) = 3) : 
  Real.tan (2 * β) = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_tan_two_beta_l959_95982


namespace NUMINAMATH_CALUDE_remainder_calculation_l959_95944

theorem remainder_calculation (P Q R D Q' R' D' D'' Q'' R'' : ℤ)
  (h1 : P = Q * D + R)
  (h2 : Q = D' * Q' + R')
  (h3 : D'' = D' + 1)
  (h4 : P = D'' * Q'' + R'') :
  R'' = R + D * R' - Q'' := by sorry

end NUMINAMATH_CALUDE_remainder_calculation_l959_95944


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l959_95909

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = f x

-- Define the concept of axis of symmetry
def AxisOfSymmetry (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x : ℝ, f (a + x) = f (a - x)

-- State the theorem
theorem symmetry_of_shifted_even_function (f : ℝ → ℝ) (h : EvenFunction f) :
  AxisOfSymmetry (fun x ↦ f (x + 1)) (-1) :=
sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l959_95909


namespace NUMINAMATH_CALUDE_functional_equation_solution_l959_95901

theorem functional_equation_solution (f : ℝ → ℝ) :
  (∀ x y : ℝ, f x * f y = f (x - y)) →
  (∀ x : ℝ, f x = 0) ∨ (∀ x : ℝ, f x = 1) :=
by sorry

end NUMINAMATH_CALUDE_functional_equation_solution_l959_95901


namespace NUMINAMATH_CALUDE_four_must_be_in_A_l959_95951

/-- A type representing the circles in the diagram -/
inductive Circle : Type
  | A | B | C | D | E | F | G

/-- The set of numbers to be placed in the circles -/
def NumberSet : Finset ℕ := {1, 2, 3, 4, 5, 6, 7}

/-- A function that assigns a number to each circle -/
def Assignment := Circle → ℕ

/-- Predicate to check if an assignment is valid -/
def IsValidAssignment (f : Assignment) : Prop :=
  (∀ n ∈ NumberSet, ∃ c : Circle, f c = n) ∧
  (∀ c : Circle, f c ∈ NumberSet) ∧
  (f Circle.A + f Circle.D + f Circle.E = 
   f Circle.A + f Circle.C + f Circle.F) ∧
  (f Circle.A + f Circle.D + f Circle.E = 
   f Circle.A + f Circle.B + f Circle.G) ∧
  (f Circle.D + f Circle.C + f Circle.B = 
   f Circle.E + f Circle.F + f Circle.G)

theorem four_must_be_in_A (f : Assignment) 
  (h : IsValidAssignment f) : 
  f Circle.A = 4 ∧ f Circle.E ≠ 4 := by
  sorry

end NUMINAMATH_CALUDE_four_must_be_in_A_l959_95951


namespace NUMINAMATH_CALUDE_flu_infection_rate_l959_95992

theorem flu_infection_rate : ∃ (x : ℝ), 
  (x > 0) ∧ 
  (1 + x + x * (1 + x) = 196) ∧ 
  (x = 13) :=
sorry

end NUMINAMATH_CALUDE_flu_infection_rate_l959_95992


namespace NUMINAMATH_CALUDE_company_a_profit_share_l959_95945

/-- Prove that Company A's share of combined profits is 60% given the conditions -/
theorem company_a_profit_share :
  ∀ (total_profit : ℝ) (company_b_profit : ℝ) (company_a_profit : ℝ),
    company_b_profit = 0.4 * total_profit →
    company_b_profit = 60000 →
    company_a_profit = 90000 →
    company_a_profit / total_profit = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_company_a_profit_share_l959_95945


namespace NUMINAMATH_CALUDE_debate_team_groups_l959_95957

/-- The number of boys on the debate team -/
def num_boys : ℕ := 28

/-- The number of girls on the debate team -/
def num_girls : ℕ := 4

/-- The minimum number of boys required in each group -/
def min_boys_per_group : ℕ := 2

/-- The minimum number of girls required in each group -/
def min_girls_per_group : ℕ := 1

/-- The maximum number of groups that can be formed -/
def max_groups : ℕ := 4

theorem debate_team_groups :
  (num_girls ≥ max_groups * min_girls_per_group) ∧
  (num_boys ≥ max_groups * min_boys_per_group) ∧
  (∀ n : ℕ, n > max_groups → 
    (num_girls < n * min_girls_per_group) ∨ 
    (num_boys < n * min_boys_per_group)) :=
sorry

end NUMINAMATH_CALUDE_debate_team_groups_l959_95957


namespace NUMINAMATH_CALUDE_concentric_circles_radii_difference_l959_95903

theorem concentric_circles_radii_difference (s L : ℝ) (h : s > 0) :
  (L^2 / s^2 = 9 / 4) → (L - s = 0.5 * s) := by
  sorry

end NUMINAMATH_CALUDE_concentric_circles_radii_difference_l959_95903


namespace NUMINAMATH_CALUDE_product_approx_six_times_number_l959_95977

-- Define a function to check if two numbers are approximately equal
def approx_equal (x y : ℝ) : Prop := abs (x - y) ≤ 1

-- Theorem 1: The product of 198 × 2 is approximately 400
theorem product_approx : approx_equal (198 * 2) 400 := by sorry

-- Theorem 2: If twice a number is 78, then six times that number is 240
theorem six_times_number (x : ℝ) (h : 2 * x = 78) : 6 * x = 240 := by sorry

end NUMINAMATH_CALUDE_product_approx_six_times_number_l959_95977


namespace NUMINAMATH_CALUDE_factorial_ratio_72_l959_95978

theorem factorial_ratio_72 : ∃! (n : ℕ), (Nat.factorial (n + 2)) / (Nat.factorial n) = 72 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_72_l959_95978


namespace NUMINAMATH_CALUDE_eleanor_cookies_l959_95953

theorem eleanor_cookies : ∃! N : ℕ, N < 100 ∧ N % 12 = 5 ∧ N % 8 = 2 ∧ N = 29 := by
  sorry

end NUMINAMATH_CALUDE_eleanor_cookies_l959_95953


namespace NUMINAMATH_CALUDE_box_max_volume_l959_95967

variable (a : ℝ) (x : ℝ)

-- Define the volume function
def V (a x : ℝ) : ℝ := (a - 2*x)^2 * x

-- State the theorem
theorem box_max_volume (h1 : a > 0) (h2 : 0 < x) (h3 : x < a/2) :
  ∃ (x_max : ℝ), x_max = a/6 ∧ 
  (∀ y, 0 < y → y < a/2 → V a y ≤ V a x_max) ∧
  V a x_max = 2*a^3/27 :=
sorry

end NUMINAMATH_CALUDE_box_max_volume_l959_95967


namespace NUMINAMATH_CALUDE_problem_statement_l959_95927

theorem problem_statement : 1^567 + 3^5 / 3^3 - 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l959_95927


namespace NUMINAMATH_CALUDE_greatest_five_digit_with_product_90_l959_95987

def is_five_digit (n : ℕ) : Prop :=
  10000 ≤ n ∧ n ≤ 99999

def digit_product (n : ℕ) : ℕ :=
  (n / 10000) * ((n / 1000) % 10) * ((n / 100) % 10) * ((n / 10) % 10) * (n % 10)

def digit_sum (n : ℕ) : ℕ :=
  (n / 10000) + ((n / 1000) % 10) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

theorem greatest_five_digit_with_product_90 :
  ∃ M : ℕ, is_five_digit M ∧ 
    digit_product M = 90 ∧ 
    (∀ n : ℕ, is_five_digit n → digit_product n = 90 → n ≤ M) ∧
    digit_sum M = 17 :=
sorry

end NUMINAMATH_CALUDE_greatest_five_digit_with_product_90_l959_95987


namespace NUMINAMATH_CALUDE_possible_values_of_a_l959_95938

theorem possible_values_of_a (a b x : ℝ) 
  (h1 : a ≠ b) 
  (h2 : a^3 - b^3 = 19*x^3) 
  (h3 : a - b = x) : 
  a = 3*x ∨ a = -2*x := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_l959_95938


namespace NUMINAMATH_CALUDE_man_walked_40_minutes_l959_95921

/-- Represents the scenario of a man meeting his wife at the train station and going home. -/
structure TrainScenario where
  T : ℕ  -- usual arrival time at the station
  X : ℕ  -- usual driving time from station to home

/-- Calculates the time spent walking in the given scenario. -/
def time_walking (s : TrainScenario) : ℕ :=
  s.X - 40

/-- Theorem stating that the man spent 40 minutes walking. -/
theorem man_walked_40_minutes (s : TrainScenario) :
  time_walking s = 40 :=
by
  sorry


end NUMINAMATH_CALUDE_man_walked_40_minutes_l959_95921


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l959_95958

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the parallel and perpendicular relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)

-- State the theorem
theorem line_parallel_perpendicular 
  (a b : Line) (α : Plane) :
  parallel a b → perpendicular a α → perpendicular b α :=
by sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l959_95958


namespace NUMINAMATH_CALUDE_dulce_has_three_points_l959_95980

-- Define the points for each person and the team
def max_points : ℕ := 5
def dulce_points : ℕ := 3  -- This is what we want to prove
def val_points (d : ℕ) : ℕ := 2 * (max_points + d)
def team_total (d : ℕ) : ℕ := max_points + d + val_points d

-- Define the opponent's points and the point difference
def opponent_points : ℕ := 40
def point_difference : ℕ := 16

-- Theorem to prove
theorem dulce_has_three_points : 
  team_total dulce_points = opponent_points - point_difference := by
  sorry


end NUMINAMATH_CALUDE_dulce_has_three_points_l959_95980


namespace NUMINAMATH_CALUDE_isosceles_trapezoid_dimensions_l959_95960

/-- An isosceles trapezoid with legs intersecting at a right angle -/
structure IsoscelesTrapezoid where
  /-- Length of the longer base -/
  longerBase : ℝ
  /-- Length of the shorter base -/
  shorterBase : ℝ
  /-- Height of the trapezoid -/
  height : ℝ
  /-- Area of the trapezoid -/
  area : ℝ
  /-- The trapezoid is isosceles -/
  isIsosceles : True
  /-- The legs intersect at a right angle -/
  legsRightAngle : True
  /-- The area is calculated correctly -/
  areaEq : area = (longerBase + shorterBase) * height / 2

/-- Theorem about the dimensions of a specific isosceles trapezoid -/
theorem isosceles_trapezoid_dimensions (t : IsoscelesTrapezoid) 
  (h_area : t.area = 12)
  (h_height : t.height = 2) :
  t.longerBase = 8 ∧ t.shorterBase = 4 := by
  sorry


end NUMINAMATH_CALUDE_isosceles_trapezoid_dimensions_l959_95960


namespace NUMINAMATH_CALUDE_prob_C_correct_prob_C_given_A_correct_l959_95965

/-- Represents a box containing red and white balls -/
structure Box where
  red : ℕ
  white : ℕ

/-- The probability of drawing a red ball from a box -/
def prob_red (b : Box) : ℚ :=
  b.red / (b.red + b.white)

/-- The probability of drawing a white ball from a box -/
def prob_white (b : Box) : ℚ :=
  b.white / (b.red + b.white)

/-- Initial state of box A -/
def box_A : Box := ⟨3, 2⟩

/-- Initial state of box B -/
def box_B : Box := ⟨2, 3⟩

/-- State of box B after transferring a ball from box A -/
def box_B_after (red_transferred : Bool) : Box :=
  if red_transferred then ⟨box_B.red + 1, box_B.white⟩
  else ⟨box_B.red, box_B.white + 1⟩

/-- Probability of event C given the initial conditions -/
def prob_C : ℚ :=
  (prob_red box_A * prob_red (box_B_after true)) +
  (prob_white box_A * prob_red (box_B_after false))

/-- Conditional probability of event C given event A -/
def prob_C_given_A : ℚ :=
  prob_red (box_B_after true)

theorem prob_C_correct : prob_C = 13 / 30 := by sorry

theorem prob_C_given_A_correct : prob_C_given_A = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_prob_C_correct_prob_C_given_A_correct_l959_95965


namespace NUMINAMATH_CALUDE_complex_number_existence_l959_95949

theorem complex_number_existence : ∃ (c : ℂ) (d : ℝ), c ≠ 0 ∧
  ∀ (z : ℂ), Complex.abs z = 1 → (1 + z + z^2 ≠ 0) →
    Complex.abs (Complex.abs (1 / (1 + z + z^2)) - Complex.abs (1 / (1 + z + z^2) - c)) = d :=
by sorry

end NUMINAMATH_CALUDE_complex_number_existence_l959_95949


namespace NUMINAMATH_CALUDE_prob_non_yellow_specific_l959_95959

/-- The probability of selecting a non-yellow jelly bean -/
def prob_non_yellow (red green yellow blue : ℕ) : ℚ :=
  (red + green + blue) / (red + green + yellow + blue)

/-- Theorem: The probability of selecting a non-yellow jelly bean from a bag
    containing 4 red, 7 green, 9 yellow, and 10 blue jelly beans is 7/10 -/
theorem prob_non_yellow_specific : prob_non_yellow 4 7 9 10 = 7/10 := by
  sorry

end NUMINAMATH_CALUDE_prob_non_yellow_specific_l959_95959


namespace NUMINAMATH_CALUDE_pet_store_theorem_l959_95976

/-- Given a ratio of cats to dogs to birds and the number of cats, 
    calculate the number of dogs and birds -/
def pet_store_count (cat_ratio dog_ratio bird_ratio num_cats : ℕ) : ℕ × ℕ :=
  let scale_factor := num_cats / cat_ratio
  (dog_ratio * scale_factor, bird_ratio * scale_factor)

/-- Theorem: Given the ratio 2:3:4 for cats:dogs:birds and 20 cats, 
    there are 30 dogs and 40 birds -/
theorem pet_store_theorem : 
  pet_store_count 2 3 4 20 = (30, 40) := by
  sorry

end NUMINAMATH_CALUDE_pet_store_theorem_l959_95976


namespace NUMINAMATH_CALUDE_complex_product_real_l959_95902

theorem complex_product_real (a : ℝ) : 
  let z₁ : ℂ := 3 - 2*I
  let z₂ : ℂ := 1 + a*I
  (z₁ * z₂).im = 0 → a = 2/3 := by sorry

end NUMINAMATH_CALUDE_complex_product_real_l959_95902


namespace NUMINAMATH_CALUDE_sequence_bounded_l959_95969

/-- Given a sequence of positive real numbers satisfying a specific condition, prove that the sequence is bounded -/
theorem sequence_bounded (a : ℕ → ℝ) (h_pos : ∀ n, a n > 0) 
  (h_cond : ∀ k n m l, k + n = m + l → 
    (a k + a n) / (1 + a k * a n) = (a m + a l) / (1 + a m * a l)) :
  ∃ M, ∀ n, a n ≤ M :=
sorry

end NUMINAMATH_CALUDE_sequence_bounded_l959_95969


namespace NUMINAMATH_CALUDE_balloon_difference_balloon_difference_proof_l959_95988

/-- Proves that the difference between the combined total of Amy, Felix, and Olivia's balloons
    and James' balloons is 373. -/
theorem balloon_difference : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun james_balloons amy_balloons felix_balloons olivia_balloons =>
    james_balloons = 1222 ∧
    amy_balloons = 513 ∧
    felix_balloons = 687 ∧
    olivia_balloons = 395 →
    (amy_balloons + felix_balloons + olivia_balloons) - james_balloons = 373

-- The proof is omitted
theorem balloon_difference_proof :
  balloon_difference 1222 513 687 395 := by sorry

end NUMINAMATH_CALUDE_balloon_difference_balloon_difference_proof_l959_95988


namespace NUMINAMATH_CALUDE_extreme_points_range_l959_95923

/-- The function f(x) = x^2 + a*ln(1+x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a * Real.log (1 + x)

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := (2 * x^2 + 2 * x + a) / (1 + x)

theorem extreme_points_range (a : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ 
    f_derivative a x = 0 ∧ 
    f_derivative a y = 0 ∧ 
    (∀ z : ℝ, f_derivative a z = 0 → z = x ∨ z = y)) →
  (0 < a ∧ a < 1/2) :=
sorry

end NUMINAMATH_CALUDE_extreme_points_range_l959_95923


namespace NUMINAMATH_CALUDE_quadrant_I_solution_condition_l959_95995

theorem quadrant_I_solution_condition (c : ℝ) :
  (∃ x y : ℝ, 2 * x - y = 5 ∧ c * x + y = 4 ∧ x > 0 ∧ y > 0) ↔ -2 < c ∧ c < 8/5 := by
  sorry

end NUMINAMATH_CALUDE_quadrant_I_solution_condition_l959_95995


namespace NUMINAMATH_CALUDE_planet_colonization_combinations_l959_95955

/-- Represents the number of habitable planets discovered -/
def total_planets : ℕ := 13

/-- Represents the number of Earth-like planets -/
def earth_like_planets : ℕ := 5

/-- Represents the number of Mars-like planets -/
def mars_like_planets : ℕ := total_planets - earth_like_planets

/-- Represents the units required to colonize an Earth-like planet -/
def earth_like_units : ℕ := 2

/-- Represents the units required to colonize a Mars-like planet -/
def mars_like_units : ℕ := 1

/-- Represents the total units available for colonization -/
def available_units : ℕ := 15

/-- Calculates the number of unique combinations of planets that can be occupied -/
def count_combinations : ℕ :=
  (Nat.choose earth_like_planets earth_like_planets * Nat.choose mars_like_planets 5) +
  (Nat.choose earth_like_planets 4 * Nat.choose mars_like_planets 7)

theorem planet_colonization_combinations :
  count_combinations = 96 :=
sorry

end NUMINAMATH_CALUDE_planet_colonization_combinations_l959_95955


namespace NUMINAMATH_CALUDE_all_graphs_different_l959_95940

-- Define the three equations
def eq_I (x y : ℝ) : Prop := y = x - 3
def eq_II (x y : ℝ) : Prop := y = (x^2 - 9) / (x + 3)
def eq_III (x y : ℝ) : Prop := (x + 3) * y = x^2 - 9

-- Define what it means for two equations to have the same graph
def same_graph (eq1 eq2 : ℝ → ℝ → Prop) : Prop :=
  ∀ x y, eq1 x y ↔ eq2 x y

-- Theorem stating that all graphs are different
theorem all_graphs_different :
  ¬(same_graph eq_I eq_II) ∧ 
  ¬(same_graph eq_I eq_III) ∧ 
  ¬(same_graph eq_II eq_III) :=
sorry

end NUMINAMATH_CALUDE_all_graphs_different_l959_95940


namespace NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l959_95973

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (parallel : Plane → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)

-- Define the theorem
theorem planes_parallel_if_perpendicular_to_same_line
  (m n : Line) (α β : Plane)
  (h_not_coincident_lines : m ≠ n)
  (h_not_coincident_planes : α ≠ β)
  (h_m_perp_α : perpendicular m α)
  (h_m_perp_β : perpendicular m β) :
  parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_if_perpendicular_to_same_line_l959_95973


namespace NUMINAMATH_CALUDE_volleyball_lineup_count_is_151200_l959_95974

/-- The number of ways to choose 6 players from a team of 10 players for 6 distinct positions -/
def volleyball_lineup_count : ℕ := 10 * 9 * 8 * 7 * 6 * 5

/-- Theorem stating that the number of ways to choose a volleyball lineup is 151,200 -/
theorem volleyball_lineup_count_is_151200 : volleyball_lineup_count = 151200 := by
  sorry

end NUMINAMATH_CALUDE_volleyball_lineup_count_is_151200_l959_95974


namespace NUMINAMATH_CALUDE_total_kids_signed_up_l959_95964

/-- The number of girls signed up for the talent show. -/
def num_girls : ℕ := 28

/-- The difference between the number of girls and boys signed up. -/
def girl_boy_difference : ℕ := 22

/-- Theorem: The total number of kids signed up for the talent show is 34. -/
theorem total_kids_signed_up : 
  num_girls + (num_girls - girl_boy_difference) = 34 := by
  sorry

end NUMINAMATH_CALUDE_total_kids_signed_up_l959_95964


namespace NUMINAMATH_CALUDE_f_value_at_7_l959_95979

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x - 5

-- State the theorem
theorem f_value_at_7 (a b : ℝ) :
  f a b (-7) = 7 → f a b 7 = -17 := by
  sorry

end NUMINAMATH_CALUDE_f_value_at_7_l959_95979


namespace NUMINAMATH_CALUDE_inequality_equivalence_l959_95983

theorem inequality_equivalence (x : ℝ) : 
  (2 * x + 3) / (x^2 - 2 * x + 4) > (4 * x + 5) / (2 * x^2 + 5 * x + 7) ↔ 
  x > (-23 - Real.sqrt 453) / 38 ∧ x < (-23 + Real.sqrt 453) / 38 :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l959_95983


namespace NUMINAMATH_CALUDE_bowling_ball_volume_l959_95947

/-- The volume of a sphere with two cylindrical holes drilled into it -/
theorem bowling_ball_volume 
  (sphere_diameter : ℝ) 
  (hole1_depth hole1_diameter : ℝ) 
  (hole2_depth hole2_diameter : ℝ) 
  (h1 : sphere_diameter = 24)
  (h2 : hole1_depth = 6)
  (h3 : hole1_diameter = 3)
  (h4 : hole2_depth = 6)
  (h5 : hole2_diameter = 4) : 
  (4 / 3 * π * (sphere_diameter / 2) ^ 3) - 
  (π * (hole1_diameter / 2) ^ 2 * hole1_depth) - 
  (π * (hole2_diameter / 2) ^ 2 * hole2_depth) = 2266.5 * π := by
  sorry

end NUMINAMATH_CALUDE_bowling_ball_volume_l959_95947


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l959_95914

-- Define the quadratic function f
def f : ℝ → ℝ := λ x => x^2 - 2*x - 1

-- State the theorem
theorem quadratic_function_properties :
  (∀ x, f x ≥ -2) ∧  -- minimum value is -2
  f 3 = 2 ∧ f (-1) = 2 ∧  -- given conditions
  (∀ t, f (2*t^2 - 4*t + 3) > f (t^2 + t + 3) ↔ t > 5 ∨ t < 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l959_95914


namespace NUMINAMATH_CALUDE_kitten_growth_ratio_l959_95936

/-- Given the initial length and final length of a kitten, and knowing that the final length is twice the intermediate length, prove that the ratio of intermediate length to initial length is 2. -/
theorem kitten_growth_ratio (L₀ L₂ L₄ : ℝ) (h₀ : L₀ = 4) (h₄ : L₄ = 16) (h_double : L₄ = 2 * L₂) : L₂ / L₀ = 2 := by
  sorry

end NUMINAMATH_CALUDE_kitten_growth_ratio_l959_95936


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_six_sqrt_five_l959_95962

theorem sqrt_sum_equals_six_sqrt_five :
  Real.sqrt ((5 - 3 * Real.sqrt 5) ^ 2) + Real.sqrt ((5 + 3 * Real.sqrt 5) ^ 2) = 6 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_six_sqrt_five_l959_95962


namespace NUMINAMATH_CALUDE_student_arrangement_count_l959_95942

/-- The number of students in the row -/
def total_students : ℕ := 4

/-- The number of students that must stand next to each other -/
def adjacent_students : ℕ := 2

/-- The number of different arrangements of students -/
def num_arrangements : ℕ := 12

/-- 
Theorem: Given 4 students standing in a row, where 2 specific students 
must stand next to each other, the number of different arrangements is 12.
-/
theorem student_arrangement_count :
  (total_students = 4) →
  (adjacent_students = 2) →
  (num_arrangements = 12) :=
by sorry

end NUMINAMATH_CALUDE_student_arrangement_count_l959_95942


namespace NUMINAMATH_CALUDE_homework_difference_l959_95986

theorem homework_difference (total : ℕ) (math : ℕ) (reading : ℕ)
  (h1 : total = 13)
  (h2 : math = 8)
  (h3 : total = math + reading) :
  math - reading = 3 :=
by sorry

end NUMINAMATH_CALUDE_homework_difference_l959_95986


namespace NUMINAMATH_CALUDE_range_of_m_l959_95916

/-- Statement p: For any real number x, the inequality x^2 - 2x + m ≥ 0 always holds -/
def statement_p (m : ℝ) : Prop :=
  ∀ x : ℝ, x^2 - 2*x + m ≥ 0

/-- Statement q: The equation (x^2)/(m-3) - (y^2)/m = 1 represents a hyperbola with foci on the x-axis -/
def statement_q (m : ℝ) : Prop :=
  m > 3 ∧ ∀ x y : ℝ, x^2 / (m - 3) - y^2 / m = 1

/-- The range of m when "p ∨ q" is true and "p ∧ q" is false -/
theorem range_of_m :
  ∀ m : ℝ, (statement_p m ∨ statement_q m) ∧ ¬(statement_p m ∧ statement_q m) →
  1 ≤ m ∧ m ≤ 3 :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l959_95916


namespace NUMINAMATH_CALUDE_trajectory_is_ray_l959_95935

/-- The set of complex numbers z satisfying |z+1| - |z-1| = 2 forms a ray in the complex plane -/
theorem trajectory_is_ray : 
  {z : ℂ | Complex.abs (z + 1) - Complex.abs (z - 1) = 2} = 
  {z : ℂ | ∃ t : ℝ, t ≥ 0 ∧ z = 1 + t} := by sorry

end NUMINAMATH_CALUDE_trajectory_is_ray_l959_95935


namespace NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l959_95905

theorem twenty_five_percent_less_than_80 (x : ℚ) : x + (1/4) * x = 60 → x = 48 := by
  sorry

end NUMINAMATH_CALUDE_twenty_five_percent_less_than_80_l959_95905


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l959_95975

theorem repeating_decimal_to_fraction : 
  ∃ (n : ℚ), n = 7 + 123 / 999 ∧ n = 593 / 111 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l959_95975


namespace NUMINAMATH_CALUDE_unique_solution_cube_equation_l959_95968

theorem unique_solution_cube_equation (x y : ℕ) :
  y^6 + 2*y^3 - y^2 + 1 = x^3 → x = 1 ∧ y = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_cube_equation_l959_95968


namespace NUMINAMATH_CALUDE_factory_machines_l959_95900

/-- The number of machines in the factory -/
def num_machines : ℕ := 7

/-- The time taken by 6 machines to fill the order -/
def time_6_machines : ℕ := 42

/-- The theorem stating the correct number of machines in the factory -/
theorem factory_machines :
  (∀ m : ℕ, m > 0 → ∃ t : ℕ, t > 0 ∧ m * t = 6 * time_6_machines) →
  (num_machines * (time_6_machines - 6) = 6 * time_6_machines) →
  num_machines = 7 := by
sorry


end NUMINAMATH_CALUDE_factory_machines_l959_95900


namespace NUMINAMATH_CALUDE_painter_problem_l959_95989

theorem painter_problem (total_rooms : ℕ) (time_per_room : ℕ) (time_left : ℕ) 
  (h1 : total_rooms = 9)
  (h2 : time_per_room = 8)
  (h3 : time_left = 32) :
  total_rooms - (time_left / time_per_room) = 5 := by
sorry

end NUMINAMATH_CALUDE_painter_problem_l959_95989


namespace NUMINAMATH_CALUDE_fruit_cost_l959_95908

/-- The cost of fruit combinations -/
theorem fruit_cost (x y z : ℚ) : 
  (2 * x + y + 4 * z = 6) →
  (4 * x + 2 * y + 2 * z = 4) →
  (4 * x + 2 * y + 5 * z = 8) :=
by sorry

end NUMINAMATH_CALUDE_fruit_cost_l959_95908


namespace NUMINAMATH_CALUDE_eighth_term_of_sequence_l959_95970

theorem eighth_term_of_sequence (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n : ℕ, S n = n^2) → a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_eighth_term_of_sequence_l959_95970


namespace NUMINAMATH_CALUDE_sum_remainder_mod_11_l959_95911

theorem sum_remainder_mod_11 : (103104 + 103105 + 103106 + 103107 + 103108 + 103109 + 103110 + 103111 + 103112) % 11 = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_mod_11_l959_95911


namespace NUMINAMATH_CALUDE_linear_system_k_values_l959_95984

/-- Given a system of linear equations in two variables x and y,
    prove the value of k under certain conditions. -/
theorem linear_system_k_values (x y k : ℝ) : 
  (3 * x + y = k + 1) →
  (x + 3 * y = 3) →
  (
    ((x * y < 0) → (k = -4)) ∧
    ((x + y < 3 ∧ x - y > 1) → (4 < k ∧ k < 8))
  ) := by sorry

end NUMINAMATH_CALUDE_linear_system_k_values_l959_95984


namespace NUMINAMATH_CALUDE_sector_area_l959_95952

/-- The area of a sector with a central angle of 60° in a circle passing through two given points -/
theorem sector_area (P Q : ℝ × ℝ) (h : P = (2, -2) ∧ Q = (8, 6)) : 
  let r := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  (1/6 : ℝ) * π * r^2 = 50*π/3 := by sorry

end NUMINAMATH_CALUDE_sector_area_l959_95952


namespace NUMINAMATH_CALUDE_statement_b_not_always_true_l959_95994

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields

/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields

/-- Parallel relation between a line and a plane -/
def parallel_line_plane (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallel relation between two lines -/
def parallel_lines (l1 l2 : Line3D) : Prop :=
  sorry

/-- Intersection of two planes -/
def plane_intersection (p1 p2 : Plane3D) : Line3D :=
  sorry

/-- Statement B is not always true -/
theorem statement_b_not_always_true :
  ∃ (a : Line3D) (α β : Plane3D),
    parallel_line_plane a α ∧
    plane_intersection α β = b ∧
    ¬ parallel_lines a b :=
  sorry

end NUMINAMATH_CALUDE_statement_b_not_always_true_l959_95994


namespace NUMINAMATH_CALUDE_pyramid_layers_l959_95932

/-- Represents a pyramid with layers of sandstone blocks -/
structure Pyramid where
  total_blocks : ℕ
  layer_ratio : ℕ
  top_layer_blocks : ℕ

/-- Calculates the number of layers in a pyramid -/
def num_layers (p : Pyramid) : ℕ :=
  sorry

/-- Theorem stating that a pyramid with 40 blocks, 3:1 layer ratio, and single top block has 4 layers -/
theorem pyramid_layers (p : Pyramid) 
  (h1 : p.total_blocks = 40)
  (h2 : p.layer_ratio = 3)
  (h3 : p.top_layer_blocks = 1) :
  num_layers p = 4 := by
  sorry

end NUMINAMATH_CALUDE_pyramid_layers_l959_95932


namespace NUMINAMATH_CALUDE_pipe_cut_theorem_l959_95948

theorem pipe_cut_theorem (total_length : ℝ) (difference : ℝ) (shorter_length : ℝ) : 
  total_length = 120 →
  difference = 22 →
  total_length = shorter_length + (shorter_length + difference) →
  shorter_length = 49 := by
sorry

end NUMINAMATH_CALUDE_pipe_cut_theorem_l959_95948


namespace NUMINAMATH_CALUDE_purchase_payment_possible_l959_95972

theorem purchase_payment_possible :
  ∃ (x y : ℕ), x ≤ 15 ∧ y ≤ 15 ∧ 3 * x - 5 * y = 19 :=
sorry

end NUMINAMATH_CALUDE_purchase_payment_possible_l959_95972


namespace NUMINAMATH_CALUDE_business_subscription_problem_l959_95922

/-- Proves that given the conditions of the business subscription problem, 
    the total amount subscribed is 50,000 Rs. -/
theorem business_subscription_problem 
  (a b c : ℕ) 
  (h1 : a = b + 4000)
  (h2 : b = c + 5000)
  (total_profit : ℕ)
  (h3 : total_profit = 36000)
  (a_profit : ℕ)
  (h4 : a_profit = 15120)
  (h5 : a_profit * (a + b + c) = a * total_profit) :
  a + b + c = 50000 := by
  sorry

end NUMINAMATH_CALUDE_business_subscription_problem_l959_95922


namespace NUMINAMATH_CALUDE_recurrence_equals_explicit_l959_95946

def recurrence_sequence (n : ℕ) : ℤ :=
  match n with
  | 0 => 5
  | 1 => 10
  | n + 2 => 5 * recurrence_sequence (n + 1) - 6 * recurrence_sequence n + 2 * (n + 2) - 3

def explicit_form (n : ℕ) : ℤ :=
  2^(n + 1) + 3^n + n + 2

theorem recurrence_equals_explicit : ∀ n : ℕ, recurrence_sequence n = explicit_form n :=
  sorry

end NUMINAMATH_CALUDE_recurrence_equals_explicit_l959_95946


namespace NUMINAMATH_CALUDE_wall_bricks_count_l959_95926

/-- Represents the number of bricks in the wall -/
def total_bricks : ℕ := 720

/-- Time taken by the first bricklayer to complete the wall alone (in hours) -/
def time_worker1 : ℕ := 12

/-- Time taken by the second bricklayer to complete the wall alone (in hours) -/
def time_worker2 : ℕ := 15

/-- Productivity decrease when working together (in bricks per hour) -/
def productivity_decrease : ℕ := 12

/-- Time taken when both workers work together (in hours) -/
def time_together : ℕ := 6

/-- Theorem stating that the number of bricks in the wall is 720 -/
theorem wall_bricks_count :
  (total_bricks / time_worker1 + total_bricks / time_worker2 - productivity_decrease) * time_together = total_bricks := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l959_95926


namespace NUMINAMATH_CALUDE_theater_line_permutations_l959_95930

theorem theater_line_permutations : Nat.factorial 8 = 40320 := by
  sorry

end NUMINAMATH_CALUDE_theater_line_permutations_l959_95930


namespace NUMINAMATH_CALUDE_survey_methods_correct_l959_95956

/-- Represents a sampling method --/
inductive SamplingMethod
  | SimpleRandom
  | Systematic
  | Stratified

/-- Represents a school survey --/
structure SchoolSurvey where
  totalStudents : Nat
  method1 : SamplingMethod
  method2 : SamplingMethod

/-- Defines the specific survey conducted by the school --/
def surveyConducted : SchoolSurvey := {
  totalStudents := 240,
  method1 := SamplingMethod.SimpleRandom,
  method2 := SamplingMethod.Systematic
}

/-- Theorem stating that the survey methods are correctly identified --/
theorem survey_methods_correct : 
  surveyConducted.method1 = SamplingMethod.SimpleRandom ∧
  surveyConducted.method2 = SamplingMethod.Systematic :=
by sorry

end NUMINAMATH_CALUDE_survey_methods_correct_l959_95956


namespace NUMINAMATH_CALUDE_megan_seashells_l959_95934

def current_seashells : ℕ := 19
def needed_seashells : ℕ := 6
def target_seashells : ℕ := 25

theorem megan_seashells : current_seashells + needed_seashells = target_seashells := by
  sorry

end NUMINAMATH_CALUDE_megan_seashells_l959_95934


namespace NUMINAMATH_CALUDE_system_solution_l959_95999

theorem system_solution (x y z t : ℤ) :
  (3 * x - 2 * y + 4 * z + 2 * t = 19) →
  (5 * x + 6 * y - 2 * z + 3 * t = 23) →
  (x = 16 * z - 18 * y - 11 ∧ t = 28 * y - 26 * z + 26) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l959_95999


namespace NUMINAMATH_CALUDE_resale_value_drops_below_target_in_four_years_l959_95971

def initial_price : ℝ := 625000
def first_year_depreciation : ℝ := 0.20
def subsequent_depreciation : ℝ := 0.08
def target_value : ℝ := 400000

def resale_value (n : ℕ) : ℝ :=
  if n = 0 then initial_price
  else if n = 1 then initial_price * (1 - first_year_depreciation)
  else initial_price * (1 - first_year_depreciation) * (1 - subsequent_depreciation) ^ (n - 1)

theorem resale_value_drops_below_target_in_four_years :
  resale_value 4 < target_value ∧ ∀ k : ℕ, k < 4 → resale_value k ≥ target_value :=
sorry

end NUMINAMATH_CALUDE_resale_value_drops_below_target_in_four_years_l959_95971


namespace NUMINAMATH_CALUDE_class_A_student_count_l959_95910

theorem class_A_student_count :
  ∀ (girls boys : ℕ),
    girls = 25 →
    girls = boys + 3 →
    girls + boys = 47 :=
by sorry

end NUMINAMATH_CALUDE_class_A_student_count_l959_95910


namespace NUMINAMATH_CALUDE_point_transformation_l959_95950

def rotate90CounterClockwise (x y cx cy : ℝ) : ℝ × ℝ :=
  (cx - (y - cy), cy + (x - cx))

def reflectAboutNegativeDiagonal (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (a b : ℝ) :
  let (x₁, y₁) := rotate90CounterClockwise a b 1 5
  let (x₂, y₂) := reflectAboutNegativeDiagonal x₁ y₁
  (x₂ = -6 ∧ y₂ = 3) → b - a = -5 := by sorry

end NUMINAMATH_CALUDE_point_transformation_l959_95950


namespace NUMINAMATH_CALUDE_direction_vector_c_value_l959_95919

/-- Given a line passing through two points and a direction vector, prove the value of c. -/
theorem direction_vector_c_value (p1 p2 : ℝ × ℝ) (h : p1 = (-3, 1) ∧ p2 = (0, 4)) :
  let v : ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2)
  v.1 = 3 → v = (3, 3) :=
by sorry

end NUMINAMATH_CALUDE_direction_vector_c_value_l959_95919


namespace NUMINAMATH_CALUDE_binomial_700_700_l959_95912

theorem binomial_700_700 : Nat.choose 700 700 = 1 := by
  sorry

end NUMINAMATH_CALUDE_binomial_700_700_l959_95912


namespace NUMINAMATH_CALUDE_astronaut_distribution_l959_95931

/-- The number of ways to distribute n astronauts among k distinct modules,
    with each module containing at least min and at most max astronauts. -/
def distribute_astronauts (n k min max : ℕ) : ℕ :=
  sorry

/-- The main theorem stating that there are 450 ways to distribute
    6 astronauts among 3 distinct modules, with each module containing
    at least 1 and at most 3 astronauts. -/
theorem astronaut_distribution :
  distribute_astronauts 6 3 1 3 = 450 :=
sorry

end NUMINAMATH_CALUDE_astronaut_distribution_l959_95931


namespace NUMINAMATH_CALUDE_jemma_grasshopper_count_l959_95918

/-- The number of grasshoppers Jemma saw on her African daisy plant -/
def grasshoppers_on_plant : ℕ := 7

/-- The number of dozens of baby grasshoppers Jemma found under the plant -/
def dozens_of_baby_grasshoppers : ℕ := 2

/-- The number of grasshoppers in a dozen -/
def grasshoppers_per_dozen : ℕ := 12

/-- The total number of grasshoppers Jemma found -/
def total_grasshoppers : ℕ := grasshoppers_on_plant + dozens_of_baby_grasshoppers * grasshoppers_per_dozen

theorem jemma_grasshopper_count : total_grasshoppers = 31 := by
  sorry

end NUMINAMATH_CALUDE_jemma_grasshopper_count_l959_95918


namespace NUMINAMATH_CALUDE_bernoulli_inequality_l959_95907

theorem bernoulli_inequality (n : ℕ+) (x : ℝ) (h : x > -1) :
  (1 + x)^(n : ℝ) ≥ 1 + n * x :=
by sorry

end NUMINAMATH_CALUDE_bernoulli_inequality_l959_95907


namespace NUMINAMATH_CALUDE_oatmeal_cookie_baggies_l959_95933

theorem oatmeal_cookie_baggies 
  (total_cookies : ℝ) 
  (chocolate_chip_cookies : ℝ) 
  (cookies_per_bag : ℝ) 
  (h1 : total_cookies = 41.0) 
  (h2 : chocolate_chip_cookies = 13.0) 
  (h3 : cookies_per_bag = 9.0) :
  ⌊(total_cookies - chocolate_chip_cookies) / cookies_per_bag⌋ = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_oatmeal_cookie_baggies_l959_95933


namespace NUMINAMATH_CALUDE_runner_position_l959_95954

/-- Represents a point on a circular track -/
inductive QuarterPoint
  | A
  | B
  | C
  | D

/-- Represents a circular track -/
structure CircularTrack where
  circumference : ℝ
  start_point : QuarterPoint

/-- Calculates the ending point after running a certain distance on a circular track -/
def end_point (track : CircularTrack) (distance : ℝ) : QuarterPoint :=
  sorry

theorem runner_position (track : CircularTrack) (distance : ℝ) :
  track.circumference = 45 →
  track.start_point = QuarterPoint.A →
  distance = 5400 →
  end_point track distance = QuarterPoint.A :=
  sorry

end NUMINAMATH_CALUDE_runner_position_l959_95954


namespace NUMINAMATH_CALUDE_puppies_sold_l959_95981

theorem puppies_sold (initial_puppies cages puppies_per_cage : ℕ) :
  initial_puppies = 78 →
  puppies_per_cage = 8 →
  cages = 6 →
  initial_puppies - (cages * puppies_per_cage) = 30 :=
by sorry

end NUMINAMATH_CALUDE_puppies_sold_l959_95981


namespace NUMINAMATH_CALUDE_spa_nail_polish_l959_95906

/-- The number of girls who went to the spa -/
def num_girls : ℕ := 8

/-- The number of fingers on each girl's hands -/
def fingers_per_girl : ℕ := 10

/-- The number of toes on each girl's feet -/
def toes_per_girl : ℕ := 10

/-- The total number of digits polished at the spa -/
def total_digits_polished : ℕ := num_girls * (fingers_per_girl + toes_per_girl)

theorem spa_nail_polish :
  total_digits_polished = 160 := by sorry

end NUMINAMATH_CALUDE_spa_nail_polish_l959_95906


namespace NUMINAMATH_CALUDE_units_digit_17_to_17_l959_95996

theorem units_digit_17_to_17 : (17^17 : ℕ) % 10 = 7 := by sorry

end NUMINAMATH_CALUDE_units_digit_17_to_17_l959_95996


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l959_95963

/-- Given a positive real number r, prove that if the line x - y = r is tangent to the circle x^2 + y^2 = r, then r = 2 -/
theorem tangent_line_to_circle (r : ℝ) (hr : r > 0) : 
  (∀ x y : ℝ, x - y = r → x^2 + y^2 ≤ r) ∧ 
  (∃ x y : ℝ, x - y = r ∧ x^2 + y^2 = r) → 
  r = 2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l959_95963


namespace NUMINAMATH_CALUDE_enclosed_area_is_3600_l959_95997

/-- The equation defining the graph -/
def graph_equation (x y : ℝ) : Prop :=
  |x - 120| + |y| = |x/3|

/-- The set of points satisfying the graph equation -/
def graph_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | graph_equation p.1 p.2}

/-- The area enclosed by the graph -/
noncomputable def enclosed_area : ℝ := sorry

/-- Theorem stating that the enclosed area is 3600 -/
theorem enclosed_area_is_3600 : enclosed_area = 3600 := by sorry

end NUMINAMATH_CALUDE_enclosed_area_is_3600_l959_95997


namespace NUMINAMATH_CALUDE_madeline_class_hours_l959_95941

/-- Calculates the number of hours Madeline spends in class per week -/
def hours_in_class (hours_per_day : ℕ) (days_per_week : ℕ) 
  (homework_hours_per_day : ℕ) (sleep_hours_per_day : ℕ) 
  (work_hours_per_week : ℕ) (leftover_hours : ℕ) : ℕ :=
  hours_per_day * days_per_week - 
  (homework_hours_per_day * days_per_week + 
   sleep_hours_per_day * days_per_week + 
   work_hours_per_week + 
   leftover_hours)

theorem madeline_class_hours : 
  hours_in_class 24 7 4 8 20 46 = 18 := by
  sorry

end NUMINAMATH_CALUDE_madeline_class_hours_l959_95941


namespace NUMINAMATH_CALUDE_largest_angle_in_triangle_l959_95913

theorem largest_angle_in_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 7/5 of a right angle
  a + b = (7 / 5) * 90 →
  -- One angle is 45° larger than the other
  b = a + 45 →
  -- All angles are positive
  a > 0 ∧ b > 0 ∧ c > 0 →
  -- Sum of all angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 85.5°
  max a (max b c) = 85.5 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_triangle_l959_95913


namespace NUMINAMATH_CALUDE_line_slope_l959_95998

/-- The slope of the line (x/2) + (y/3) = 2 is -3/2 -/
theorem line_slope (x y : ℝ) :
  (x / 2 + y / 3 = 2) → (∃ b : ℝ, y = (-3/2) * x + b) := by
  sorry

end NUMINAMATH_CALUDE_line_slope_l959_95998


namespace NUMINAMATH_CALUDE_place_mat_length_l959_95993

theorem place_mat_length (r : ℝ) (n : ℕ) (y : ℝ) : 
  r = 5 → n = 8 → y = 2 * r * Real.sin (π / (2 * n)) → y = 5 * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

#check place_mat_length

end NUMINAMATH_CALUDE_place_mat_length_l959_95993


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l959_95939

theorem complex_arithmetic_equality : (90 + 5) * (12 / (180 / (3^2))) = 57 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l959_95939


namespace NUMINAMATH_CALUDE_min_money_required_l959_95924

/-- Represents the number of candies of each type -/
structure CandyCounts where
  apple : ℕ
  orange : ℕ
  strawberry : ℕ
  grape : ℕ

/-- Represents the vending machine with given conditions -/
def VendingMachine (c : CandyCounts) : Prop :=
  c.apple = 2 * c.orange ∧
  c.strawberry = 2 * c.grape ∧
  c.apple = 2 * c.strawberry ∧
  c.apple + c.orange + c.strawberry + c.grape = 90

/-- The cost of a single candy -/
def candy_cost : ℚ := 1/10

/-- The minimum number of candies to buy -/
def min_candies_to_buy (c : CandyCounts) : ℕ :=
  min c.grape 10 + 3 + 3 + 3

/-- The theorem to prove -/
theorem min_money_required (c : CandyCounts) (h : VendingMachine c) :
  (min_candies_to_buy c : ℚ) * candy_cost = 19/10 := by
  sorry


end NUMINAMATH_CALUDE_min_money_required_l959_95924


namespace NUMINAMATH_CALUDE_player_matches_l959_95929

/-- The number of matches played by a player -/
def num_matches : ℕ := sorry

/-- The current average runs per match -/
def current_average : ℚ := 32

/-- The runs to be scored in the next match -/
def next_match_runs : ℕ := 98

/-- The increase in average after the next match -/
def average_increase : ℚ := 6

theorem player_matches :
  (current_average * num_matches + next_match_runs) / (num_matches + 1) = current_average + average_increase →
  num_matches = 10 := by sorry

end NUMINAMATH_CALUDE_player_matches_l959_95929
