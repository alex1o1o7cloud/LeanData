import Mathlib

namespace systematic_sampling_proof_l2189_218976

/-- The total number of technical personnel --/
def total_personnel : ℕ := 37

/-- The number of attendees --/
def n : ℕ := 18

/-- Proves that n satisfies the conditions of the systematic sampling problem --/
theorem systematic_sampling_proof :
  (total_personnel - 1) % n = 0 ∧ 
  (total_personnel - 3) % (n - 1) = 0 := by
  sorry

end systematic_sampling_proof_l2189_218976


namespace hexagon_angle_measure_l2189_218918

-- Define the hexagon and its angles
structure ConvexHexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

-- Define the theorem
theorem hexagon_angle_measure (h : ConvexHexagon) :
  h.A = h.B ∧ h.B = h.C ∧  -- A, B, C are congruent
  h.D = h.E ∧ h.E = h.F ∧  -- D, E, F are congruent
  h.A + 20 = h.D ∧         -- A is 20° less than D
  h.A + h.B + h.C + h.D + h.E + h.F = 720 -- Sum of angles in a hexagon
  →
  h.D = 130 := by sorry

end hexagon_angle_measure_l2189_218918


namespace asterisk_replacement_l2189_218933

theorem asterisk_replacement : (42 / 21) * (42 / 84) = 1 := by
  sorry

end asterisk_replacement_l2189_218933


namespace expression_value_l2189_218986

theorem expression_value : 
  let x : ℕ := 2
  2 + 2 * (2 * 2) = 10 := by sorry

end expression_value_l2189_218986


namespace sine_inequality_holds_only_at_zero_l2189_218957

theorem sine_inequality_holds_only_at_zero (y : Real) :
  (y ∈ Set.Icc 0 (Real.pi / 2)) →
  (∀ x ∈ Set.Icc 0 (Real.pi / 2), Real.sin (x + y) ≤ Real.sin x + Real.sin y) ↔
  y = 0 := by
sorry

end sine_inequality_holds_only_at_zero_l2189_218957


namespace smallest_number_with_remainders_l2189_218913

theorem smallest_number_with_remainders : ∃ N : ℕ, 
  N > 0 ∧ 
  N % 13 = 2 ∧ 
  N % 15 = 4 ∧ 
  (∀ M : ℕ, M > 0 → M % 13 = 2 → M % 15 = 4 → N ≤ M) ∧
  N = 184 := by
sorry

end smallest_number_with_remainders_l2189_218913


namespace rectangle_area_l2189_218999

/-- A rectangle with specific properties -/
structure Rectangle where
  width : ℝ
  length : ℝ
  length_eq : length = 3 * width + 15
  perimeter_eq : 2 * (width + length) = 800

/-- The area of a rectangle with the given properties is 29234.375 square feet -/
theorem rectangle_area (rect : Rectangle) : rect.width * rect.length = 29234.375 := by
  sorry

end rectangle_area_l2189_218999


namespace assignment_methods_eq_twelve_l2189_218983

/-- The number of ways to assign doctors and nurses to schools. -/
def assignment_methods (num_doctors num_nurses num_schools : ℕ) : ℕ :=
  (num_doctors.choose 1) * (num_nurses.choose 2)

/-- Theorem stating the number of assignment methods for the given problem. -/
theorem assignment_methods_eq_twelve :
  assignment_methods 2 4 2 = 12 :=
by sorry

end assignment_methods_eq_twelve_l2189_218983


namespace brother_payment_l2189_218944

/-- Margaux's daily earnings from her money lending company -/
structure DailyEarnings where
  friend : ℝ
  brother : ℝ
  cousin : ℝ

/-- The total earnings after a given number of days -/
def total_earnings (e : DailyEarnings) (days : ℝ) : ℝ :=
  (e.friend + e.brother + e.cousin) * days

/-- Theorem stating that Margaux's brother pays $8 per day -/
theorem brother_payment (e : DailyEarnings) :
  e.friend = 5 ∧ e.cousin = 4 ∧ total_earnings e 7 = 119 → e.brother = 8 := by
  sorry

end brother_payment_l2189_218944


namespace last_two_digits_of_2007_power_20077_l2189_218980

theorem last_two_digits_of_2007_power_20077 : 2007^20077 % 100 = 7 := by
  sorry

end last_two_digits_of_2007_power_20077_l2189_218980


namespace branch_fraction_l2189_218936

theorem branch_fraction (L : ℝ) (F : ℝ) : 
  L = 3 →  -- The branch length is 3 meters
  0 < F → F < 1 →  -- F is a proper fraction
  L - (L / 3 + F * L) = 0.6 * L →  -- Remaining length after removal
  F = 1 / 15 := by
sorry

end branch_fraction_l2189_218936


namespace product_of_sum_and_sum_of_cubes_l2189_218901

theorem product_of_sum_and_sum_of_cubes (c d : ℝ) 
  (h1 : c + d = 10) 
  (h2 : c^3 + d^3 = 370) : 
  c * d = 21 := by
  sorry

end product_of_sum_and_sum_of_cubes_l2189_218901


namespace r_value_when_n_is_3_l2189_218931

theorem r_value_when_n_is_3 :
  let n : ℕ := 3
  let t : ℕ := 3^n + n^2
  let r : ℕ := 4^t - t^2
  r = 2^72 - 1296 := by
  sorry

end r_value_when_n_is_3_l2189_218931


namespace apple_sale_profit_percentage_l2189_218974

/-- Represents the shopkeeper's apple selling scenario -/
structure AppleSale where
  total_apples : ℝ
  sell_percent_1 : ℝ
  profit_percent_1 : ℝ
  sell_percent_2 : ℝ
  profit_percent_2 : ℝ
  sell_percent_3 : ℝ
  profit_percent_3 : ℝ
  unsold_percent : ℝ
  additional_expenses : ℝ

/-- Calculates the effective profit percentage for the given apple sale scenario -/
def effectiveProfitPercentage (sale : AppleSale) : ℝ :=
  sorry

/-- Theorem stating the effective profit percentage for the given scenario -/
theorem apple_sale_profit_percentage :
  let sale : AppleSale := {
    total_apples := 120,
    sell_percent_1 := 0.4,
    profit_percent_1 := 0.25,
    sell_percent_2 := 0.3,
    profit_percent_2 := 0.35,
    sell_percent_3 := 0.2,
    profit_percent_3 := 0.2,
    unsold_percent := 0.1,
    additional_expenses := 20
  }
  ∃ (ε : ℝ), ε > 0 ∧ abs (effectiveProfitPercentage sale + 2.407) < ε :=
sorry

end apple_sale_profit_percentage_l2189_218974


namespace employee_age_when_hired_l2189_218925

/-- Represents the retirement eligibility rule where age plus years of employment must equal 70 -/
def retirement_rule (age : ℕ) (years_employed : ℕ) : Prop :=
  age + years_employed = 70

/-- Represents the fact that the employee worked for 19 years before retirement eligibility -/
def years_worked : ℕ := 19

/-- Proves that the employee's age when hired was 51 -/
theorem employee_age_when_hired :
  ∃ (age_when_hired : ℕ),
    retirement_rule (age_when_hired + years_worked) years_worked ∧
    age_when_hired = 51 := by
  sorry

end employee_age_when_hired_l2189_218925


namespace sallys_class_size_l2189_218928

theorem sallys_class_size (school_money : ℕ) (book_cost : ℕ) (out_of_pocket : ℕ) :
  school_money = 320 →
  book_cost = 12 →
  out_of_pocket = 40 →
  (school_money + out_of_pocket) / book_cost = 30 := by
sorry

end sallys_class_size_l2189_218928


namespace junk_items_after_transactions_l2189_218900

/-- Represents the composition of items in the attic -/
structure AtticComposition where
  useful : Rat
  valuable : Rat
  junk : Rat

/-- Represents the number of items in each category -/
structure AtticItems where
  useful : ℕ
  valuable : ℕ
  junk : ℕ

/-- The theorem to prove -/
theorem junk_items_after_transactions 
  (initial_composition : AtticComposition)
  (initial_items : AtticItems)
  (items_removed : AtticItems)
  (final_composition : AtticComposition)
  (final_useful_items : ℕ) :
  (initial_composition.useful = 1/5) →
  (initial_composition.valuable = 1/10) →
  (initial_composition.junk = 7/10) →
  (items_removed.useful = 4) →
  (items_removed.valuable = 3) →
  (final_composition.useful = 1/4) →
  (final_composition.valuable = 3/20) →
  (final_composition.junk = 3/5) →
  (final_useful_items = 20) →
  ∃ (final_items : AtticItems), final_items.junk = 48 := by
  sorry

end junk_items_after_transactions_l2189_218900


namespace area_of_triangle_MOI_l2189_218934

/-- Triangle ABC with given side lengths -/
structure Triangle where
  AB : ℝ
  AC : ℝ
  BC : ℝ

/-- The given triangle -/
def givenTriangle : Triangle := { AB := 15, AC := 8, BC := 17 }

/-- Point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Circumcenter of the triangle -/
noncomputable def O : Point := sorry

/-- Incenter of the triangle -/
noncomputable def I : Point := sorry

/-- Center of the circle tangent to AC, BC, and the circumcircle -/
noncomputable def M : Point := sorry

/-- Area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ := sorry

/-- The main theorem -/
theorem area_of_triangle_MOI (t : Triangle) (h : t = givenTriangle) : 
  triangleArea O I M = 3.4 := by sorry

end area_of_triangle_MOI_l2189_218934


namespace cubic_sum_theorem_l2189_218967

theorem cubic_sum_theorem (a b c : ℝ) (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) 
  (h_eq : (a^3 + 9) / a = (b^3 + 9) / b ∧ (b^3 + 9) / b = (c^3 + 9) / c) : 
  a^3 + b^3 + c^3 = -27 := by
  sorry

end cubic_sum_theorem_l2189_218967


namespace parallelogram_count_l2189_218917

/-- The number of parallelograms formed by lines passing through each grid point in a triangle -/
def f (n : ℕ) : ℕ := 3 * ((n + 2) * (n + 1) * n * (n - 1)) / 24

/-- Theorem stating that f(n) correctly calculates the number of parallelograms -/
theorem parallelogram_count (n : ℕ) : 
  f n = 3 * ((n + 2) * (n + 1) * n * (n - 1)) / 24 := by sorry

end parallelogram_count_l2189_218917


namespace johns_age_l2189_218977

theorem johns_age (j d : ℕ) 
  (h1 : j + 28 = d)
  (h2 : j + d = 76)
  (h3 : d = 2 * (j - 4)) :
  j = 24 := by
  sorry

end johns_age_l2189_218977


namespace inequality_solution_implies_a_value_l2189_218941

theorem inequality_solution_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, 2 * x - a ≤ -1 ↔ x ≤ 1) → a = 3 :=
by
  sorry

end inequality_solution_implies_a_value_l2189_218941


namespace common_chord_length_l2189_218961

/-- The length of the common chord of two circles -/
theorem common_chord_length (c1 c2 : ℝ × ℝ → Prop) : 
  (∀ x y, c1 (x, y) ↔ x^2 + y^2 = 4) →
  (∀ x y, c2 (x, y) ↔ x^2 + y^2 - 2*y - 6 = 0) →
  ∃ l : ℝ, l = 2 * Real.sqrt 3 ∧ 
    ∀ x y, (c1 (x, y) ∧ c2 (x, y)) → 
      (x^2 + y^2 = 4 ∧ y = -1) ∨ 
      (x^2 + y^2 - 2*y - 6 = 0 ∧ y = -1) :=
by sorry

end common_chord_length_l2189_218961


namespace correct_amount_paid_l2189_218975

/-- The amount paid by Mr. Doré given the costs of items and change received -/
def amount_paid (pants_cost shirt_cost tie_cost change : ℕ) : ℕ :=
  pants_cost + shirt_cost + tie_cost + change

/-- Theorem stating that the amount paid is correct given the problem conditions -/
theorem correct_amount_paid :
  amount_paid 140 43 15 2 = 200 := by
  sorry

end correct_amount_paid_l2189_218975


namespace pauls_garage_sale_l2189_218912

/-- The number of books Paul sold in the garage sale -/
def books_sold (initial : ℕ) (given_away : ℕ) (remaining : ℕ) : ℕ :=
  initial - given_away - remaining

/-- Proof that Paul sold 27 books in the garage sale -/
theorem pauls_garage_sale : books_sold 134 39 68 = 27 := by
  sorry

end pauls_garage_sale_l2189_218912


namespace tangent_circles_count_l2189_218991

/-- A circle in a plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Two circles are tangent if the distance between their centers equals the sum or difference of their radii -/
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  ((x2 - x1)^2 + (y2 - y1)^2) = (c1.radius + c2.radius)^2 ∨
  ((x2 - x1)^2 + (y2 - y1)^2) = (c1.radius - c2.radius)^2

/-- A circle is tangent to two other circles if it's tangent to both of them -/
def is_tangent_to_both (c : Circle) (c1 c2 : Circle) : Prop :=
  are_tangent c c1 ∧ are_tangent c c2

/-- The main theorem: there are exactly 6 circles of radius 5 tangent to two tangent circles of radius 2 -/
theorem tangent_circles_count (c1 c2 : Circle) 
  (h1 : c1.radius = 2)
  (h2 : c2.radius = 2)
  (h3 : are_tangent c1 c2) :
  ∃! (s : Finset Circle), (∀ c ∈ s, c.radius = 5 ∧ is_tangent_to_both c c1 c2) ∧ s.card = 6 :=
sorry

end tangent_circles_count_l2189_218991


namespace basketball_surface_area_l2189_218998

/-- The surface area of a sphere with circumference 30 inches is 900/π square inches -/
theorem basketball_surface_area :
  let circumference : ℝ := 30
  let radius : ℝ := circumference / (2 * Real.pi)
  let surface_area : ℝ := 4 * Real.pi * radius ^ 2
  surface_area = 900 / Real.pi := by
  sorry

end basketball_surface_area_l2189_218998


namespace one_third_of_twelve_x_plus_five_l2189_218914

theorem one_third_of_twelve_x_plus_five (x : ℚ) : (1 / 3) * (12 * x + 5) = 4 * x + 5 / 3 := by
  sorry

end one_third_of_twelve_x_plus_five_l2189_218914


namespace train_journey_theorem_l2189_218907

/-- Represents the properties of a train journey -/
structure TrainJourney where
  reducedSpeed : ℝ  -- Speed at which the train actually travels
  speedFraction : ℝ  -- Fraction of the train's own speed at which it travels
  time : ℝ  -- Time taken for the journey
  distance : ℝ  -- Distance traveled

/-- The problem setup -/
def trainProblem (trainA trainB : TrainJourney) : Prop :=
  trainA.speedFraction = 2/3 ∧
  trainA.time = 12 ∧
  trainA.distance = 360 ∧
  trainB.speedFraction = 1/2 ∧
  trainB.time = 8 ∧
  trainB.distance = trainA.distance

/-- The theorem to be proved -/
theorem train_journey_theorem (trainA trainB : TrainJourney) 
  (h : trainProblem trainA trainB) : 
  (trainA.time * (1 - trainA.speedFraction) + trainB.time * (1 - trainB.speedFraction) = 8) ∧
  (trainB.distance = 360) := by
  sorry

end train_journey_theorem_l2189_218907


namespace height_difference_after_growth_spurt_l2189_218942

theorem height_difference_after_growth_spurt 
  (uncle_height : ℝ) 
  (james_initial_ratio : ℝ) 
  (sarah_initial_ratio : ℝ) 
  (james_growth : ℝ) 
  (sarah_growth : ℝ) 
  (h1 : uncle_height = 72) 
  (h2 : james_initial_ratio = 2/3) 
  (h3 : sarah_initial_ratio = 3/4) 
  (h4 : james_growth = 10) 
  (h5 : sarah_growth = 12) : 
  (james_initial_ratio * uncle_height + james_growth + 
   sarah_initial_ratio * james_initial_ratio * uncle_height + sarah_growth) - uncle_height = 34 := by
  sorry

end height_difference_after_growth_spurt_l2189_218942


namespace share_difference_l2189_218962

/-- Represents the distribution of money among five people -/
structure MoneyDistribution where
  faruk : ℕ
  vasim : ℕ
  ranjith : ℕ
  priya : ℕ
  elina : ℕ

/-- Theorem stating the difference in shares based on the given conditions -/
theorem share_difference (d : MoneyDistribution) :
  d.faruk = 3 * 600 ∧
  d.vasim = 5 * 600 ∧
  d.ranjith = 9 * 600 ∧
  d.priya = 7 * 600 ∧
  d.elina = 11 * 600 ∧
  d.vasim = 3000 →
  (d.faruk + d.ranjith + d.elina) - (d.vasim + d.priya) = 6600 := by
  sorry


end share_difference_l2189_218962


namespace arithmetic_sequence_sum_l2189_218945

/-- An arithmetic sequence with integer common ratio -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  q : ℤ
  seq_def : ∀ n, a (n + 1) = a n + q

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  seq.a 1 * n + seq.q * (n * (n - 1) / 2)

theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h1 : seq.a 2 - seq.a 3 = -2)
  (h2 : seq.a 1 + seq.a 3 = 10/3) :
  sum_n seq 4 = 40/3 := by
  sorry

end arithmetic_sequence_sum_l2189_218945


namespace computer_pricing_l2189_218932

/-- Represents the prices of computer components -/
structure Prices where
  basic_computer : ℝ
  printer : ℝ
  regular_monitor : ℝ

/-- Proves the correct prices given the problem conditions -/
theorem computer_pricing (prices : Prices) 
  (total_basic : prices.basic_computer + prices.printer + prices.regular_monitor = 3000)
  (enhanced_printer_ratio : prices.printer = (1/4) * (prices.basic_computer + 500 + prices.printer + prices.regular_monitor + 300)) :
  prices.printer = 950 ∧ prices.basic_computer + prices.regular_monitor = 2050 := by
  sorry


end computer_pricing_l2189_218932


namespace scientific_notation_32000000_l2189_218903

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- Converts a real number to scientific notation -/
noncomputable def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_32000000 :
  toScientificNotation 32000000 = ScientificNotation.mk 3.2 7 (by norm_num) :=
sorry

end scientific_notation_32000000_l2189_218903


namespace sqrt_abs_equation_l2189_218966

theorem sqrt_abs_equation (a b : ℤ) :
  (Real.sqrt (a - 2023 : ℝ) + |b + 2023| - 1 = 0) → (a + b = 1 ∨ a + b = -1) := by
  sorry

end sqrt_abs_equation_l2189_218966


namespace simplified_expression_l2189_218910

theorem simplified_expression (x : ℝ) (h1 : x ≠ 3) (h2 : x ≠ -2) :
  (3 * x^2 - 2 * x - 5) / ((x - 3) * (x + 2)) - (5 * x - 6) / ((x - 3) * (x + 2)) =
  3 * (x - (7 + Real.sqrt 37) / 6) * (x - (7 - Real.sqrt 37) / 6) / ((x - 3) * (x + 2)) := by
  sorry

end simplified_expression_l2189_218910


namespace jake_not_drop_coffee_l2189_218984

/-- The probability of Jake tripping over his dog on any given morning. -/
def prob_trip : ℝ := 0.40

/-- The probability of Jake dropping his coffee when he trips over his dog. -/
def prob_drop_when_trip : ℝ := 0.25

/-- The probability of Jake missing a step on the stairs on any given morning. -/
def prob_miss_step : ℝ := 0.30

/-- The probability of Jake spilling his coffee when he misses a step. -/
def prob_spill_when_miss : ℝ := 0.20

/-- Theorem: The probability of Jake not dropping his coffee on any given morning is 0.846. -/
theorem jake_not_drop_coffee :
  (1 - prob_trip * prob_drop_when_trip) * (1 - prob_miss_step * prob_spill_when_miss) = 0.846 := by
  sorry

end jake_not_drop_coffee_l2189_218984


namespace cat_count_l2189_218939

/-- The number of cats that can jump -/
def jump : ℕ := 45

/-- The number of cats that can fetch -/
def fetch : ℕ := 25

/-- The number of cats that can meow -/
def meow : ℕ := 40

/-- The number of cats that can jump and fetch -/
def jump_fetch : ℕ := 15

/-- The number of cats that can fetch and meow -/
def fetch_meow : ℕ := 20

/-- The number of cats that can jump and meow -/
def jump_meow : ℕ := 23

/-- The number of cats that can do all three tricks -/
def all_three : ℕ := 10

/-- The number of cats that can do no tricks -/
def no_tricks : ℕ := 5

/-- The total number of cats in the training center -/
def total_cats : ℕ := 67

theorem cat_count : 
  jump + fetch + meow - jump_fetch - fetch_meow - jump_meow + all_three + no_tricks = total_cats :=
by sorry

end cat_count_l2189_218939


namespace diophantine_equation_unique_solution_l2189_218997

theorem diophantine_equation_unique_solution :
  ∀ a b c : ℤ, 5 * a^2 + 9 * b^2 = 13 * c^2 → a = 0 ∧ b = 0 ∧ c = 0 :=
by sorry

end diophantine_equation_unique_solution_l2189_218997


namespace transaction_gain_per_year_l2189_218956

/-- Calculate simple interest -/
def simpleInterest (principal rate time : ℚ) : ℚ :=
  principal * rate * time / 100

theorem transaction_gain_per_year
  (principal : ℚ)
  (borrowRate lendRate : ℚ)
  (time : ℚ)
  (h1 : principal = 8000)
  (h2 : borrowRate = 4)
  (h3 : lendRate = 6)
  (h4 : time = 2) :
  (simpleInterest principal lendRate time - simpleInterest principal borrowRate time) / time = 160 := by
  sorry

end transaction_gain_per_year_l2189_218956


namespace trig_expression_equals_one_l2189_218969

theorem trig_expression_equals_one : 
  (Real.cos (36 * π / 180) * Real.sin (24 * π / 180) + 
   Real.sin (144 * π / 180) * Real.sin (84 * π / 180)) / 
  (Real.cos (44 * π / 180) * Real.sin (16 * π / 180) + 
   Real.sin (136 * π / 180) * Real.sin (76 * π / 180)) = 1 := by
  sorry

end trig_expression_equals_one_l2189_218969


namespace sphere_radius_is_zero_l2189_218948

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- The configuration of points and lines in the problem -/
structure Configuration where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  m : Line3D
  n : Line3D
  a : ℝ
  b : ℝ
  θ : ℝ

/-- Checks if two points are distinct -/
def are_distinct (p q : Point3D) : Prop :=
  p ≠ q

/-- Checks if a line is perpendicular to another line -/
def is_perpendicular (l1 l2 : Line3D) : Prop :=
  sorry

/-- Checks if a point is on a line -/
def point_on_line (p : Point3D) (l : Line3D) : Prop :=
  sorry

/-- Calculates the distance between two points -/
def distance (p q : Point3D) : ℝ :=
  sorry

/-- Calculates the angle between two lines -/
def angle_between_lines (l1 l2 : Line3D) : ℝ :=
  sorry

/-- Calculates the radius of a sphere passing through four points -/
def sphere_radius (p1 p2 p3 p4 : Point3D) : ℝ :=
  sorry

/-- The main theorem stating that the radius of the sphere is zero -/
theorem sphere_radius_is_zero (config : Configuration) :
  are_distinct config.A config.B ∧
  is_perpendicular config.m (Line3D.mk config.A config.B) ∧
  is_perpendicular config.n (Line3D.mk config.A config.B) ∧
  point_on_line config.C config.m ∧
  are_distinct config.A config.C ∧
  point_on_line config.D config.n ∧
  are_distinct config.B config.D ∧
  distance config.A config.B = config.a ∧
  distance config.C config.D = config.b ∧
  angle_between_lines config.m config.n = config.θ
  →
  sphere_radius config.A config.B config.C config.D = 0 :=
sorry

end sphere_radius_is_zero_l2189_218948


namespace limit_one_minus_x_squared_over_sin_pi_x_l2189_218978

/-- The limit of (1 - x^2) / sin(πx) as x approaches 1 is 2/π -/
theorem limit_one_minus_x_squared_over_sin_pi_x (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ x : ℝ, 0 < |x - 1| ∧ |x - 1| < δ →
    |(1 - x^2) / Real.sin (π * x) - 2/π| < ε :=
sorry

end limit_one_minus_x_squared_over_sin_pi_x_l2189_218978


namespace express_y_in_terms_of_x_l2189_218993

theorem express_y_in_terms_of_x (p : ℝ) (x y : ℝ) 
  (hx : x = 2 + 2^p) (hy : y = 1 + 2^(-p)) : 
  y = (x - 1) / (x - 2) := by
  sorry

end express_y_in_terms_of_x_l2189_218993


namespace contrapositive_equivalence_l2189_218996

theorem contrapositive_equivalence (a b : ℤ) :
  ((Odd a ∧ Odd b) → Even (a + b)) ↔
  (¬Even (a + b) → ¬(Odd a ∧ Odd b)) :=
by sorry

end contrapositive_equivalence_l2189_218996


namespace isosceles_triangle_perimeter_l2189_218920

/-- An isosceles triangle with side lengths 4 and 8 has a perimeter of 20 -/
theorem isosceles_triangle_perimeter : ∀ (a b c : ℝ), 
  a = 8 ∧ b = 8 ∧ c = 4 → -- Two sides are 8, one side is 4
  a + b > c ∧ b + c > a ∧ c + a > b → -- Triangle inequality
  a + b + c = 20 := by
  sorry


end isosceles_triangle_perimeter_l2189_218920


namespace unique_solution_l2189_218959

theorem unique_solution : ∃! (x : ℕ), 
  x > 0 ∧ 
  let n := x^2 + 4*x + 23
  let d := 3*x + 7
  n = d*x + 2 := by
  sorry

end unique_solution_l2189_218959


namespace triathlete_swimming_speed_l2189_218979

/-- Calculates the swimming speed of a triathlete given the conditions of the problem -/
theorem triathlete_swimming_speed
  (distance : ℝ)
  (running_speed : ℝ)
  (average_rate : ℝ)
  (h1 : distance = 2)
  (h2 : running_speed = 10)
  (h3 : average_rate = 0.1111111111111111)
  : ∃ (swimming_speed : ℝ), swimming_speed = 5 := by
  sorry

end triathlete_swimming_speed_l2189_218979


namespace amount_lent_to_C_l2189_218971

/-- Amount lent to B in rupees -/
def amount_B : ℝ := 5000

/-- Duration of loan to B in years -/
def duration_B : ℝ := 2

/-- Duration of loan to C in years -/
def duration_C : ℝ := 4

/-- Annual interest rate as a decimal -/
def interest_rate : ℝ := 0.08

/-- Total interest received from both B and C in rupees -/
def total_interest : ℝ := 1760

/-- Calculates simple interest -/
def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time

theorem amount_lent_to_C : ∃ (amount_C : ℝ),
  amount_C = 3000 ∧
  simple_interest amount_B interest_rate duration_B +
  simple_interest amount_C interest_rate duration_C = total_interest :=
sorry

end amount_lent_to_C_l2189_218971


namespace specific_triangle_angle_l2189_218926

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem stating the properties of the specific triangle -/
theorem specific_triangle_angle (t : Triangle) 
  (h1 : t.a = 2)
  (h2 : t.b = 2)
  (h3 : t.A = 45) :
  t.B = 67.5 := by
  sorry


end specific_triangle_angle_l2189_218926


namespace expression_simplification_l2189_218987

theorem expression_simplification (x y : ℝ) :
  (-2 * x^2 * y) * (-3 * x * y)^2 / (3 * x * y^2) = -6 * x^3 * y := by sorry

end expression_simplification_l2189_218987


namespace sin_cos_equality_l2189_218916

theorem sin_cos_equality (θ : Real) (h : Real.sin θ * Real.cos θ = 1/2) :
  Real.sin θ - Real.cos θ = 0 := by
  sorry

end sin_cos_equality_l2189_218916


namespace log_sawing_time_l2189_218924

theorem log_sawing_time (log_length : ℕ) (section_length : ℕ) (saw_time : ℕ) 
  (h1 : log_length = 10)
  (h2 : section_length = 1)
  (h3 : saw_time = 3) :
  (log_length - 1) * saw_time = 27 :=
by sorry

end log_sawing_time_l2189_218924


namespace equation_solution_l2189_218927

theorem equation_solution :
  ∃ x : ℚ, 5 * (x - 9) = 6 * (3 - 3 * x) + 9 ∧ x = 72 / 23 := by
  sorry

end equation_solution_l2189_218927


namespace ellipse_from_hyperbola_l2189_218951

/-- Given a hyperbola with equation x²/4 - y²/12 = -1, prove that the equation of the ellipse
    whose foci are the vertices of the hyperbola and whose vertices are the foci of the hyperbola
    is x²/4 + y²/16 = 1 -/
theorem ellipse_from_hyperbola (x y : ℝ) :
  (x^2 / 4 - y^2 / 12 = -1) →
  ∃ (a b : ℝ), (a > 0 ∧ b > 0 ∧ a > b) ∧
  ((x^2 / a^2 + y^2 / b^2 = 1) ↔ (x^2 / 4 + y^2 / 16 = 1)) :=
by sorry

end ellipse_from_hyperbola_l2189_218951


namespace units_digit_sum_powers_l2189_218930

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- The result of raising a number to a power, considering only the units digit -/
def powerUnitsDigit (base : ℕ) (exp : ℕ) : ℕ :=
  (unitsDigit base ^ exp) % 10

theorem units_digit_sum_powers : unitsDigit (powerUnitsDigit 53 107 + powerUnitsDigit 97 59) = 0 := by
  sorry

end units_digit_sum_powers_l2189_218930


namespace line_mb_equals_two_l2189_218990

/-- Given a line passing through points (0, -1) and (-1, 1) with equation y = mx + b, prove that mb = 2 -/
theorem line_mb_equals_two (m b : ℝ) : 
  (∀ x y : ℝ, y = m * x + b) →  -- Line equation
  ((-1 : ℝ) = m * 0 + b) →      -- Point (0, -1)
  (1 : ℝ) = m * (-1) + b →      -- Point (-1, 1)
  m * b = 2 := by
sorry

end line_mb_equals_two_l2189_218990


namespace range_of_a_l2189_218995

theorem range_of_a (a : ℝ) : 
  (a - 3*3 < 4*a*3 + 2) → 
  (a - 3*0 < 4*a*0 + 2) → 
  (-1 < a ∧ a < 2) := by
sorry

end range_of_a_l2189_218995


namespace product_and_remainder_problem_l2189_218992

theorem product_and_remainder_problem :
  ∃ (a b c d : ℤ),
    d = a * b * c ∧
    1 < a ∧ a < b ∧ b < c ∧
    233 % d = 79 ∧
    a + c = 13 := by
  sorry

end product_and_remainder_problem_l2189_218992


namespace erased_number_theorem_l2189_218981

theorem erased_number_theorem :
  ∀ x : ℕ, x ∈ Finset.range 21 \ {0} →
  (∃ y ∈ Finset.range 21 \ {0, x},
    19 * y = (Finset.sum (Finset.range 21 \ {0, x}) id)) ↔
  x = 1 ∨ x = 20 :=
by sorry

end erased_number_theorem_l2189_218981


namespace cut_cube_theorem_l2189_218973

/-- Represents a cube cut into smaller cubes -/
structure CutCube where
  /-- The number of small cubes along each edge of the large cube -/
  edge_count : ℕ
  /-- All faces of the large cube are painted -/
  all_faces_painted : Bool
  /-- The number of small cubes with three faces colored -/
  three_face_colored_count : ℕ

/-- Theorem: If a cube is cut so that 8 small cubes have three faces colored, 
    then the total number of small cubes is 8 -/
theorem cut_cube_theorem (c : CutCube) 
  (h1 : c.all_faces_painted = true) 
  (h2 : c.three_face_colored_count = 8) : 
  c.edge_count ^ 3 = 8 := by
  sorry

#check cut_cube_theorem

end cut_cube_theorem_l2189_218973


namespace parabola_directrix_l2189_218954

/-- The equation of the directrix of the parabola y = x^2 is 4y + 1 = 0 -/
theorem parabola_directrix : 
  ∀ (x y : ℝ), y = x^2 → (∃ (k : ℝ), k * y + 1 = 0 ∧ k = 4) :=
by sorry

end parabola_directrix_l2189_218954


namespace min_buses_required_l2189_218955

theorem min_buses_required (total_students : ℕ) (bus_capacity : ℕ) (h1 : total_students = 625) (h2 : bus_capacity = 47) :
  ∃ (n : ℕ), n * bus_capacity ≥ total_students ∧ ∀ (m : ℕ), m * bus_capacity ≥ total_students → n ≤ m :=
by
  sorry

end min_buses_required_l2189_218955


namespace milk_water_ratio_after_addition_l2189_218994

def initial_volume : ℚ := 45
def initial_milk_ratio : ℚ := 4
def initial_water_ratio : ℚ := 1
def added_water : ℚ := 23

theorem milk_water_ratio_after_addition :
  let initial_milk := initial_volume * initial_milk_ratio / (initial_milk_ratio + initial_water_ratio)
  let initial_water := initial_volume * initial_water_ratio / (initial_milk_ratio + initial_water_ratio)
  let final_water := initial_water + added_water
  (initial_milk : ℚ) / final_water = 9 / 8 := by sorry

end milk_water_ratio_after_addition_l2189_218994


namespace expected_sides_is_four_l2189_218923

/-- The number of cuts made in one hour -/
def k : ℕ := 3600

/-- The initial number of sides of the rectangular sheet -/
def initial_sides : ℕ := 4

/-- The total number of sides after k cuts -/
def total_sides (k : ℕ) : ℕ := initial_sides + 4 * k

/-- The total number of polygons after k cuts -/
def total_polygons (k : ℕ) : ℕ := k + 1

/-- The expected number of sides of a randomly picked polygon after k cuts -/
def expected_sides (k : ℕ) : ℚ :=
  (total_sides k : ℚ) / (total_polygons k : ℚ)

theorem expected_sides_is_four :
  expected_sides k = 4 := by sorry

end expected_sides_is_four_l2189_218923


namespace max_expectation_exp_l2189_218982

-- Define a random variable X
variable (X : ℝ → ℝ)

-- Define probability measure P
variable (P : Set ℝ → ℝ)

-- Define expectation E
variable (E : (ℝ → ℝ) → ℝ)

-- Define variance D
variable (D : (ℝ → ℝ) → ℝ)

-- Constants σ and b
variable (σ b : ℝ)

-- Conditions
variable (h1 : P {x | |X x| ≤ b} = 1)
variable (h2 : E X = 0)
variable (h3 : D X = σ^2)
variable (h4 : σ > 0)
variable (h5 : b > 0)

-- Theorem statement
theorem max_expectation_exp :
  (∀ Y : ℝ → ℝ, P {x | |Y x| ≤ b} = 1 → E Y = 0 → D Y = σ^2 →
    E (fun x => Real.exp (Y x)) ≤ (Real.exp b * σ^2 + Real.exp (-σ^2 / b) * b^2) / (σ^2 + b^2)) ∧
  (E (fun x => Real.exp (X x)) = (Real.exp b * σ^2 + Real.exp (-σ^2 / b) * b^2) / (σ^2 + b^2)) :=
sorry

end max_expectation_exp_l2189_218982


namespace sqrt_fraction_equality_l2189_218937

theorem sqrt_fraction_equality : Real.sqrt (9/4) - Real.sqrt (4/9) + 1/6 = 1 := by
  sorry

end sqrt_fraction_equality_l2189_218937


namespace equation_solution_l2189_218963

theorem equation_solution : 
  ∀ t : ℝ, t ≠ 6 ∧ t ≠ -4 →
  ((t^2 - 3*t - 18) / (t - 6) = 2 / (t + 4)) ↔ (t = -2 ∨ t = -5) :=
by sorry

end equation_solution_l2189_218963


namespace smallest_bob_number_l2189_218989

def alice_number : ℕ := 30

def is_valid_bob_number (n : ℕ) : Prop :=
  ∀ p : ℕ, Nat.Prime p → p ∣ alice_number → p ∣ n

theorem smallest_bob_number :
  ∃ (bob_number : ℕ), is_valid_bob_number bob_number ∧
    ∀ (m : ℕ), is_valid_bob_number m → bob_number ≤ m ∧ bob_number = 15 :=
by sorry

end smallest_bob_number_l2189_218989


namespace jessica_withdrawal_l2189_218985

/-- Proves that given the conditions of Jessica's bank transactions, 
    the amount she initially withdrew was $200. -/
theorem jessica_withdrawal (B : ℚ) : 
  (3/5 * B + 1/5 * (3/5 * B) = 360) → 
  (2/5 * B = 200) := by
  sorry

#eval (2/5 : ℚ) * 500 -- Optional: to verify the result

end jessica_withdrawal_l2189_218985


namespace dropped_student_score_l2189_218968

theorem dropped_student_score
  (initial_students : ℕ)
  (initial_average : ℚ)
  (remaining_students : ℕ)
  (remaining_average : ℚ)
  (h1 : initial_students = 16)
  (h2 : initial_average = 62.5)
  (h3 : remaining_students = 15)
  (h4 : remaining_average = 62)
  (h5 : initial_students = remaining_students + 1) :
  (initial_students : ℚ) * initial_average - (remaining_students : ℚ) * remaining_average = 70 := by
  sorry

#check dropped_student_score

end dropped_student_score_l2189_218968


namespace fibonacci_150_mod_9_l2189_218964

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci (n + 1) + fibonacci n

theorem fibonacci_150_mod_9 : fibonacci 150 % 9 = 8 := by
  sorry

end fibonacci_150_mod_9_l2189_218964


namespace prob_A_and_B_l2189_218909

/-- The probability of event A occurring -/
def prob_A : ℝ := 0.75

/-- The probability of event B occurring -/
def prob_B : ℝ := 0.60

/-- The theorem stating that the probability of both A and B occurring is 0.45 -/
theorem prob_A_and_B : prob_A * prob_B = 0.45 := by
  sorry

end prob_A_and_B_l2189_218909


namespace power_inequality_l2189_218946

theorem power_inequality (x t : ℝ) (hx : x ≥ 3) :
  (0 < t ∧ t < 1 → x^t - (x-1)^t < (x-2)^t - (x-3)^t) ∧
  (t > 1 → x^t - (x-1)^t > (x-2)^t - (x-3)^t) := by
  sorry

end power_inequality_l2189_218946


namespace line_passes_through_circle_center_l2189_218953

/-- The line equation: x - y + 1 = 0 -/
def line_equation (x y : ℝ) : Prop := x - y + 1 = 0

/-- The circle equation: (x + 1)^2 + y^2 = 1 -/
def circle_equation (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 1

/-- The center of the circle -/
def circle_center : ℝ × ℝ := (-1, 0)

theorem line_passes_through_circle_center :
  line_equation (circle_center.1) (circle_center.2) := by
  sorry

end line_passes_through_circle_center_l2189_218953


namespace book_sale_revenue_book_sale_revenue_proof_l2189_218960

theorem book_sale_revenue : ℕ → ℕ → ℕ → Prop :=
  fun total_books sold_price remaining_books =>
    (2 * total_books = 3 * remaining_books) →
    (total_books - remaining_books) * sold_price = 288

-- Proof
theorem book_sale_revenue_proof :
  book_sale_revenue 108 4 36 := by
  sorry

end book_sale_revenue_book_sale_revenue_proof_l2189_218960


namespace derivative_at_pi_third_l2189_218950

theorem derivative_at_pi_third (f : ℝ → ℝ) (h : f = λ x => Real.cos x + Real.sqrt 3 * Real.sin x) :
  deriv f (π / 3) = 0 := by sorry

end derivative_at_pi_third_l2189_218950


namespace simplify_expression_l2189_218943

theorem simplify_expression (z : ℝ) : (3 - 5 * z^2) - (5 + 7 * z^2) = -2 - 12 * z^2 := by
  sorry

end simplify_expression_l2189_218943


namespace exercise_book_count_l2189_218915

/-- Given a ratio of pencils to exercise books and the number of pencils,
    calculate the number of exercise books -/
def calculate_exercise_books (pencil_ratio : ℕ) (book_ratio : ℕ) (num_pencils : ℕ) : ℕ :=
  (num_pencils / pencil_ratio) * book_ratio

/-- Theorem: In a shop with 140 pencils and a pencil to exercise book ratio of 14:3,
    there are 30 exercise books -/
theorem exercise_book_count :
  calculate_exercise_books 14 3 140 = 30 := by
  sorry

end exercise_book_count_l2189_218915


namespace new_person_weight_l2189_218911

/-- Given a group of 7 people, if replacing one person weighing 95 kg with a new person
    increases the average weight by 12.3 kg, then the weight of the new person is 181.1 kg. -/
theorem new_person_weight (group_size : ℕ) (weight_increase : ℝ) (old_weight : ℝ) :
  group_size = 7 →
  weight_increase = 12.3 →
  old_weight = 95 →
  (group_size : ℝ) * weight_increase + old_weight = 181.1 := by
  sorry

end new_person_weight_l2189_218911


namespace money_distribution_l2189_218902

theorem money_distribution (A B C : ℝ) 
  (total : A + B + C = 450)
  (ac_sum : A + C = 200)
  (bc_sum : B + C = 350) :
  C = 100 := by
sorry

end money_distribution_l2189_218902


namespace tomatoes_left_theorem_l2189_218906

/-- Calculates the number of tomatoes left after processing -/
def tomatoes_left (plants : ℕ) (tomatoes_per_plant : ℕ) : ℕ :=
  let total := plants * tomatoes_per_plant
  let dried := total / 2
  let remaining := total - dried
  let marinara := remaining / 3
  remaining - marinara

/-- Theorem: Given 18 plants with 7 tomatoes each, after processing, 42 tomatoes are left -/
theorem tomatoes_left_theorem : tomatoes_left 18 7 = 42 := by
  sorry

end tomatoes_left_theorem_l2189_218906


namespace proposition_values_l2189_218921

theorem proposition_values (p q : Prop) 
  (h1 : ¬(p ∧ q)) 
  (h2 : ¬p) : 
  ¬p ∧ (q ∨ ¬q) :=
sorry

end proposition_values_l2189_218921


namespace problem_solution_l2189_218904

/-- A geometric sequence with the given property -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ), ∀ n, a (n + 1) = r * a n

/-- The property of the sequence given in the problem -/
def sequence_property (a : ℕ → ℝ) : Prop :=
  ∀ n, a n + a (n + 1) = 3 * (1/2)^n

theorem problem_solution (a : ℕ → ℝ) 
  (h1 : geometric_sequence a) 
  (h2 : sequence_property a) : 
  a 5 = 1/16 := by
  sorry

end problem_solution_l2189_218904


namespace monic_quartic_polynomial_value_l2189_218972

theorem monic_quartic_polynomial_value (p : ℝ → ℝ) :
  (∃ a b c : ℝ, ∀ x, p x = x^4 + a*x^3 + b*x^2 + c*x + (3 - 1 - a - b - c)) →
  p 1 = 3 →
  p 2 = 7 →
  p 3 = 13 →
  p 4 = 21 →
  p 5 = 51 := by
sorry

end monic_quartic_polynomial_value_l2189_218972


namespace solution_implies_expression_value_l2189_218940

theorem solution_implies_expression_value
  (a b : ℝ)
  (h : a * (-2) - b = 1) :
  4 * a + 2 * b + 7 = 5 :=
by sorry

end solution_implies_expression_value_l2189_218940


namespace rmb_notes_problem_l2189_218970

theorem rmb_notes_problem (x y z : ℕ) : 
  x + y + z = 33 →
  x + 5 * y + 10 * z = 187 →
  y = x - 5 →
  (x = 12 ∧ y = 7 ∧ z = 14) :=
by sorry

end rmb_notes_problem_l2189_218970


namespace angle_in_fourth_quadrant_l2189_218908

theorem angle_in_fourth_quadrant (θ : Real) 
  (h1 : Real.sin θ < Real.cos θ) 
  (h2 : Real.sin θ * Real.cos θ < 0) : 
  0 < θ ∧ θ < Real.pi / 2 ∧ Real.sin θ < 0 ∧ Real.cos θ > 0 := by
  sorry

end angle_in_fourth_quadrant_l2189_218908


namespace problem_statement_l2189_218919

theorem problem_statement (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a + 2*b = 1) :
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + 2*y = 1 ∧ a^2 + b^2 ≥ x^2 + y^2) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → a^2 + b^2 ≤ x^2 + y^2) ∧
  (a^2 + b^2 = 1/5) ∧
  (a*b ≤ 1/8) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → x*y ≤ 1/8) ∧
  (1/a + 1/b ≥ 3 + 2*Real.sqrt 2) ∧
  (∀ (x y : ℝ), x > 0 → y > 0 → x + 2*y = 1 → 1/x + 1/y ≥ 3 + 2*Real.sqrt 2) := by
sorry

end problem_statement_l2189_218919


namespace math_book_cost_l2189_218947

theorem math_book_cost (total_books : ℕ) (history_book_cost : ℕ) (total_price : ℕ) (math_books : ℕ) :
  total_books = 80 →
  history_book_cost = 5 →
  total_price = 373 →
  math_books = 27 →
  ∃ (math_book_cost : ℕ),
    math_book_cost * math_books + history_book_cost * (total_books - math_books) = total_price ∧
    math_book_cost = 4 :=
by sorry

end math_book_cost_l2189_218947


namespace ellipse_foci_distance_l2189_218905

/-- The distance between the foci of an ellipse given by the equation
    9x^2 - 36x + 4y^2 + 16y + 16 = 0 is 2√5 -/
theorem ellipse_foci_distance : 
  ∃ (a b c : ℝ), 
    (∀ x y : ℝ, 9*x^2 - 36*x + 4*y^2 + 16*y + 16 = 0 ↔ 
      (x - 2)^2 / a^2 + (y + 2)^2 / b^2 = 1) ∧
    a^2 - b^2 = c^2 ∧
    2 * c = 2 * Real.sqrt 5 := by
  sorry

end ellipse_foci_distance_l2189_218905


namespace pentagonal_pillar_faces_l2189_218922

/-- Represents a pentagonal pillar -/
structure PentagonalPillar :=
  (rectangular_faces : Nat)
  (pentagonal_faces : Nat)

/-- The total number of faces of a pentagonal pillar -/
def total_faces (p : PentagonalPillar) : Nat :=
  p.rectangular_faces + p.pentagonal_faces

/-- Theorem stating that a pentagonal pillar has 7 faces -/
theorem pentagonal_pillar_faces :
  ∀ (p : PentagonalPillar),
  p.rectangular_faces = 5 ∧ p.pentagonal_faces = 2 →
  total_faces p = 7 := by
  sorry

#check pentagonal_pillar_faces

end pentagonal_pillar_faces_l2189_218922


namespace chrysanthemum_arrangement_count_l2189_218965

/-- Represents the number of pots for each color of chrysanthemums -/
structure ChrysanthemumPots where
  yellow : Nat
  white : Nat
  red : Nat

/-- Calculates the number of arrangements for chrysanthemum pots -/
def countArrangements (pots : ChrysanthemumPots) : Nat :=
  sorry

/-- Theorem stating the number of arrangements for the given conditions -/
theorem chrysanthemum_arrangement_count :
  let pots : ChrysanthemumPots := { yellow := 2, white := 2, red := 1 }
  countArrangements pots = 24 := by
  sorry

end chrysanthemum_arrangement_count_l2189_218965


namespace triangle_isosceles_from_side_condition_l2189_218988

theorem triangle_isosceles_from_side_condition (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_condition : a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b) = 0) :
  a = b ∨ b = c ∨ c = a := by
sorry


end triangle_isosceles_from_side_condition_l2189_218988


namespace arithmetic_sequence_sum_l2189_218952

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 2 + a 10 = 16 → a 4 + a 6 + a 8 = 24 := by
  sorry

end arithmetic_sequence_sum_l2189_218952


namespace min_value_trig_expression_l2189_218938

theorem min_value_trig_expression (θ φ : ℝ) :
  (3 * Real.cos θ + 4 * Real.sin φ - 10)^2 + (3 * Real.sin θ + 4 * Real.cos φ - 20)^2 ≥ 549 - 140 * Real.sqrt 5 := by
  sorry

end min_value_trig_expression_l2189_218938


namespace point_outside_region_l2189_218958

def planar_region (x y : ℝ) : Prop := 2 * x + 3 * y < 6

theorem point_outside_region :
  ¬(planar_region 0 2) ∧
  (planar_region 0 0) ∧
  (planar_region 1 1) ∧
  (planar_region 2 0) :=
sorry

end point_outside_region_l2189_218958


namespace video_game_points_l2189_218935

/-- 
Given a video game level with the following conditions:
- There are 6 enemies in total
- Each defeated enemy gives 3 points
- 2 enemies are not defeated

Prove that the total points earned is 12.
-/
theorem video_game_points (total_enemies : ℕ) (points_per_enemy : ℕ) (undefeated_enemies : ℕ) :
  total_enemies = 6 →
  points_per_enemy = 3 →
  undefeated_enemies = 2 →
  (total_enemies - undefeated_enemies) * points_per_enemy = 12 :=
by sorry

end video_game_points_l2189_218935


namespace zeros_properties_l2189_218929

noncomputable def f (x : ℝ) : ℝ := (2 * x / (x - 2))^2 - 3^x

noncomputable def g (x : ℝ) : ℝ := 2 * (Real.log x / Real.log 3) - 4 / (x - 2) - 2

theorem zeros_properties (x₁ x₂ : ℝ) 
  (h₁ : x₁ > 2) (h₂ : x₂ > 2) 
  (hf₁ : f x₁ = 0) (hf₂ : f x₂ = 0) 
  (hg₁ : g x₁ = 0) (hg₂ : g x₂ = 0) :
  x₂ > 3 ∧ 2*x₁ + 2*x₂ = x₁*x₂ ∧ x₁*x₂ > 16 :=
by sorry

end zeros_properties_l2189_218929


namespace find_number_l2189_218949

theorem find_number : ∃! x : ℝ, 0.6 * ((x / 1.2) - 22.5) + 10.5 = 30 := by
  sorry

end find_number_l2189_218949
