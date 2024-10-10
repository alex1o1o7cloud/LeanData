import Mathlib

namespace root_sum_fraction_l183_18347

/-- Given a, b, c are roots of x^3 - 20x^2 + 22, prove bc/a^2 + ac/b^2 + ab/c^2 = -40 -/
theorem root_sum_fraction (a b c : ℝ) : 
  (a^3 - 20*a^2 + 22 = 0) → 
  (b^3 - 20*b^2 + 22 = 0) → 
  (c^3 - 20*c^2 + 22 = 0) → 
  (b*c/a^2 + a*c/b^2 + a*b/c^2 = -40) := by
  sorry

end root_sum_fraction_l183_18347


namespace no_real_solution_l183_18352

theorem no_real_solution : ¬∃ (x y : ℝ), x ≠ 0 ∧ y ≠ 0 ∧ x^2 = 1 + 1/y^2 ∧ y^2 = 1 + 1/x^2 := by
  sorry

end no_real_solution_l183_18352


namespace smallest_sum_of_squares_l183_18313

theorem smallest_sum_of_squares (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  ∃ x y z : ℕ, x > 0 ∧ y > 0 ∧ z > 0 ∧ 
    a + b = x^2 ∧ b + c = y^2 ∧ c + a = z^2 →
  55 ≤ a + b + c :=
by sorry

end smallest_sum_of_squares_l183_18313


namespace remainder_zero_l183_18355

theorem remainder_zero : (88144 * 88145 + 88146 + 88147 + 88148 + 88149 + 88150) % 5 = 0 := by
  sorry

end remainder_zero_l183_18355


namespace first_page_drawings_count_l183_18320

/-- The number of drawings on the first page of an art book. -/
def first_page_drawings : ℕ := 5

/-- The increase in the number of drawings on each subsequent page. -/
def drawing_increase : ℕ := 5

/-- The total number of pages considered. -/
def total_pages : ℕ := 5

/-- The total number of drawings on the first five pages. -/
def total_drawings : ℕ := 75

/-- Theorem stating that the number of drawings on the first page is 5,
    given the conditions of the problem. -/
theorem first_page_drawings_count :
  (first_page_drawings +
   (first_page_drawings + drawing_increase) +
   (first_page_drawings + 2 * drawing_increase) +
   (first_page_drawings + 3 * drawing_increase) +
   (first_page_drawings + 4 * drawing_increase)) = total_drawings :=
by sorry

end first_page_drawings_count_l183_18320


namespace second_polygon_sides_l183_18383

/-- Given two regular polygons with equal perimeters, where one has 50 sides
    and its side length is three times the other's, prove that the number of
    sides of the second polygon is 150. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) : 
  s > 0 →  -- Ensure positive side length
  50 * (3 * s) = n * s →  -- Equal perimeters
  n = 150 := by
  sorry


end second_polygon_sides_l183_18383


namespace set_B_elements_l183_18333

def A : Set Int := {-2, 0, 1, 3}

def B : Set Int := {x | -x ∈ A ∧ (1 - x) ∉ A}

theorem set_B_elements : B = {-3, -1, 2} := by sorry

end set_B_elements_l183_18333


namespace fraction_and_sum_problem_l183_18322

theorem fraction_and_sum_problem :
  (5 : ℚ) / 40 = 0.125 ∧ 0.125 + 0.375 = 0.500 := by
  sorry

end fraction_and_sum_problem_l183_18322


namespace difference_between_decimal_and_fraction_l183_18336

theorem difference_between_decimal_and_fraction : 0.127 - (1 / 8 : ℚ) = 0.002 := by
  sorry

end difference_between_decimal_and_fraction_l183_18336


namespace angle_A_is_30_degrees_l183_18377

/-- In a triangle ABC, given that the side opposite to angle B is twice the length of the side opposite to angle A, and angle B is 60° greater than angle A, prove that angle A measures 30°. -/
theorem angle_A_is_30_degrees (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧  -- Angles are positive
  A + B + C = π ∧  -- Sum of angles in a triangle
  b = 2 * a ∧  -- Given condition
  B = A + π / 3 →  -- B = A + 60° (in radians)
  A = π / 6 :=  -- A = 30° (in radians)
by sorry

end angle_A_is_30_degrees_l183_18377


namespace investment_interest_theorem_l183_18365

/-- Calculates the total interest paid in an 18-month investment contract with specified conditions -/
def totalInterest (initialInvestment : ℝ) : ℝ :=
  let interestRate1 := 0.02
  let interestRate2 := 0.03
  let interestRate3 := 0.04
  
  let interest1 := initialInvestment * interestRate1
  let newInvestment1 := initialInvestment + interest1
  
  let interest2 := newInvestment1 * interestRate2
  let newInvestment2 := newInvestment1 + interest2
  
  let interest3 := newInvestment2 * interestRate3
  
  interest1 + interest2 + interest3

/-- Theorem stating that the total interest paid in the given investment scenario is $926.24 -/
theorem investment_interest_theorem :
  totalInterest 10000 = 926.24 := by sorry

end investment_interest_theorem_l183_18365


namespace geometric_sequence_property_l183_18339

/-- A geometric sequence -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_property (a : ℕ → ℝ) :
  geometric_sequence a →
  a 2 * a 3 * a 7 = 8 →
  a 4 = 2 := by
sorry

end geometric_sequence_property_l183_18339


namespace triangle_problem_l183_18372

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  (2 * Real.sqrt 3 / 3 * b * c * Real.sin A = b^2 + c^2 - a^2) →
  (c = 5) →
  (Real.cos B = 1 / 7) →
  (A = π / 3 ∧ b = 8) := by
  sorry

end triangle_problem_l183_18372


namespace reasoning_is_inductive_l183_18326

/-- Represents different types of reasoning methods -/
inductive ReasoningMethod
  | Analogical
  | Inductive
  | Deductive
  | Analytical

/-- Represents a metal -/
structure Metal where
  name : String

/-- Represents the property of conducting electricity -/
def conductsElectricity (m : Metal) : Prop := sorry

/-- The set of metals mentioned in the statement -/
def mentionedMetals : List Metal := [
  { name := "Gold" },
  { name := "Silver" },
  { name := "Copper" },
  { name := "Iron" }
]

/-- The statement that all mentioned metals conduct electricity -/
def allMentionedMetalsConduct : Prop :=
  ∀ m ∈ mentionedMetals, conductsElectricity m

/-- The conclusion that all metals conduct electricity -/
def allMetalsConduct : Prop :=
  ∀ m : Metal, conductsElectricity m

/-- The reasoning method used in the given statement -/
def reasoningMethodUsed : ReasoningMethod := sorry

/-- Theorem stating that the reasoning method used is inductive -/
theorem reasoning_is_inductive :
  allMentionedMetalsConduct →
  reasoningMethodUsed = ReasoningMethod.Inductive :=
sorry

end reasoning_is_inductive_l183_18326


namespace inequality_proof_l183_18364

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (b + c - a)^2 / (a^2 + (b + c)^2) +
  (c + a - b)^2 / (b^2 + (c + a)^2) +
  (a + b - c)^2 / (c^2 + (a + b)^2) ≥ 3/5 := by
  sorry

end inequality_proof_l183_18364


namespace complex_sum_and_reciprocal_l183_18317

theorem complex_sum_and_reciprocal (z : ℂ) : z = 1 + I → z + 2 / z = 2 := by
  sorry

end complex_sum_and_reciprocal_l183_18317


namespace geometric_sequence_product_l183_18363

/-- Given a geometric sequence {a_n}, prove that a_4 * a_7 = -6 
    when a_1 and a_10 are roots of x^2 - x - 6 = 0 -/
theorem geometric_sequence_product (a : ℕ → ℝ) : 
  (∀ n : ℕ, ∃ r : ℝ, a (n + 1) = a n * r) →  -- geometric sequence condition
  (a 1)^2 - (a 1) - 6 = 0 →  -- a_1 is a root of x^2 - x - 6 = 0
  (a 10)^2 - (a 10) - 6 = 0 →  -- a_10 is a root of x^2 - x - 6 = 0
  a 4 * a 7 = -6 := by
sorry

end geometric_sequence_product_l183_18363


namespace range_of_a_min_value_of_fraction_l183_18331

-- Define the function f
def f (a b x : ℝ) : ℝ := a * x^2 + 4 * x + b

-- Theorem 1
theorem range_of_a (a : ℝ) :
  (∀ x ∈ Set.Icc 1 2, f a 2 x ≥ 0) → a ≥ -5/2 :=
sorry

-- Theorem 2
theorem min_value_of_fraction (a b : ℝ) :
  a > b →
  (∀ x : ℝ, f a b x ≥ 0) →
  (∃ x₀ : ℝ, f a b x₀ = 0) →
  (a^2 + b^2) / (a - b) ≥ 4 * Real.sqrt 2 :=
sorry

end range_of_a_min_value_of_fraction_l183_18331


namespace domain_of_g_l183_18381

noncomputable def g (x : ℝ) : ℝ := 1 / ⌊x^2 - 8*x + 18⌋

theorem domain_of_g : Set.range g = Set.univ :=
sorry

end domain_of_g_l183_18381


namespace intersection_A_and_naturals_l183_18396

def A : Set ℝ := {x : ℝ | -1 < x ∧ x < 2}

theorem intersection_A_and_naturals :
  A ∩ Set.univ.image (Nat.cast : ℕ → ℝ) = {0, 1} := by sorry

end intersection_A_and_naturals_l183_18396


namespace equal_numbers_from_different_sequences_l183_18340

/-- Represents a sequence of consecutive natural numbers -/
def ConsecutiveSequence (start : ℕ) (length : ℕ) : List ℕ :=
  List.range length |>.map (· + start)

/-- Concatenates a list of natural numbers into a single number -/
def concatenateToNumber (list : List ℕ) : ℕ := sorry

theorem equal_numbers_from_different_sequences :
  ∃ (a b : ℕ) (orderA : List ℕ → List ℕ) (orderB : List ℕ → List ℕ),
    let seqA := ConsecutiveSequence a 20
    let seqB := ConsecutiveSequence b 21
    concatenateToNumber (orderA seqA) = concatenateToNumber (orderB seqB) := by
  sorry

end equal_numbers_from_different_sequences_l183_18340


namespace martha_black_butterflies_l183_18312

/-- The number of black butterflies in Martha's collection --/
def num_black_butterflies (total : ℕ) (blue : ℕ) (red : ℕ) : ℕ :=
  total - (blue + red)

/-- Proof that Martha has 34 black butterflies --/
theorem martha_black_butterflies :
  num_black_butterflies 56 12 10 = 34 := by
  sorry

end martha_black_butterflies_l183_18312


namespace shaded_area_calculation_l183_18309

theorem shaded_area_calculation (square_side : ℝ) (triangle1_base triangle1_height : ℝ) (triangle2_base triangle2_height : ℝ) :
  square_side = 40 →
  triangle1_base = 15 →
  triangle1_height = 20 →
  triangle2_base = 15 →
  triangle2_height = 10 →
  square_side * square_side - (0.5 * triangle1_base * triangle1_height + 0.5 * triangle2_base * triangle2_height) = 1375 := by
  sorry

end shaded_area_calculation_l183_18309


namespace number_equation_l183_18307

theorem number_equation (x : ℝ) : (40 / 100) * x = (10 / 100) * 70 → x = 17.5 := by
  sorry

end number_equation_l183_18307


namespace jenn_bike_purchase_l183_18351

/-- Calculates the amount left over after buying a bike, given the number of jars of quarters,
    quarters per jar, and the cost of the bike. -/
def money_left_over (num_jars : ℕ) (quarters_per_jar : ℕ) (bike_cost : ℚ) : ℚ :=
  (num_jars * quarters_per_jar * (1/4 : ℚ)) - bike_cost

/-- Proves that given 5 jars of quarters with 160 quarters per jar, and a bike costing $180,
    the amount left over after buying the bike is $20. -/
theorem jenn_bike_purchase : money_left_over 5 160 180 = 20 := by
  sorry

end jenn_bike_purchase_l183_18351


namespace total_leaves_l183_18303

/-- The number of leaves Sabrina needs for her poultice --/
structure HerbLeaves where
  basil : ℕ
  sage : ℕ
  verbena : ℕ
  chamomile : ℕ
  lavender : ℕ

/-- The conditions for Sabrina's herb collection --/
def validHerbCollection (h : HerbLeaves) : Prop :=
  h.basil = 3 * h.sage ∧
  h.verbena = h.sage + 8 ∧
  h.chamomile = 2 * h.sage + 7 ∧
  h.lavender = (h.basil + h.chamomile + 1) / 2 ∧
  h.basil = 48

/-- The theorem stating the total number of leaves needed --/
theorem total_leaves (h : HerbLeaves) (hvalid : validHerbCollection h) :
  h.basil + h.sage + h.verbena + h.chamomile + h.lavender = 171 := by
  sorry

#check total_leaves

end total_leaves_l183_18303


namespace probability_point_between_C_and_E_l183_18346

/-- Given a line segment AB with points C, D, and E, where AB = 4AD = 8BC and E divides CD into two equal parts,
    the probability of a randomly selected point on AB falling between C and E is 5/16. -/
theorem probability_point_between_C_and_E (A B C D E : ℝ) : 
  A < C ∧ C < D ∧ D < E ∧ E < B →  -- Points are ordered on the line
  B - A = 4 * (D - A) →            -- AB = 4AD
  B - A = 8 * (C - B) →            -- AB = 8BC
  E - C = D - E →                  -- E divides CD into two equal parts
  (E - C) / (B - A) = 5 / 16 :=     -- Probability is 5/16
by sorry

end probability_point_between_C_and_E_l183_18346


namespace ellipse_segment_length_l183_18375

/-- The length of segment AB for a given ellipse -/
theorem ellipse_segment_length : 
  ∀ (x y : ℝ), 
  (x^2 / 25 + y^2 / 16 = 1) →  -- Ellipse equation
  (∃ (a b : ℝ), 
    (a^2 / 25 + b^2 / 16 = 1) ∧  -- Points A and B satisfy ellipse equation
    (a = 3) ∧  -- x-coordinate of right focus
    (b = 16/5 ∨ b = -16/5)) →  -- y-coordinates of intersection points
  (16/5 - (-16/5) = 32/5) :=  -- Length of segment AB
by
  sorry

end ellipse_segment_length_l183_18375


namespace triangle_congruence_problem_l183_18395

theorem triangle_congruence_problem (x y z : ℝ) : 
  (x + y + z = 3) →
  (z + 6 = 2*y - z) →
  (x + 8*z = y + 2) →
  x^2 + y^2 + z^2 = 21 :=
by
  sorry

end triangle_congruence_problem_l183_18395


namespace some_students_not_club_members_l183_18357

-- Define the universe
variable (U : Type)

-- Define predicates
variable (Student : U → Prop)
variable (Punctual : U → Prop)
variable (ClubMember : U → Prop)
variable (FraternityMember : U → Prop)

-- Define the conditions
variable (h1 : ∃ x, Student x ∧ ¬Punctual x)
variable (h2 : ∀ x, ClubMember x → Punctual x)
variable (h3 : ∀ x, FraternityMember x → ¬ClubMember x)

-- State the theorem
theorem some_students_not_club_members :
  ∃ x, Student x ∧ ¬ClubMember x :=
sorry

end some_students_not_club_members_l183_18357


namespace equation_equality_l183_18306

theorem equation_equality 
  (p q r x y z a b c : ℝ) 
  (h1 : p / x = q / y ∧ q / y = r / z) 
  (h2 : x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 1) :
  p^2 / a^2 + q^2 / b^2 + r^2 / c^2 = (p^2 + q^2 + r^2) / (x^2 + y^2 + z^2) := by
sorry

end equation_equality_l183_18306


namespace hex_numeric_count_and_sum_l183_18356

/-- Represents a hexadecimal digit --/
inductive HexDigit
| Numeric (n : Fin 10)
| Alpha (a : Fin 6)

/-- Converts a natural number to its hexadecimal representation --/
def toHex (n : ℕ) : List HexDigit :=
  sorry

/-- Checks if a hexadecimal representation contains only numeric digits --/
def isAllNumeric (hex : List HexDigit) : Bool :=
  sorry

/-- Counts numbers up to n (exclusive) with only numeric hexadecimal digits --/
def countNumericHex (n : ℕ) : ℕ :=
  sorry

/-- Sums the digits of a natural number --/
def sumDigits (n : ℕ) : ℕ :=
  sorry

theorem hex_numeric_count_and_sum :
  countNumericHex 512 = 200 ∧ sumDigits 200 = 2 := by
  sorry

end hex_numeric_count_and_sum_l183_18356


namespace rowing_distance_calculation_l183_18374

/-- Represents the problem of calculating the distance to a destination given rowing conditions. -/
theorem rowing_distance_calculation 
  (rowing_speed : ℝ) 
  (current_speed : ℝ) 
  (total_time : ℝ) 
  (h1 : rowing_speed = 10) 
  (h2 : current_speed = 2) 
  (h3 : total_time = 5) : 
  ∃ (distance : ℝ), distance = 24 ∧ 
    distance / (rowing_speed + current_speed) + 
    distance / (rowing_speed - current_speed) = total_time :=
by sorry

end rowing_distance_calculation_l183_18374


namespace complete_square_quadratic_l183_18390

theorem complete_square_quadratic (x : ℝ) : 
  ∃ (a b : ℝ), x^2 - 6*x + 7 = 0 ↔ (x + a)^2 = b ∧ b = 2 := by
sorry

end complete_square_quadratic_l183_18390


namespace large_marshmallows_count_l183_18329

/-- Represents the number of Rice Krispie Treats made -/
def rice_krispie_treats : ℕ := 5

/-- Represents the total number of marshmallows used -/
def total_marshmallows : ℕ := 18

/-- Represents the number of mini marshmallows used -/
def mini_marshmallows : ℕ := 10

/-- Represents the number of large marshmallows used -/
def large_marshmallows : ℕ := total_marshmallows - mini_marshmallows

theorem large_marshmallows_count : large_marshmallows = 8 := by
  sorry

end large_marshmallows_count_l183_18329


namespace martin_purchase_cost_l183_18367

/-- The cost of items at a store -/
structure StorePrices where
  pencil : ℕ
  notebook : ℕ
  eraser : ℕ

/-- The conditions of the problem -/
def store_conditions (prices : StorePrices) : Prop :=
  prices.notebook + prices.eraser = 85 ∧
  prices.pencil + prices.eraser = 45 ∧
  3 * prices.pencil + 3 * prices.notebook + 3 * prices.eraser = 315

/-- The theorem stating that Martin's purchase costs 80 cents -/
theorem martin_purchase_cost (prices : StorePrices) 
  (h : store_conditions prices) : 
  prices.pencil + prices.notebook = 80 := by
  sorry

end martin_purchase_cost_l183_18367


namespace unique_perfect_square_grid_l183_18300

/-- A type representing a 2x3 grid of natural numbers -/
def Grid := Fin 2 → Fin 3 → ℕ

/-- Check if a natural number is a perfect square -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- Check if a Grid forms valid perfect squares horizontally and vertically -/
def is_valid_grid (g : Grid) : Prop :=
  (is_perfect_square (g 0 0 * 100 + g 0 1 * 10 + g 0 2)) ∧
  (is_perfect_square (g 1 0 * 100 + g 1 1 * 10 + g 1 2)) ∧
  (is_perfect_square (g 0 0 * 10 + g 1 0)) ∧
  (is_perfect_square (g 0 1 * 10 + g 1 1)) ∧
  (is_perfect_square (g 0 2 * 10 + g 1 2)) ∧
  (∀ i j, g i j < 10)

/-- The main theorem stating the existence and uniqueness of the solution -/
theorem unique_perfect_square_grid :
  ∃! g : Grid, is_valid_grid g ∧ g 0 0 = 8 ∧ g 0 1 = 4 ∧ g 0 2 = 1 ∧
                               g 1 0 = 1 ∧ g 1 1 = 9 ∧ g 1 2 = 6 :=
sorry

end unique_perfect_square_grid_l183_18300


namespace percentage_relation_l183_18362

theorem percentage_relation (A B C : ℝ) (h1 : A = 1.71 * C) (h2 : A = 0.05 * B) : B = 14.2 * C := by
  sorry

end percentage_relation_l183_18362


namespace remainder_of_sum_l183_18358

theorem remainder_of_sum (x y u v : ℕ) (h1 : y > 0) (h2 : x = u * y + v) (h3 : v < y) :
  (x + 3 * u * y) % y = v :=
by sorry

end remainder_of_sum_l183_18358


namespace max_fraction_of_three_numbers_l183_18361

/-- Two-digit natural number -/
def TwoDigitNat : Type := {n : ℕ // 10 ≤ n ∧ n ≤ 99}

theorem max_fraction_of_three_numbers (x y z : TwoDigitNat) 
  (h : (x.val + y.val + z.val) / 3 = 60) :
  (x.val + y.val) / z.val ≤ 17 := by
  sorry

end max_fraction_of_three_numbers_l183_18361


namespace fold_and_punch_theorem_l183_18366

/-- Represents a rectangular piece of paper -/
structure Paper :=
  (width : ℕ)
  (height : ℕ)

/-- Represents the state of the paper after folding and punching -/
inductive FoldedPaper
  | Unfolded (p : Paper)
  | FoldedOnce (p : Paper)
  | FoldedTwice (p : Paper)
  | FoldedThrice (p : Paper)
  | Punched (p : Paper)

/-- Folds the paper from bottom to top -/
def foldBottomToTop (p : Paper) : FoldedPaper :=
  FoldedPaper.FoldedOnce p

/-- Folds the paper from right to left -/
def foldRightToLeft (p : FoldedPaper) : FoldedPaper :=
  match p with
  | FoldedPaper.FoldedOnce p => FoldedPaper.FoldedTwice p
  | _ => p

/-- Folds the paper from top to bottom -/
def foldTopToBottom (p : FoldedPaper) : FoldedPaper :=
  match p with
  | FoldedPaper.FoldedTwice p => FoldedPaper.FoldedThrice p
  | _ => p

/-- Punches a hole in the center of the folded paper -/
def punchHole (p : FoldedPaper) : FoldedPaper :=
  match p with
  | FoldedPaper.FoldedThrice p => FoldedPaper.Punched p
  | _ => p

/-- Counts the number of holes in the unfolded paper -/
def countHoles (p : FoldedPaper) : ℕ :=
  match p with
  | FoldedPaper.Punched _ => 8
  | _ => 0

/-- Theorem stating that folding a rectangular paper three times and punching a hole results in 8 holes when unfolded -/
theorem fold_and_punch_theorem (p : Paper) :
  countHoles (punchHole (foldTopToBottom (foldRightToLeft (foldBottomToTop p)))) = 8 := by
  sorry


end fold_and_punch_theorem_l183_18366


namespace two_digit_multiplication_trick_l183_18386

theorem two_digit_multiplication_trick (a b c : ℕ) 
  (h1 : b + c = 10) 
  (h2 : 0 ≤ a ∧ a ≤ 9) 
  (h3 : 0 ≤ b ∧ b ≤ 9) 
  (h4 : 0 ≤ c ∧ c ≤ 9) :
  (10 * a + b) * (10 * a + c) = 100 * a * (a + 1) + b * c :=
by sorry

end two_digit_multiplication_trick_l183_18386


namespace point_on_axes_with_inclination_l183_18345

/-- Given point A(-2, 1) and the angle of inclination of line PA is 30°,
    prove that point P on the coordinate axes is either (-4, 0) or (0, 2). -/
theorem point_on_axes_with_inclination (A : ℝ × ℝ) (P : ℝ × ℝ) :
  A = (-2, 1) →
  (P.1 = 0 ∨ P.2 = 0) →
  (P.2 - A.2) / (P.1 - A.1) = Real.tan (30 * π / 180) →
  (P = (-4, 0) ∨ P = (0, 2)) :=
by sorry

end point_on_axes_with_inclination_l183_18345


namespace identical_lines_condition_no_identical_lines_l183_18380

/-- Two lines are identical if and only if they have the same slope and y-intercept -/
theorem identical_lines_condition (a b : ℝ) : 
  (∀ x y : ℝ, 2*x + a*y + b = 0 ↔ b*x - 3*y + 15 = 0) ↔ 
  ((-2/a = b/3) ∧ (-b/a = -5)) :=
sorry

/-- There are no real pairs (a, b) such that the lines 2x + ay + b = 0 and bx - 3y + 15 = 0 have the same graph -/
theorem no_identical_lines : ¬∃ a b : ℝ, ∀ x y : ℝ, 2*x + a*y + b = 0 ↔ b*x - 3*y + 15 = 0 :=
sorry

end identical_lines_condition_no_identical_lines_l183_18380


namespace unique_solution_system_of_equations_l183_18315

theorem unique_solution_system_of_equations :
  ∃! (x y : ℝ), x + 2 * y = 2 ∧ 3 * x - 4 * y = -24 :=
by
  -- The proof goes here
  sorry

end unique_solution_system_of_equations_l183_18315


namespace linear_functions_intersection_l183_18327

theorem linear_functions_intersection (a b c d : ℝ) (h : a ≠ b) :
  (∃ x y : ℝ, y = a * x + a ∧ y = b * x + b ∧ y = c * x + d) → c = d := by
  sorry

end linear_functions_intersection_l183_18327


namespace smallest_good_is_correct_l183_18388

/-- The operation described in the problem -/
def operation (n : ℕ) : ℕ :=
  (n / 10) + 2 * (n % 10)

/-- A number is 'good' if it's unchanged by the operation -/
def is_good (n : ℕ) : Prop :=
  operation n = n

/-- The smallest 'good' number -/
def smallest_good : ℕ :=
  10^99 + 1

theorem smallest_good_is_correct :
  is_good smallest_good ∧ ∀ m : ℕ, m < smallest_good → ¬ is_good m :=
sorry

end smallest_good_is_correct_l183_18388


namespace monochromatic_rectangle_exists_l183_18382

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a coloring function type
def Coloring := ℤ × ℤ → Color

-- Define a rectangle type
structure Rectangle where
  x1 : ℤ
  y1 : ℤ
  x2 : ℤ
  y2 : ℤ
  h_x : x1 < x2
  h_y : y1 < y2

-- State the theorem
theorem monochromatic_rectangle_exists (c : Coloring) :
  ∃ (r : Rectangle), 
    c (r.x1, r.y1) = c (r.x1, r.y2) ∧
    c (r.x1, r.y1) = c (r.x2, r.y1) ∧
    c (r.x1, r.y1) = c (r.x2, r.y2) :=
by sorry

end monochromatic_rectangle_exists_l183_18382


namespace days_missed_proof_l183_18301

/-- The total number of days missed by Vanessa, Mike, and Sarah -/
def total_days_missed (v m s : ℕ) : ℕ := v + m + s

/-- Theorem: Given the conditions, the total number of days missed is 17 -/
theorem days_missed_proof (v m s : ℕ) 
  (h1 : v + m = 14)  -- Vanessa and Mike have missed 14 days total
  (h2 : m + s = 12)  -- Mike and Sarah have missed 12 days total
  (h3 : v = 5)       -- Vanessa missed 5 days of school alone
  : total_days_missed v m s = 17 := by
  sorry

#check days_missed_proof

end days_missed_proof_l183_18301


namespace parallel_lines_a_value_l183_18384

/-- Two lines in the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if their slopes are equal -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem parallel_lines_a_value :
  ∀ a : ℝ,
  let l1 : Line := { a := a, b := 2, c := 6 }
  let l2 : Line := { a := 1, b := a - 1, c := 3 }
  parallel l1 l2 → a = -1 := by
  sorry

end parallel_lines_a_value_l183_18384


namespace negation_of_universal_positive_quadratic_l183_18349

theorem negation_of_universal_positive_quadratic (p : Prop) :
  (p ↔ ∀ x : ℝ, x^2 - x + 1 > 0) →
  (¬p ↔ ∃ x : ℝ, x^2 - x + 1 ≤ 0) :=
by sorry

end negation_of_universal_positive_quadratic_l183_18349


namespace second_half_speed_l183_18373

/-- Proves that given a journey of 224 km completed in 10 hours, where the first half is traveled at 21 km/hr, the speed of the second half of the journey is 24 km/hr. -/
theorem second_half_speed (total_distance : ℝ) (total_time : ℝ) (first_half_speed : ℝ) 
    (h1 : total_distance = 224)
    (h2 : total_time = 10)
    (h3 : first_half_speed = 21) : 
  (total_distance / 2) / (total_time - (total_distance / 2) / first_half_speed) = 24 := by
  sorry

#check second_half_speed

end second_half_speed_l183_18373


namespace books_remaining_on_shelf_l183_18344

theorem books_remaining_on_shelf (initial_books : Real) (books_taken : Real) 
  (h1 : initial_books = 38.0) (h2 : books_taken = 10.0) : 
  initial_books - books_taken = 28.0 := by
  sorry

end books_remaining_on_shelf_l183_18344


namespace geometric_sequence_sum_l183_18348

/-- Given a geometric sequence, prove that if the sum of the first n terms is 48
    and the sum of the first 2n terms is 60, then the sum of the first 3n terms is 63. -/
theorem geometric_sequence_sum (n : ℕ) (S : ℕ → ℝ) : 
  S n = 48 → S (2*n) = 60 → S (3*n) = 63 := by
  sorry

end geometric_sequence_sum_l183_18348


namespace five_solutions_l183_18337

/-- The number of positive integer solutions to 2x + y = 11 -/
def solution_count : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 2 * p.1 + p.2 = 11 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 11) (Finset.range 11))).card

/-- Theorem stating that there are exactly 5 positive integer solutions to 2x + y = 11 -/
theorem five_solutions : solution_count = 5 := by
  sorry

end five_solutions_l183_18337


namespace negative_abs_of_negative_one_l183_18332

theorem negative_abs_of_negative_one : -|-1| = 1 := by
  sorry

end negative_abs_of_negative_one_l183_18332


namespace rhombus_perimeter_l183_18385

/-- The perimeter of a rhombus with diagonals measuring 24 feet and 16 feet is 16√13 feet. -/
theorem rhombus_perimeter (d1 d2 : ℝ) (h1 : d1 = 24) (h2 : d2 = 16) :
  4 * Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2) = 16 * Real.sqrt 13 :=
by sorry

end rhombus_perimeter_l183_18385


namespace x_value_for_y_4_l183_18353

/-- The relationship between x, y, and z -/
def x_relation (x y z k : ℚ) : Prop :=
  x = k * (z / y^2)

/-- The function defining z in terms of y -/
def z_function (y : ℚ) : ℚ :=
  2 * y - 1

theorem x_value_for_y_4 (k : ℚ) :
  (∃ x₀ : ℚ, x_relation x₀ 3 (z_function 3) k ∧ x₀ = 1) →
  (∃ x : ℚ, x_relation x 4 (z_function 4) k ∧ x = 63/80) :=
by sorry

end x_value_for_y_4_l183_18353


namespace sqrt_two_minus_half_sqrt_two_equals_half_sqrt_two_l183_18370

theorem sqrt_two_minus_half_sqrt_two_equals_half_sqrt_two :
  Real.sqrt 2 - (Real.sqrt 2) / 2 = (Real.sqrt 2) / 2 := by
  sorry

end sqrt_two_minus_half_sqrt_two_equals_half_sqrt_two_l183_18370


namespace kim_cherry_saplings_l183_18316

/-- Given that Kim plants 80 cherry pits, 25% of them sprout, and she sells 6 saplings,
    prove that she has 14 cherry saplings left. -/
theorem kim_cherry_saplings (total_pits : ℕ) (sprout_rate : ℚ) (sold_saplings : ℕ) :
  total_pits = 80 →
  sprout_rate = 1/4 →
  sold_saplings = 6 →
  (total_pits : ℚ) * sprout_rate - sold_saplings = 14 := by
  sorry

end kim_cherry_saplings_l183_18316


namespace circle_slash_problem_l183_18302

/-- Custom operation ⊘ defined as (a ⊘ b) = (√(k*a + b))^3 -/
noncomputable def circle_slash (k : ℝ) (a b : ℝ) : ℝ := (Real.sqrt (k * a + b)) ^ 3

/-- Theorem: If 9 ⊘ x = 64 and k = 3, then x = -11 -/
theorem circle_slash_problem (x : ℝ) (h1 : circle_slash 3 9 x = 64) : x = -11 := by
  sorry

end circle_slash_problem_l183_18302


namespace smallest_four_digit_divisible_by_53_l183_18321

theorem smallest_four_digit_divisible_by_53 :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by sorry

end smallest_four_digit_divisible_by_53_l183_18321


namespace linear_function_unique_solution_l183_18359

/-- Given a linear function f(x) = ax + 19 where f(3) = 7, 
    prove that if f(t) = 15, then t = 1 -/
theorem linear_function_unique_solution 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = a * x + 19) 
  (h2 : f 3 = 7) 
  (t : ℝ) 
  (h3 : f t = 15) : 
  t = 1 := by
sorry

end linear_function_unique_solution_l183_18359


namespace second_markdown_percentage_l183_18343

theorem second_markdown_percentage
  (P : ℝ)  -- Original price
  (first_markdown_percent : ℝ := 50)  -- First markdown percentage
  (final_price_percent : ℝ := 45)  -- Final price as percentage of original
  (h_P_pos : P > 0)  -- Assumption that the original price is positive
  : ∃ (second_markdown_percent : ℝ),
    (1 - second_markdown_percent / 100) * ((100 - first_markdown_percent) / 100 * P) = 
    final_price_percent / 100 * P ∧
    second_markdown_percent = 10 := by
sorry

end second_markdown_percentage_l183_18343


namespace bc_length_fraction_l183_18368

/-- Given a line segment AD with points B and C on it, prove that BC = 5/36 * AD -/
theorem bc_length_fraction (A B C D : ℝ) : 
  (B - A) = 3 * (D - B) →  -- AB = 3 * BD
  (C - A) = 8 * (D - C) →  -- AC = 8 * CD
  (C - B) = (5 / 36) * (D - A) := by sorry

end bc_length_fraction_l183_18368


namespace inequality_proof_l183_18335

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 3) : 
  1 / (x + y + z^2) + 1 / (y + z + x^2) + 1 / (z + x + y^2) ≤ 1 := by
  sorry

end inequality_proof_l183_18335


namespace quadratic_two_roots_l183_18354

theorem quadratic_two_roots
  (a b c d e : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > d)
  (h4 : e ≠ -1) :
  ∃ (x y : ℝ), x ≠ y ∧
  (e + 1) * x^2 - (a + c + b*e + d*e) * x + a*c + e*b*d = 0 ∧
  (e + 1) * y^2 - (a + c + b*e + d*e) * y + a*c + e*b*d = 0 :=
by sorry

end quadratic_two_roots_l183_18354


namespace divide_eight_by_repeating_third_l183_18397

theorem divide_eight_by_repeating_third (x : ℚ) : x = 1/3 → 8 / x = 24 := by
  sorry

end divide_eight_by_repeating_third_l183_18397


namespace paint_room_combinations_l183_18308

theorem paint_room_combinations (n : ℕ) (k : ℕ) (h1 : n = 9) (h2 : k = 2) :
  (Nat.choose n k) * k.factorial = 72 := by
  sorry

end paint_room_combinations_l183_18308


namespace first_digit_base8_725_l183_18394

/-- The first digit of the base 8 representation of a natural number -/
def firstDigitBase8 (n : ℕ) : ℕ :=
  sorry

/-- The base 10 number we're converting -/
def base10Number : ℕ := 725

/-- Theorem stating that the first digit of 725 in base 8 is 1 -/
theorem first_digit_base8_725 : firstDigitBase8 base10Number = 1 := by
  sorry

end first_digit_base8_725_l183_18394


namespace dans_remaining_money_l183_18328

/-- Proves that if Dan has $3 and spends $1 on a candy bar, the amount of money left is $2. -/
theorem dans_remaining_money (initial_amount : ℕ) (candy_bar_cost : ℕ) (remaining_amount : ℕ) :
  initial_amount = 3 →
  candy_bar_cost = 1 →
  remaining_amount = initial_amount - candy_bar_cost →
  remaining_amount = 2 :=
by
  sorry

end dans_remaining_money_l183_18328


namespace building_occupancy_ratio_l183_18392

/-- Calculates the occupancy ratio of a building given the number of units,
    monthly rent per unit, and total annual rent received. -/
theorem building_occupancy_ratio
  (num_units : ℕ)
  (monthly_rent : ℝ)
  (annual_rent_received : ℝ)
  (h1 : num_units = 100)
  (h2 : monthly_rent = 400)
  (h3 : annual_rent_received = 360000) :
  annual_rent_received / (num_units * monthly_rent * 12) = 0.75 := by
  sorry

end building_occupancy_ratio_l183_18392


namespace reciprocal_sum_diff_l183_18338

theorem reciprocal_sum_diff : (1 / (1/4 + 1/6 - 1/12) : ℚ) = 3 := by
  sorry

end reciprocal_sum_diff_l183_18338


namespace value_difference_l183_18378

theorem value_difference (n : ℝ) (h : n = 40) : 
  (n * 1.25) - (n * 0.7) = 22 := by
  sorry

end value_difference_l183_18378


namespace arccos_neg_one_eq_pi_l183_18350

theorem arccos_neg_one_eq_pi : Real.arccos (-1) = π := by
  sorry

end arccos_neg_one_eq_pi_l183_18350


namespace final_amount_is_47500_l183_18324

def income : ℝ := 200000

def children_share : ℝ := 0.15
def num_children : ℕ := 3
def wife_share : ℝ := 0.30
def donation_rate : ℝ := 0.05

def final_amount : ℝ :=
  let children_total := children_share * num_children * income
  let wife_total := wife_share * income
  let remaining_after_family := income - children_total - wife_total
  let donation := donation_rate * remaining_after_family
  remaining_after_family - donation

theorem final_amount_is_47500 :
  final_amount = 47500 := by sorry

end final_amount_is_47500_l183_18324


namespace partner_numbers_problem_l183_18391

/-- Definition of "partner numbers" -/
def partner_numbers (m n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    m = 100 * a + 10 * b + c ∧
    n = 100 * d + 10 * e + f ∧
    1 ≤ a ∧ a ≤ 9 ∧
    1 ≤ b ∧ b ≤ 9 ∧
    1 ≤ c ∧ c ≤ 9 ∧
    1 ≤ d ∧ d ≤ 9 ∧
    1 ≤ e ∧ e ≤ 9 ∧
    1 ≤ f ∧ f ≤ 9 ∧
    ∃ (k : ℤ), k * (b - c) = a + 4 * d + 4 * e + 4 * f

theorem partner_numbers_problem (x y z : ℕ) 
  (hx : x ≤ 3) 
  (hy : 0 < y ∧ y ≤ 4) 
  (hz : 3 < z ∧ z ≤ 9) 
  (h_partner : partner_numbers (467 + 110 * x) (200 * y + z + 37))
  (h_sum : (2 * y + z + 1) % 12 = 0) :
  467 + 110 * x = 467 ∨ 467 + 110 * x = 687 := by
  sorry

end partner_numbers_problem_l183_18391


namespace farmer_potatoes_l183_18305

theorem farmer_potatoes (initial_tomatoes picked_tomatoes total_left : ℕ) 
  (h1 : initial_tomatoes = 177)
  (h2 : picked_tomatoes = 53)
  (h3 : total_left = 136) :
  initial_tomatoes - picked_tomatoes + (total_left - (initial_tomatoes - picked_tomatoes)) = 12 := by
  sorry

end farmer_potatoes_l183_18305


namespace wrapping_paper_area_l183_18318

/-- The area of a square wrapping paper used to wrap a rectangular box with a square base. -/
theorem wrapping_paper_area (w h x : ℝ) (hw : w > 0) (hh : h > 0) (hx : x ≥ 0) :
  let s := Real.sqrt ((h + x)^2 + (w/2)^2)
  s^2 = (h + x)^2 + w^2/4 :=
by sorry

end wrapping_paper_area_l183_18318


namespace rosies_pies_l183_18319

/-- Given that Rosie can make 3 pies out of 12 apples, 
    this theorem proves how many pies she can make out of 36 apples. -/
theorem rosies_pies (apples_per_three_pies : ℕ) (total_apples : ℕ) : 
  apples_per_three_pies = 12 → total_apples = 36 → (total_apples / apples_per_three_pies) * 3 = 27 := by
  sorry

#check rosies_pies

end rosies_pies_l183_18319


namespace purely_imaginary_iff_a_nonzero_b_zero_l183_18393

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem purely_imaginary_iff_a_nonzero_b_zero (a b : ℝ) :
  is_purely_imaginary (Complex.mk b (a)) ↔ a ≠ 0 ∧ b = 0 := by
  sorry

end purely_imaginary_iff_a_nonzero_b_zero_l183_18393


namespace smallest_square_partition_l183_18341

theorem smallest_square_partition : ∃ (n : ℕ), 
  (40 ∣ n) ∧ 
  (49 ∣ n) ∧ 
  (∀ (m : ℕ), (40 ∣ m) ∧ (49 ∣ m) → m ≥ n) ∧
  n = 1960 := by
  sorry

end smallest_square_partition_l183_18341


namespace student_a_final_score_l183_18311

/-- Calculate the final score for a test -/
def finalScore (totalQuestions : ℕ) (correctAnswers : ℕ) : ℕ :=
  let incorrectAnswers := totalQuestions - correctAnswers
  correctAnswers - 2 * incorrectAnswers

/-- Theorem: The final score for a test with 100 questions and 92 correct answers is 76 -/
theorem student_a_final_score :
  finalScore 100 92 = 76 := by
  sorry

end student_a_final_score_l183_18311


namespace continued_fraction_sum_l183_18314

theorem continued_fraction_sum (x y z : ℕ+) :
  (30 : ℚ) / 7 = x + 1 / (y + 1 / z) →
  x + y + z = 9 := by
  sorry

end continued_fraction_sum_l183_18314


namespace bridget_fruits_count_bridget_fruits_proof_l183_18376

theorem bridget_fruits_count : ℕ → ℕ → Prop :=
  fun apples oranges =>
    apples / oranges = 2 →
    apples / 2 - 3 = 4 →
    oranges - 3 = 5 →
    apples + oranges = 21

theorem bridget_fruits_proof : ∃ a o : ℕ, bridget_fruits_count a o := by
  sorry

end bridget_fruits_count_bridget_fruits_proof_l183_18376


namespace absolute_value_equation_solution_l183_18369

theorem absolute_value_equation_solution :
  ∃! y : ℝ, |y - 8| + 3 * y = 12 :=
by
  -- Proof goes here
  sorry

end absolute_value_equation_solution_l183_18369


namespace max_x_value_l183_18342

theorem max_x_value (x : ℤ) 
  (h : Real.log (2 * x + 1) / Real.log (1/4) < Real.log (x - 1) / Real.log (1/2)) : 
  x ≤ 3 ∧ ∃ y : ℤ, y ≤ 3 ∧ Real.log (2 * y + 1) / Real.log (1/4) < Real.log (y - 1) / Real.log (1/2) :=
sorry

end max_x_value_l183_18342


namespace a_in_M_l183_18323

def M : Set ℝ := { x | x ≤ 5 }

def a : ℝ := 2

theorem a_in_M : a ∈ M := by sorry

end a_in_M_l183_18323


namespace non_equilateral_combinations_l183_18334

/-- The number of dots evenly spaced on the circle's circumference -/
def n : ℕ := 6

/-- The number of dots to be selected in each combination -/
def k : ℕ := 3

/-- The total number of combinations of k dots from n dots -/
def total_combinations : ℕ := Nat.choose n k

/-- The number of equilateral triangles that can be formed -/
def equilateral_triangles : ℕ := 2

/-- Theorem: The number of combinations of 3 dots that do not form an equilateral triangle
    is equal to the total number of 3-dot combinations minus the number of equilateral triangles -/
theorem non_equilateral_combinations :
  total_combinations - equilateral_triangles = 18 := by sorry

end non_equilateral_combinations_l183_18334


namespace triangle_problem_l183_18379

theorem triangle_problem (a b c A B C : Real) (h1 : (2*b - c) * Real.cos A = a * Real.cos C)
  (h2 : a = Real.sqrt 13) (h3 : b + c = 5) :
  A = π/3 ∧ (1/2 * b * c * Real.sin A = Real.sqrt 3) :=
by sorry

end triangle_problem_l183_18379


namespace two_digit_number_interchange_l183_18389

theorem two_digit_number_interchange (a b k : ℕ) (h1 : a ≥ 1 ∧ a ≤ 9) (h2 : b ≤ 9) 
  (h3 : 10 * a + b = k * (a + b)) :
  10 * b + a = (11 - k) * (a + b) := by
  sorry

end two_digit_number_interchange_l183_18389


namespace largest_four_digit_divisible_by_35_l183_18399

theorem largest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, n ≤ 9999 ∧ n ≥ 1000 ∧ n % 35 = 0 → n ≤ 9985 :=
by
  sorry

end largest_four_digit_divisible_by_35_l183_18399


namespace train_speed_l183_18398

def train_length : ℝ := 100
def tunnel_length : ℝ := 2300
def time_seconds : ℝ := 120

theorem train_speed :
  let total_distance := tunnel_length + train_length
  let speed_ms := total_distance / time_seconds
  let speed_kmh := speed_ms * 3.6
  speed_kmh = 72 := by sorry

end train_speed_l183_18398


namespace valid_numbers_are_unique_l183_18371

/-- Represents a six-digit number of the form 387abc --/
def SixDigitNumber (a b c : Nat) : Nat :=
  387000 + a * 100 + b * 10 + c

/-- Checks if a natural number is divisible by 5, 6, and 7 --/
def isDivisibleBy567 (n : Nat) : Prop :=
  n % 5 = 0 ∧ n % 6 = 0 ∧ n % 7 = 0

/-- The set of valid six-digit numbers --/
def ValidNumbers : Set Nat :=
  {387000, 387210, 387420, 387630, 387840}

/-- Theorem stating that the ValidNumbers are the only six-digit numbers
    of the form 387abc that are divisible by 5, 6, and 7 --/
theorem valid_numbers_are_unique :
  ∀ a b c : Nat, a < 10 ∧ b < 10 ∧ c < 10 →
  isDivisibleBy567 (SixDigitNumber a b c) ↔ SixDigitNumber a b c ∈ ValidNumbers :=
by sorry

end valid_numbers_are_unique_l183_18371


namespace democrat_ratio_l183_18304

theorem democrat_ratio (total_participants male_participants female_participants male_democrats female_democrats : ℕ) :
  total_participants = 720 ∧
  female_participants = 240 ∧
  male_participants = 480 ∧
  female_democrats = 120 ∧
  2 * female_democrats = female_participants ∧
  3 * (male_democrats + female_democrats) = total_participants ∧
  male_participants + female_participants = total_participants →
  4 * male_democrats = male_participants :=
by
  sorry

end democrat_ratio_l183_18304


namespace expression_simplification_l183_18387

theorem expression_simplification (a x : ℝ) 
  (h1 : x ≠ a / 3) (h2 : x ≠ -a / 3) (h3 : x ≠ -a) : 
  (3 * a^2 + 2 * a * x - x^2) / ((3 * x + a) * (a + x)) - 2 + 
  10 * (a * x - 3 * x^2) / (a^2 - 9 * x^2) = 1 := by
  sorry

end expression_simplification_l183_18387


namespace eight_divided_by_repeating_third_l183_18310

/-- Represents the repeating decimal 0.333... -/
def repeating_third : ℚ := 1 / 3

/-- Proves that 8 divided by 0.333... equals 24 -/
theorem eight_divided_by_repeating_third : 8 / repeating_third = 24 := by sorry

end eight_divided_by_repeating_third_l183_18310


namespace sin_2alpha_value_l183_18330

theorem sin_2alpha_value (α : Real) (h : Real.cos (α - Real.pi / 4) = Real.sqrt 3 / 3) :
  Real.sin (2 * α) = -1 / 3 := by
  sorry

end sin_2alpha_value_l183_18330


namespace root_expression_value_l183_18360

theorem root_expression_value (a : ℝ) : 
  (2 * a^2 - 7 * a - 1 = 0) → (a * (2 * a - 7) + 5 = 6) := by
  sorry

end root_expression_value_l183_18360


namespace triple_root_values_l183_18325

/-- A polynomial with integer coefficients of the form x^5 + b₄x^4 + b₃x^3 + b₂x^2 + b₁x + 24 -/
def IntPolynomial (b₄ b₃ b₂ b₁ : ℤ) (x : ℤ) : ℤ :=
  x^5 + b₄*x^4 + b₃*x^3 + b₂*x^2 + b₁*x + 24

/-- r is a triple root of the polynomial if (x - r)^3 divides the polynomial -/
def IsTripleRoot (r : ℤ) (b₄ b₃ b₂ b₁ : ℤ) : Prop :=
  ∃ (q : ℤ → ℤ), ∀ x, IntPolynomial b₄ b₃ b₂ b₁ x = (x - r)^3 * q x

theorem triple_root_values (r : ℤ) :
  (∃ b₄ b₃ b₂ b₁ : ℤ, IsTripleRoot r b₄ b₃ b₂ b₁) ↔ r ∈ ({-2, -1, 1, 2} : Set ℤ) :=
sorry

end triple_root_values_l183_18325
