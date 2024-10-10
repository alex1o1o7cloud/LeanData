import Mathlib

namespace circle_intersection_problem_l1132_113271

theorem circle_intersection_problem (k : ℝ) :
  let center : ℝ × ℝ := ((27 - 3) / 2 + -3, 0)
  let radius : ℝ := (27 - (-3)) / 2
  let circle_equation (x y : ℝ) : Prop := (x - center.1)^2 + (y - center.2)^2 = radius^2
  (∃ y₁ y₂ : ℝ, y₁ ≠ y₂ ∧ circle_equation k y₁ ∧ circle_equation k y₂) →
  (∃ y : ℝ, circle_equation k y ∧ y = 12) →
  k = 3 ∨ k = 21 :=
by sorry


end circle_intersection_problem_l1132_113271


namespace cost_of_500_apples_l1132_113265

/-- The cost of a single apple in cents -/
def apple_cost : ℕ := 5

/-- The number of cents in a dollar -/
def cents_per_dollar : ℕ := 100

/-- The number of apples we want to calculate the cost for -/
def apple_quantity : ℕ := 500

/-- Theorem stating that the cost of 500 apples is 25.00 dollars -/
theorem cost_of_500_apples : 
  (apple_quantity * apple_cost : ℚ) / cents_per_dollar = 25 := by
  sorry

end cost_of_500_apples_l1132_113265


namespace g_of_x_plus_3_l1132_113225

/-- Given a function g(x) = x(x+3)/3, prove that g(x+3) = (x^2 + 9x + 18) / 3 -/
theorem g_of_x_plus_3 (x : ℝ) : 
  let g : ℝ → ℝ := fun x => x * (x + 3) / 3
  g (x + 3) = (x^2 + 9*x + 18) / 3 := by
sorry

end g_of_x_plus_3_l1132_113225


namespace correct_calculation_l1132_113229

theorem correct_calculation (a : ℝ) : (-2 * a^3)^2 = 4 * a^6 := by
  sorry

end correct_calculation_l1132_113229


namespace shaded_squares_correct_l1132_113215

/-- Given a square grid with odd side length, calculates the number of shaded squares along the two diagonals -/
def shadedSquares (n : ℕ) : ℕ :=
  2 * n - 1

theorem shaded_squares_correct (n : ℕ) (h : Odd n) :
  shadedSquares n = 2 * n - 1 := by
  sorry

#eval shadedSquares 7  -- Expected: 13
#eval shadedSquares 101  -- Expected: 201

end shaded_squares_correct_l1132_113215


namespace al_sandwich_count_l1132_113216

/-- The number of different types of bread available. -/
def num_bread : ℕ := 5

/-- The number of different types of meat available. -/
def num_meat : ℕ := 6

/-- The number of different types of cheese available. -/
def num_cheese : ℕ := 5

/-- Represents whether French bread is available. -/
def french_bread_available : Prop := True

/-- Represents whether turkey is available. -/
def turkey_available : Prop := True

/-- Represents whether Swiss cheese is available. -/
def swiss_cheese_available : Prop := True

/-- Represents whether white bread is available. -/
def white_bread_available : Prop := True

/-- Represents whether rye bread is available. -/
def rye_bread_available : Prop := True

/-- Represents whether chicken is available. -/
def chicken_available : Prop := True

/-- The number of sandwich combinations with turkey and Swiss cheese. -/
def turkey_swiss_combos : ℕ := num_bread

/-- The number of sandwich combinations with white bread and chicken. -/
def white_chicken_combos : ℕ := num_cheese

/-- The number of sandwich combinations with rye bread and turkey. -/
def rye_turkey_combos : ℕ := num_cheese

/-- The total number of sandwich combinations Al can order. -/
def al_sandwich_options : ℕ := num_bread * num_meat * num_cheese - turkey_swiss_combos - white_chicken_combos - rye_turkey_combos

theorem al_sandwich_count :
  french_bread_available ∧ 
  turkey_available ∧ 
  swiss_cheese_available ∧ 
  white_bread_available ∧
  rye_bread_available ∧
  chicken_available →
  al_sandwich_options = 135 := by
  sorry

end al_sandwich_count_l1132_113216


namespace point_on_number_line_l1132_113280

theorem point_on_number_line (A : ℝ) : 
  (|A| = 5) ↔ (A = 5 ∨ A = -5) := by sorry

end point_on_number_line_l1132_113280


namespace marks_of_a_l1132_113211

theorem marks_of_a (a b c d e : ℝ) : 
  (a + b + c) / 3 = 48 →
  (a + b + c + d) / 4 = 47 →
  e = d + 3 →
  (b + c + d + e) / 4 = 48 →
  a = 43 := by
sorry

end marks_of_a_l1132_113211


namespace adults_attending_play_l1132_113266

/-- Proves the number of adults attending a play given ticket prices and total receipts --/
theorem adults_attending_play (adult_price children_price total_receipts total_attendance : ℕ) 
  (h1 : adult_price = 25)
  (h2 : children_price = 15)
  (h3 : total_receipts = 7200)
  (h4 : total_attendance = 400) :
  ∃ (adults children : ℕ), 
    adults + children = total_attendance ∧ 
    adult_price * adults + children_price * children = total_receipts ∧
    adults = 120 := by
  sorry


end adults_attending_play_l1132_113266


namespace sequence_problem_l1132_113217

def arithmetic_sequence (a b c d : ℝ) : Prop :=
  b - a = c - b ∧ c - b = d - c

def geometric_sequence (a b c d : ℝ) : Prop :=
  b / a = c / b ∧ c / b = d / c

theorem sequence_problem (a₁ a₂ a₃ b₁ b₂ b₃ : ℝ) 
  (h1 : arithmetic_sequence 1 a₁ a₂ a₃)
  (h2 : arithmetic_sequence a₁ a₂ a₃ 9)
  (h3 : geometric_sequence (-9) b₁ b₂ b₃)
  (h4 : geometric_sequence b₁ b₂ b₃ (-1)) :
  b₂ / (a₁ + a₃) = -3/10 := by
  sorry

end sequence_problem_l1132_113217


namespace no_infinite_line_family_l1132_113263

theorem no_infinite_line_family : ¬ ∃ (k : ℕ → ℝ), 
  (∀ n, k (n + 1) ≥ k n - 1 / k n) ∧ 
  (∀ n, k n * k (n + 1) ≥ 0) := by
  sorry

end no_infinite_line_family_l1132_113263


namespace fifteenth_term_of_sequence_l1132_113223

def inverse_proportional_sequence (a : ℕ → ℝ) : Prop :=
  ∃ k : ℝ, k > 0 ∧ ∀ n : ℕ, n > 0 → a n * a (n + 1) = k

theorem fifteenth_term_of_sequence 
  (a : ℕ → ℝ)
  (h_inv_prop : inverse_proportional_sequence a)
  (h_first_term : a 1 = 3)
  (h_second_term : a 2 = 4) :
  a 15 = 3 := by
  sorry

end fifteenth_term_of_sequence_l1132_113223


namespace squares_in_figure_150_l1132_113246

/-- The number of squares in figure n -/
def f (n : ℕ) : ℕ := 3 * n^2 + 3 * n + 1

/-- The sequence of squares for the first four figures -/
def initial_sequence : List ℕ := [1, 7, 19, 37]

theorem squares_in_figure_150 :
  f 150 = 67951 ∧
  (∀ n : Fin 4, f n.val = initial_sequence.get n) :=
sorry

end squares_in_figure_150_l1132_113246


namespace max_vector_norm_l1132_113239

theorem max_vector_norm (θ : ℝ) : 
  (‖(2 * Real.cos θ - Real.sqrt 3, 2 * Real.sin θ + 1)‖ : ℝ) ≤ 4 ∧ 
  ∃ θ₀ : ℝ, ‖(2 * Real.cos θ₀ - Real.sqrt 3, 2 * Real.sin θ₀ + 1)‖ = 4 :=
sorry

end max_vector_norm_l1132_113239


namespace arithmetic_sequence_common_difference_l1132_113264

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) :=
  ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ) (d : ℝ)
  (h_arithmetic : arithmetic_sequence a d)
  (h_a1 : a 1 = 2)
  (h_sum : a 2 + a 3 = 13) :
  d = 3 := by
sorry

end arithmetic_sequence_common_difference_l1132_113264


namespace hyperbola_vertex_distance_l1132_113279

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  4 * x^2 - 8 * x - 16 * y^2 + 32 * y - 12 = 0

/-- The distance between the vertices of the hyperbola -/
def vertex_distance : ℝ := 2

/-- Theorem: The distance between the vertices of the hyperbola is 2 -/
theorem hyperbola_vertex_distance :
  ∀ x y : ℝ, hyperbola_equation x y → vertex_distance = 2 := by
  sorry


end hyperbola_vertex_distance_l1132_113279


namespace weight_replacement_l1132_113273

theorem weight_replacement (total_weight : ℝ) (replaced_weight : ℝ) : 
  (8 : ℝ) * ((total_weight - replaced_weight + 77) / 8 - total_weight / 8) = 1.5 →
  replaced_weight = 65 := by
sorry

end weight_replacement_l1132_113273


namespace ellipse_and_line_theorem_l1132_113277

-- Define the ellipse C
def ellipse_C (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the line l
def line_l (m : ℝ) (x y : ℝ) : Prop := x = m*y + 1

-- Define the circle with diameter AB passing through origin
def circle_AB_origin (xA yA xB yB : ℝ) : Prop := xA*xB + yA*yB = 0

theorem ellipse_and_line_theorem :
  -- Given conditions
  let a : ℝ := Real.sqrt 3
  let e : ℝ := Real.sqrt 6 / 3
  let c : ℝ := e * a

  -- Part 1: Prove the standard equation of ellipse C
  (∀ x y : ℝ, ellipse_C x y ↔ x^2/3 + y^2 = 1) ∧

  -- Part 2: Prove the equation of line l
  (∃ m : ℝ, m = Real.sqrt 3 / 3 ∨ m = -Real.sqrt 3 / 3) ∧
  (∀ m : ℝ, (m = Real.sqrt 3 / 3 ∨ m = -Real.sqrt 3 / 3) →
    (∃ xA yA xB yB : ℝ,
      ellipse_C xA yA ∧ ellipse_C xB yB ∧
      line_l m xA yA ∧ line_l m xB yB ∧
      circle_AB_origin xA yA xB yB)) :=
by sorry

end ellipse_and_line_theorem_l1132_113277


namespace larger_number_is_72_l1132_113299

theorem larger_number_is_72 (a b : ℝ) : 
  5 * b = 6 * a ∧ b - a = 12 → b = 72 := by
  sorry

end larger_number_is_72_l1132_113299


namespace root_configurations_l1132_113234

-- Define the polynomial
def polynomial (a b c x : ℂ) : ℂ := x^4 - a*x^3 - b*x + c

-- Define the theorem
theorem root_configurations (a b c : ℂ) :
  (∃ d : ℂ, d ≠ a ∧ d ≠ b ∧ d ≠ c ∧
    (∀ x : ℂ, polynomial a b c x = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d)) →
  ((a = 0 ∧ b = 0 ∧ c ≠ 0) ∨
   (a ≠ 0 ∧ b = c ∧ c ≠ 0 ∧ c^2 + c + 1 = 0)) :=
by sorry


end root_configurations_l1132_113234


namespace vector_equation_solution_l1132_113281

theorem vector_equation_solution :
  ∃ (u v : ℝ), (![3, 1] : Fin 2 → ℝ) + u • ![8, -6] = ![2, -2] + v • ![-3, 4] ∧ 
  u = -13/14 ∧ v = 15/7 := by
  sorry

end vector_equation_solution_l1132_113281


namespace carrot_calories_l1132_113230

/-- The number of calories in a pound of carrots -/
def calories_per_pound_carrots : ℕ := 51

/-- The number of pounds of carrots Tom eats -/
def pounds_carrots : ℕ := 1

/-- The number of pounds of broccoli Tom eats -/
def pounds_broccoli : ℕ := 2

/-- The ratio of calories in broccoli compared to carrots -/
def broccoli_calorie_ratio : ℚ := 1/3

/-- The total number of calories Tom ate -/
def total_calories : ℕ := 85

theorem carrot_calories :
  calories_per_pound_carrots * pounds_carrots +
  (calories_per_pound_carrots : ℚ) * broccoli_calorie_ratio * pounds_broccoli = total_calories := by
  sorry

end carrot_calories_l1132_113230


namespace tangent_line_problem_l1132_113255

noncomputable def g (b : ℝ) (x : ℝ) : ℝ := x^3 + (5/2) * x^2 + 3 * Real.log x + b

theorem tangent_line_problem (b : ℝ) :
  (∃ (m : ℝ), (g b 1 = m * 1 - 5) ∧ 
              (∀ (x : ℝ), x ≠ 1 → (g b x - g b 1) / (x - 1) < m)) →
  b = 5/2 := by sorry

end tangent_line_problem_l1132_113255


namespace last_two_digits_product_l1132_113214

/-- Given an integer n, returns its last two digits as a pair -/
def lastTwoDigits (n : ℤ) : ℤ × ℤ :=
  let tens := (n / 10) % 10
  let ones := n % 10
  (tens, ones)

/-- Given an integer n, returns true if it's divisible by 8 -/
def divisibleBy8 (n : ℤ) : Prop :=
  n % 8 = 0

theorem last_two_digits_product (n : ℤ) :
  divisibleBy8 n ∧ (let (a, b) := lastTwoDigits n; a + b = 14) →
  (let (a, b) := lastTwoDigits n; a * b = 48) :=
by sorry

end last_two_digits_product_l1132_113214


namespace prob_five_is_one_thirteenth_l1132_113235

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (card_count : cards.card = 52)
  (rank_count : (cards.image (·.1)).card = 13)
  (suit_count : (cards.image (·.2)).card = 4)
  (unique_cards : ∀ r s, (r, s) ∈ cards ↔ r ∈ Finset.range 13 ∧ s ∈ Finset.range 4)

/-- The probability of drawing a specific rank from a standard deck -/
def prob_rank (d : Deck) (rank : Nat) : ℚ :=
  (d.cards.filter (·.1 = rank)).card / d.cards.card

/-- Theorem: The probability of drawing a 5 from a standard deck is 1/13 -/
theorem prob_five_is_one_thirteenth (d : Deck) : prob_rank d 5 = 1 / 13 := by
  sorry

end prob_five_is_one_thirteenth_l1132_113235


namespace geometric_sequence_product_l1132_113249

theorem geometric_sequence_product (a b : ℝ) :
  (∃ r : ℝ, r ≠ 0 ∧ -1 = -1 ∧ a = -1 * r ∧ b = a * r ∧ 2 = b * r) →
  a * b = -2 := by
sorry

end geometric_sequence_product_l1132_113249


namespace sequence_problem_l1132_113227

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, b (n + 1) = b n * r

theorem sequence_problem (a : ℕ → ℝ) (b : ℕ → ℝ) 
  (h_arith : arithmetic_sequence a)
  (h_geom : geometric_sequence b)
  (h_sum_a : a 1 + a 3 + a 5 + a 7 + a 9 = 50)
  (h_prod_b : b 4 * b 6 * b 14 * b 16 = 625) :
  (a 2 + a 8) / b 10 = 4 ∨ (a 2 + a 8) / b 10 = -4 :=
sorry

end sequence_problem_l1132_113227


namespace circle_equation_l1132_113260

-- Define the line l
def line_l (x y : ℝ) : Prop := x - 2*y = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x-2)^2 + (y-1)^2 = 1

-- Define that a point is on the line
def point_on_line (x y : ℝ) : Prop := line_l x y

-- Define that a point is on the circle
def point_on_circle (x y : ℝ) : Prop := circle_C x y

-- Theorem statement
theorem circle_equation : 
  (point_on_line 2 1) ∧ 
  (point_on_line 6 3) ∧ 
  (∃ h k : ℝ, point_on_line h k ∧ point_on_circle h k) ∧
  (point_on_circle 2 0) ∧ 
  (point_on_circle 3 1) → 
  ∀ x y : ℝ, circle_C x y ↔ (x-2)^2 + (y-1)^2 = 1 :=
sorry

end circle_equation_l1132_113260


namespace chairs_subset_count_l1132_113269

/-- The number of chairs arranged in a circle -/
def n : ℕ := 12

/-- The minimum number of adjacent chairs required in a subset -/
def k : ℕ := 4

/-- The number of subsets of n chairs arranged in a circle that contain at least k adjacent chairs -/
def subsets_with_adjacent_chairs (n k : ℕ) : ℕ := sorry

theorem chairs_subset_count : subsets_with_adjacent_chairs n k = 1610 := by sorry

end chairs_subset_count_l1132_113269


namespace apples_left_after_pie_l1132_113287

def apples_left (initial : ℝ) (contribution : ℝ) (pie_requirement : ℝ) : ℝ :=
  initial + contribution - pie_requirement

theorem apples_left_after_pie : apples_left 10 5 4 = 11 := by
  sorry

end apples_left_after_pie_l1132_113287


namespace inspector_rejection_l1132_113221

-- Define the rejection rate
def rejection_rate : ℝ := 0.15

-- Define the number of meters examined
def meters_examined : ℝ := 66.67

-- Define the function to calculate the number of rejected meters
def rejected_meters (rate : ℝ) (total : ℝ) : ℝ := rate * total

-- Theorem statement
theorem inspector_rejection :
  rejected_meters rejection_rate meters_examined = 10 := by
  sorry

end inspector_rejection_l1132_113221


namespace market_fruit_count_l1132_113298

theorem market_fruit_count (apples oranges bananas : ℕ) 
  (h1 : apples = oranges + 27)
  (h2 : oranges = bananas + 11)
  (h3 : apples + oranges + bananas = 301) :
  apples = 122 := by
sorry

end market_fruit_count_l1132_113298


namespace max_value_of_f_l1132_113201

def f (x a : ℝ) : ℝ := -x^2 + 4*x + a

theorem max_value_of_f (a : ℝ) :
  (∀ x ∈ Set.Icc 0 1, f x a ≥ -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = -2) →
  (∃ x ∈ Set.Icc 0 1, f x a = 1) ∧
  (∀ x ∈ Set.Icc 0 1, f x a ≤ 1) :=
by sorry

end max_value_of_f_l1132_113201


namespace product_of_difference_and_sum_of_squares_l1132_113220

theorem product_of_difference_and_sum_of_squares (a b : ℝ) 
  (h1 : a - b = 3) 
  (h2 : a^2 + b^2 = 25) : 
  a * b = 8 := by
sorry

end product_of_difference_and_sum_of_squares_l1132_113220


namespace apple_crates_delivered_l1132_113258

/-- The number of crates delivered to a factory, given the conditions of the apple delivery problem. -/
theorem apple_crates_delivered : ℕ := by
  -- Define the number of apples per crate
  let apples_per_crate : ℕ := 180

  -- Define the number of rotten apples
  let rotten_apples : ℕ := 160

  -- Define the number of boxes and apples per box for the remaining apples
  let num_boxes : ℕ := 100
  let apples_per_box : ℕ := 20

  -- Calculate the total number of good apples
  let good_apples : ℕ := num_boxes * apples_per_box

  -- Calculate the total number of apples delivered
  let total_apples : ℕ := good_apples + rotten_apples

  -- Calculate the number of crates delivered
  let crates_delivered : ℕ := total_apples / apples_per_crate

  -- Prove that the number of crates delivered is 12
  have : crates_delivered = 12 := by sorry

  -- Return the result
  exact 12


end apple_crates_delivered_l1132_113258


namespace workers_in_first_group_l1132_113233

/-- The number of workers in the first group -/
def W : ℕ := 70

/-- The time taken by the first group to complete the job (in hours) -/
def T1 : ℕ := 3

/-- The number of workers in the second group -/
def W2 : ℕ := 30

/-- The time taken by the second group to complete the job (in hours) -/
def T2 : ℕ := 7

/-- The amount of work done (assumed to be constant for both groups) -/
def work : ℕ := W * T1

theorem workers_in_first_group :
  (W * T1 = W2 * T2) ∧ (W * T2 = W2 * T1) → W = 70 := by
  sorry

end workers_in_first_group_l1132_113233


namespace binomial_coefficient_ratio_sum_l1132_113261

theorem binomial_coefficient_ratio_sum (n k : ℕ) : 
  (2 : ℚ) / 5 = (n.choose k : ℚ) / (n.choose (k + 1) : ℚ) →
  (∃ m l : ℕ, m ≠ l ∧ (m = n ∧ l = k ∨ l = n ∧ m = k) ∧ m + l = 23) :=
by sorry

end binomial_coefficient_ratio_sum_l1132_113261


namespace i_times_one_plus_i_l1132_113248

-- Define the imaginary unit i
def i : ℂ := Complex.I

-- State the theorem
theorem i_times_one_plus_i : i * (1 + i) = i - 1 := by
  sorry

end i_times_one_plus_i_l1132_113248


namespace committee_formation_with_previous_member_l1132_113218

def total_members : ℕ := 18
def committee_size : ℕ := 6
def previous_members : ℕ := 5

theorem committee_formation_with_previous_member :
  (Nat.choose total_members committee_size) - 
  (Nat.choose (total_members - previous_members) committee_size) = 16848 := by
  sorry

end committee_formation_with_previous_member_l1132_113218


namespace circle_radius_for_equal_areas_l1132_113288

/-- The radius of a circle satisfying the given conditions for a right-angled triangle --/
theorem circle_radius_for_equal_areas (a b c : ℝ) (h_right_triangle : a^2 + b^2 = c^2)
  (h_side_lengths : a = 6 ∧ b = 8 ∧ c = 10) : 
  ∃ r : ℝ, r^2 = 24 / Real.pi ∧ 
    (π * r^2 = a * b / 2) ∧
    (π * r^2 - a * b / 2 = a * b / 2 - π * r^2) :=
sorry

end circle_radius_for_equal_areas_l1132_113288


namespace perpendicular_planes_l1132_113204

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- State the theorem
theorem perpendicular_planes 
  (a b : Line) (α β : Plane) :
  perpendicular a β → 
  parallel a b → 
  contained_in b α → 
  plane_perpendicular α β :=
sorry

end perpendicular_planes_l1132_113204


namespace birds_joined_fence_l1132_113247

/-- Given initial numbers of storks and birds on a fence, and the fact that after some birds
    joined there are 2 more birds than storks, prove that 4 birds joined the fence. -/
theorem birds_joined_fence (initial_storks initial_birds : ℕ) 
  (h1 : initial_storks = 5)
  (h2 : initial_birds = 3)
  (h3 : ∃ (joined : ℕ), initial_birds + joined = initial_storks + 2) :
  ∃ (joined : ℕ), joined = 4 ∧ initial_birds + joined = initial_storks + 2 :=
by sorry

end birds_joined_fence_l1132_113247


namespace smallest_all_blue_count_l1132_113250

/-- Represents the colors of chameleons -/
inductive Color
| Red
| C2
| C3
| C4
| Blue

/-- Represents the result of a bite interaction between two chameleons -/
def bite_result (biter bitten : Color) : Color :=
  match biter, bitten with
  | Color.Red, Color.Red => Color.C2
  | Color.Red, Color.C2 => Color.C3
  | Color.Red, Color.C3 => Color.C4
  | Color.Red, Color.C4 => Color.Blue
  | Color.C2, Color.Red => Color.C2
  | Color.C3, Color.Red => Color.C3
  | Color.C4, Color.Red => Color.C4
  | Color.Blue, Color.Red => Color.Blue
  | _, Color.Blue => Color.Blue
  | _, _ => bitten  -- For all other cases, no color change

/-- A sequence of bites that transforms all chameleons to blue -/
def all_blue_sequence (n : ℕ) : List (Fin n × Fin n) → Prop := sorry

/-- The theorem stating that 5 is the smallest number of red chameleons that can guarantee becoming all blue -/
theorem smallest_all_blue_count :
  (∃ (seq : List (Fin 5 × Fin 5)), all_blue_sequence 5 seq) ∧
  (∀ k < 5, ¬∃ (seq : List (Fin k × Fin k)), all_blue_sequence k seq) :=
sorry

end smallest_all_blue_count_l1132_113250


namespace exists_probability_outside_range_l1132_113213

/-- Represents a packet of candies -/
structure Packet :=
  (total : ℕ)
  (blue : ℕ)
  (h : blue ≤ total)

/-- Represents a box containing two packets of candies -/
structure Box :=
  (packet1 : Packet)
  (packet2 : Packet)

/-- Calculates the probability of drawing a blue candy from the box -/
def blueProbability (box : Box) : ℚ :=
  (box.packet1.blue + box.packet2.blue : ℚ) / (box.packet1.total + box.packet2.total)

/-- Theorem stating that there exists a box configuration where the probability
    of drawing a blue candy is not between 3/8 and 2/5 -/
theorem exists_probability_outside_range :
  ∃ (box : Box), ¬(3/8 < blueProbability box ∧ blueProbability box < 2/5) :=
sorry

end exists_probability_outside_range_l1132_113213


namespace decimal_23_to_binary_l1132_113267

def decimal_to_binary (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec to_binary_helper (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else to_binary_helper (m / 2) ((m % 2) :: acc)
    to_binary_helper n []

theorem decimal_23_to_binary :
  decimal_to_binary 23 = [1, 0, 1, 1, 1] := by
  sorry

end decimal_23_to_binary_l1132_113267


namespace lion_king_star_wars_profit_ratio_l1132_113270

/-- The ratio of profits between two movies -/
def profit_ratio (cost1 revenue1 cost2 revenue2 : ℚ) : ℚ :=
  (revenue1 - cost1) / (revenue2 - cost2)

/-- Theorem: The ratio of The Lion King's profit to Star Wars' profit is 1:2 -/
theorem lion_king_star_wars_profit_ratio :
  profit_ratio 10 200 25 405 = 1/2 := by
sorry

end lion_king_star_wars_profit_ratio_l1132_113270


namespace inequality_proof_l1132_113206

theorem inequality_proof (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^3 + y^3 + z^3 = 1) :
  (x^2 / (1 - x^2)) + (y^2 / (1 - y^2)) + (z^2 / (1 - z^2)) ≥ (3 * Real.sqrt 3) / 2 := by
  sorry

end inequality_proof_l1132_113206


namespace max_roses_is_316_l1132_113240

/-- The price of an individual rose in cents -/
def individual_price : ℕ := 730

/-- The price of one dozen roses in cents -/
def dozen_price : ℕ := 3600

/-- The price of two dozen roses in cents -/
def two_dozen_price : ℕ := 5000

/-- The total budget in cents -/
def budget : ℕ := 68000

/-- The function to calculate the maximum number of roses that can be purchased -/
def max_roses : ℕ :=
  let two_dozen_sets := budget / two_dozen_price
  let remaining := budget % two_dozen_price
  let individual_roses := remaining / individual_price
  two_dozen_sets * 24 + individual_roses

/-- Theorem stating that the maximum number of roses that can be purchased is 316 -/
theorem max_roses_is_316 : max_roses = 316 := by
  sorry

end max_roses_is_316_l1132_113240


namespace tank_fill_problem_l1132_113268

theorem tank_fill_problem (tank_capacity : ℚ) (added_amount : ℚ) (final_fraction : ℚ) :
  tank_capacity = 54 →
  added_amount = 9 →
  final_fraction = 9/10 →
  (tank_capacity * final_fraction - added_amount) / tank_capacity = 7/10 := by
  sorry

end tank_fill_problem_l1132_113268


namespace base6_divisibility_by_11_l1132_113278

/-- Converts a base-6 number of the form 2dd5₆ to base 10 --/
def base6ToBase10 (d : Nat) : Nat :=
  2 * 6^3 + d * 6^2 + d * 6^1 + 5

/-- Checks if a number is divisible by 11 --/
def isDivisibleBy11 (n : Nat) : Prop :=
  n % 11 = 0

/-- Represents a base-6 digit --/
def isBase6Digit (d : Nat) : Prop :=
  d < 6

theorem base6_divisibility_by_11 :
  ∃ (d : Nat), isBase6Digit d ∧ isDivisibleBy11 (base6ToBase10 d) ↔ d = 4 := by
  sorry

end base6_divisibility_by_11_l1132_113278


namespace f_is_odd_and_satisfies_conditions_l1132_113259

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then 2^x + x - 2
  else if x = 0 then 0
  else -2^(-x) + x + 2

-- Theorem statement
theorem f_is_odd_and_satisfies_conditions :
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x > 0, f x = 2^x + x - 2) ∧
  (f 0 = 0) ∧
  (∀ x < 0, f x = -2^(-x) + x + 2) := by
  sorry

end f_is_odd_and_satisfies_conditions_l1132_113259


namespace triangle_theorem_l1132_113241

noncomputable section

open Real

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the given condition
def condition (t : Triangle) : Prop :=
  (t.c / (t.a + t.b)) + (sin t.A / (sin t.B + sin t.C)) = 1

-- Define the theorem
theorem triangle_theorem (t : Triangle) 
  (h1 : condition t) 
  (h2 : t.b = sqrt 2) : 
  t.B = π/3 ∧ (∀ (x y : ℝ), x^2 + y^2 ≤ 4) ∧ (∃ (x y : ℝ), x^2 + y^2 = 4) :=
sorry

end

end triangle_theorem_l1132_113241


namespace sqrt_equation_solution_l1132_113274

theorem sqrt_equation_solution (y : ℝ) :
  (y > 2) →
  (Real.sqrt (7 * y) / Real.sqrt (4 * (y - 2)) = 3) →
  y = 72 / 29 := by
sorry

end sqrt_equation_solution_l1132_113274


namespace equation_solutions_l1132_113242

theorem equation_solutions (m : ℕ+) :
  ∀ x y z : ℕ+, (x^2 + y^2)^m.val = (x * y)^z.val →
  ∃ k n : ℕ+, x = 2^k.val ∧ y = 2^k.val ∧ z = (1 + 2*k.val)*n.val ∧ m = 2*k.val*n.val :=
sorry

end equation_solutions_l1132_113242


namespace sqrt_three_times_five_to_fourth_l1132_113297

theorem sqrt_three_times_five_to_fourth (x : ℝ) : 
  x = Real.sqrt (5^4 + 5^4 + 5^4) → x = 75 := by
  sorry

end sqrt_three_times_five_to_fourth_l1132_113297


namespace zebras_total_games_l1132_113291

theorem zebras_total_games : 
  ∀ (initial_games : ℕ) (initial_wins : ℕ),
    initial_wins = (2 * initial_games / 5) →  -- 40% win rate initially
    ∀ (final_games : ℕ) (final_wins : ℕ),
      final_games = initial_games + 11 →  -- 8 won + 3 lost = 11 more games
      final_wins = initial_wins + 8 →     -- 8 more wins
      final_wins = (11 * final_games / 20) →  -- 55% win rate finally
      final_games = 24 := by
sorry

end zebras_total_games_l1132_113291


namespace correct_metal_ratio_l1132_113245

/-- Represents the ratio of two metals in an alloy -/
structure MetalRatio where
  a : ℚ
  b : ℚ

/-- Calculates the cost of an alloy given the ratio of metals and their individual costs -/
def alloyCost (ratio : MetalRatio) (costA costB : ℚ) : ℚ :=
  (ratio.a * costA + ratio.b * costB) / (ratio.a + ratio.b)

/-- Theorem stating the correct ratio of metals to achieve the desired alloy cost -/
theorem correct_metal_ratio :
  let desiredRatio : MetalRatio := ⟨3, 1⟩
  let costA : ℚ := 68
  let costB : ℚ := 96
  let desiredCost : ℚ := 75
  alloyCost desiredRatio costA costB = desiredCost := by sorry

end correct_metal_ratio_l1132_113245


namespace cos_150_degrees_l1132_113224

theorem cos_150_degrees : Real.cos (150 * π / 180) = -1/2 := by sorry

end cos_150_degrees_l1132_113224


namespace chord_intersection_probability_l1132_113293

/-- A circle with n evenly spaced points -/
structure CirclePoints where
  n : ℕ
  h : n ≥ 4

/-- Four distinct points selected from the circle -/
structure FourPoints (c : CirclePoints) where
  A : Fin c.n
  B : Fin c.n
  C : Fin c.n
  D : Fin c.n
  distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

/-- The probability that chord AB intersects chord CD -/
def intersectionProbability (c : CirclePoints) : ℚ :=
  1 / 3

/-- Theorem: The probability of chord AB intersecting chord CD is 1/3 -/
theorem chord_intersection_probability (c : CirclePoints) :
  intersectionProbability c = 1 / 3 := by
  sorry


end chord_intersection_probability_l1132_113293


namespace jenny_ate_65_chocolates_l1132_113256

/-- The number of chocolates Mike ate -/
def mike_chocolates : ℕ := 20

/-- The number of chocolates John ate -/
def john_chocolates : ℕ := mike_chocolates / 2

/-- The combined number of chocolates Mike and John ate -/
def combined_chocolates : ℕ := mike_chocolates + john_chocolates

/-- The number of chocolates Jenny ate -/
def jenny_chocolates : ℕ := 2 * combined_chocolates + 5

/-- Theorem stating that Jenny ate 65 chocolates -/
theorem jenny_ate_65_chocolates : jenny_chocolates = 65 := by
  sorry

end jenny_ate_65_chocolates_l1132_113256


namespace perpendicular_lines_a_value_l1132_113207

/-- Two lines in the xy-plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of perpendicular lines -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_lines_a_value :
  ∀ a : ℝ,
  let l1 : Line := ⟨2, a, -7⟩
  let l2 : Line := ⟨a - 3, 1, 4⟩
  perpendicular l1 l2 → a = 2 := by
sorry

end perpendicular_lines_a_value_l1132_113207


namespace max_d_is_three_l1132_113292

/-- Represents a 7-digit number of the form 5d5,22e1 -/
def SevenDigitNumber (d e : Nat) : Nat :=
  5000000 + d * 100000 + 500000 + 22000 + e * 10 + 1

/-- Checks if a number is divisible by 33 -/
def isDivisibleBy33 (n : Nat) : Prop :=
  n % 33 = 0

/-- Checks if d and e are single digits -/
def areSingleDigits (d e : Nat) : Prop :=
  d ≤ 9 ∧ e ≤ 9

/-- The main theorem stating that the maximum value of d is 3 -/
theorem max_d_is_three :
  ∃ (d e : Nat), areSingleDigits d e ∧ 
    isDivisibleBy33 (SevenDigitNumber d e) ∧
    d = 3 ∧
    ∀ (d' e' : Nat), areSingleDigits d' e' → 
      isDivisibleBy33 (SevenDigitNumber d' e') → 
      d' ≤ d :=
by sorry

end max_d_is_three_l1132_113292


namespace b_invested_after_six_months_l1132_113244

/-- Represents the investment scenario and calculates when B invested -/
def calculate_b_investment_time (a_investment : ℕ) (b_investment : ℕ) (total_profit : ℕ) (a_profit : ℕ) : ℕ :=
  let a_time := 12
  let b_time := 12 - (a_investment * a_time * total_profit) / (a_profit * (a_investment + b_investment))
  b_time

/-- Theorem stating that B invested 6 months after A, given the problem conditions -/
theorem b_invested_after_six_months :
  calculate_b_investment_time 300 200 100 75 = 6 := by
  sorry

end b_invested_after_six_months_l1132_113244


namespace final_number_after_ten_steps_l1132_113222

/-- Performs one step of the sequence operation -/
def step (n : ℕ) (i : ℕ) : ℕ :=
  if i % 2 = 0 then n * 3 else n / 4

/-- Performs n steps of the sequence operation -/
def iterate_steps (start : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => start
  | k + 1 => step (iterate_steps start k) k

theorem final_number_after_ten_steps :
  iterate_steps 800000 10 = 1518750 := by
  sorry

end final_number_after_ten_steps_l1132_113222


namespace rectangle_longest_side_l1132_113232

/-- A rectangle with perimeter 240 feet and area eight times its perimeter has its longest side equal to 101 feet. -/
theorem rectangle_longest_side : ∀ l w : ℝ,
  l > 0 → w > 0 →
  2 * (l + w) = 240 →
  l * w = 8 * 240 →
  max l w = 101 :=
by sorry

end rectangle_longest_side_l1132_113232


namespace sufficient_not_necessary_l1132_113252

theorem sufficient_not_necessary : 
  (∃ a : ℝ, a^2 > 16 ∧ a ≤ 4) ∧ 
  (∀ a : ℝ, a > 4 → a^2 > 16) := by
  sorry

end sufficient_not_necessary_l1132_113252


namespace cloth_cost_price_theorem_l1132_113243

/-- Represents the cost price of one meter of cloth given the selling conditions --/
def cost_price_per_meter (total_meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (selling_price - profit_per_meter * total_meters) / total_meters

/-- Theorem stating that under the given conditions, the cost price per meter is 88 --/
theorem cloth_cost_price_theorem (total_meters : ℕ) (selling_price : ℕ) (profit_per_meter : ℕ)
    (h1 : total_meters = 45)
    (h2 : selling_price = 4500)
    (h3 : profit_per_meter = 12) :
    cost_price_per_meter total_meters selling_price profit_per_meter = 88 := by
  sorry

end cloth_cost_price_theorem_l1132_113243


namespace constant_term_theorem_l1132_113205

theorem constant_term_theorem (m : ℝ) : 
  (∀ x, (x - m) * (x + 7) = x^2 + (7 - m) * x - 7 * m) →
  -7 * m = 14 →
  m = -2 := by
sorry

end constant_term_theorem_l1132_113205


namespace lena_calculation_l1132_113282

def double (n : ℕ) : ℕ := 2 * n

def roundToNearestTen (n : ℕ) : ℕ :=
  let remainder := n % 10
  if remainder < 5 then n - remainder else n + (10 - remainder)

theorem lena_calculation : roundToNearestTen (63 + double 29) = 120 := by
  sorry

end lena_calculation_l1132_113282


namespace inequality_range_difference_l1132_113285

-- Define g as a strictly increasing function
variable (g : ℝ → ℝ)

-- Define the property of g being strictly increasing
def StrictlyIncreasing (g : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → g x < g y

-- Define the theorem
theorem inequality_range_difference
  (h1 : StrictlyIncreasing g)
  (h2 : ∀ x, x ≥ 0 → g x ≠ 0)
  (h3 : ∃ a b, ∀ t, (g (2*t^2 + t + 5) < g (t^2 - 3*t + 2)) ↔ (b < t ∧ t < a)) :
  ∃ a b, (∀ t, (g (2*t^2 + t + 5) < g (t^2 - 3*t + 2)) ↔ (b < t ∧ t < a)) ∧ a - b = 2 :=
by sorry

end inequality_range_difference_l1132_113285


namespace total_hair_product_usage_l1132_113253

/-- Represents the daily usage of hair products and calculates the total usage over 14 days. -/
def HairProductUsage (S C H R : ℚ) : Prop :=
  S = 1 ∧
  C = 1/2 * S ∧
  H = 2/3 * S ∧
  R = 1/4 * C ∧
  S * 14 = 14 ∧
  C * 14 = 7 ∧
  H * 14 = 28/3 ∧
  R * 14 = 7/4

/-- Theorem stating the total usage of hair products over 14 days. -/
theorem total_hair_product_usage (S C H R : ℚ) :
  HairProductUsage S C H R →
  S * 14 = 14 ∧ C * 14 = 7 ∧ H * 14 = 28/3 ∧ R * 14 = 7/4 :=
by sorry

end total_hair_product_usage_l1132_113253


namespace number_of_persons_l1132_113208

theorem number_of_persons (total_amount : ℕ) (amount_per_person : ℕ) 
  (h1 : total_amount = 42900)
  (h2 : amount_per_person = 1950) :
  total_amount / amount_per_person = 22 := by
  sorry

end number_of_persons_l1132_113208


namespace matrix_power_difference_l1132_113272

theorem matrix_power_difference (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B = ![![2, 4], ![0, 1]]) : 
  B^20 - 3 • B^19 = ![![-1, 4], ![0, -2]] := by
  sorry

end matrix_power_difference_l1132_113272


namespace folk_song_competition_probability_l1132_113296

/-- The number of provinces in the competition -/
def num_provinces : ℕ := 6

/-- The number of singers per province -/
def singers_per_province : ℕ := 2

/-- The total number of singers in the competition -/
def total_singers : ℕ := num_provinces * singers_per_province

/-- The number of winners to be selected -/
def num_winners : ℕ := 4

/-- The probability of selecting 4 winners such that exactly two of them are from the same province -/
theorem folk_song_competition_probability :
  (num_provinces.choose 1 * singers_per_province.choose 2 * (total_singers - singers_per_province).choose 1 * (num_provinces - 1).choose 1) / total_singers.choose num_winners = 16 / 33 := by
  sorry

end folk_song_competition_probability_l1132_113296


namespace ceiling_floor_product_range_l1132_113284

theorem ceiling_floor_product_range (y : ℝ) :
  y < 0 → (Int.ceil y * Int.floor y = 72) → y ∈ Set.Icc (-9 : ℝ) (-8 : ℝ) := by
  sorry

end ceiling_floor_product_range_l1132_113284


namespace geometric_sequence_sum_l1132_113237

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) := ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_sum (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 + a 2 = 2 →
  a 4 + a 5 = 4 →
  a 10 + a 11 = 16 :=
by sorry

end geometric_sequence_sum_l1132_113237


namespace sum_of_matching_positions_is_322_l1132_113203

def array_size : Nat × Nat := (16, 10)

def esther_fill (r c : Nat) : Nat :=
  16 * (r - 1) + c

def frida_fill (r c : Nat) : Nat :=
  10 * (c - 1) + r

def is_same_position (r c : Nat) : Prop :=
  esther_fill r c = frida_fill r c

def sum_of_matching_positions : Nat :=
  (esther_fill 1 1) + (esther_fill 4 6) + (esther_fill 7 11) + (esther_fill 10 16)

theorem sum_of_matching_positions_is_322 :
  sum_of_matching_positions = 322 :=
sorry

end sum_of_matching_positions_is_322_l1132_113203


namespace no_solution_when_m_zero_infinite_solutions_when_m_neg_three_unique_solution_when_m_not_zero_and_not_neg_three_l1132_113228

-- Define the system of linear equations
def system (m x y : ℝ) : Prop :=
  m * x + y = -1 ∧ 3 * m * x - m * y = 2 * m + 3

-- Define the determinant of the coefficient matrix
def det_coeff (m : ℝ) : ℝ := -m * (m + 3)

-- Define the determinants for x and y
def det_x (m : ℝ) : ℝ := -m - 3
def det_y (m : ℝ) : ℝ := 2 * m * (m + 3)

-- Theorem for the case when m = 0
theorem no_solution_when_m_zero :
  ¬∃ x y : ℝ, system 0 x y :=
sorry

-- Theorem for the case when m = -3
theorem infinite_solutions_when_m_neg_three :
  ∃ x y : ℝ, system (-3) x y ∧ ∀ t : ℝ, system (-3) (x + t) (y - 3*t) :=
sorry

-- Theorem for the case when m ≠ 0 and m ≠ -3
theorem unique_solution_when_m_not_zero_and_not_neg_three (m : ℝ) (hm : m ≠ 0 ∧ m ≠ -3) :
  ∃! x y : ℝ, system m x y ∧ x = 1/m ∧ y = -2 :=
sorry

end no_solution_when_m_zero_infinite_solutions_when_m_neg_three_unique_solution_when_m_not_zero_and_not_neg_three_l1132_113228


namespace equation_solution_l1132_113294

theorem equation_solution :
  ∃ x : ℝ, x ≠ 0 ∧ (2 / x + 3 * ((4 / x) / (8 / x)) = 1.2) ∧ x = -20 / 3 := by
  sorry

end equation_solution_l1132_113294


namespace smallest_number_l1132_113276

def base_2_to_10 (n : ℕ) : ℕ := n

def base_4_to_10 (n : ℕ) : ℕ := n

def base_8_to_10 (n : ℕ) : ℕ := n

theorem smallest_number :
  let a := base_4_to_10 321
  let b := 58
  let c := base_2_to_10 111000
  let d := base_8_to_10 73
  c < a ∧ c < b ∧ c < d :=
by sorry

end smallest_number_l1132_113276


namespace arithmetic_sequence_a7_l1132_113286

/-- An arithmetic sequence with given conditions -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧
  a 4 = 4 ∧
  a 3 + a 8 = 5

/-- Theorem stating that a_7 = 1 for the given arithmetic sequence -/
theorem arithmetic_sequence_a7 (a : ℕ → ℚ) (h : ArithmeticSequence a) : a 7 = 1 := by
  sorry

end arithmetic_sequence_a7_l1132_113286


namespace can_finish_typing_l1132_113236

/-- Proves that given a passage of 300 characters and a typing speed of 52 characters per minute, 
    it is possible to finish typing the passage in 6 minutes. -/
theorem can_finish_typing (passage_length : ℕ) (typing_speed : ℕ) (time : ℕ) : 
  passage_length = 300 → 
  typing_speed = 52 → 
  time = 6 → 
  typing_speed * time ≥ passage_length := by
sorry

end can_finish_typing_l1132_113236


namespace M_squared_equals_36_50_times_144_36_and_sum_of_digits_75_l1132_113254

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Definition of M -/
def M : ℕ := sorry

theorem M_squared_equals_36_50_times_144_36_and_sum_of_digits_75 :
  M^2 = 36^50 * 144^36 ∧ sum_of_digits M = 75 := by
  sorry

end M_squared_equals_36_50_times_144_36_and_sum_of_digits_75_l1132_113254


namespace impossible_tiling_l1132_113238

/-- Represents a rectangular board -/
structure Board :=
  (rows : ℕ)
  (cols : ℕ)

/-- Represents a tile that can be placed on the board -/
inductive Tile
  | Domino    : Tile  -- 1 × 2 horizontal domino
  | Rectangle : Tile  -- 1 × 3 vertical rectangle

/-- Represents a tiling of the board -/
def Tiling := List (Tile × ℕ × ℕ)  -- List of (tile type, row, column)

/-- Check if a tiling is valid for the given board -/
def is_valid_tiling (board : Board) (tiling : Tiling) : Prop :=
  sorry

/-- The main theorem stating that it's impossible to tile the 2003 × 2003 board -/
theorem impossible_tiling :
  ∀ (tiling : Tiling), ¬(is_valid_tiling (Board.mk 2003 2003) tiling) :=
sorry

end impossible_tiling_l1132_113238


namespace volume_sphere_minus_cylinder_l1132_113219

/-- The volume of space inside a sphere and outside an inscribed right cylinder -/
theorem volume_sphere_minus_cylinder (r_sphere : ℝ) (r_cylinder : ℝ) : 
  r_sphere = 6 → r_cylinder = 4 → 
  ∃ V : ℝ, V = (288 - 64 * Real.sqrt 5) * Real.pi ∧
    V = (4 / 3 * Real.pi * r_sphere^3) - (Real.pi * r_cylinder^2 * Real.sqrt (r_sphere^2 - r_cylinder^2)) :=
by sorry

end volume_sphere_minus_cylinder_l1132_113219


namespace sum_difference_equals_210_l1132_113210

theorem sum_difference_equals_210 : 152 + 29 + 25 + 14 - 10 = 210 := by
  sorry

end sum_difference_equals_210_l1132_113210


namespace ball_hits_middle_pocket_l1132_113200

/-- Represents a rectangular billiard table -/
structure BilliardTable where
  p : ℕ
  q : ℕ
  p_odd : Odd p
  q_odd : Odd q

/-- Represents the trajectory of a ball on the billiard table -/
def ball_trajectory (table : BilliardTable) : ℕ → ℕ → Prop :=
  fun x y => y = x

/-- Represents a middle pocket on the long side of the table -/
def middle_pocket (table : BilliardTable) : ℕ → ℕ → Prop :=
  fun x y => (x = table.p / 2 ∧ (y = 0 ∨ y = 2 * table.q)) ∨ 
             (y = table.q ∧ (x = 0 ∨ x = table.p))

/-- The main theorem stating that the ball will hit a middle pocket -/
theorem ball_hits_middle_pocket (table : BilliardTable) :
  ∃ (x y : ℕ), ball_trajectory table x y ∧ middle_pocket table x y :=
sorry

end ball_hits_middle_pocket_l1132_113200


namespace inequality_proof_l1132_113226

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^4 + y^4 + z^2 ≥ x*y*z*Real.sqrt 8 := by
  sorry

end inequality_proof_l1132_113226


namespace first_term_of_sequence_l1132_113290

def fibonacci_like_sequence (a b : ℕ) : ℕ → ℕ
  | 0 => a
  | 1 => b
  | (n + 2) => fibonacci_like_sequence a b n + fibonacci_like_sequence a b (n + 1)

theorem first_term_of_sequence (a b : ℕ) :
  fibonacci_like_sequence a b 5 = 21 ∧
  fibonacci_like_sequence a b 6 = 34 ∧
  fibonacci_like_sequence a b 7 = 55 →
  a = 2 := by
sorry

end first_term_of_sequence_l1132_113290


namespace min_value_sum_of_fractions_l1132_113209

theorem min_value_sum_of_fractions (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / b + b / c + c / a + a / c ≥ 4 ∧
  (a / b + b / c + c / a + a / c = 4 ↔ a = b ∧ b = c) :=
by sorry

end min_value_sum_of_fractions_l1132_113209


namespace theater_seats_count_l1132_113275

/-- Represents a theater with a specific seating arrangement. -/
structure Theater where
  rows : ℕ
  first_row_seats : ℕ
  seat_increment : ℕ
  last_row_seats : ℕ

/-- Calculates the total number of seats in the theater. -/
def total_seats (t : Theater) : ℕ :=
  (t.rows * (2 * t.first_row_seats + (t.rows - 1) * t.seat_increment)) / 2

/-- Theorem stating that a theater with the given properties has 720 seats. -/
theorem theater_seats_count :
  ∀ (t : Theater),
    t.rows = 15 →
    t.first_row_seats = 20 →
    t.seat_increment = 4 →
    t.last_row_seats = 76 →
    total_seats t = 720 :=
by
  sorry


end theater_seats_count_l1132_113275


namespace joey_caught_one_kg_more_than_peter_l1132_113289

/-- Given three fishers Ali, Peter, and Joey, prove that Joey caught 1 kg more fish than Peter -/
theorem joey_caught_one_kg_more_than_peter 
  (total_catch : ℝ)
  (ali_catch : ℝ)
  (peter_catch : ℝ)
  (joey_catch : ℝ)
  (h1 : total_catch = 25)
  (h2 : ali_catch = 12)
  (h3 : ali_catch = 2 * peter_catch)
  (h4 : joey_catch = peter_catch + (joey_catch - peter_catch))
  (h5 : total_catch = ali_catch + peter_catch + joey_catch) :
  joey_catch - peter_catch = 1 := by
  sorry

end joey_caught_one_kg_more_than_peter_l1132_113289


namespace train_speed_problem_l1132_113295

/-- The speed of Train A in miles per hour -/
def speed_A : ℝ := 30

/-- The time difference between Train A and Train B's departure in hours -/
def time_diff : ℝ := 2

/-- The distance at which Train B overtakes Train A in miles -/
def overtake_distance : ℝ := 360

/-- The speed of Train B in miles per hour -/
def speed_B : ℝ := 42

theorem train_speed_problem :
  speed_A * (overtake_distance / speed_A) = 
  speed_B * (overtake_distance / speed_B - time_diff) ∧
  speed_B * time_diff + speed_A * time_diff = overtake_distance := by
  sorry

end train_speed_problem_l1132_113295


namespace money_distribution_l1132_113283

theorem money_distribution (a b c : ℝ) : 
  a + b + c = 360 ∧
  a = (1/3) * (b + c) ∧
  b = (2/7) * (a + c) ∧
  a > b
  →
  a - b = 10 := by
sorry

end money_distribution_l1132_113283


namespace fraction_equality_l1132_113257

theorem fraction_equality (a b : ℚ) (h : b ≠ 0) (h1 : a / b = 2 / 3) :
  (a - b) / b = -1 / 3 := by
  sorry

end fraction_equality_l1132_113257


namespace sum_of_absolute_roots_l1132_113202

theorem sum_of_absolute_roots (m : ℤ) (a b c d : ℤ) : 
  (∀ x : ℤ, x^4 - x^3 - 4023*x + m = 0 ↔ x = a ∨ x = b ∨ x = c ∨ x = d) →
  |a| + |b| + |c| + |d| = 621 := by
sorry

end sum_of_absolute_roots_l1132_113202


namespace quoted_price_calculation_l1132_113262

/-- Calculates the quoted price of shares given investment details -/
theorem quoted_price_calculation (investment : ℚ) (face_value : ℚ) (dividend_rate : ℚ) (annual_income : ℚ) : 
  investment = 4455 ∧ 
  face_value = 10 ∧ 
  dividend_rate = 12 / 100 ∧ 
  annual_income = 648 → 
  (investment / (annual_income / (dividend_rate * face_value))) = 33 / 4 :=
by sorry

end quoted_price_calculation_l1132_113262


namespace tan_alpha_2_implies_expression_zero_l1132_113231

theorem tan_alpha_2_implies_expression_zero (α : Real) (h : Real.tan α = 2) :
  2 * (Real.sin α)^2 - 3 * (Real.sin α) * (Real.cos α) - 2 * (Real.cos α)^2 = 0 := by
  sorry

end tan_alpha_2_implies_expression_zero_l1132_113231


namespace absolute_value_equation_roots_l1132_113212

theorem absolute_value_equation_roots (a : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ |x| = a * x - a) ∧ 
  (∀ x : ℝ, x < 0 → |x| ≠ a * x - a) → 
  a > 1 :=
sorry

end absolute_value_equation_roots_l1132_113212


namespace cone_volume_and_surface_area_l1132_113251

/-- Cone with given slant height and height -/
structure Cone where
  slant_height : ℝ
  height : ℝ

/-- Volume of a cone -/
def volume (c : Cone) : ℝ := sorry

/-- Surface area of a cone -/
def surface_area (c : Cone) : ℝ := sorry

/-- Theorem stating the volume and surface area of a specific cone -/
theorem cone_volume_and_surface_area :
  let c : Cone := { slant_height := 17, height := 15 }
  (volume c = 320 * Real.pi) ∧ (surface_area c = 200 * Real.pi) := by
  sorry

end cone_volume_and_surface_area_l1132_113251
