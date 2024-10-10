import Mathlib

namespace y_value_at_27_l3411_341138

-- Define the relation between y and x
def y (k : ℝ) (x : ℝ) : ℝ := k * x^(1/3)

-- State the theorem
theorem y_value_at_27 (k : ℝ) :
  y k 8 = 4 → y k 27 = 6 := by
  sorry

end y_value_at_27_l3411_341138


namespace set_union_equality_implies_m_range_l3411_341158

theorem set_union_equality_implies_m_range (m : ℝ) : 
  let A : Set ℝ := {x | x^2 - 3*x - 10 ≤ 0}
  let B : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}
  A ∪ B = A → m ∈ Set.Icc (-3) 3 := by
  sorry

end set_union_equality_implies_m_range_l3411_341158


namespace bike_distance_proof_l3411_341139

/-- Calculates the distance traveled given speed and time -/
def distance (speed : ℝ) (time : ℝ) : ℝ := speed * time

/-- Proves that a bike traveling at 8 m/s for 6 seconds covers 48 meters -/
theorem bike_distance_proof :
  let speed : ℝ := 8
  let time : ℝ := 6
  distance speed time = 48 := by
  sorry

end bike_distance_proof_l3411_341139


namespace other_communities_count_l3411_341123

theorem other_communities_count (total : ℕ) (muslim_percent : ℚ) (hindu_percent : ℚ) (sikh_percent : ℚ)
  (h_total : total = 850)
  (h_muslim : muslim_percent = 44 / 100)
  (h_hindu : hindu_percent = 32 / 100)
  (h_sikh : sikh_percent = 10 / 100) :
  ↑total * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 119 := by
  sorry

end other_communities_count_l3411_341123


namespace prime_triplet_theorem_l3411_341108

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_geometric_progression (a b c : ℕ) : Prop := (b + 1)^2 = (a + 1) * (c + 1)

def valid_prime_triplet (a b c : ℕ) : Prop :=
  is_prime a ∧ is_prime b ∧ is_prime c ∧
  a < b ∧ b < c ∧ c < 100 ∧
  is_geometric_progression a b c

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(2, 5, 11), (2, 11, 47), (5, 11, 23), (5, 17, 53), (7, 11, 17), (7, 23, 71),
   (11, 23, 47), (17, 23, 31), (17, 41, 97), (31, 47, 71), (71, 83, 97)}

theorem prime_triplet_theorem :
  {x : ℕ × ℕ × ℕ | valid_prime_triplet x.1 x.2.1 x.2.2} = solution_set :=
by sorry

end prime_triplet_theorem_l3411_341108


namespace least_positive_difference_l3411_341174

def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ := a₁ * r ^ (n - 1)

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1) * d

def sequence_C (n : ℕ) : ℝ := geometric_sequence 3 3 n

def sequence_D (n : ℕ) : ℝ := arithmetic_sequence 10 20 n

def valid_C (n : ℕ) : Prop := sequence_C n ≤ 200

def valid_D (n : ℕ) : Prop := sequence_D n ≤ 200

theorem least_positive_difference :
  ∃ (m n : ℕ) (h₁ : valid_C m) (h₂ : valid_D n),
    ∀ (p q : ℕ) (h₃ : valid_C p) (h₄ : valid_D q),
      |sequence_C m - sequence_D n| ≤ |sequence_C p - sequence_D q| ∧
      |sequence_C m - sequence_D n| > 0 ∧
      |sequence_C m - sequence_D n| = 9 :=
sorry

end least_positive_difference_l3411_341174


namespace adult_ticket_cost_l3411_341184

theorem adult_ticket_cost (child_ticket_cost : ℝ) (total_tickets : ℕ) (total_revenue : ℝ) (child_tickets : ℕ) : ℝ :=
  let adult_tickets := total_tickets - child_tickets
  let child_revenue := child_ticket_cost * child_tickets
  let adult_revenue := total_revenue - child_revenue
  let adult_ticket_cost := adult_revenue / adult_tickets
  by
    have h1 : child_ticket_cost = 4.5 := by sorry
    have h2 : total_tickets = 400 := by sorry
    have h3 : total_revenue = 2100 := by sorry
    have h4 : child_tickets = 200 := by sorry
    sorry

end adult_ticket_cost_l3411_341184


namespace sphere_cylinder_volume_difference_l3411_341102

theorem sphere_cylinder_volume_difference (r_sphere r_cylinder : ℝ) 
  (h_sphere : r_sphere = 7)
  (h_cylinder : r_cylinder = 4) :
  let h_cylinder := Real.sqrt (r_sphere^2 - r_cylinder^2)
  let v_sphere := (4/3) * π * r_sphere^3
  let v_cylinder := π * r_cylinder^2 * h_cylinder
  v_sphere - v_cylinder = ((1372/3) - 16 * Real.sqrt 132) * π :=
by sorry

end sphere_cylinder_volume_difference_l3411_341102


namespace unique_solution_l3411_341179

-- Define the type for digits from 2 to 9
def Digit := Fin 8

-- Define a function to map letters to digits
def LetterToDigit := Char → Digit

-- Define the letters used in the problem
def Letters : List Char := ['N', 'I', 'E', 'O', 'T', 'W', 'S', 'X']

-- Define the function to convert a word to a number
def wordToNumber (f : LetterToDigit) (word : List Char) : Nat :=
  word.foldl (fun acc c => 10 * acc + (f c).val.succ.succ) 0

-- State the theorem
theorem unique_solution :
  ∃! f : LetterToDigit,
    (∀ c₁ c₂ : Char, c₁ ∈ Letters → c₂ ∈ Letters → c₁ ≠ c₂ → f c₁ ≠ f c₂) ∧
    (wordToNumber f ['O', 'N', 'E']) + 
    (wordToNumber f ['T', 'W', 'O']) + 
    (wordToNumber f ['S', 'I', 'X']) = 
    (wordToNumber f ['N', 'I', 'N', 'E']) ∧
    (wordToNumber f ['N', 'I', 'N', 'E']) = 2526 :=
  sorry

end unique_solution_l3411_341179


namespace extra_money_spent_theorem_l3411_341129

/-- Represents the price of radishes and pork ribs last month and this month --/
structure PriceData where
  radish_last : ℝ
  pork_last : ℝ
  radish_this : ℝ
  pork_this : ℝ

/-- Calculates the extra money spent given the price data and quantities --/
def extra_money_spent (p : PriceData) (radish_qty : ℝ) (pork_qty : ℝ) : ℝ :=
  radish_qty * (p.radish_this - p.radish_last) + pork_qty * (p.pork_this - p.pork_last)

/-- Theorem stating the extra money spent on radishes and pork ribs --/
theorem extra_money_spent_theorem (a : ℝ) :
  let p : PriceData := {
    radish_last := a,
    pork_last := 7 * a + 2,
    radish_this := 1.25 * a,
    pork_this := 1.2 * (7 * a + 2)
  }
  extra_money_spent p 3 2 = 3.55 * a + 0.8 := by
  sorry

#check extra_money_spent_theorem

end extra_money_spent_theorem_l3411_341129


namespace triangle_area_from_squares_l3411_341130

/-- Given four squares with areas 256, 64, 225, and 49, prove that the area of the triangle formed by three of these squares is 60 -/
theorem triangle_area_from_squares (s₁ s₂ s₃ s₄ : ℝ) 
  (h₁ : s₁ = 256) (h₂ : s₂ = 64) (h₃ : s₃ = 225) (h₄ : s₄ = 49) : 
  ∃ (a b c : ℝ), a^2 + b^2 = c^2 ∧ a^2 = s₃ ∧ b^2 = s₂ ∧ c^2 = s₁ ∧ (1/2 * a * b = 60) :=
by sorry

end triangle_area_from_squares_l3411_341130


namespace exists_quadratic_through_point_l3411_341103

-- Define a quadratic function
def quadratic_function (a b c : ℝ) : ℝ → ℝ := λ x => a * x^2 + b * x + c

-- State the theorem
theorem exists_quadratic_through_point :
  ∃ (a b c : ℝ), a > 0 ∧ quadratic_function a b c 0 = 1 := by
  sorry

end exists_quadratic_through_point_l3411_341103


namespace fraction_ordering_l3411_341111

def t₁ : ℚ := (100^100 + 1) / (100^90 + 1)
def t₂ : ℚ := (100^99 + 1) / (100^89 + 1)
def t₃ : ℚ := (100^101 + 1) / (100^91 + 1)
def t₄ : ℚ := (101^101 + 1) / (101^91 + 1)
def t₅ : ℚ := (101^100 + 1) / (101^90 + 1)
def t₆ : ℚ := (99^99 + 1) / (99^89 + 1)
def t₇ : ℚ := (99^100 + 1) / (99^90 + 1)

theorem fraction_ordering :
  t₆ < t₇ ∧ t₇ < t₂ ∧ t₂ < t₁ ∧ t₁ < t₃ ∧ t₃ < t₅ ∧ t₅ < t₄ :=
by sorry

end fraction_ordering_l3411_341111


namespace parallel_lines_m_value_l3411_341165

/-- Two lines are parallel if and only if they have the same slope -/
axiom parallel_lines_same_slope {m₁ m₂ b₁ b₂ : ℝ} :
  (∀ x y : ℝ, y = m₁ * x + b₁ ↔ y = m₂ * x + b₂) ↔ m₁ = m₂

/-- The equation of line l₁ -/
def line_l₁ (m : ℝ) (x y : ℝ) : Prop := m * x + y - 2 = 0

/-- The equation of line l₂ -/
def line_l₂ (x y : ℝ) : Prop := y = 2 * x - 1

theorem parallel_lines_m_value :
  (∀ x y : ℝ, line_l₁ m x y ↔ line_l₂ x y) → m = -2 :=
by sorry

end parallel_lines_m_value_l3411_341165


namespace margo_age_in_three_years_l3411_341161

/-- Margo's age in three years given Benjie's current age and their age difference -/
def margos_future_age (benjies_age : ℕ) (age_difference : ℕ) : ℕ :=
  (benjies_age - age_difference) + 3

theorem margo_age_in_three_years :
  margos_future_age 6 5 = 4 := by
  sorry

end margo_age_in_three_years_l3411_341161


namespace consecutive_sum_18_l3411_341160

def consecutive_sum (start : ℕ) (length : ℕ) : ℕ :=
  (length * (2 * start + length - 1)) / 2

theorem consecutive_sum_18 :
  ∃! (start length : ℕ), 2 ≤ length ∧ consecutive_sum start length = 18 :=
sorry

end consecutive_sum_18_l3411_341160


namespace five_pointed_star_angle_sum_l3411_341180

/-- An irregular five-pointed star -/
structure FivePointedStar where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  D : ℝ × ℝ
  E : ℝ × ℝ

/-- The angles at the vertices of a five-pointed star -/
structure StarAngles where
  α : ℝ
  β : ℝ
  γ : ℝ
  δ : ℝ
  ε : ℝ

/-- The sum of angles in a five-pointed star is 180 degrees -/
theorem five_pointed_star_angle_sum (star : FivePointedStar) (angles : StarAngles) :
  angles.α + angles.β + angles.γ + angles.δ + angles.ε = 180 := by
  sorry

#check five_pointed_star_angle_sum

end five_pointed_star_angle_sum_l3411_341180


namespace sine_equality_implies_equal_arguments_l3411_341186

theorem sine_equality_implies_equal_arguments
  (α β γ τ : ℝ)
  (h_pos : α > 0 ∧ β > 0 ∧ γ > 0 ∧ τ > 0)
  (h_eq : ∀ x : ℝ, Real.sin (α * x) + Real.sin (β * x) = Real.sin (γ * x) + Real.sin (τ * x)) :
  α = γ ∨ α = τ :=
sorry

end sine_equality_implies_equal_arguments_l3411_341186


namespace tan_product_pi_ninths_l3411_341188

theorem tan_product_pi_ninths : 
  Real.tan (π / 9) * Real.tan (2 * π / 9) * Real.tan (4 * π / 9) = 3 := by sorry

end tan_product_pi_ninths_l3411_341188


namespace no_rain_probability_l3411_341167

theorem no_rain_probability (p : ℚ) (h : p = 2/3) :
  (1 - p)^5 = 1/243 := by
  sorry

end no_rain_probability_l3411_341167


namespace even_pairs_ge_odd_pairs_l3411_341121

/-- A sequence of binary digits (0 or 1) -/
def BinarySequence := List Nat

/-- Count the number of (1,0) pairs with even number of digits between them -/
def countEvenPairs (seq : BinarySequence) : Nat :=
  sorry

/-- Count the number of (1,0) pairs with odd number of digits between them -/
def countOddPairs (seq : BinarySequence) : Nat :=
  sorry

/-- The main theorem: For any binary sequence, the number of (1,0) pairs
    with even number of digits between is greater than or equal to
    the number of (1,0) pairs with odd number of digits between -/
theorem even_pairs_ge_odd_pairs (seq : BinarySequence) :
  countEvenPairs seq ≥ countOddPairs seq :=
sorry

end even_pairs_ge_odd_pairs_l3411_341121


namespace class_overlap_difference_l3411_341156

theorem class_overlap_difference (total students_geometry students_biology : ℕ) 
  (h1 : total = 232)
  (h2 : students_geometry = 144)
  (h3 : students_biology = 119) :
  (min students_geometry students_biology) - 
  (students_geometry + students_biology - total) = 88 :=
by sorry

end class_overlap_difference_l3411_341156


namespace mice_elimination_time_l3411_341125

/-- Represents the rate at which cats hunt mice -/
def hunting_rate : ℝ := 0.1

/-- Represents the total amount of work to eliminate all mice -/
def total_work : ℝ := 1

/-- Represents the number of days taken by initial cats -/
def initial_days : ℕ := 5

/-- Represents the initial number of cats -/
def initial_cats : ℕ := 2

/-- Represents the final number of cats -/
def final_cats : ℕ := 5

theorem mice_elimination_time :
  let initial_work := hunting_rate * initial_cats * initial_days
  let remaining_work := total_work - initial_work
  let final_rate := hunting_rate * final_cats
  initial_days + (remaining_work / final_rate) = 7 := by sorry

end mice_elimination_time_l3411_341125


namespace circle_sequence_circumference_sum_l3411_341113

theorem circle_sequence_circumference_sum (r₁ r₂ r₃ r₄ : ℝ) : 
  r₁ = 1 →                           -- radius of first circle is 1
  r₁ < r₂ ∧ r₂ < r₃ ∧ r₃ < r₄ →       -- radii are increasing
  r₂ / r₁ = r₃ / r₂ ∧ r₃ / r₂ = r₄ / r₃ →  -- circles form a geometric progression
  r₄^2 * Real.pi = 64 * Real.pi →    -- area of fourth circle is 64π
  2 * Real.pi * r₂ + 2 * Real.pi * r₃ = 12 * Real.pi := by
sorry

end circle_sequence_circumference_sum_l3411_341113


namespace absolute_value_inequality_l3411_341176

theorem absolute_value_inequality (x : ℝ) : 
  abs (x - 3) + abs (x - 5) ≥ 4 ↔ x ≥ 6 ∨ x ≤ 2 := by
  sorry

end absolute_value_inequality_l3411_341176


namespace function_characterization_l3411_341119

/-- A function satisfying the given functional equation -/
def SatisfiesEquation (f : ℝ → ℝ) : Prop :=
  f 0 ≠ 0 ∧
  ∀ x y : ℝ, f (x + y)^2 = 2 * f x * f y + max (f (x^2) + f (y^2)) (f (x^2 + y^2))

/-- The theorem stating that any function satisfying the equation must be either constant -1 or x - 1 -/
theorem function_characterization (f : ℝ → ℝ) (h : SatisfiesEquation f) :
  (∀ x, f x = -1) ∨ (∀ x, f x = x - 1) := by sorry

end function_characterization_l3411_341119


namespace supermarket_spending_l3411_341134

theorem supermarket_spending (total : ℚ) : 
  (1/4 : ℚ) * total + (1/3 : ℚ) * total + (1/6 : ℚ) * total + 6 = total →
  total = 24 := by
  sorry

end supermarket_spending_l3411_341134


namespace accessory_percentage_l3411_341149

def computer_cost : ℝ := 3000
def initial_money : ℝ := 3 * computer_cost
def money_left : ℝ := 2700

theorem accessory_percentage :
  let total_spent := initial_money - money_left
  let accessory_cost := total_spent - computer_cost
  (accessory_cost / computer_cost) * 100 = 110 := by sorry

end accessory_percentage_l3411_341149


namespace freds_shopping_cost_l3411_341166

/-- Calculates the total cost of Fred's shopping trip --/
def calculate_shopping_cost (orange_price : ℚ) (orange_quantity : ℕ)
                            (cereal_price : ℚ) (cereal_quantity : ℕ)
                            (bread_price : ℚ) (bread_quantity : ℕ)
                            (cookie_price : ℚ) (cookie_quantity : ℕ)
                            (bakery_discount_threshold : ℚ)
                            (bakery_discount_rate : ℚ)
                            (coupon_threshold : ℚ)
                            (coupon_value : ℚ) : ℚ :=
  let orange_total := orange_price * orange_quantity
  let cereal_total := cereal_price * cereal_quantity
  let bread_total := bread_price * bread_quantity
  let cookie_total := cookie_price * cookie_quantity
  let bakery_total := bread_total + cookie_total
  let total_before_discounts := orange_total + cereal_total + bakery_total
  let bakery_discount := if bakery_total > bakery_discount_threshold then bakery_total * bakery_discount_rate else 0
  let total_after_bakery_discount := total_before_discounts - bakery_discount
  let final_total := if total_before_discounts ≥ coupon_threshold then total_after_bakery_discount - coupon_value else total_after_bakery_discount
  final_total

/-- Theorem stating that Fred's shopping cost is $30.5 --/
theorem freds_shopping_cost :
  calculate_shopping_cost 2 4 4 3 3 3 6 1 15 (1/10) 30 3 = 30.5 := by
  sorry

end freds_shopping_cost_l3411_341166


namespace transform_invariant_l3411_341198

def initial_point : ℝ × ℝ × ℝ := (1, 1, 1)

def rotate_z_90 (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-y, x, z)

def reflect_yz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (-x, y, z)

def reflect_xz (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, -y, z)

def reflect_xy (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := p
  (x, y, -z)

def transform (p : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  p
  |> rotate_z_90
  |> reflect_yz
  |> reflect_xz
  |> rotate_z_90
  |> reflect_xy
  |> reflect_xz

theorem transform_invariant : transform initial_point = initial_point := by
  sorry

end transform_invariant_l3411_341198


namespace equidistant_centers_l3411_341100

-- Define the structure for a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the structure for a triangle
structure Triangle where
  A : Point2D
  B : Point2D
  C : Point2D

-- Define the structure for a circle
structure Circle where
  center : Point2D
  radius : ℝ

def is_right_triangle (t : Triangle) : Prop := sorry

def altitude_to_hypotenuse (t : Triangle) : Point2D := sorry

def inscribed_circle (t : Triangle) : Circle := sorry

def touch_point_on_hypotenuse (c : Circle) (t : Triangle) : Point2D := sorry

def distance (p1 p2 : Point2D) : ℝ := sorry

theorem equidistant_centers (ABC : Triangle) (H₃ : Point2D) :
  is_right_triangle ABC →
  H₃ = altitude_to_hypotenuse ABC →
  let O := (inscribed_circle ABC).center
  let O₁ := (inscribed_circle ⟨ABC.A, ABC.C, H₃⟩).center
  let O₂ := (inscribed_circle ⟨ABC.B, ABC.C, H₃⟩).center
  let T := touch_point_on_hypotenuse (inscribed_circle ABC) ABC
  distance O T = distance O₁ T ∧ distance O T = distance O₂ T :=
by sorry

end equidistant_centers_l3411_341100


namespace circle_point_x_coordinate_l3411_341126

theorem circle_point_x_coordinate 
  (x : ℝ) 
  (h1 : (x - 6)^2 + 10^2 = 12^2) : 
  x = 6 + 2 * Real.sqrt 11 ∨ x = 6 - 2 * Real.sqrt 11 := by
  sorry


end circle_point_x_coordinate_l3411_341126


namespace candy_store_revenue_l3411_341181

/-- Represents the revenue calculation for a candy store sale --/
theorem candy_store_revenue : 
  let fudge_pounds : ℝ := 37
  let fudge_price : ℝ := 2.5
  let truffle_count : ℝ := 82
  let truffle_price : ℝ := 1.5
  let pretzel_count : ℝ := 48
  let pretzel_price : ℝ := 2
  let fudge_discount : ℝ := 0.1
  let sales_tax : ℝ := 0.05

  let fudge_revenue := fudge_pounds * fudge_price
  let truffle_revenue := truffle_count * truffle_price
  let pretzel_revenue := pretzel_count * pretzel_price

  let total_before_discount := fudge_revenue + truffle_revenue + pretzel_revenue
  let fudge_discount_amount := fudge_revenue * fudge_discount
  let total_after_discount := total_before_discount - fudge_discount_amount
  let tax_amount := total_after_discount * sales_tax
  let final_revenue := total_after_discount + tax_amount

  final_revenue = 317.36
  := by sorry


end candy_store_revenue_l3411_341181


namespace odd_function_property_l3411_341196

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : f (-3) = -2) :
  f 3 + f 0 = 2 := by
  sorry

end odd_function_property_l3411_341196


namespace plums_picked_total_l3411_341159

/-- The number of plums Alyssa picked -/
def alyssas_plums : ℕ := 17

/-- The number of plums Jason picked -/
def jasons_plums : ℕ := 10

/-- The total number of plums picked -/
def total_plums : ℕ := alyssas_plums + jasons_plums

theorem plums_picked_total :
  total_plums = 27 := by sorry

end plums_picked_total_l3411_341159


namespace greatest_negative_value_x_minus_y_l3411_341193

theorem greatest_negative_value_x_minus_y :
  ∃ (x y : ℝ), 
    (Real.sin x + Real.sin y) * (Real.cos x - Real.cos y) = 1/2 + Real.sin (x - y) * Real.cos (x + y) ∧
    x - y = -π/6 ∧
    ∀ (a b : ℝ), 
      (Real.sin a + Real.sin b) * (Real.cos a - Real.cos b) = 1/2 + Real.sin (a - b) * Real.cos (a + b) →
      a - b < 0 →
      a - b ≤ -π/6 :=
by sorry

end greatest_negative_value_x_minus_y_l3411_341193


namespace container_capacity_l3411_341104

theorem container_capacity (C : ℝ) 
  (h1 : 0.3 * C + 36 = 0.75 * C) : C = 80 := by
  sorry

end container_capacity_l3411_341104


namespace arithmetic_sequence_problem_l3411_341143

/-- An arithmetic sequence is a sequence where the difference between each consecutive term is constant. -/
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence a with a₁ = 3 and a₃ = 5, prove that a₅ = 7 -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) 
  (h_arith : is_arithmetic_sequence a) 
  (h_a1 : a 1 = 3) 
  (h_a3 : a 3 = 5) : 
  a 5 = 7 := by
sorry

end arithmetic_sequence_problem_l3411_341143


namespace total_chips_bags_l3411_341182

theorem total_chips_bags (total_bags : ℕ) (doritos_bags : ℕ) : 
  (4 * doritos_bags = total_bags) →  -- One quarter of the bags are Doritos
  (4 * 5 = doritos_bags) →           -- Doritos bags can be split into 4 equal piles with 5 bags in each
  total_bags = 80 :=                 -- Prove that the total number of bags is 80
by
  sorry

end total_chips_bags_l3411_341182


namespace cistern_wet_surface_area_l3411_341105

/-- Calculates the total wet surface area of a rectangular cistern -/
def wetSurfaceArea (length width depth : ℝ) : ℝ :=
  length * width + 2 * (length * depth + width * depth)

/-- Theorem stating the total wet surface area of the given cistern -/
theorem cistern_wet_surface_area :
  let length : ℝ := 6
  let width : ℝ := 4
  let depth : ℝ := 1.25
  wetSurfaceArea length width depth = 49 := by
  sorry

end cistern_wet_surface_area_l3411_341105


namespace second_part_speed_l3411_341122

/-- Proves that given a total distance of 20 miles, where the first 10 miles are traveled at 12 miles per hour,
    and the average speed for the entire trip is 10.909090909090908 miles per hour,
    the speed for the second part of the trip is 10 miles per hour. -/
theorem second_part_speed
  (total_distance : ℝ)
  (first_part_distance : ℝ)
  (first_part_speed : ℝ)
  (average_speed : ℝ)
  (h1 : total_distance = 20)
  (h2 : first_part_distance = 10)
  (h3 : first_part_speed = 12)
  (h4 : average_speed = 10.909090909090908)
  : ∃ (second_part_speed : ℝ),
    second_part_speed = 10 ∧
    average_speed = (first_part_distance / first_part_speed + (total_distance - first_part_distance) / second_part_speed) / (total_distance / average_speed) :=
by
  sorry

end second_part_speed_l3411_341122


namespace no_integer_solution_l3411_341147

theorem no_integer_solution : ¬ ∃ (a b c d : ℤ), 
  (a * 19^3 + b * 19^2 + c * 19 + d = 1) ∧ 
  (a * 62^3 + b * 62^2 + c * 62 + d = 2) := by
sorry

end no_integer_solution_l3411_341147


namespace cube_sum_equals_407_l3411_341171

theorem cube_sum_equals_407 (x y : ℝ) (h1 : x + y = 11) (h2 : x^2 * y = 36) :
  x^3 + y^3 = 407 := by
sorry

end cube_sum_equals_407_l3411_341171


namespace basketball_game_theorem_l3411_341152

/-- Represents the score of a team for each quarter -/
structure GameScore :=
  (q1 q2 q3 q4 : ℚ)

/-- Checks if a GameScore forms a geometric sequence with ratio 3 -/
def isGeometricSequence (score : GameScore) : Prop :=
  score.q2 = 3 * score.q1 ∧ score.q3 = 3 * score.q2 ∧ score.q4 = 3 * score.q3

/-- Checks if a GameScore forms an arithmetic sequence with difference 12 -/
def isArithmeticSequence (score : GameScore) : Prop :=
  score.q2 = score.q1 + 12 ∧ score.q3 = score.q2 + 12 ∧ score.q4 = score.q3 + 12

/-- Calculates the total score for a GameScore -/
def totalScore (score : GameScore) : ℚ :=
  score.q1 + score.q2 + score.q3 + score.q4

/-- Calculates the first half score for a GameScore -/
def firstHalfScore (score : GameScore) : ℚ :=
  score.q1 + score.q2

theorem basketball_game_theorem (eagles lions : GameScore) : 
  eagles.q1 = lions.q1 →  -- Tied at the end of first quarter
  isGeometricSequence eagles →
  isArithmeticSequence lions →
  totalScore eagles = totalScore lions + 3 →  -- Eagles won by 3 points
  totalScore eagles ≤ 120 →
  totalScore lions ≤ 120 →
  firstHalfScore eagles + firstHalfScore lions = 15 :=
by sorry

end basketball_game_theorem_l3411_341152


namespace rice_sales_problem_l3411_341177

/-- Represents the daily rice sales function -/
structure RiceSales where
  k : ℝ
  b : ℝ
  y : ℝ → ℝ
  h1 : ∀ x, y x = k * x + b
  h2 : y 5 = 950
  h3 : y 6 = 900

/-- Calculates the profit for a given price -/
def profit (price : ℝ) (sales : ℝ) : ℝ := (price - 4) * sales

theorem rice_sales_problem (rs : RiceSales) :
  (rs.y = λ x => -50 * x + 1200) ∧
  (∃ x ∈ Set.Icc 4 7, profit x (rs.y x) = 1800 ∧ x = 6) ∧
  (∀ x ∈ Set.Icc 4 7, profit x (rs.y x) ≤ 2550) ∧
  (profit 7 (rs.y 7) = 2550) := by
  sorry

end rice_sales_problem_l3411_341177


namespace simplify_fraction_l3411_341118

theorem simplify_fraction : 18 * (8 / 12) * (1 / 27) = 4 / 9 := by sorry

end simplify_fraction_l3411_341118


namespace intersection_complement_equality_l3411_341190

def U : Set ℕ := {1,2,3,4,5,6,7}
def A : Set ℕ := {2,4,6}
def B : Set ℕ := {1,3,5,7}

theorem intersection_complement_equality : A ∩ (U \ B) = {2,4,6} := by
  sorry

end intersection_complement_equality_l3411_341190


namespace central_angle_regular_hexagon_l3411_341148

/-- The central angle of a regular hexagon is 60 degrees. -/
theorem central_angle_regular_hexagon :
  ∀ (full_circle_degrees : ℝ) (num_sides : ℕ),
    full_circle_degrees = 360 →
    num_sides = 6 →
    full_circle_degrees / num_sides = 60 := by
  sorry

end central_angle_regular_hexagon_l3411_341148


namespace perpendicular_a_parallel_distance_l3411_341124

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := 2 * a * x + y - 1 = 0
def l₂ (a x y : ℝ) : Prop := a * x + (a - 1) * y + 1 = 0

-- Define perpendicularity of lines
def perpendicular (a : ℝ) : Prop := 
  ∃ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ ∧ (x₁ - x₂) * (y₁ - y₂) = -1

-- Define parallelism of lines
def parallel (a : ℝ) : Prop := 
  ∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ → 2 * a * (a - 1) = a

-- Theorem for perpendicular case
theorem perpendicular_a : ∀ a : ℝ, perpendicular a → a = -1 ∨ a = 1/2 :=
sorry

-- Theorem for parallel case
theorem parallel_distance : ∀ a : ℝ, parallel a → a ≠ 1 → 
  ∃ d : ℝ, d = (3 * Real.sqrt 10) / 10 ∧ 
  (∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ → 
    ((x₁ - x₂)^2 + (y₁ - y₂)^2 = d^2)) :=
sorry

end perpendicular_a_parallel_distance_l3411_341124


namespace students_walking_home_l3411_341168

theorem students_walking_home (bus car bicycle skateboard : ℚ) 
  (h1 : bus = 3/8)
  (h2 : car = 2/5)
  (h3 : bicycle = 1/8)
  (h4 : skateboard = 5/100)
  : 1 - (bus + car + bicycle + skateboard) = 1/20 := by
  sorry

end students_walking_home_l3411_341168


namespace A_intersect_B_l3411_341155

def A : Set ℕ := {2, 3, 4}
def B : Set ℕ := {0, 1, 2}

theorem A_intersect_B : A ∩ B = {2} := by sorry

end A_intersect_B_l3411_341155


namespace simplify_polynomial_l3411_341136

theorem simplify_polynomial (x : ℝ) : 
  (x - 2)^4 + 4*(x - 2)^3 + 6*(x - 2)^2 + 4*(x - 2) + 1 = (x - 1)^4 := by
  sorry

end simplify_polynomial_l3411_341136


namespace percentage_of_absent_students_l3411_341114

theorem percentage_of_absent_students (total : ℕ) (present : ℕ) : 
  total = 50 → present = 44 → (total - present) * 100 / total = 12 := by
sorry

end percentage_of_absent_students_l3411_341114


namespace calculation_proof_l3411_341131

theorem calculation_proof : (2.5 * (30.1 + 0.5)) / 1.5 = 51 := by
  sorry

end calculation_proof_l3411_341131


namespace complex_root_quadratic_l3411_341146

theorem complex_root_quadratic (a : ℝ) : 
  (∃ x : ℂ, x^2 - 2*a*x + a^2 - 4*a + 6 = 0) ∧ 
  (Complex.I^2 = -1) ∧
  ((1 : ℂ) + Complex.I * Real.sqrt 2)^2 - 2*a*((1 : ℂ) + Complex.I * Real.sqrt 2) + a^2 - 4*a + 6 = 0
  → a = 1 := by
sorry

end complex_root_quadratic_l3411_341146


namespace f_derivative_at_2_l3411_341187

def f (x : ℝ) : ℝ := x^3 + 4*x - 5

theorem f_derivative_at_2 : (deriv f) 2 = 16 := by sorry

end f_derivative_at_2_l3411_341187


namespace cafeteria_cottage_pies_l3411_341128

theorem cafeteria_cottage_pies :
  ∀ (lasagna_count : ℕ) (lasagna_mince : ℕ) (cottage_pie_mince : ℕ) (total_mince : ℕ),
    lasagna_count = 100 →
    lasagna_mince = 2 →
    cottage_pie_mince = 3 →
    total_mince = 500 →
    ∃ (cottage_pie_count : ℕ),
      cottage_pie_count * cottage_pie_mince + lasagna_count * lasagna_mince = total_mince ∧
      cottage_pie_count = 100 :=
by
  sorry

end cafeteria_cottage_pies_l3411_341128


namespace mike_working_time_l3411_341191

/-- Calculates the total working time in hours for Mike's car service tasks. -/
def calculate_working_time (wash_time min_per_car : ℕ) (oil_change_time min_per_car : ℕ)
  (tire_change_time min_per_set : ℕ) (cars_washed : ℕ) (oil_changes : ℕ)
  (tire_sets_changed : ℕ) : ℚ :=
  let total_minutes := wash_time * cars_washed +
                       oil_change_time * oil_changes +
                       tire_change_time * tire_sets_changed
  total_minutes / 60

theorem mike_working_time :
  calculate_working_time 10 15 30 9 6 2 = 4 := by
  sorry

end mike_working_time_l3411_341191


namespace fraction_equality_implies_x_equals_four_l3411_341110

theorem fraction_equality_implies_x_equals_four (x : ℝ) :
  (x ≠ 0) → (x ≠ -2) → (6 / (x + 2) = 4 / x) → x = 4 := by
  sorry

end fraction_equality_implies_x_equals_four_l3411_341110


namespace division_chain_l3411_341112

theorem division_chain : (88 / 4) / 2 = 11 := by
  sorry

end division_chain_l3411_341112


namespace tan_sum_simplification_l3411_341144

theorem tan_sum_simplification :
  (∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)) →
  Real.tan (45 * π / 180) = 1 →
  (1 + Real.tan (10 * π / 180)) * (1 + Real.tan (35 * π / 180)) = 2 := by
  sorry

end tan_sum_simplification_l3411_341144


namespace sum_of_cubes_of_roots_l3411_341117

theorem sum_of_cubes_of_roots (P : ℝ → ℝ) (x₁ x₂ x₃ : ℝ) :
  P = (fun x ↦ x^3 - 3*x - 1) →
  P x₁ = 0 →
  P x₂ = 0 →
  P x₃ = 0 →
  x₁^3 + x₂^3 + x₃^3 = 3 := by
  sorry

end sum_of_cubes_of_roots_l3411_341117


namespace push_mower_rate_l3411_341137

/-- Proves that the push mower's cutting rate is 1 acre per hour given the conditions of Jerry's lawn mowing scenario. -/
theorem push_mower_rate (total_acres : ℝ) (riding_mower_fraction : ℝ) (riding_mower_rate : ℝ) (total_mowing_time : ℝ) : 
  total_acres = 8 ∧ 
  riding_mower_fraction = 3/4 ∧ 
  riding_mower_rate = 2 ∧ 
  total_mowing_time = 5 → 
  (total_acres * (1 - riding_mower_fraction)) / (total_mowing_time - (total_acres * riding_mower_fraction) / riding_mower_rate) = 1 := by
  sorry

end push_mower_rate_l3411_341137


namespace good_numbers_up_to_17_and_18_not_good_l3411_341115

/-- The number of positive divisors of n -/
def d (n : ℕ+) : ℕ := sorry

/-- A number m is "good" if there exists a positive integer n such that m = n / d(n) -/
def is_good (m : ℕ+) : Prop :=
  ∃ n : ℕ+, (n : ℚ) / d n = m

theorem good_numbers_up_to_17_and_18_not_good :
  (∀ m : ℕ+, m ≤ 17 → is_good m) ∧ ¬ is_good 18 := by sorry

end good_numbers_up_to_17_and_18_not_good_l3411_341115


namespace complex_fraction_simplification_l3411_341145

theorem complex_fraction_simplification (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (x / (y + 1)) / (y / (x + 2)) = 3 / 4 := by
  sorry

end complex_fraction_simplification_l3411_341145


namespace trig_inequality_l3411_341199

theorem trig_inequality : Real.tan (55 * π / 180) > Real.cos (55 * π / 180) ∧ Real.cos (55 * π / 180) > Real.sin (33 * π / 180) := by
  sorry

end trig_inequality_l3411_341199


namespace total_cars_is_32_l3411_341178

/-- The number of cars owned by each person -/
structure CarOwnership where
  cathy : ℕ
  lindsey : ℕ
  carol : ℕ
  susan : ℕ

/-- The conditions of car ownership -/
def car_ownership_conditions (c : CarOwnership) : Prop :=
  c.lindsey = c.cathy + 4 ∧
  c.susan = c.carol - 2 ∧
  c.carol = 2 * c.cathy ∧
  c.cathy = 5

/-- The total number of cars owned by all four people -/
def total_cars (c : CarOwnership) : ℕ :=
  c.cathy + c.lindsey + c.carol + c.susan

/-- Theorem stating that the total number of cars is 32 -/
theorem total_cars_is_32 (c : CarOwnership) (h : car_ownership_conditions c) :
  total_cars c = 32 := by
  sorry

end total_cars_is_32_l3411_341178


namespace wood_measurement_problem_l3411_341151

theorem wood_measurement_problem (x y : ℝ) :
  (x + 4.5 = y ∧ x + 1 = (1/2) * y) ↔
  (∃ (wood_length rope_length : ℝ),
    wood_length = x ∧
    rope_length = y ∧
    wood_length + 4.5 = rope_length ∧
    wood_length + 1 = (1/2) * rope_length) :=
by sorry

end wood_measurement_problem_l3411_341151


namespace binomial_coefficient_formula_l3411_341109

theorem binomial_coefficient_formula (n k : ℕ) (h1 : k < n) (h2 : 0 < k) :
  Nat.choose n k = (Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))) :=
by sorry

end binomial_coefficient_formula_l3411_341109


namespace min_f_tetrahedron_l3411_341185

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron ABCD -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Distance between two points -/
def distance (p q : Point3D) : ℝ := sorry

/-- Function f(P) for a given tetrahedron and point P -/
def f (t : Tetrahedron) (P : Point3D) : ℝ :=
  distance P t.A + distance P t.B + distance P t.C + distance P t.D

/-- Theorem: Minimum value of f(P) for a tetrahedron with given properties -/
theorem min_f_tetrahedron (t : Tetrahedron) (a b c : ℝ) :
  (distance t.A t.D = a) →
  (distance t.B t.C = a) →
  (distance t.A t.C = b) →
  (distance t.B t.D = b) →
  (distance t.A t.B * distance t.C t.D = c^2) →
  ∃ (min_val : ℝ), (∀ (P : Point3D), f t P ≥ min_val) ∧ (min_val = Real.sqrt ((a^2 + b^2 + c^2) / 2)) :=
sorry

end min_f_tetrahedron_l3411_341185


namespace odd_function_properties_l3411_341170

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x

-- State the theorem
theorem odd_function_properties :
  ∀ (a b c : ℝ),
  (∀ x, f a b c (-x) = -(f a b c x)) →  -- f is odd
  f a b c (-Real.sqrt 2) = Real.sqrt 2 →  -- f(-√2) = √2
  f a b c (2 * Real.sqrt 2) = 10 * Real.sqrt 2 →  -- f(2√2) = 10√2
  (∃ (f' : ℝ → ℝ),
    (∀ x, f a b c x = x^3 - 3*x) ∧  -- f(x) = x³ - 3x
    (∀ x, x < -1 → f' x > 0) ∧  -- f is increasing on (-∞, -1)
    (∀ x, -1 < x ∧ x < 1 → f' x < 0) ∧  -- f is decreasing on (-1, 1)
    (∀ x, 1 < x → f' x > 0) ∧  -- f is increasing on (1, +∞)
    (∀ m, (∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
           f a b c x + m = 0 ∧ f a b c y + m = 0 ∧ f a b c z + m = 0) ↔
          -2 < m ∧ m < 2)) -- f(x) + m = 0 has three distinct roots iff m ∈ (-2, 2)
  := by sorry

end odd_function_properties_l3411_341170


namespace twenty_factorial_digits_sum_l3411_341101

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem twenty_factorial_digits_sum (B A : ℕ) : 
  B < 10 → A < 10 → 
  ∃ k : ℕ, factorial 20 = k * 10000 + B * 100 + A * 10 → 
  B + A = 10 := by
  sorry

end twenty_factorial_digits_sum_l3411_341101


namespace cyclic_sum_inequality_l3411_341172

theorem cyclic_sum_inequality (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  ((2*a + b + c)^2) / (2*a^2 + (b + c)^2) +
  ((2*b + c + a)^2) / (2*b^2 + (c + a)^2) +
  ((2*c + a + b)^2) / (2*c^2 + (a + b)^2) ≤ 8 := by
  sorry

end cyclic_sum_inequality_l3411_341172


namespace coordinates_of_N_l3411_341157

-- Define the point M
def M : ℝ × ℝ := (-1, 3)

-- Define the length of MN
def MN_length : ℝ := 4

-- Define the property that MN is parallel to y-axis
def parallel_to_y_axis (N : ℝ × ℝ) : Prop :=
  N.1 = M.1

-- Define the distance between M and N
def distance (N : ℝ × ℝ) : ℝ :=
  |N.2 - M.2|

-- Theorem statement
theorem coordinates_of_N :
  ∃ N : ℝ × ℝ, parallel_to_y_axis N ∧ distance N = MN_length ∧ (N = (-1, -1) ∨ N = (-1, 7)) :=
sorry

end coordinates_of_N_l3411_341157


namespace complement_of_union_equals_four_l3411_341189

universe u

def U : Set ℕ := {1, 2, 3, 4}
def M : Set ℕ := {1, 2}
def N : Set ℕ := {2, 3}

theorem complement_of_union_equals_four :
  (M ∪ N)ᶜ = {4} := by sorry

end complement_of_union_equals_four_l3411_341189


namespace false_propositions_count_l3411_341153

-- Define the original proposition
def original_prop (m n : ℝ) : Prop := m > -n → m^2 > n^2

-- Define the contrapositive
def contrapositive (m n : ℝ) : Prop := ¬(m^2 > n^2) → ¬(m > -n)

-- Define the inverse
def inverse (m n : ℝ) : Prop := m^2 > n^2 → m > -n

-- Define the negation
def negation (m n : ℝ) : Prop := ¬(m > -n → m^2 > n^2)

-- Theorem statement
theorem false_propositions_count :
  ∃ (m n : ℝ), (¬(original_prop m n) ∧ ¬(contrapositive m n) ∧ ¬(inverse m n) ∧ ¬(negation m n)) :=
by sorry

end false_propositions_count_l3411_341153


namespace min_cars_with_racing_stripes_l3411_341140

theorem min_cars_with_racing_stripes 
  (total_cars : ℕ) 
  (cars_without_ac : ℕ) 
  (max_ac_no_stripes : ℕ) 
  (h1 : total_cars = 100)
  (h2 : cars_without_ac = 47)
  (h3 : max_ac_no_stripes = 45) :
  ∃ (min_cars_with_stripes : ℕ), 
    min_cars_with_stripes = 8 ∧ 
    ∀ (cars_with_stripes : ℕ), 
      cars_with_stripes ≥ min_cars_with_stripes →
      cars_with_stripes + max_ac_no_stripes ≥ total_cars - cars_without_ac :=
by
  sorry

end min_cars_with_racing_stripes_l3411_341140


namespace quadratic_roots_l3411_341169

/-- A quadratic function passing through specific points -/
def f (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- The theorem stating that the quadratic equation has specific roots -/
theorem quadratic_roots (a b c : ℝ) : 
  f a b c (-2) = 21 ∧ 
  f a b c (-1) = 12 ∧ 
  f a b c 1 = 0 ∧ 
  f a b c 2 = -3 ∧ 
  f a b c 4 = -3 → 
  (∀ x, f a b c x = 0 ↔ x = 1 ∨ x = 5) :=
by sorry

end quadratic_roots_l3411_341169


namespace number_of_factors_of_30_l3411_341154

theorem number_of_factors_of_30 : Finset.card (Nat.divisors 30) = 8 := by
  sorry

end number_of_factors_of_30_l3411_341154


namespace basketball_not_tabletennis_l3411_341116

theorem basketball_not_tabletennis (total : ℕ) (basketball : ℕ) (tabletennis : ℕ) (neither : ℕ)
  (h1 : total = 40)
  (h2 : basketball = 24)
  (h3 : tabletennis = 16)
  (h4 : neither = 6) :
  basketball - (basketball + tabletennis - (total - neither)) = 18 :=
by sorry

end basketball_not_tabletennis_l3411_341116


namespace smallest_triangle_leg_l3411_341135

-- Define the properties of a 30-60-90 triangle
def thirty_sixty_ninety_triangle (short_leg long_leg hypotenuse : ℝ) : Prop :=
  short_leg = hypotenuse / 2 ∧ long_leg = short_leg * Real.sqrt 3

-- Define the sequence of four connected triangles
def connected_triangles (h1 h2 h3 h4 : ℝ) : Prop :=
  ∃ (s1 l1 s2 l2 s3 l3 s4 l4 : ℝ),
    thirty_sixty_ninety_triangle s1 l1 h1 ∧
    thirty_sixty_ninety_triangle s2 l2 h2 ∧
    thirty_sixty_ninety_triangle s3 l3 h3 ∧
    thirty_sixty_ninety_triangle s4 l4 h4 ∧
    l1 = h2 ∧ l2 = h3 ∧ l3 = h4

theorem smallest_triangle_leg (h1 h2 h3 h4 : ℝ) :
  h1 = 10 → connected_triangles h1 h2 h3 h4 → l4 = 45 / 8 :=
by sorry

end smallest_triangle_leg_l3411_341135


namespace remaining_oil_after_350km_distance_when_8_liters_left_l3411_341173

-- Define the initial conditions
def initial_oil : ℝ := 56
def oil_consumption_rate : ℝ := 0.08

-- Define the relationship between remaining oil and distance traveled
def remaining_oil (x : ℝ) : ℝ := initial_oil - oil_consumption_rate * x

-- Theorem to prove the remaining oil after 350 km
theorem remaining_oil_after_350km :
  remaining_oil 350 = 28 := by sorry

-- Theorem to prove the distance traveled when 8 liters are left
theorem distance_when_8_liters_left :
  ∃ x : ℝ, remaining_oil x = 8 ∧ x = 600 := by sorry

end remaining_oil_after_350km_distance_when_8_liters_left_l3411_341173


namespace tenth_pebble_count_l3411_341141

def pebble_sequence : ℕ → ℕ
  | 0 => 1
  | 1 => 5
  | 2 => 12
  | 3 => 22
  | (n + 4) => pebble_sequence (n + 3) + 3 * (n + 4) - 2

theorem tenth_pebble_count : pebble_sequence 9 = 145 := by
  sorry

end tenth_pebble_count_l3411_341141


namespace final_fruit_juice_percentage_l3411_341133

/-- Given an initial mixture of punch and some added pure fruit juice,
    calculate the final percentage of fruit juice in the punch. -/
theorem final_fruit_juice_percentage
  (initial_volume : ℝ)
  (initial_percentage : ℝ)
  (added_juice : ℝ)
  (h1 : initial_volume = 2)
  (h2 : initial_percentage = 0.1)
  (h3 : added_juice = 0.4)
  : (initial_volume * initial_percentage + added_juice) / (initial_volume + added_juice) = 0.25 := by
  sorry

end final_fruit_juice_percentage_l3411_341133


namespace pipe_fill_time_l3411_341197

theorem pipe_fill_time (T : ℝ) (h1 : T > 0) (h2 : 1/T - 1/4.5 = 1/9) : T = 3 := by
  sorry

end pipe_fill_time_l3411_341197


namespace solve_simultaneous_equations_l3411_341120

theorem solve_simultaneous_equations (a u : ℝ) 
  (eq1 : 3 / a + 1 / u = 7 / 2)
  (eq2 : 2 / a - 3 / u = 6) :
  a = 2 / 3 := by
sorry

end solve_simultaneous_equations_l3411_341120


namespace B_60_is_identity_l3411_341175

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, -1, 0],
    ![1,  0, 0],
    ![0,  0, 1]]

theorem B_60_is_identity :
  B^60 = 1 := by sorry

end B_60_is_identity_l3411_341175


namespace inner_hexagon_area_lower_bound_l3411_341162

/-- A regular hexagon -/
structure RegularHexagon where
  vertices : Fin 6 → ℝ × ℝ
  is_regular : sorry

/-- A point inside a regular hexagon -/
structure PointInHexagon (h : RegularHexagon) where
  point : ℝ × ℝ
  is_inside : sorry

/-- The hexagon formed by connecting a point to the vertices of a regular hexagon -/
def inner_hexagon (h : RegularHexagon) (p : PointInHexagon h) : Set (ℝ × ℝ) :=
  sorry

/-- The area of a set in ℝ² -/
noncomputable def area (s : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- The theorem stating that the area of the inner hexagon is at least 2/3 of the original hexagon -/
theorem inner_hexagon_area_lower_bound (h : RegularHexagon) (p : PointInHexagon h) :
  area (inner_hexagon h p) ≥ (2/3) * area (Set.range h.vertices) :=
sorry

end inner_hexagon_area_lower_bound_l3411_341162


namespace final_candy_count_l3411_341194

-- Define the variables
def initial_candy : ℕ := 47
def eaten_candy : ℕ := 25
def received_candy : ℕ := 40

-- State the theorem
theorem final_candy_count :
  initial_candy - eaten_candy + received_candy = 62 := by
  sorry

end final_candy_count_l3411_341194


namespace converse_and_inverse_false_l3411_341150

-- Define the properties of triangles
def Equilateral (t : Type) : Prop := sorry
def Isosceles (t : Type) : Prop := sorry

-- State the given true statement
axiom equilateral_implies_isosceles : ∀ t : Type, Equilateral t → Isosceles t

-- Define the converse and inverse
def converse : Prop := ∀ t : Type, Isosceles t → Equilateral t
def inverse : Prop := ∀ t : Type, ¬(Equilateral t) → ¬(Isosceles t)

-- Theorem to prove
theorem converse_and_inverse_false : ¬converse ∧ ¬inverse := by
  sorry

end converse_and_inverse_false_l3411_341150


namespace expression_simplification_l3411_341127

theorem expression_simplification (a b : ℝ) (ha : a ≠ 0) :
  (a^(7/3) - 2*a^(5/3)*b^(2/3) + a*b^(4/3)) / (a^(5/3) - a^(4/3)*b^(1/3) - a*b^(2/3) + a^(2/3)*b) / a^(1/3) = a^(1/3) + b^(1/3) := by
  sorry

end expression_simplification_l3411_341127


namespace smallest_solution_congruence_l3411_341142

theorem smallest_solution_congruence :
  ∃ (x : ℕ), x > 0 ∧ (6 * x) % 35 = 17 % 35 ∧
  ∀ (y : ℕ), y > 0 ∧ (6 * y) % 35 = 17 % 35 → x ≤ y :=
by
  -- The proof goes here
  sorry

end smallest_solution_congruence_l3411_341142


namespace multiply_fractions_l3411_341132

theorem multiply_fractions (a b : ℝ) : (1 / 3) * a^2 * (-6 * a * b) = -2 * a^3 * b := by
  sorry

end multiply_fractions_l3411_341132


namespace carriage_hourly_rate_l3411_341164

/-- Calculates the hourly rate for a carriage given the journey details and costs. -/
theorem carriage_hourly_rate
  (distance : ℝ)
  (speed : ℝ)
  (flat_fee : ℝ)
  (total_cost : ℝ)
  (h1 : distance = 20)
  (h2 : speed = 10)
  (h3 : flat_fee = 20)
  (h4 : total_cost = 80) :
  (total_cost - flat_fee) / (distance / speed) = 30 := by
  sorry

#check carriage_hourly_rate

end carriage_hourly_rate_l3411_341164


namespace toys_per_day_l3411_341195

-- Define the weekly toy production
def weekly_production : ℕ := 4340

-- Define the number of working days per week
def working_days : ℕ := 2

-- Define the daily toy production
def daily_production : ℕ := weekly_production / working_days

-- Theorem to prove
theorem toys_per_day : daily_production = 2170 := by
  sorry

end toys_per_day_l3411_341195


namespace solution_set_x_abs_x_leq_one_l3411_341183

theorem solution_set_x_abs_x_leq_one (x : ℝ) : x * |x| ≤ 1 ↔ x ≤ 1 := by sorry

end solution_set_x_abs_x_leq_one_l3411_341183


namespace circle_passes_through_points_l3411_341163

/-- The equation of a circle passing through points A(2, 0), B(4, 0), and C(1, 2) -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - (7/2)*y + 8 = 0

/-- Point A -/
def point_A : ℝ × ℝ := (2, 0)

/-- Point B -/
def point_B : ℝ × ℝ := (4, 0)

/-- Point C -/
def point_C : ℝ × ℝ := (1, 2)

/-- Theorem stating that the circle equation passes through points A, B, and C -/
theorem circle_passes_through_points :
  circle_equation point_A.1 point_A.2 ∧
  circle_equation point_B.1 point_B.2 ∧
  circle_equation point_C.1 point_C.2 :=
by sorry

end circle_passes_through_points_l3411_341163


namespace sarah_game_multiple_l3411_341107

/-- The game's formula to predict marriage age -/
def marriage_age_formula (name_length : ℕ) (current_age : ℕ) (multiple : ℕ) : ℕ :=
  name_length + multiple * current_age

/-- Proof that the multiple in Sarah's game is 2 -/
theorem sarah_game_multiple : ∃ (multiple : ℕ), 
  marriage_age_formula 5 9 multiple = 23 ∧ multiple = 2 :=
by sorry

end sarah_game_multiple_l3411_341107


namespace polynomial_factorization_l3411_341106

theorem polynomial_factorization (y : ℝ) : 
  y^8 - 4*y^6 + 6*y^4 - 4*y^2 + 1 = (y-1)^4 * (y+1)^4 := by
sorry

end polynomial_factorization_l3411_341106


namespace binomial_coefficient_28_5_l3411_341192

theorem binomial_coefficient_28_5 (h1 : Nat.choose 26 3 = 2600)
                                  (h2 : Nat.choose 26 4 = 14950)
                                  (h3 : Nat.choose 26 5 = 65780) :
  Nat.choose 28 5 = 98280 := by
  sorry

end binomial_coefficient_28_5_l3411_341192
