import Mathlib

namespace NUMINAMATH_CALUDE_sandy_age_l2671_267138

/-- Given that Molly is 20 years older than Sandy and their ages are in the ratio 7:9, prove that Sandy is 70 years old. -/
theorem sandy_age (sandy molly : ℕ) 
  (h1 : molly = sandy + 20) 
  (h2 : sandy * 9 = molly * 7) : 
  sandy = 70 := by sorry

end NUMINAMATH_CALUDE_sandy_age_l2671_267138


namespace NUMINAMATH_CALUDE_xy_value_l2671_267175

theorem xy_value (x y : ℝ) (h : (x - 3)^2 + |y + 2| = 0) : x * y = -6 := by
  sorry

end NUMINAMATH_CALUDE_xy_value_l2671_267175


namespace NUMINAMATH_CALUDE_equidistant_points_on_axes_l2671_267156

/-- Given points A(1, 5) and B(2, 4), this theorem states that (0, 3) and (-3, 0) are the only points
    on the coordinate axes that are equidistant from A and B. -/
theorem equidistant_points_on_axes (A B P : ℝ × ℝ) : 
  A = (1, 5) → B = (2, 4) → 
  (P.1 = 0 ∨ P.2 = 0) →  -- P is on a coordinate axis
  (dist A P = dist B P) →  -- P is equidistant from A and B
  (P = (0, 3) ∨ P = (-3, 0)) :=
by sorry

#check equidistant_points_on_axes

end NUMINAMATH_CALUDE_equidistant_points_on_axes_l2671_267156


namespace NUMINAMATH_CALUDE_solve_system_l2671_267100

theorem solve_system (x z : ℝ) (hx : x ≠ 0) (hz : z ≠ 0)
  (eq1 : x = 2 + 1/z) (eq2 : z = 3 + 1/x) :
  z = (3 + Real.sqrt 15) / 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_system_l2671_267100


namespace NUMINAMATH_CALUDE_max_area_rectangle_l2671_267177

/-- A rectangle represents a playground with length and width. -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- The perimeter of a rectangle is twice the sum of its length and width. -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- The area of a rectangle is the product of its length and width. -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- Theorem: The maximum area of a rectangle with perimeter 400 feet, 
    length at least 100 feet, and width at least 60 feet is 10,000 square feet. -/
theorem max_area_rectangle : 
  ∃ (r : Rectangle), 
    perimeter r = 400 ∧ 
    r.length ≥ 100 ∧ 
    r.width ≥ 60 ∧ 
    area r = 10000 ∧ 
    ∀ (s : Rectangle), 
      perimeter s = 400 → 
      s.length ≥ 100 → 
      s.width ≥ 60 → 
      area s ≤ 10000 := by
  sorry

end NUMINAMATH_CALUDE_max_area_rectangle_l2671_267177


namespace NUMINAMATH_CALUDE_apple_difference_l2671_267140

def apple_contest (aaron bella claire daniel edward fiona george hannah : ℕ) : Prop :=
  aaron = 5 ∧ bella = 3 ∧ claire = 7 ∧ daniel = 2 ∧ edward = 4 ∧ fiona = 3 ∧ george = 1 ∧ hannah = 6 ∧
  claire ≥ aaron ∧ claire ≥ bella ∧ claire ≥ daniel ∧ claire ≥ edward ∧ claire ≥ fiona ∧ claire ≥ george ∧ claire ≥ hannah ∧
  aaron ≥ bella ∧ aaron ≥ daniel ∧ aaron ≥ edward ∧ aaron ≥ fiona ∧ aaron ≥ george ∧ aaron ≥ hannah ∧
  george ≤ aaron ∧ george ≤ bella ∧ george ≤ claire ∧ george ≤ daniel ∧ george ≤ edward ∧ george ≤ fiona ∧ george ≤ hannah

theorem apple_difference (aaron bella claire daniel edward fiona george hannah : ℕ) :
  apple_contest aaron bella claire daniel edward fiona george hannah →
  claire - george = 6 := by
  sorry

end NUMINAMATH_CALUDE_apple_difference_l2671_267140


namespace NUMINAMATH_CALUDE_baseball_theorem_l2671_267103

def baseball_problem (team_scores : List Nat) (lost_games : Nat) : Prop :=
  let total_games := team_scores.length
  let won_games := total_games - lost_games
  let opponent_scores := team_scores.map (λ score =>
    if score ∈ [2, 4, 6, 8] then score + 2 else score / 3)
  
  (total_games = 8) ∧
  (team_scores = [2, 3, 4, 5, 6, 7, 8, 9]) ∧
  (lost_games = 4) ∧
  (opponent_scores.sum = 36)

theorem baseball_theorem :
  baseball_problem [2, 3, 4, 5, 6, 7, 8, 9] 4 := by
  sorry

end NUMINAMATH_CALUDE_baseball_theorem_l2671_267103


namespace NUMINAMATH_CALUDE_train_passing_time_l2671_267101

/-- The time it takes for a faster train to completely pass a slower train -/
theorem train_passing_time (v_fast v_slow : ℝ) (length : ℝ) (h_fast : v_fast = 50) (h_slow : v_slow = 32) (h_length : length = 75) :
  (length / ((v_fast - v_slow) * (1000 / 3600))) = 15 :=
sorry

end NUMINAMATH_CALUDE_train_passing_time_l2671_267101


namespace NUMINAMATH_CALUDE_initial_cells_count_l2671_267159

/-- Calculates the number of cells after one hour given the initial number -/
def cellsAfterOneHour (initialCells : ℕ) : ℕ :=
  2 * (initialCells - 2)

/-- Calculates the number of cells after n hours given the initial number -/
def cellsAfterNHours (initialCells n : ℕ) : ℕ :=
  match n with
  | 0 => initialCells
  | m + 1 => cellsAfterOneHour (cellsAfterNHours initialCells m)

/-- Theorem stating that if there are 164 cells after 5 hours, the initial number of cells was 9 -/
theorem initial_cells_count (initialCells : ℕ) :
  cellsAfterNHours initialCells 5 = 164 → initialCells = 9 :=
by
  sorry

#check initial_cells_count

end NUMINAMATH_CALUDE_initial_cells_count_l2671_267159


namespace NUMINAMATH_CALUDE_jamie_quiz_performance_l2671_267193

theorem jamie_quiz_performance (y : ℕ) : 
  let total_questions : ℕ := 8 * y
  let missed_questions : ℕ := 2 * y
  let correct_questions : ℕ := total_questions - missed_questions
  (correct_questions : ℚ) / (total_questions : ℚ) * 100 = 75 := by
sorry

end NUMINAMATH_CALUDE_jamie_quiz_performance_l2671_267193


namespace NUMINAMATH_CALUDE_gcd_lcm_product_l2671_267152

theorem gcd_lcm_product (a b : ℕ) (h1 : a = 150) (h2 : b = 180) :
  (Nat.gcd a b) * (Nat.lcm a b) = 54000 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_l2671_267152


namespace NUMINAMATH_CALUDE_three_intersection_range_l2671_267196

def f (x : ℝ) := x^3 - 3*x

theorem three_intersection_range :
  ∃ (a_min a_max : ℝ), a_min < a_max ∧
  (∀ a : ℝ, (∃ (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ ∧
                               f x₁ = a ∧ f x₂ = a ∧ f x₃ = a) ↔
              a_min < a ∧ a < a_max) ∧
  a_min = -2 ∧ a_max = 2 :=
sorry

end NUMINAMATH_CALUDE_three_intersection_range_l2671_267196


namespace NUMINAMATH_CALUDE_smallest_block_volume_l2671_267118

theorem smallest_block_volume (a b c : ℕ) (h : (a - 1) * (b - 1) * (c - 1) = 143) :
  a * b * c ≥ 336 :=
by sorry

end NUMINAMATH_CALUDE_smallest_block_volume_l2671_267118


namespace NUMINAMATH_CALUDE_sticker_collection_l2671_267165

theorem sticker_collection (karl_stickers : ℕ) : 
  (∃ (ryan_stickers ben_stickers : ℕ),
    ryan_stickers = karl_stickers + 20 ∧
    ben_stickers = ryan_stickers - 10 ∧
    karl_stickers + ryan_stickers + ben_stickers = 105) →
  karl_stickers = 25 := by
sorry

end NUMINAMATH_CALUDE_sticker_collection_l2671_267165


namespace NUMINAMATH_CALUDE_max_average_growth_rate_l2671_267108

theorem max_average_growth_rate
  (P₁ P₂ M : ℝ)
  (h_sum : P₁ + P₂ = M)
  (h_nonneg : 0 ≤ P₁ ∧ 0 ≤ P₂)
  (P : ℝ)
  (h_avg_growth : (1 + P)^2 = (1 + P₁) * (1 + P₂)) :
  P ≤ M / 2 :=
sorry

end NUMINAMATH_CALUDE_max_average_growth_rate_l2671_267108


namespace NUMINAMATH_CALUDE_abs_x_minus_5_lt_3_iff_2_lt_x_lt_8_l2671_267104

theorem abs_x_minus_5_lt_3_iff_2_lt_x_lt_8 :
  ∀ x : ℝ, |x - 5| < 3 ↔ 2 < x ∧ x < 8 := by
  sorry

end NUMINAMATH_CALUDE_abs_x_minus_5_lt_3_iff_2_lt_x_lt_8_l2671_267104


namespace NUMINAMATH_CALUDE_set_conditions_l2671_267125

def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2 * m - 1}

theorem set_conditions (m : ℝ) :
  (B m = ∅ ↔ m < 2) ∧
  (A ∩ B m = ∅ ↔ m > 4 ∨ m < 2) := by
  sorry

end NUMINAMATH_CALUDE_set_conditions_l2671_267125


namespace NUMINAMATH_CALUDE_exponent_division_l2671_267147

theorem exponent_division (a : ℝ) (h : a ≠ 0) : a^5 / a^3 = a^2 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l2671_267147


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2671_267122

theorem inequality_and_equality_condition (n : ℕ) (hn : n ≥ 1) :
  (1 / 3 : ℝ) * n^2 + (1 / 2 : ℝ) * n + (1 / 6 : ℝ) ≥ (n.factorial : ℝ)^((2 : ℝ) / n) ∧
  ((1 / 3 : ℝ) * n^2 + (1 / 2 : ℝ) * n + (1 / 6 : ℝ) = (n.factorial : ℝ)^((2 : ℝ) / n) ↔ n = 1) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2671_267122


namespace NUMINAMATH_CALUDE_joe_fruit_probability_l2671_267114

/-- The number of fruit types Joe can choose from -/
def num_fruit_types : ℕ := 4

/-- The number of meals Joe has in a day -/
def num_meals : ℕ := 4

/-- The probability of choosing a specific fruit for one meal -/
def prob_one_fruit : ℚ := 1 / num_fruit_types

/-- The probability of eating the same fruit for all meals -/
def prob_same_fruit : ℚ := num_fruit_types * (prob_one_fruit ^ num_meals)

/-- The probability of eating at least two different kinds of fruit in one day -/
def prob_different_fruits : ℚ := 1 - prob_same_fruit

theorem joe_fruit_probability : prob_different_fruits = 63 / 64 := by
  sorry

end NUMINAMATH_CALUDE_joe_fruit_probability_l2671_267114


namespace NUMINAMATH_CALUDE_gcd_18_30_l2671_267174

theorem gcd_18_30 : Nat.gcd 18 30 = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_18_30_l2671_267174


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l2671_267144

/-- Reflects a point across the x-axis -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The point to be reflected -/
def original_point : ℝ × ℝ := (-2, -3)

/-- The expected result after reflection -/
def expected_reflection : ℝ × ℝ := (-2, 3)

theorem reflection_across_x_axis :
  reflect_x original_point = expected_reflection := by sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l2671_267144


namespace NUMINAMATH_CALUDE_second_purchase_profit_less_than_first_l2671_267110

/-- Represents a type of T-shirt -/
structure TShirt where
  purchasePrice : ℕ
  sellingPrice : ℕ

/-- Represents the store's inventory and sales -/
structure Store where
  typeA : TShirt
  typeB : TShirt
  firstPurchaseQuantityA : ℕ
  firstPurchaseQuantityB : ℕ
  secondPurchaseQuantityA : ℕ
  secondPurchaseQuantityB : ℕ

/-- Calculate the profit from the first purchase -/
def firstPurchaseProfit (s : Store) : ℕ :=
  (s.typeA.sellingPrice - s.typeA.purchasePrice) * s.firstPurchaseQuantityA +
  (s.typeB.sellingPrice - s.typeB.purchasePrice) * s.firstPurchaseQuantityB

/-- Calculate the maximum profit from the second purchase -/
def maxSecondPurchaseProfit (s : Store) : ℕ :=
  let newTypeA := TShirt.mk (s.typeA.purchasePrice + 5) s.typeA.sellingPrice
  let newTypeB := TShirt.mk (s.typeB.purchasePrice + 10) s.typeB.sellingPrice
  (newTypeA.sellingPrice - newTypeA.purchasePrice) * s.secondPurchaseQuantityA +
  (newTypeB.sellingPrice - newTypeB.purchasePrice) * s.secondPurchaseQuantityB

/-- The theorem to be proved -/
theorem second_purchase_profit_less_than_first (s : Store) :
  s.firstPurchaseQuantityA + s.firstPurchaseQuantityB = 120 →
  s.typeA.purchasePrice * s.firstPurchaseQuantityA + s.typeB.purchasePrice * s.firstPurchaseQuantityB = 6000 →
  s.secondPurchaseQuantityA + s.secondPurchaseQuantityB = 150 →
  s.secondPurchaseQuantityB ≤ 2 * s.secondPurchaseQuantityA →
  maxSecondPurchaseProfit s < firstPurchaseProfit s :=
by
  sorry

#check second_purchase_profit_less_than_first

end NUMINAMATH_CALUDE_second_purchase_profit_less_than_first_l2671_267110


namespace NUMINAMATH_CALUDE_order_of_f_l2671_267145

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_even : ∀ x : ℝ, f x = f (-x)
axiom f_increasing_nonneg : ∀ x y : ℝ, 0 ≤ x → x ≤ y → f x ≤ f y

-- Theorem statement
theorem order_of_f : f (-2) < f 3 ∧ f 3 < f (-π) :=
sorry

end NUMINAMATH_CALUDE_order_of_f_l2671_267145


namespace NUMINAMATH_CALUDE_per_minute_charge_plan_a_l2671_267116

/-- Represents the per-minute charge after the first 4 minutes under plan A -/
def x : ℝ := sorry

/-- The cost of an 18-minute call under plan A -/
def cost_plan_a : ℝ := 0.60 + 14 * x

/-- The cost of an 18-minute call under plan B -/
def cost_plan_b : ℝ := 0.08 * 18

/-- Theorem stating that the per-minute charge after the first 4 minutes under plan A is $0.06 -/
theorem per_minute_charge_plan_a : x = 0.06 := by
  have h1 : cost_plan_a = cost_plan_b := by sorry
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_per_minute_charge_plan_a_l2671_267116


namespace NUMINAMATH_CALUDE_owen_profit_l2671_267132

/-- Calculate Owen's overall profit from selling face masks --/
theorem owen_profit : 
  let cheap_boxes := 8
  let expensive_boxes := 4
  let cheap_box_price := 9
  let expensive_box_price := 12
  let masks_per_box := 50
  let small_packets := 100
  let small_packet_price := 5
  let small_packet_size := 25
  let large_packets := 28
  let large_packet_price := 12
  let large_packet_size := 100
  let remaining_cheap := 150
  let remaining_expensive := 150
  let remaining_cheap_price := 3
  let remaining_expensive_price := 4

  let total_cost := cheap_boxes * cheap_box_price + expensive_boxes * expensive_box_price
  let total_masks := (cheap_boxes + expensive_boxes) * masks_per_box
  let repacked_revenue := small_packets * small_packet_price + large_packets * large_packet_price
  let remaining_revenue := remaining_cheap * remaining_cheap_price + remaining_expensive * remaining_expensive_price
  let total_revenue := repacked_revenue + remaining_revenue
  let profit := total_revenue - total_cost

  profit = 1766 := by sorry

end NUMINAMATH_CALUDE_owen_profit_l2671_267132


namespace NUMINAMATH_CALUDE_next_multiple_remainder_l2671_267192

theorem next_multiple_remainder (N : ℕ) (h : N = 44 * 432) :
  (N + 432) % 39 = 12 := by
  sorry

end NUMINAMATH_CALUDE_next_multiple_remainder_l2671_267192


namespace NUMINAMATH_CALUDE_solution_of_exponential_equation_l2671_267128

theorem solution_of_exponential_equation :
  {x : ℝ | (4 : ℝ) ^ (x^2 + 1) = 16} = {-1, 1} := by sorry

end NUMINAMATH_CALUDE_solution_of_exponential_equation_l2671_267128


namespace NUMINAMATH_CALUDE_ada_original_seat_l2671_267197

/-- Represents the seats in the theater --/
inductive Seat
| one
| two
| three
| four
| five
| six

/-- Represents the friends --/
inductive Friend
| ada
| bea
| ceci
| dee
| edie
| fi

/-- Represents the direction of movement --/
inductive Direction
| left
| right

/-- Defines a movement of a friend --/
structure Movement where
  friend : Friend
  distance : Nat
  direction : Direction

/-- Defines the seating arrangement --/
def SeatingArrangement := Friend → Seat

/-- Defines the set of movements --/
def Movements := List Movement

/-- Function to apply a movement to a seating arrangement --/
def applyMovement (arrangement : SeatingArrangement) (move : Movement) : SeatingArrangement :=
  sorry

/-- Function to apply all movements to a seating arrangement --/
def applyMovements (arrangement : SeatingArrangement) (moves : Movements) : SeatingArrangement :=
  sorry

/-- Theorem stating Ada's original seat --/
theorem ada_original_seat 
  (initial_arrangement : SeatingArrangement)
  (moves : Movements)
  (final_arrangement : SeatingArrangement) :
  (moves = [
    ⟨Friend.bea, 3, Direction.right⟩,
    ⟨Friend.ceci, 1, Direction.left⟩,
    ⟨Friend.dee, 1, Direction.right⟩,
    ⟨Friend.edie, 1, Direction.left⟩
  ]) →
  (final_arrangement = applyMovements initial_arrangement moves) →
  (final_arrangement Friend.ada = Seat.one ∨ final_arrangement Friend.ada = Seat.six) →
  (initial_arrangement Friend.ada = Seat.three) :=
sorry

end NUMINAMATH_CALUDE_ada_original_seat_l2671_267197


namespace NUMINAMATH_CALUDE_exponent_product_simplification_l2671_267154

theorem exponent_product_simplification :
  (5 ^ 0.4) * (5 ^ 0.1) * (5 ^ 0.5) * (5 ^ 0.3) * (5 ^ 0.7) = 25 := by
  sorry

end NUMINAMATH_CALUDE_exponent_product_simplification_l2671_267154


namespace NUMINAMATH_CALUDE_petes_total_distance_l2671_267185

/-- Represents the distance Pete traveled in blocks for each leg of his journey -/
structure Journey where
  house_to_garage : ℕ
  garage_to_post_office : ℕ
  post_office_to_friend : ℕ

/-- Calculates the total distance traveled for a round trip -/
def total_distance (j : Journey) : ℕ :=
  2 * (j.house_to_garage + j.garage_to_post_office + j.post_office_to_friend)

/-- Pete's actual journey -/
def petes_journey : Journey :=
  { house_to_garage := 5
  , garage_to_post_office := 20
  , post_office_to_friend := 10 }

/-- Theorem stating that Pete traveled 70 blocks in total -/
theorem petes_total_distance : total_distance petes_journey = 70 := by
  sorry

end NUMINAMATH_CALUDE_petes_total_distance_l2671_267185


namespace NUMINAMATH_CALUDE_four_digit_equal_digits_l2671_267149

theorem four_digit_equal_digits (n : ℤ) : 12 * n^2 + 12 * n + 11 = 5555 ↔ n = 21 ∨ n = -22 := by
  sorry

end NUMINAMATH_CALUDE_four_digit_equal_digits_l2671_267149


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l2671_267189

/-- Given a number c that forms a geometric sequence when added to 20, 50, and 100, 
    the common ratio of this sequence is 5/3 -/
theorem geometric_sequence_ratio (c : ℝ) : 
  (∃ r : ℝ, (50 + c) / (20 + c) = r ∧ (100 + c) / (50 + c) = r) → 
  (50 + c) / (20 + c) = 5/3 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l2671_267189


namespace NUMINAMATH_CALUDE_unique_prime_303509_l2671_267124

theorem unique_prime_303509 :
  ∃! (B : ℕ), B < 10 ∧ Nat.Prime (303500 + B) :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_CALUDE_unique_prime_303509_l2671_267124


namespace NUMINAMATH_CALUDE_students_not_eating_lunch_proof_l2671_267181

def students_not_eating_lunch (total_students cafeteria_students : ℕ) : ℕ :=
  total_students - (cafeteria_students + 3 * cafeteria_students)

theorem students_not_eating_lunch_proof :
  students_not_eating_lunch 60 10 = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_not_eating_lunch_proof_l2671_267181


namespace NUMINAMATH_CALUDE_ratio_change_l2671_267195

theorem ratio_change (x y : ℤ) (n : ℤ) : 
  y = 48 → x / y = 1 / 4 → (x + n) / y = 1 / 2 → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_ratio_change_l2671_267195


namespace NUMINAMATH_CALUDE_crayons_difference_proof_l2671_267113

-- Define the given conditions
def initial_crayons : ℕ := 4 * 8
def crayons_to_mae : ℕ := 5
def crayons_left : ℕ := 15

-- Define the theorem
theorem crayons_difference_proof : 
  (initial_crayons - crayons_to_mae - crayons_left) - crayons_to_mae = 7 := by
  sorry

end NUMINAMATH_CALUDE_crayons_difference_proof_l2671_267113


namespace NUMINAMATH_CALUDE_brothers_age_fraction_l2671_267143

/-- Given three brothers with ages M, O, and Y, prove that Y/O = 1/3 -/
theorem brothers_age_fraction (M O Y : ℕ) : 
  Y = 5 → 
  M + O + Y = 28 → 
  O = 2 * (M - 1) + 1 → 
  Y / O = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_brothers_age_fraction_l2671_267143


namespace NUMINAMATH_CALUDE_inequality_proof_l2671_267164

theorem inequality_proof (a b : ℝ) (ha : |a| < 2) (hb : |b| < 2) : 2*|a + b| < |4 + a*b| := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2671_267164


namespace NUMINAMATH_CALUDE_smaller_number_puzzle_l2671_267117

theorem smaller_number_puzzle (x y : ℝ) (h_sum : x + y = 18) (h_product : x * y = 80) :
  min x y = 8 := by sorry

end NUMINAMATH_CALUDE_smaller_number_puzzle_l2671_267117


namespace NUMINAMATH_CALUDE_negation_of_square_nonnegative_l2671_267180

theorem negation_of_square_nonnegative :
  (¬ ∀ x : ℝ, x ^ 2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀ ^ 2 < 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_square_nonnegative_l2671_267180


namespace NUMINAMATH_CALUDE_solution_is_two_lines_l2671_267170

-- Define the equation
def equation (x y : ℝ) : Prop := (x - 2*y)^2 = x^2 - 4*y^2

-- Define the solution set
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | equation p.1 p.2}

-- Define the two lines
def x_axis : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.2 = 0}
def diagonal_line : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1 = 2 * p.2}

-- Theorem statement
theorem solution_is_two_lines :
  solution_set = x_axis ∪ diagonal_line :=
sorry

end NUMINAMATH_CALUDE_solution_is_two_lines_l2671_267170


namespace NUMINAMATH_CALUDE_sequence_increasing_iff_a0_eq_one_fifth_l2671_267126

/-- The sequence defined by a(n+1) = 2^n - 3*a(n) -/
def a : ℕ → ℝ → ℝ 
  | 0, a₀ => a₀
  | n + 1, a₀ => 2^n - 3 * a n a₀

/-- The sequence is increasing -/
def is_increasing (a₀ : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) a₀ > a n a₀

/-- Theorem: The sequence is increasing if and only if a₀ = 1/5 -/
theorem sequence_increasing_iff_a0_eq_one_fifth :
  ∀ a₀ : ℝ, is_increasing a₀ ↔ a₀ = 1/5 := by sorry

end NUMINAMATH_CALUDE_sequence_increasing_iff_a0_eq_one_fifth_l2671_267126


namespace NUMINAMATH_CALUDE_simplify_radical_sum_l2671_267107

theorem simplify_radical_sum : Real.sqrt 72 + Real.sqrt 32 = 10 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_sum_l2671_267107


namespace NUMINAMATH_CALUDE_unique_mod_equivalence_l2671_267168

theorem unique_mod_equivalence :
  ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ -2023 [ZMOD 10] := by
  sorry

end NUMINAMATH_CALUDE_unique_mod_equivalence_l2671_267168


namespace NUMINAMATH_CALUDE_sum_of_two_equals_third_l2671_267166

theorem sum_of_two_equals_third (x y z : ℤ) 
  (h1 : x + y = z) (h2 : y + z = x) (h3 : z + x = y) : 
  x = 0 ∧ y = 0 ∧ z = 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_equals_third_l2671_267166


namespace NUMINAMATH_CALUDE_arithmetic_sum_odd_numbers_l2671_267171

theorem arithmetic_sum_odd_numbers : ∀ (a₁ aₙ d n : ℕ),
  a₁ = 1 →
  aₙ = 99 →
  d = 2 →
  aₙ = a₁ + (n - 1) * d →
  n * (a₁ + aₙ) / 2 = 2500 :=
by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sum_odd_numbers_l2671_267171


namespace NUMINAMATH_CALUDE_pm25_decrease_theorem_l2671_267191

/-- Calculates the PM2.5 concentration after two consecutive years of 10% decrease -/
def pm25_concentration (initial : ℝ) : ℝ :=
  initial * (1 - 0.1)^2

/-- Theorem stating that given an initial PM2.5 concentration of 50 micrograms per cubic meter
    two years ago, with a 10% decrease each year for two consecutive years,
    the resulting concentration is 40.5 micrograms per cubic meter -/
theorem pm25_decrease_theorem (initial : ℝ) (h : initial = 50) :
  pm25_concentration initial = 40.5 := by
  sorry

#eval pm25_concentration 50

end NUMINAMATH_CALUDE_pm25_decrease_theorem_l2671_267191


namespace NUMINAMATH_CALUDE_negation_of_existence_real_root_l2671_267182

theorem negation_of_existence_real_root : 
  (¬ ∃ x : ℝ, x^3 - 2*x + 1 = 0) ↔ (∀ x : ℝ, x^3 - 2*x + 1 ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_existence_real_root_l2671_267182


namespace NUMINAMATH_CALUDE_card_probability_l2671_267151

def standard_deck : ℕ := 52
def num_jacks : ℕ := 4
def num_queens : ℕ := 4

theorem card_probability : 
  let p_two_queens := (num_queens / standard_deck) * ((num_queens - 1) / (standard_deck - 1))
  let p_one_jack := 2 * (num_jacks / standard_deck) * ((standard_deck - num_jacks) / (standard_deck - 1))
  let p_two_jacks := (num_jacks / standard_deck) * ((num_jacks - 1) / (standard_deck - 1))
  p_two_queens + p_one_jack + p_two_jacks = 2 / 13 := by
  sorry

end NUMINAMATH_CALUDE_card_probability_l2671_267151


namespace NUMINAMATH_CALUDE_linear_system_determinant_l2671_267135

/-- 
Given integers a, b, c, d such that the system of equations
  ax + by = m
  cx + dy = n
has integer solutions for all integer values of m and n,
prove that |ad - bc| = 1
-/
theorem linear_system_determinant (a b c d : ℤ) 
  (h : ∀ (m n : ℤ), ∃ (x y : ℤ), a * x + b * y = m ∧ c * x + d * y = n) :
  |a * d - b * c| = 1 :=
sorry

end NUMINAMATH_CALUDE_linear_system_determinant_l2671_267135


namespace NUMINAMATH_CALUDE_student_count_l2671_267187

theorem student_count (x : ℕ) (h : x > 0) :
  (Nat.choose x 4 : ℚ) / (x * (x - 1)) = 13 / 2 → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_student_count_l2671_267187


namespace NUMINAMATH_CALUDE_largest_fraction_l2671_267146

theorem largest_fraction : 
  let f1 := 5 / 11
  let f2 := 6 / 13
  let f3 := 19 / 39
  let f4 := 101 / 203
  let f5 := 152 / 303
  let f6 := 80 / 159
  (f6 > f1) ∧ (f6 > f2) ∧ (f6 > f3) ∧ (f6 > f4) ∧ (f6 > f5) := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_l2671_267146


namespace NUMINAMATH_CALUDE_sixth_year_fee_l2671_267121

def membership_fee (initial_fee : ℕ) (annual_increase : ℕ) (year : ℕ) : ℕ :=
  initial_fee + (year - 1) * annual_increase

theorem sixth_year_fee :
  membership_fee 80 10 6 = 130 := by
  sorry

end NUMINAMATH_CALUDE_sixth_year_fee_l2671_267121


namespace NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l2671_267123

theorem sum_of_numbers_in_ratio (x : ℝ) :
  x > 0 →
  x^2 + (2*x)^2 + (4*x)^2 = 1701 →
  x + 2*x + 4*x = 63 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l2671_267123


namespace NUMINAMATH_CALUDE_binomial_square_coefficient_l2671_267176

theorem binomial_square_coefficient (a : ℚ) : 
  (∃ r s : ℚ, ∀ x : ℚ, a * x^2 + 18 * x + 16 = (r * x + s)^2) → a = 81 / 16 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_coefficient_l2671_267176


namespace NUMINAMATH_CALUDE_triangle_groups_count_l2671_267106

/-- The number of groups of 3 points from 12 points that can form a triangle -/
def triangle_groups : ℕ := 200

/-- The total number of points -/
def total_points : ℕ := 12

/-- Theorem stating that the number of groups of 3 points from 12 points 
    that can form a triangle is equal to 200 -/
theorem triangle_groups_count : 
  triangle_groups = 200 ∧ total_points = 12 := by sorry

end NUMINAMATH_CALUDE_triangle_groups_count_l2671_267106


namespace NUMINAMATH_CALUDE_cloth_trimming_l2671_267148

/-- Given a square piece of cloth with side length 22 feet, prove that after trimming 6 feet from two opposite edges and 5 feet from the other two edges, the remaining area is 272 square feet. -/
theorem cloth_trimming (original_length : ℕ) (trim_1 : ℕ) (trim_2 : ℕ) : 
  original_length = 22 → 
  trim_1 = 6 → 
  trim_2 = 5 → 
  (original_length - trim_1) * (original_length - trim_2) = 272 := by
sorry

end NUMINAMATH_CALUDE_cloth_trimming_l2671_267148


namespace NUMINAMATH_CALUDE_complex_equation_to_parabola_l2671_267127

/-- The set of points (x, y) satisfying the complex equation is equivalent to a parabola with two holes -/
theorem complex_equation_to_parabola (x y : ℝ) :
  (Complex.I + x^2 - 2*x + 2*y*Complex.I = 
   (y - 1 : ℂ) + ((4*y^2 - 1)/(2*y - 1) : ℝ)*Complex.I) ↔ 
  (y = (x - 1)^2 ∧ y ≠ (1/2 : ℝ)) :=
sorry

end NUMINAMATH_CALUDE_complex_equation_to_parabola_l2671_267127


namespace NUMINAMATH_CALUDE_max_popsicles_l2671_267172

def lucy_budget : ℚ := 15
def popsicle_cost : ℚ := 2.4

theorem max_popsicles : 
  ∀ n : ℕ, (n : ℚ) * popsicle_cost ≤ lucy_budget → n ≤ 6 :=
by sorry

end NUMINAMATH_CALUDE_max_popsicles_l2671_267172


namespace NUMINAMATH_CALUDE_B_power_98_l2671_267109

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![0, 1, 0],
    ![-1, 0, 0],
    ![0, 0, 1]]

theorem B_power_98 : B^98 = ![![-1, 0, 0],
                              ![0, -1, 0],
                              ![0, 0, 1]] := by sorry

end NUMINAMATH_CALUDE_B_power_98_l2671_267109


namespace NUMINAMATH_CALUDE_sum_f_positive_l2671_267157

def f (x : ℝ) : ℝ := x + x^3

theorem sum_f_positive (x₁ x₂ x₃ : ℝ) 
  (h₁ : x₁ + x₂ > 0) (h₂ : x₂ + x₃ > 0) (h₃ : x₃ + x₁ > 0) : 
  f x₁ + f x₂ + f x₃ > 0 := by
  sorry

end NUMINAMATH_CALUDE_sum_f_positive_l2671_267157


namespace NUMINAMATH_CALUDE_economics_test_correct_answers_l2671_267129

theorem economics_test_correct_answers 
  (total_students : ℕ) 
  (correct_q1 : ℕ) 
  (correct_q2 : ℕ) 
  (not_taken : ℕ) 
  (h1 : total_students = 25) 
  (h2 : correct_q1 = 22) 
  (h3 : correct_q2 = 20) 
  (h4 : not_taken = 3) :
  (correct_q1 + correct_q2) - (total_students - not_taken) = 20 := by
sorry

end NUMINAMATH_CALUDE_economics_test_correct_answers_l2671_267129


namespace NUMINAMATH_CALUDE_percent_relation_l2671_267199

theorem percent_relation (x y : ℝ) (h : 0.3 * (x - y) = 0.2 * (x + y)) :
  y = 0.2 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l2671_267199


namespace NUMINAMATH_CALUDE_solve_for_Y_l2671_267160

theorem solve_for_Y : ∃ Y : ℤ, 80 - (Y - (6 + 2 * (7 - 8 - 5))) = 89 ∧ Y = -15 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_Y_l2671_267160


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2671_267167

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ+ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ+, a (n + 1) = a n + d

/-- Partial sum of an arithmetic sequence -/
def partial_sum (a : ℕ+ → ℝ) (n : ℕ+) : ℝ :=
  (Finset.range n.val).sum (λ i => a ⟨i + 1, Nat.succ_pos i⟩)

theorem arithmetic_sequence_properties
  (a : ℕ+ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : partial_sum a 6 > partial_sum a 7 ∧ partial_sum a 7 > partial_sum a 5) :
  (∃ d : ℝ, d < 0 ∧ ∀ n : ℕ+, a (n + 1) = a n + d) ∧
  partial_sum a 11 > 0 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l2671_267167


namespace NUMINAMATH_CALUDE_triangle_properties_l2671_267111

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the conditions
def is_valid_triangle (t : Triangle) : Prop :=
  -- a and b are roots of x^2 - 2√3x + 2 = 0
  t.a^2 - 2 * Real.sqrt 3 * t.a + 2 = 0 ∧
  t.b^2 - 2 * Real.sqrt 3 * t.b + 2 = 0 ∧
  -- 2cos(A+B) = 1
  2 * Real.cos (t.A + t.B) = 1

-- State the theorem
theorem triangle_properties (t : Triangle) (h : is_valid_triangle t) :
  t.C = 2 * π / 3 ∧  -- 120° in radians
  t.c = Real.sqrt 10 ∧
  (1 / 2 : ℝ) * t.a * t.b * Real.sin t.C = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l2671_267111


namespace NUMINAMATH_CALUDE_sum_areas_externally_tangent_circles_l2671_267136

/-- Given a 5-12-13 right triangle with vertices as centers of three mutually externally tangent circles,
    the sum of the areas of these circles is 113π. -/
theorem sum_areas_externally_tangent_circles (r s t : ℝ) : 
  r + s = 5 →
  s + t = 12 →
  r + t = 13 →
  π * (r^2 + s^2 + t^2) = 113 * π := by
  sorry

end NUMINAMATH_CALUDE_sum_areas_externally_tangent_circles_l2671_267136


namespace NUMINAMATH_CALUDE_product_abcd_is_zero_l2671_267130

theorem product_abcd_is_zero
  (a b c d : ℤ)
  (eq1 : 3*a + 2*b + 4*c + 8*d = 40)
  (eq2 : 4*(d+c) = b)
  (eq3 : 2*b + 2*c = a)
  (eq4 : c + 1 = d) :
  a * b * c * d = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_abcd_is_zero_l2671_267130


namespace NUMINAMATH_CALUDE_probability_select_one_coastal_l2671_267186

/-- Represents a city that can be either coastal or inland -/
inductive City
| coastal : City
| inland : City

/-- The set of all cities -/
def allCities : Finset City := sorry

/-- The set of coastal cities -/
def coastalCities : Finset City := sorry

theorem probability_select_one_coastal :
  (2 : ℕ) = Finset.card coastalCities →
  (4 : ℕ) = Finset.card allCities →
  (1 : ℚ) / 2 = Finset.card coastalCities / Finset.card allCities := by
  sorry

end NUMINAMATH_CALUDE_probability_select_one_coastal_l2671_267186


namespace NUMINAMATH_CALUDE_equation_root_l2671_267190

theorem equation_root (a b c d : ℝ) (h1 : a + d = 2015) (h2 : b + c = 2015) (h3 : a ≠ c) :
  (∀ x : ℝ, (x - a) * (x - b) = (x - c) * (x - d)) ↔ x = 1007.5 :=
by sorry

end NUMINAMATH_CALUDE_equation_root_l2671_267190


namespace NUMINAMATH_CALUDE_converse_x_gt_abs_y_implies_x_gt_y_l2671_267142

theorem converse_x_gt_abs_y_implies_x_gt_y : ∀ x y : ℝ, x > |y| → x > y := by
  sorry

end NUMINAMATH_CALUDE_converse_x_gt_abs_y_implies_x_gt_y_l2671_267142


namespace NUMINAMATH_CALUDE_remainder_problem_l2671_267158

theorem remainder_problem : (7 * 10^20 + 2^20) % 11 = 8 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l2671_267158


namespace NUMINAMATH_CALUDE_geologists_probability_l2671_267134

/-- Represents a circular field with radial roads -/
structure CircularField where
  numRoads : ℕ
  radius : ℝ

/-- Represents a geologist's position -/
structure GeologistPosition where
  road : ℕ
  distance : ℝ

/-- Calculates the distance between two geologists -/
def distanceBetweenGeologists (field : CircularField) (pos1 pos2 : GeologistPosition) : ℝ :=
  sorry

/-- Calculates the probability of two geologists being at least a certain distance apart -/
def probabilityOfDistance (field : CircularField) (speed time minDistance : ℝ) : ℝ :=
  sorry

theorem geologists_probability (field : CircularField) :
  field.numRoads = 6 →
  probabilityOfDistance field 4 1 6 = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_geologists_probability_l2671_267134


namespace NUMINAMATH_CALUDE_quadratic_inequality_and_hyperbola_l2671_267141

-- Define the solution set condition
def solution_set (a b : ℝ) : Prop :=
  ∀ x : ℝ, (a * x^2 - 3*x + 2 > 0) ↔ (x < 1 ∨ x > b)

-- Define the main theorem
theorem quadratic_inequality_and_hyperbola (a b : ℝ) :
  solution_set a b →
  (∀ x y : ℝ, x > 0 → y > 0 → a/x + b/y = 1 →
    (∀ k : ℝ, (∀ x y : ℝ, x > 0 → y > 0 → a/x + b/y = 1 → 2*x + y ≥ k) → k ≤ 8)) →
  a = 1 ∧ b = 2 := by
sorry


end NUMINAMATH_CALUDE_quadratic_inequality_and_hyperbola_l2671_267141


namespace NUMINAMATH_CALUDE_class_average_mark_l2671_267115

theorem class_average_mark (total_students : ℕ) (excluded_students : ℕ) 
  (excluded_average : ℝ) (remaining_average : ℝ) : 
  total_students = 35 →
  excluded_students = 5 →
  excluded_average = 20 →
  remaining_average = 90 →
  (total_students * (total_students * remaining_average - 
    excluded_students * excluded_average)) / 
    (total_students * (total_students - excluded_students)) = 80 := by
  sorry

end NUMINAMATH_CALUDE_class_average_mark_l2671_267115


namespace NUMINAMATH_CALUDE_car_travel_distance_l2671_267178

/-- Proves that Car X travels 294 miles from when Car Y starts until both cars stop -/
theorem car_travel_distance (speed_x speed_y : ℝ) (head_start : ℝ) : 
  speed_x = 35 →
  speed_y = 40 →
  head_start = 1.2 →
  (speed_y * (head_start + (294 / speed_x))) = (speed_x * (294 / speed_x) + speed_x * head_start) →
  294 = speed_x * (294 / speed_x) :=
by
  sorry

#check car_travel_distance

end NUMINAMATH_CALUDE_car_travel_distance_l2671_267178


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l2671_267150

-- Define the universal set U
def U : Set ℝ := {x | -Real.sqrt 3 < x}

-- Define set A
def A : Set ℝ := {x | 2^x > Real.sqrt 2}

-- Statement to prove
theorem complement_of_A_in_U :
  Set.compl A ∩ U = Set.Icc (-Real.sqrt 3) (1/2) := by sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l2671_267150


namespace NUMINAMATH_CALUDE_trajectory_of_point_B_l2671_267155

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in 2D space of the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Definition of a parallelogram ABCD -/
def is_parallelogram (A B C D : Point) : Prop := sorry

/-- Definition of a point lying on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Theorem: Trajectory of point B in parallelogram ABCD -/
theorem trajectory_of_point_B 
  (A B C D : Point)
  (h_parallelogram : is_parallelogram A B C D)
  (h_A : A = ⟨3, -1⟩)
  (h_C : C = ⟨2, -3⟩)
  (l : Line)
  (h_l : l = ⟨3, -1, 1⟩)
  (h_D_on_l : point_on_line D l) :
  point_on_line B ⟨3, -1, -20⟩ := by
    sorry

end NUMINAMATH_CALUDE_trajectory_of_point_B_l2671_267155


namespace NUMINAMATH_CALUDE_nested_expression_value_l2671_267184

theorem nested_expression_value : 4 * (1 + 4 * (1 + 4 * (1 + 4 * (1 + 4)))) = 1364 := by
  sorry

end NUMINAMATH_CALUDE_nested_expression_value_l2671_267184


namespace NUMINAMATH_CALUDE_line_l_passes_through_fixed_point_line_l_not_in_fourth_quadrant_min_area_of_triangle_AOB_l2671_267163

-- Define the line l: kx - y + 1 + 2k = 0
def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0

-- Define the fixed point
def fixed_point : ℝ × ℝ := (-2, 1)

-- Define the fourth quadrant
def fourth_quadrant (x y : ℝ) : Prop := x > 0 ∧ y < 0

-- Define the negative half of x-axis and positive half of y-axis
def neg_x_axis (x : ℝ) : Prop := x < 0
def pos_y_axis (y : ℝ) : Prop := y > 0

-- Define the origin
def origin : ℝ × ℝ := (0, 0)

-- Define the area of a triangle given three points
def triangle_area (a b c : ℝ × ℝ) : ℝ := sorry

-- Theorem statements
theorem line_l_passes_through_fixed_point :
  ∀ k : ℝ, line_l k (fixed_point.1) (fixed_point.2) := by sorry

theorem line_l_not_in_fourth_quadrant :
  ∀ k x y : ℝ, line_l k x y → ¬(fourth_quadrant x y) → k ≥ 0 := by sorry

theorem min_area_of_triangle_AOB :
  ∀ k x y : ℝ,
  line_l k x y →
  neg_x_axis x →
  pos_y_axis y →
  let a := (x, 0)
  let b := (0, y)
  triangle_area a origin b ≥ 4 ∧
  (triangle_area a origin b = 4 ↔ line_l (1/2) x y) := by sorry

end NUMINAMATH_CALUDE_line_l_passes_through_fixed_point_line_l_not_in_fourth_quadrant_min_area_of_triangle_AOB_l2671_267163


namespace NUMINAMATH_CALUDE_min_value_sqrt_sum_equality_condition_l2671_267169

theorem min_value_sqrt_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2) / c + (b^2 + c^2) / a + (c^2 + a^2) / b) ≥ Real.sqrt 6 :=
by sorry

theorem equality_condition (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2) / c + (b^2 + c^2) / a + (c^2 + a^2) / b) = Real.sqrt 6 ↔ a = b ∧ b = c :=
by sorry

end NUMINAMATH_CALUDE_min_value_sqrt_sum_equality_condition_l2671_267169


namespace NUMINAMATH_CALUDE_prob_at_least_one_second_class_l2671_267183

/-- The probability of selecting at least one second-class item when randomly choosing 3 items
    from a set of 10 items, where 6 are first-class and 4 are second-class. -/
theorem prob_at_least_one_second_class (total : Nat) (first_class : Nat) (second_class : Nat) (selected : Nat)
    (h1 : total = 10)
    (h2 : first_class = 6)
    (h3 : second_class = 4)
    (h4 : selected = 3)
    (h5 : total = first_class + second_class) :
    (1 : ℚ) - (Nat.choose first_class selected : ℚ) / (Nat.choose total selected : ℚ) = 5/6 := by
  sorry

end NUMINAMATH_CALUDE_prob_at_least_one_second_class_l2671_267183


namespace NUMINAMATH_CALUDE_evaluate_fraction_l2671_267153

theorem evaluate_fraction : (18 : ℝ) / (14 * 5.3) = 1.8 / 7.42 := by sorry

end NUMINAMATH_CALUDE_evaluate_fraction_l2671_267153


namespace NUMINAMATH_CALUDE_square_divisibility_l2671_267188

theorem square_divisibility (n : ℕ) (h1 : n > 0) (h2 : ∀ d : ℕ, d ∣ n → d ≤ 6) :
  36 ∣ n^2 := by
sorry

end NUMINAMATH_CALUDE_square_divisibility_l2671_267188


namespace NUMINAMATH_CALUDE_return_journey_speed_l2671_267119

/-- Given a round trip with the following conditions:
    - The distance between home and the retreat is 300 miles each way
    - The average speed to the retreat was 50 miles per hour
    - The round trip took 10 hours
    - The same route was taken both ways
    Prove that the average speed on the return journey is 75 mph. -/
theorem return_journey_speed (distance : ℝ) (speed_to : ℝ) (total_time : ℝ) :
  distance = 300 →
  speed_to = 50 →
  total_time = 10 →
  let time_to : ℝ := distance / speed_to
  let time_from : ℝ := total_time - time_to
  let speed_from : ℝ := distance / time_from
  speed_from = 75 := by sorry

end NUMINAMATH_CALUDE_return_journey_speed_l2671_267119


namespace NUMINAMATH_CALUDE_largest_integer_less_than_sqrt7_plus_sqrt3_power6_l2671_267120

theorem largest_integer_less_than_sqrt7_plus_sqrt3_power6 :
  ⌊(Real.sqrt 7 + Real.sqrt 3)^6⌋ = 7039 := by sorry

end NUMINAMATH_CALUDE_largest_integer_less_than_sqrt7_plus_sqrt3_power6_l2671_267120


namespace NUMINAMATH_CALUDE_probability_information_both_clubs_l2671_267131

def total_students : ℕ := 30
def art_club_students : ℕ := 22
def music_club_students : ℕ := 25

def probability_both_clubs : ℚ := 397 / 435

theorem probability_information_both_clubs :
  let students_in_both := art_club_students + music_club_students - total_students
  let students_only_art := art_club_students - students_in_both
  let students_only_music := music_club_students - students_in_both
  let total_combinations := total_students.choose 2
  let incompatible_combinations := students_only_art.choose 2 + students_only_music.choose 2
  (1 : ℚ) - (incompatible_combinations : ℚ) / total_combinations = probability_both_clubs :=
by sorry

end NUMINAMATH_CALUDE_probability_information_both_clubs_l2671_267131


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l2671_267112

-- Define the function f
def f (a x : ℝ) : ℝ := |a - x| + |x + 2|

-- Part I
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x < 7} = {x : ℝ | -3 < x ∧ x < 4} := by sorry

-- Part II
theorem range_of_a :
  {a : ℝ | ∀ x ∈ Set.Icc 1 2, |f a x| ≤ |x + 4|} = Set.Icc 0 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_when_a_is_3_range_of_a_l2671_267112


namespace NUMINAMATH_CALUDE_quadratic_no_real_roots_l2671_267173

theorem quadratic_no_real_roots : 
  ∀ x : ℝ, x^2 + 2*x + 5 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_no_real_roots_l2671_267173


namespace NUMINAMATH_CALUDE_julie_school_year_work_hours_l2671_267198

/-- Julie's summer work scenario -/
structure SummerWork where
  hoursPerWeek : ℕ
  weeks : ℕ
  earnings : ℕ

/-- Julie's school year work scenario -/
structure SchoolYearWork where
  weeks : ℕ
  targetEarnings : ℕ

/-- Calculate required hours per week for school year -/
def requiredHoursPerWeek (summer : SummerWork) (schoolYear : SchoolYearWork) : ℕ :=
  let hourlyWage := summer.earnings / (summer.hoursPerWeek * summer.weeks)
  let totalHours := schoolYear.targetEarnings / hourlyWage
  totalHours / schoolYear.weeks

/-- Theorem: Julie needs to work 15 hours per week during school year -/
theorem julie_school_year_work_hours 
  (summer : SummerWork) 
  (schoolYear : SchoolYearWork) 
  (h1 : summer.hoursPerWeek = 60)
  (h2 : summer.weeks = 10)
  (h3 : summer.earnings = 6000)
  (h4 : schoolYear.weeks = 40)
  (h5 : schoolYear.targetEarnings = 6000) : 
  requiredHoursPerWeek summer schoolYear = 15 := by
  sorry

end NUMINAMATH_CALUDE_julie_school_year_work_hours_l2671_267198


namespace NUMINAMATH_CALUDE_tates_education_years_l2671_267102

/-- The total years Tate spent in high school and college -/
def total_education_years (normal_hs_duration : ℕ) (hs_reduction : ℕ) (college_multiplier : ℕ) : ℕ :=
  let hs_duration := normal_hs_duration - hs_reduction
  let college_duration := hs_duration * college_multiplier
  hs_duration + college_duration

/-- Theorem stating that Tate's total education years is 12 -/
theorem tates_education_years :
  total_education_years 4 1 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_tates_education_years_l2671_267102


namespace NUMINAMATH_CALUDE_second_year_increase_is_25_percent_l2671_267179

/-- Calculates the percentage increase in the second year given the initial population,
    first year increase percentage, and final population after two years. -/
def second_year_increase (initial_population : ℕ) (first_year_increase : ℚ) (final_population : ℕ) : ℚ :=
  let population_after_first_year := initial_population * (1 + first_year_increase)
  let second_year_factor := final_population / population_after_first_year
  (second_year_factor - 1) * 100

theorem second_year_increase_is_25_percent :
  second_year_increase 800 (22/100) 1220 = 25 := by
  sorry

#eval second_year_increase 800 (22/100) 1220

end NUMINAMATH_CALUDE_second_year_increase_is_25_percent_l2671_267179


namespace NUMINAMATH_CALUDE_playstation_payment_l2671_267105

theorem playstation_payment (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h_total : x₁ + x₂ + x₃ + x₄ + x₅ = 120)
  (h_x₁ : x₁ = (1/3) * (x₂ + x₃ + x₄ + x₅))
  (h_x₂ : x₂ = (1/4) * (x₁ + x₃ + x₄ + x₅))
  (h_x₃ : x₃ = (1/5) * (x₁ + x₂ + x₄ + x₅))
  (h_x₄ : x₄ = (1/6) * (x₁ + x₂ + x₃ + x₅)) :
  x₅ = 40 := by
sorry

end NUMINAMATH_CALUDE_playstation_payment_l2671_267105


namespace NUMINAMATH_CALUDE_rational_solutions_quadratic_l2671_267137

theorem rational_solutions_quadratic (k : ℕ+) : 
  (∃ x : ℚ, k * x^2 + 24 * x + 3 * k = 0) ↔ k = 6 :=
sorry

end NUMINAMATH_CALUDE_rational_solutions_quadratic_l2671_267137


namespace NUMINAMATH_CALUDE_price_crossover_year_l2671_267161

def price_X (year : ℕ) : ℚ :=
  5.20 + 0.45 * (year - 2001 : ℚ)

def price_Y (year : ℕ) : ℚ :=
  7.30 + 0.20 * (year - 2001 : ℚ)

theorem price_crossover_year :
  (∀ y : ℕ, y < 2010 → price_X y ≤ price_Y y) ∧
  price_X 2010 > price_Y 2010 := by
  sorry

end NUMINAMATH_CALUDE_price_crossover_year_l2671_267161


namespace NUMINAMATH_CALUDE_unique_n_congruence_l2671_267162

theorem unique_n_congruence : ∃! n : ℕ, 0 ≤ n ∧ n ≤ 9 ∧ n ≡ 123456 [MOD 8] := by
  sorry

end NUMINAMATH_CALUDE_unique_n_congruence_l2671_267162


namespace NUMINAMATH_CALUDE_sequence_geometric_iff_d_eq_3m_l2671_267139

/-- Given a sequence {a_n} with the following properties:
    1. 0 < a₁ < 1/m, where m is a positive integer
    2. a_{n+1} = a_n + 1/m if a_n < 3
    3. a_{n+1} = a_n / d if a_n ≥ 3
    4. d ≥ 3m

    This theorem states that the sequence with terms 
    a₂ - 1/m, a_{3m+2} - 1/m, a_{6m+2} - 1/m, a_{9m+2} - 1/m 
    forms a geometric sequence if and only if d = 3m. -/
theorem sequence_geometric_iff_d_eq_3m (m : ℕ) (d : ℝ) (a : ℕ → ℝ) 
    (h1 : 0 < a 1 ∧ a 1 < 1 / m) 
    (h2 : ∀ n, a n < 3 → a (n + 1) = a n + 1 / m) 
    (h3 : ∀ n, a n ≥ 3 → a (n + 1) = a n / d) 
    (h4 : d ≥ 3 * m) :
  (∃ r, a 2 - 1/m = r * (a 1 - 1/m) ∧ 
        a (3*m + 2) - 1/m = r * (a 2 - 1/m) ∧
        a (6*m + 2) - 1/m = r * (a (3*m + 2) - 1/m) ∧
        a (9*m + 2) - 1/m = r * (a (6*m + 2) - 1/m)) ↔ 
  d = 3 * m := by
sorry


end NUMINAMATH_CALUDE_sequence_geometric_iff_d_eq_3m_l2671_267139


namespace NUMINAMATH_CALUDE_combined_tax_rate_l2671_267133

/-- Combined tax rate calculation -/
theorem combined_tax_rate 
  (john_tax_rate : ℝ) 
  (ingrid_tax_rate : ℝ) 
  (john_income : ℝ) 
  (ingrid_income : ℝ) 
  (h1 : john_tax_rate = 0.30) 
  (h2 : ingrid_tax_rate = 0.40) 
  (h3 : john_income = 56000) 
  (h4 : ingrid_income = 74000) : 
  ∃ (combined_rate : ℝ), 
    combined_rate = (john_tax_rate * john_income + ingrid_tax_rate * ingrid_income) / (john_income + ingrid_income) :=
by
  sorry

#eval (0.30 * 56000 + 0.40 * 74000) / (56000 + 74000)

end NUMINAMATH_CALUDE_combined_tax_rate_l2671_267133


namespace NUMINAMATH_CALUDE_image_of_4_neg2_l2671_267194

/-- The mapping f from ℝ² to ℝ² defined by f(x, y) = (xy, x+y) -/
def f : ℝ × ℝ → ℝ × ℝ := λ (x, y) => (x * y, x + y)

/-- The theorem stating that the image of (4, -2) under f is (-8, 2) -/
theorem image_of_4_neg2 : f (4, -2) = (-8, 2) := by sorry

end NUMINAMATH_CALUDE_image_of_4_neg2_l2671_267194
