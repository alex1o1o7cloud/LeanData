import Mathlib

namespace NUMINAMATH_CALUDE_brother_age_l3445_344533

theorem brother_age (man_age brother_age : ℕ) : 
  man_age = brother_age + 12 →
  man_age + 2 = 2 * (brother_age + 2) →
  brother_age = 10 := by
sorry

end NUMINAMATH_CALUDE_brother_age_l3445_344533


namespace NUMINAMATH_CALUDE_craig_dave_bench_press_ratio_l3445_344503

/-- Proves that Craig's bench press is 20% of Dave's bench press -/
theorem craig_dave_bench_press_ratio :
  let dave_weight : ℝ := 175
  let dave_bench_press : ℝ := 3 * dave_weight
  let mark_bench_press : ℝ := 55
  let craig_bench_press : ℝ := mark_bench_press + 50
  (craig_bench_press / dave_bench_press) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_craig_dave_bench_press_ratio_l3445_344503


namespace NUMINAMATH_CALUDE_safe_elixir_preparations_l3445_344560

/-- Represents the number of magical herbs available. -/
def num_herbs : ℕ := 4

/-- Represents the number of mystical crystals available. -/
def num_crystals : ℕ := 6

/-- Represents the number of forbidden herb-crystal combinations. -/
def num_forbidden : ℕ := 3

/-- Calculates the number of safe elixir preparations. -/
def safe_preparations : ℕ := num_herbs * num_crystals - num_forbidden

/-- Theorem stating that the number of safe elixir preparations is 21. -/
theorem safe_elixir_preparations :
  safe_preparations = 21 := by sorry

end NUMINAMATH_CALUDE_safe_elixir_preparations_l3445_344560


namespace NUMINAMATH_CALUDE_sally_reading_time_l3445_344595

/-- The number of pages Sally reads on a weekday -/
def weekday_pages : ℕ := 10

/-- The number of pages Sally reads on a weekend day -/
def weekend_pages : ℕ := 20

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The total number of pages in Sally's book -/
def book_pages : ℕ := 180

/-- The number of weeks it takes Sally to finish the book -/
def weeks_to_finish : ℕ := 2

/-- Theorem stating that it takes Sally 2 weeks to finish the book -/
theorem sally_reading_time :
  weekday_pages * weekdays + weekend_pages * weekend_days = book_pages / weeks_to_finish :=
by sorry

end NUMINAMATH_CALUDE_sally_reading_time_l3445_344595


namespace NUMINAMATH_CALUDE_thabo_owns_160_books_l3445_344599

/-- Represents the number of books Thabo owns in each category -/
structure BookCollection where
  hardcoverNonfiction : ℕ
  paperbackNonfiction : ℕ
  paperbackFiction : ℕ

/-- Thabo's book collection satisfying the given conditions -/
def thabosBooks : BookCollection where
  hardcoverNonfiction := 25
  paperbackNonfiction := 25 + 20
  paperbackFiction := 2 * (25 + 20)

/-- The total number of books in a collection -/
def totalBooks (books : BookCollection) : ℕ :=
  books.hardcoverNonfiction + books.paperbackNonfiction + books.paperbackFiction

/-- Theorem stating that Thabo owns 160 books in total -/
theorem thabo_owns_160_books : totalBooks thabosBooks = 160 := by
  sorry


end NUMINAMATH_CALUDE_thabo_owns_160_books_l3445_344599


namespace NUMINAMATH_CALUDE_transportation_cost_calculation_l3445_344526

def transportation_cost (initial_amount dress_cost pants_cost jacket_cost dress_count pants_count jacket_count remaining_amount : ℕ) : ℕ :=
  let clothes_cost := dress_cost * dress_count + pants_cost * pants_count + jacket_cost * jacket_count
  let total_spent := initial_amount - remaining_amount
  total_spent - clothes_cost

theorem transportation_cost_calculation :
  transportation_cost 400 20 12 30 5 3 4 139 = 5 :=
by sorry

end NUMINAMATH_CALUDE_transportation_cost_calculation_l3445_344526


namespace NUMINAMATH_CALUDE_lcm_hcf_relation_l3445_344575

theorem lcm_hcf_relation (x : ℕ) :
  Nat.lcm 4 x = 36 ∧ Nat.gcd 4 x = 2 → x = 18 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_relation_l3445_344575


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l3445_344542

/-- Given a principal amount where the compound interest at 5% per annum for 2 years is $51.25,
    prove that the simple interest at the same rate and time is $50 -/
theorem simple_interest_calculation (P : ℝ) : 
  P * ((1 + 0.05)^2 - 1) = 51.25 → P * 0.05 * 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l3445_344542


namespace NUMINAMATH_CALUDE_photocopy_cost_calculation_l3445_344522

/-- The cost of one photocopy -/
def photocopy_cost : ℝ := sorry

/-- The discount rate for orders over 100 copies -/
def discount_rate : ℝ := 0.25

/-- The number of copies each person needs -/
def copies_per_person : ℕ := 80

/-- The total number of copies when combining orders -/
def total_copies : ℕ := 2 * copies_per_person

/-- The amount saved per person when combining orders -/
def savings_per_person : ℝ := 0.40

theorem photocopy_cost_calculation : 
  photocopy_cost = 0.02 :=
by
  sorry

end NUMINAMATH_CALUDE_photocopy_cost_calculation_l3445_344522


namespace NUMINAMATH_CALUDE_ascending_order_l3445_344547

theorem ascending_order (a b c : ℝ) 
  (ha : a = Real.rpow 0.8 0.7)
  (hb : b = Real.rpow 0.8 0.9)
  (hc : c = Real.rpow 1.2 0.8) :
  b < a ∧ a < c := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_l3445_344547


namespace NUMINAMATH_CALUDE_tv_sale_net_effect_l3445_344577

/-- Given a TV set with an original price P, this theorem proves the net effect on total sale value
    after applying discounts, considering sales increase and variable costs. -/
theorem tv_sale_net_effect (P : ℝ) (original_volume : ℝ) (h_pos : P > 0) (h_vol_pos : original_volume > 0) :
  let price_after_initial_reduction := P * (1 - 0.22)
  let bulk_discount := price_after_initial_reduction * 0.05
  let loyalty_discount := price_after_initial_reduction * 0.10
  let price_after_all_discounts := price_after_initial_reduction - bulk_discount - loyalty_discount
  let new_sales_volume := original_volume * 1.86
  let variable_cost_per_unit := price_after_all_discounts * 0.10
  let net_price_after_costs := price_after_all_discounts - variable_cost_per_unit
  let original_total_sale := P * original_volume
  let new_total_sale := net_price_after_costs * new_sales_volume
  let net_effect := new_total_sale - original_total_sale
  ∃ ε > 0, |net_effect / original_total_sale - 0.109862| < ε :=
by sorry

end NUMINAMATH_CALUDE_tv_sale_net_effect_l3445_344577


namespace NUMINAMATH_CALUDE_other_side_heads_probability_is_two_thirds_l3445_344546

/-- Represents the three types of coins -/
inductive Coin
  | Normal
  | TwoHeads
  | TwoTails

/-- Represents the possible outcomes of a coin toss -/
inductive CoinSide
  | Heads
  | Tails

/-- The probability of selecting each type of coin -/
def coinProbability (c : Coin) : ℚ :=
  match c with
  | Coin.Normal => 1/3
  | Coin.TwoHeads => 1/3
  | Coin.TwoTails => 1/3

/-- The probability of getting heads when tossing a specific coin -/
def headsUpProbability (c : Coin) : ℚ :=
  match c with
  | Coin.Normal => 1/2
  | Coin.TwoHeads => 1
  | Coin.TwoTails => 0

/-- The probability that the other side is heads given that heads is showing -/
def otherSideHeadsProbability : ℚ := by
  sorry

theorem other_side_heads_probability_is_two_thirds :
  otherSideHeadsProbability = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_other_side_heads_probability_is_two_thirds_l3445_344546


namespace NUMINAMATH_CALUDE_minimum_jumps_circle_l3445_344586

/-- Represents a jump on the circle of points -/
inductive Jump
| Two  : Jump  -- Jump of 2 points
| Three : Jump  -- Jump of 3 points

/-- Represents a sequence of jumps -/
def JumpSequence := List Jump

/-- Function to check if a sequence of jumps visits all points and returns to start -/
def validSequence (n : Nat) (seq : JumpSequence) : Prop :=
  -- Implementation details omitted
  sorry

theorem minimum_jumps_circle :
  ∀ (seq : JumpSequence),
    validSequence 2016 seq →
    seq.length ≥ 2017 :=
by sorry

end NUMINAMATH_CALUDE_minimum_jumps_circle_l3445_344586


namespace NUMINAMATH_CALUDE_sum_difference_even_odd_100_l3445_344544

/-- Sum of first n positive even integers -/
def sumEvenIntegers (n : ℕ) : ℕ := n * (n + 1)

/-- Sum of first n positive odd integers -/
def sumOddIntegers (n : ℕ) : ℕ := n^2

theorem sum_difference_even_odd_100 :
  sumEvenIntegers 100 - sumOddIntegers 100 = 100 := by
  sorry

end NUMINAMATH_CALUDE_sum_difference_even_odd_100_l3445_344544


namespace NUMINAMATH_CALUDE_marias_blueberries_l3445_344573

/-- Proves that Maria has 8 cartons of blueberries given the problem conditions -/
theorem marias_blueberries 
  (total_needed : ℕ) 
  (strawberries : ℕ) 
  (to_buy : ℕ) 
  (h1 : total_needed = 21)
  (h2 : strawberries = 4)
  (h3 : to_buy = 9) :
  total_needed - (strawberries + to_buy) = 8 := by
  sorry

#eval 21 - (4 + 9)  -- Should output 8

end NUMINAMATH_CALUDE_marias_blueberries_l3445_344573


namespace NUMINAMATH_CALUDE_yellow_second_draw_probability_l3445_344549

/-- The probability of drawing a yellow ball on the second draw -/
def prob_yellow_second_draw (yellow white : ℕ) : ℚ :=
  (white : ℚ) / (yellow + white) * yellow / (yellow + white - 1)

/-- Theorem: The probability of drawing a yellow ball on the second draw
    is 4/15 when there are 6 yellow and 4 white balls -/
theorem yellow_second_draw_probability :
  prob_yellow_second_draw 6 4 = 4 / 15 := by
  sorry

end NUMINAMATH_CALUDE_yellow_second_draw_probability_l3445_344549


namespace NUMINAMATH_CALUDE_bride_groom_age_difference_oldest_bride_problem_l3445_344570

theorem bride_groom_age_difference : ℕ → ℕ → ℕ → Prop :=
  fun total_age groom_age age_difference =>
    let bride_age := total_age - groom_age
    bride_age - groom_age = age_difference

theorem oldest_bride_problem (total_age groom_age : ℕ) 
  (h1 : total_age = 185) 
  (h2 : groom_age = 83) : 
  bride_groom_age_difference total_age groom_age 19 := by
  sorry

end NUMINAMATH_CALUDE_bride_groom_age_difference_oldest_bride_problem_l3445_344570


namespace NUMINAMATH_CALUDE_even_function_property_l3445_344578

-- Define an even function
def EvenFunction (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the theorem
theorem even_function_property (f : ℝ → ℝ) 
  (h1 : EvenFunction f) 
  (h2 : ∀ x < 0, f x = x * (x + 1)) : 
  ∀ x > 0, f x = x * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_even_function_property_l3445_344578


namespace NUMINAMATH_CALUDE_rectangular_cross_section_shapes_l3445_344523

/-- Enumeration of the geometric shapes in question -/
inductive GeometricShape
  | RectangularPrism
  | Cylinder
  | Cone
  | Cube

/-- Predicate to determine if a shape can have a rectangular cross-section -/
def has_rectangular_cross_section (shape : GeometricShape) : Prop :=
  match shape with
  | GeometricShape.RectangularPrism => true
  | GeometricShape.Cylinder => true
  | GeometricShape.Cone => false
  | GeometricShape.Cube => true

/-- The set of shapes that can have a rectangular cross-section -/
def shapes_with_rectangular_cross_section : Set GeometricShape :=
  {shape | has_rectangular_cross_section shape}

/-- Theorem stating which shapes can have a rectangular cross-section -/
theorem rectangular_cross_section_shapes :
  shapes_with_rectangular_cross_section =
    {GeometricShape.RectangularPrism, GeometricShape.Cylinder, GeometricShape.Cube} :=
by sorry


end NUMINAMATH_CALUDE_rectangular_cross_section_shapes_l3445_344523


namespace NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l3445_344598

theorem five_digit_divisible_by_nine :
  ∀ B : ℕ,
  (0 ≤ B ∧ B ≤ 9) →
  (40000 + 10000*B + 1000*B + 100 + 10 + 3) % 9 = 0 →
  B = 5 := by
sorry

end NUMINAMATH_CALUDE_five_digit_divisible_by_nine_l3445_344598


namespace NUMINAMATH_CALUDE_concentric_circles_annulus_area_l3445_344567

theorem concentric_circles_annulus_area (r R : ℝ) (h : r > 0) (H : R > 0) (eq : π * r^2 = π * R^2 / 2) :
  let annulus_area := π * R^2 / 2 - 2 * (π * R^2 / 4 - R^2 / 2)
  annulus_area = 2 * r^2 :=
by sorry

end NUMINAMATH_CALUDE_concentric_circles_annulus_area_l3445_344567


namespace NUMINAMATH_CALUDE_sqrt_meaningful_range_l3445_344525

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 * x - 4) ↔ x ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_meaningful_range_l3445_344525


namespace NUMINAMATH_CALUDE_existsNonSymmetricalEqualTriangles_l3445_344516

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle in 2D space -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents an ellipse -/
structure Ellipse :=
  (center : Point)
  (semiMajorAxis : ℝ)
  (semiMinorAxis : ℝ)

/-- Checks if a point is inside or on the ellipse -/
def isPointInEllipse (p : Point) (e : Ellipse) : Prop :=
  (p.x - e.center.x)^2 / e.semiMajorAxis^2 + (p.y - e.center.y)^2 / e.semiMinorAxis^2 ≤ 1

/-- Checks if a triangle is inscribed in an ellipse -/
def isTriangleInscribed (t : Triangle) (e : Ellipse) : Prop :=
  isPointInEllipse t.a e ∧ isPointInEllipse t.b e ∧ isPointInEllipse t.c e

/-- Checks if two triangles are equal -/
def areTrianglesEqual (t1 t2 : Triangle) : Prop :=
  -- Definition of triangle equality (e.g., same side lengths)
  sorry

/-- Checks if two triangles are symmetrical with respect to the x-axis -/
def areTrianglesSymmetricalXAxis (t1 t2 : Triangle) : Prop :=
  -- Definition of symmetry with respect to x-axis
  sorry

/-- Checks if two triangles are symmetrical with respect to the y-axis -/
def areTrianglesSymmetricalYAxis (t1 t2 : Triangle) : Prop :=
  -- Definition of symmetry with respect to y-axis
  sorry

/-- Checks if two triangles are symmetrical with respect to the center -/
def areTrianglesSymmetricalCenter (t1 t2 : Triangle) (e : Ellipse) : Prop :=
  -- Definition of symmetry with respect to center
  sorry

/-- Main theorem: There exist two equal triangles inscribed in an ellipse that are not symmetrical -/
theorem existsNonSymmetricalEqualTriangles :
  ∃ (e : Ellipse) (t1 t2 : Triangle),
    isTriangleInscribed t1 e ∧
    isTriangleInscribed t2 e ∧
    areTrianglesEqual t1 t2 ∧
    ¬(areTrianglesSymmetricalXAxis t1 t2 ∨
      areTrianglesSymmetricalYAxis t1 t2 ∨
      areTrianglesSymmetricalCenter t1 t2 e) :=
by
  sorry

end NUMINAMATH_CALUDE_existsNonSymmetricalEqualTriangles_l3445_344516


namespace NUMINAMATH_CALUDE_valerie_stamps_l3445_344591

/-- Calculates the total number of stamps needed for mailing various items. -/
def total_stamps (thank_you_cards : ℕ) (bills : ℕ) (extra_rebates : ℕ) : ℕ :=
  let rebates := bills + extra_rebates
  let job_applications := 2 * rebates
  let regular_stamps := thank_you_cards + bills - 1 + rebates + job_applications
  regular_stamps + 1  -- Add 1 for the extra stamp on the electric bill

theorem valerie_stamps :
  total_stamps 3 2 3 = 21 :=
by sorry

end NUMINAMATH_CALUDE_valerie_stamps_l3445_344591


namespace NUMINAMATH_CALUDE_movie_ticket_ratio_l3445_344553

def monday_cost : ℚ := 5
def wednesday_cost : ℚ := 2 * monday_cost

theorem movie_ticket_ratio :
  ∃ (saturday_cost : ℚ),
    wednesday_cost + saturday_cost = 35 ∧
    saturday_cost / monday_cost = 5 := by
  sorry

end NUMINAMATH_CALUDE_movie_ticket_ratio_l3445_344553


namespace NUMINAMATH_CALUDE_sqrt_product_equality_l3445_344565

theorem sqrt_product_equality (x : ℝ) (h1 : x > 0) 
  (h2 : Real.sqrt (16 * x) * Real.sqrt (5 * x) * Real.sqrt (6 * x) * Real.sqrt (30 * x) = 30) : 
  x = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_sqrt_product_equality_l3445_344565


namespace NUMINAMATH_CALUDE_negation_equivalence_l3445_344574

theorem negation_equivalence :
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 - 5*x₀ + 6 > 0) ↔ 
  (∀ x : ℝ, x > 0 → x^2 - 5*x + 6 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3445_344574


namespace NUMINAMATH_CALUDE_nested_f_evaluation_l3445_344571

/-- The function f(x) = x^2 - 3x + 1 -/
def f (x : ℤ) : ℤ := x^2 - 3*x + 1

/-- Theorem stating that f(f(f(f(f(f(-1)))))) = 3432163846882600 -/
theorem nested_f_evaluation : f (f (f (f (f (f (-1)))))) = 3432163846882600 := by
  sorry

end NUMINAMATH_CALUDE_nested_f_evaluation_l3445_344571


namespace NUMINAMATH_CALUDE_stating_min_rows_for_150_cans_l3445_344529

/-- 
Represents the number of cans in a row given its position
-/
def cans_in_row (n : ℕ) : ℕ := 3 * n

/-- 
Calculates the total number of cans for a given number of rows
-/
def total_cans (n : ℕ) : ℕ := n * (cans_in_row 1 + cans_in_row n) / 2

/-- 
Theorem stating that 10 is the minimum number of rows needed to have at least 150 cans
-/
theorem min_rows_for_150_cans : 
  (∀ k < 10, total_cans k < 150) ∧ total_cans 10 ≥ 150 := by
  sorry

end NUMINAMATH_CALUDE_stating_min_rows_for_150_cans_l3445_344529


namespace NUMINAMATH_CALUDE_top_square_after_folds_l3445_344535

/-- Represents a 6x6 grid of numbers -/
def Grid := Fin 6 → Fin 6 → Nat

/-- Initial grid configuration -/
def initial_grid : Grid :=
  fun i j => 6 * i.val + j.val + 1

/-- Fold operation types -/
inductive FoldType
  | TopOver
  | BottomOver
  | RightOver
  | LeftOver

/-- Apply a single fold operation to the grid -/
def apply_fold (g : Grid) (ft : FoldType) : Grid :=
  sorry  -- Implementation of folding logic

/-- Sequence of folds as described in the problem -/
def fold_sequence : List FoldType :=
  [FoldType.TopOver, FoldType.BottomOver, FoldType.RightOver, 
   FoldType.LeftOver, FoldType.TopOver, FoldType.RightOver]

/-- Apply a sequence of folds to the grid -/
def apply_fold_sequence (g : Grid) (folds : List FoldType) : Grid :=
  sorry  -- Implementation of applying multiple folds

theorem top_square_after_folds (g : Grid) :
  g = initial_grid →
  (apply_fold_sequence g fold_sequence) 0 0 = 22 :=
sorry

end NUMINAMATH_CALUDE_top_square_after_folds_l3445_344535


namespace NUMINAMATH_CALUDE_consecutive_integers_count_l3445_344530

def list_K : List ℤ := sorry

theorem consecutive_integers_count :
  (list_K.head? = some (-3)) ∧ 
  (∀ i j, i ∈ list_K → j ∈ list_K → i < j → ∀ k, i < k ∧ k < j → k ∈ list_K) ∧
  (∃ max_pos ∈ list_K, max_pos > 0 ∧ ∀ x ∈ list_K, x > 0 → x ≤ max_pos) ∧
  (∃ min_pos ∈ list_K, min_pos > 0 ∧ ∀ x ∈ list_K, x > 0 → x ≥ min_pos) ∧
  (∃ max_pos min_pos, max_pos ∈ list_K ∧ min_pos ∈ list_K ∧ 
    max_pos > 0 ∧ min_pos > 0 ∧ max_pos - min_pos = 4) →
  list_K.length = 9 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_count_l3445_344530


namespace NUMINAMATH_CALUDE_minimum_speed_x_l3445_344597

/-- Minimum speed problem for vehicle X --/
theorem minimum_speed_x (distance_xy distance_xz speed_y speed_z : ℝ) 
  (h1 : distance_xy = 500)
  (h2 : distance_xz = 300)
  (h3 : speed_y = 40)
  (h4 : speed_z = 30)
  (h5 : speed_y > speed_z)
  (speed_x : ℝ) :
  speed_x > 135 ↔ distance_xz / (speed_x - speed_z) < distance_xy / (speed_x + speed_y) :=
by sorry

end NUMINAMATH_CALUDE_minimum_speed_x_l3445_344597


namespace NUMINAMATH_CALUDE_sin_2theta_value_l3445_344508

theorem sin_2theta_value (θ : Real) (h : Real.sin θ + Real.cos θ = Real.sqrt 7 / 2) :
  Real.sin (2 * θ) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_sin_2theta_value_l3445_344508


namespace NUMINAMATH_CALUDE_swap_correct_specific_swap_l3445_344502

def swap_values (a b : ℕ) : ℕ × ℕ := 
  let c := a
  let a' := b
  let b' := c
  (a', b')

theorem swap_correct (a b : ℕ) : 
  let (a', b') := swap_values a b
  a' = b ∧ b' = a := by
sorry

theorem specific_swap : 
  let (a', b') := swap_values 10 20
  a' = 20 ∧ b' = 10 := by
sorry

end NUMINAMATH_CALUDE_swap_correct_specific_swap_l3445_344502


namespace NUMINAMATH_CALUDE_dance_workshop_avg_age_children_l3445_344581

theorem dance_workshop_avg_age_children (total_participants : ℕ) 
  (overall_avg_age : ℚ) (num_women : ℕ) (num_men : ℕ) (num_children : ℕ) 
  (avg_age_women : ℚ) (avg_age_men : ℚ) 
  (h1 : total_participants = 50)
  (h2 : overall_avg_age = 20)
  (h3 : num_women = 30)
  (h4 : num_men = 10)
  (h5 : num_children = 10)
  (h6 : avg_age_women = 22)
  (h7 : avg_age_men = 25)
  (h8 : total_participants = num_women + num_men + num_children) :
  (total_participants * overall_avg_age - num_women * avg_age_women - num_men * avg_age_men) / num_children = 9 := by
  sorry

end NUMINAMATH_CALUDE_dance_workshop_avg_age_children_l3445_344581


namespace NUMINAMATH_CALUDE_square_perimeter_l3445_344572

/-- The perimeter of a square with side length 11 cm is 44 cm. -/
theorem square_perimeter : 
  ∀ (s : ℝ), s = 11 → 4 * s = 44 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l3445_344572


namespace NUMINAMATH_CALUDE_dave_tickets_left_l3445_344592

def tickets_left (won : ℕ) (lost : ℕ) (used : ℕ) : ℕ :=
  won - lost - used

theorem dave_tickets_left : tickets_left 14 2 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_dave_tickets_left_l3445_344592


namespace NUMINAMATH_CALUDE_beef_for_community_event_l3445_344537

/-- The amount of beef needed for a given number of hamburgers -/
def beef_needed (hamburgers : ℕ) : ℚ :=
  (4 : ℚ) / 10 * hamburgers

theorem beef_for_community_event : beef_needed 35 = 14 := by
  sorry

end NUMINAMATH_CALUDE_beef_for_community_event_l3445_344537


namespace NUMINAMATH_CALUDE_sin_300_degrees_l3445_344511

theorem sin_300_degrees : Real.sin (300 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_300_degrees_l3445_344511


namespace NUMINAMATH_CALUDE_equation_proof_l3445_344515

theorem equation_proof : 10 * 6 - (9 - 3) * 2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_equation_proof_l3445_344515


namespace NUMINAMATH_CALUDE_cuboid_height_l3445_344576

/-- Proves that a rectangular parallelepiped with given dimensions has a specific height -/
theorem cuboid_height (width length sum_of_edges : ℝ) (h : ℝ) : 
  width = 30 →
  length = 22 →
  sum_of_edges = 224 →
  4 * length + 4 * width + 4 * h = sum_of_edges →
  h = 4 := by
sorry

end NUMINAMATH_CALUDE_cuboid_height_l3445_344576


namespace NUMINAMATH_CALUDE_symmetry_about_y_equals_x_l3445_344548

/-- The set of points (x, y) satisfying the given conditions is symmetric about y = x -/
theorem symmetry_about_y_equals_x (r : ℝ) :
  ∀ (x y : ℝ), x^2 + y^2 ≤ r^2 ∧ x + y > 0 →
  ∃ (x' y' : ℝ), x'^2 + y'^2 ≤ r^2 ∧ x' + y' > 0 ∧ x' = y ∧ y' = x :=
by sorry

end NUMINAMATH_CALUDE_symmetry_about_y_equals_x_l3445_344548


namespace NUMINAMATH_CALUDE_frustum_properties_l3445_344532

/-- Frustum properties -/
structure Frustum where
  r₁ : ℝ  -- radius of top base
  r₂ : ℝ  -- radius of bottom base
  l : ℝ   -- slant height
  h : ℝ   -- height

/-- Theorem about a specific frustum -/
theorem frustum_properties (f : Frustum) (h_r₁ : f.r₁ = 2) (h_r₂ : f.r₂ = 6)
    (h_lateral_area : π * (f.r₁ + f.r₂) * f.l = π * f.r₁^2 + π * f.r₂^2) :
    f.l = 5 ∧ π * f.h * (f.r₁^2 + f.r₂^2 + f.r₁ * f.r₂) / 3 = 52 * π := by
  sorry


end NUMINAMATH_CALUDE_frustum_properties_l3445_344532


namespace NUMINAMATH_CALUDE_shaded_area_is_24_l3445_344593

structure Rectangle where
  width : ℝ
  height : ℝ

structure Triangle where
  base : ℝ
  height : ℝ

def shaded_area (rect : Rectangle) (tri : Triangle) : ℝ :=
  sorry

theorem shaded_area_is_24 (rect : Rectangle) (tri : Triangle) :
  rect.width = 8 ∧ rect.height = 12 ∧ tri.base = 8 ∧ tri.height = rect.height →
  shaded_area rect tri = 24 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_is_24_l3445_344593


namespace NUMINAMATH_CALUDE_difference_of_squares_l3445_344512

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_l3445_344512


namespace NUMINAMATH_CALUDE_people_liking_neither_sport_l3445_344596

/-- Given a class with the following properties:
  * There are 16 people in total
  * 5 people like both baseball and football
  * 2 people only like baseball
  * 3 people only like football
  Prove that 6 people like neither baseball nor football -/
theorem people_liking_neither_sport (total : Nat) (both : Nat) (only_baseball : Nat) (only_football : Nat)
  (h_total : total = 16)
  (h_both : both = 5)
  (h_only_baseball : only_baseball = 2)
  (h_only_football : only_football = 3) :
  total - (both + only_baseball + only_football) = 6 := by
sorry

end NUMINAMATH_CALUDE_people_liking_neither_sport_l3445_344596


namespace NUMINAMATH_CALUDE_cell_phone_production_ambiguity_l3445_344569

/-- Represents the production of cell phones in a factory --/
structure CellPhoneProduction where
  machines_count : ℕ
  phones_per_machine : ℕ
  total_production : ℕ

/-- The production scenario described in the problem --/
def factory_scenario : CellPhoneProduction :=
  { machines_count := 10
  , phones_per_machine := 5
  , total_production := 50 }

/-- The production rate for some machines described in the problem --/
def some_machines_rate : ℕ := 10

/-- Theorem stating the ambiguity in the production calculation --/
theorem cell_phone_production_ambiguity :
  (factory_scenario.machines_count * factory_scenario.phones_per_machine = factory_scenario.total_production) ∧
  (factory_scenario.phones_per_machine ≠ some_machines_rate) :=
by sorry

end NUMINAMATH_CALUDE_cell_phone_production_ambiguity_l3445_344569


namespace NUMINAMATH_CALUDE_arun_weight_average_l3445_344559

def arun_weight_range (w : ℝ) : Prop :=
  64 < w ∧ w < 72 ∧  -- Arun's opinion
  60 < w ∧ w < 70 ∧  -- Brother's opinion
  w ≤ 67 ∧           -- Mother's opinion
  63 ≤ w ∧ w ≤ 71 ∧  -- Sister's opinion
  62 < w ∧ w ≤ 73    -- Father's opinion

theorem arun_weight_average :
  (∃ a b : ℝ, a < b ∧
    (∀ w, a < w ∧ w ≤ b ↔ arun_weight_range w) ∧
    (b - a + 1) / 2 + a = 66) :=
sorry

end NUMINAMATH_CALUDE_arun_weight_average_l3445_344559


namespace NUMINAMATH_CALUDE_h_in_terms_of_c_l3445_344580

theorem h_in_terms_of_c (a b c d e f g h : ℚ) : 
  8 = (6 / 100) * a →
  6 = (8 / 100) * b →
  9 = (5 / 100) * d →
  7 = (3 / 100) * e →
  c = b / a →
  f = d / a →
  g = e / b →
  h = f + g →
  h = (803 / 20) * c := by
sorry

end NUMINAMATH_CALUDE_h_in_terms_of_c_l3445_344580


namespace NUMINAMATH_CALUDE_rectangle_area_l3445_344594

theorem rectangle_area (a b : ℕ) : 
  (2 * (a + b) = 16) →
  (a^2 + b^2 - 2*a*b - 4 = 0) →
  (a * b = 15) := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l3445_344594


namespace NUMINAMATH_CALUDE_area_of_grid_with_cutouts_l3445_344534

/-- The area of a square grid with triangular cutouts -/
theorem area_of_grid_with_cutouts (grid_side : ℕ) (cell_side : ℝ) 
  (dark_grey_area : ℝ) (light_grey_area : ℝ) : 
  grid_side = 6 → 
  cell_side = 1 → 
  dark_grey_area = 3 → 
  light_grey_area = 6 → 
  (grid_side : ℝ) * (grid_side : ℝ) * cell_side * cell_side - dark_grey_area - light_grey_area = 27 := by
sorry

end NUMINAMATH_CALUDE_area_of_grid_with_cutouts_l3445_344534


namespace NUMINAMATH_CALUDE_sprint_medal_theorem_l3445_344501

/-- Represents the number of ways to award medals in a specific sprinting competition scenario. -/
def medalAwardingWays (totalSprinters : ℕ) (americanSprinters : ℕ) (canadianSprinters : ℕ) : ℕ :=
  -- The actual computation is not provided here
  sorry

/-- Theorem stating the number of ways to award medals in the given scenario. -/
theorem sprint_medal_theorem :
  medalAwardingWays 10 4 3 = 552 := by
  sorry

end NUMINAMATH_CALUDE_sprint_medal_theorem_l3445_344501


namespace NUMINAMATH_CALUDE_soap_brand_survey_l3445_344519

theorem soap_brand_survey (total : ℕ) (neither : ℕ) (only_a : ℕ) (both_to_only_b_ratio : ℕ) 
  (h1 : total = 180)
  (h2 : neither = 80)
  (h3 : only_a = 60)
  (h4 : both_to_only_b_ratio = 3) :
  ∃ (both : ℕ), 
    neither + only_a + both + both_to_only_b_ratio * both = total ∧ 
    both = 10 := by
  sorry

end NUMINAMATH_CALUDE_soap_brand_survey_l3445_344519


namespace NUMINAMATH_CALUDE_cookie_cost_cookie_cost_is_65_l3445_344539

/-- The cost of a package of cookies, given the amount Diane has and the additional amount she needs. -/
theorem cookie_cost (diane_has : ℕ) (diane_needs : ℕ) : ℕ :=
  diane_has + diane_needs

/-- Proof that the cost of the cookies is 65 cents. -/
theorem cookie_cost_is_65 : cookie_cost 27 38 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cost_cookie_cost_is_65_l3445_344539


namespace NUMINAMATH_CALUDE_triangle_perimeter_bound_l3445_344541

theorem triangle_perimeter_bound (a b s : ℝ) : 
  a = 7 → b = 23 → a > 0 → b > 0 → s > 0 → 
  a + b > s → a + s > b → b + s > a → 
  ∃ n : ℕ, n = 60 ∧ ∀ m : ℕ, (a + b + s < m ∧ m < n) → False :=
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_bound_l3445_344541


namespace NUMINAMATH_CALUDE_largest_number_l3445_344540

theorem largest_number (a b c d e f : ℝ) 
  (ha : a = 0.986) 
  (hb : b = 0.9859) 
  (hc : c = 0.98609) 
  (hd : d = 0.896) 
  (he : e = 0.8979) 
  (hf : f = 0.987) : 
  f = max a (max b (max c (max d (max e f)))) :=
sorry

end NUMINAMATH_CALUDE_largest_number_l3445_344540


namespace NUMINAMATH_CALUDE_area_regular_octagon_in_circle_l3445_344557

/-- The area of a regular octagon inscribed in a circle -/
theorem area_regular_octagon_in_circle (r : ℝ) (h : r^2 * Real.pi = 256 * Real.pi) :
  8 * ((2 * r * Real.sin (Real.pi / 8))^2 * Real.sqrt 2 / 4) = 
    8 * (2 * 16 * Real.sin (Real.pi / 8))^2 * Real.sqrt 2 / 4 := by
  sorry

#check area_regular_octagon_in_circle

end NUMINAMATH_CALUDE_area_regular_octagon_in_circle_l3445_344557


namespace NUMINAMATH_CALUDE_perfect_square_sum_l3445_344589

theorem perfect_square_sum : ∃ k : ℕ, 2^8 + 2^11 + 2^12 = k^2 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_sum_l3445_344589


namespace NUMINAMATH_CALUDE_sine_function_period_l3445_344531

theorem sine_function_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x ∈ Set.Icc (-π) (5*π), ∃ y, y = a * Real.sin (b * x + c) + d) →
  (∃ n : ℕ, n = 5 ∧ (6*π) / n = (2*π) / b) →
  b = 5/3 := by
sorry

end NUMINAMATH_CALUDE_sine_function_period_l3445_344531


namespace NUMINAMATH_CALUDE_eleven_divides_six_digit_repeating_l3445_344588

/-- A 6-digit positive integer where the first three digits are the same as its last three digits -/
def SixDigitRepeating (z : ℕ) : Prop :=
  ∃ (a b c : ℕ), 
    0 < a ∧ a ≤ 9 ∧ 
    b ≤ 9 ∧ 
    c ≤ 9 ∧ 
    z = 100000 * a + 10000 * b + 1000 * c + 100 * a + 10 * b + c

theorem eleven_divides_six_digit_repeating (z : ℕ) (h : SixDigitRepeating z) : 
  11 ∣ z := by
  sorry

end NUMINAMATH_CALUDE_eleven_divides_six_digit_repeating_l3445_344588


namespace NUMINAMATH_CALUDE_distance_is_sqrt_1501_div_17_l3445_344554

/-- The distance from a point to a line in 3D space -/
def distance_point_to_line (point : ℝ × ℝ × ℝ) (line_point : ℝ × ℝ × ℝ) (line_direction : ℝ × ℝ × ℝ) : ℝ :=
  sorry

/-- The given point -/
def given_point : ℝ × ℝ × ℝ := (2, 3, 4)

/-- A point on the given line -/
def line_point : ℝ × ℝ × ℝ := (5, 6, 8)

/-- The direction vector of the given line -/
def line_direction : ℝ × ℝ × ℝ := (4, 3, -3)

/-- Theorem stating that the distance from the given point to the line is √1501 / 17 -/
theorem distance_is_sqrt_1501_div_17 : 
  distance_point_to_line given_point line_point line_direction = Real.sqrt 1501 / 17 := by
  sorry

end NUMINAMATH_CALUDE_distance_is_sqrt_1501_div_17_l3445_344554


namespace NUMINAMATH_CALUDE_carpool_gas_expense_l3445_344587

/-- Calculates the monthly gas expense per person in a carpool scenario -/
theorem carpool_gas_expense
  (one_way_commute : ℝ)
  (gas_cost_per_gallon : ℝ)
  (car_efficiency : ℝ)
  (days_per_week : ℕ)
  (weeks_per_month : ℕ)
  (num_people : ℕ)
  (h1 : one_way_commute = 21)
  (h2 : gas_cost_per_gallon = 2.5)
  (h3 : car_efficiency = 30)
  (h4 : days_per_week = 5)
  (h5 : weeks_per_month = 4)
  (h6 : num_people = 5) :
  (2 * one_way_commute * days_per_week * weeks_per_month / car_efficiency * gas_cost_per_gallon) / num_people = 14 := by
  sorry


end NUMINAMATH_CALUDE_carpool_gas_expense_l3445_344587


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l3445_344500

theorem arithmetic_calculation : 2 + 5 * 4 - 6 + 3 = 19 := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l3445_344500


namespace NUMINAMATH_CALUDE_perfect_square_quadratic_l3445_344528

theorem perfect_square_quadratic (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 - a*x + 16 = (x - b)^2) → (a = 8 ∨ a = -8) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_quadratic_l3445_344528


namespace NUMINAMATH_CALUDE_coterminal_angle_correct_l3445_344556

/-- The angle in degrees that is coterminal with 1000° and lies between 0° and 360° -/
def coterminal_angle : ℝ := 280

/-- Proof that the coterminal angle is correct -/
theorem coterminal_angle_correct :
  0 ≤ coterminal_angle ∧ 
  coterminal_angle < 360 ∧
  ∃ (k : ℤ), coterminal_angle = 1000 - 360 * k :=
by sorry

end NUMINAMATH_CALUDE_coterminal_angle_correct_l3445_344556


namespace NUMINAMATH_CALUDE_rectangular_plot_width_l3445_344518

theorem rectangular_plot_width (length width area : ℝ) : 
  length = 3 * width →
  area = length * width →
  area = 432 →
  width = 12 := by
sorry

end NUMINAMATH_CALUDE_rectangular_plot_width_l3445_344518


namespace NUMINAMATH_CALUDE_monica_students_l3445_344561

/-- Represents the number of students in each class and the overlaps between classes -/
structure ClassData where
  class1 : ℕ
  class2 : ℕ
  class3 : ℕ
  class4 : ℕ
  class5 : ℕ
  class6 : ℕ
  overlap12 : ℕ
  overlap45 : ℕ
  overlap236 : ℕ
  overlap56 : ℕ

/-- Calculates the number of individual students Monica sees each day -/
def individualStudents (data : ClassData) : ℕ :=
  data.class1 + data.class2 + data.class3 + data.class4 + data.class5 + data.class6 -
  (data.overlap12 + data.overlap45 + data.overlap236 + data.overlap56)

/-- Theorem stating that Monica sees 114 individual students each day -/
theorem monica_students :
  ∀ (data : ClassData),
    data.class1 = 20 ∧
    data.class2 = 25 ∧
    data.class3 = 25 ∧
    data.class4 = 10 ∧
    data.class5 = 28 ∧
    data.class6 = 28 ∧
    data.overlap12 = 5 ∧
    data.overlap45 = 3 ∧
    data.overlap236 = 6 ∧
    data.overlap56 = 8 →
    individualStudents data = 114 :=
by
  sorry


end NUMINAMATH_CALUDE_monica_students_l3445_344561


namespace NUMINAMATH_CALUDE_discriminant_nonnegativity_l3445_344510

theorem discriminant_nonnegativity (x : ℤ) :
  x^2 * (81 - 56 * x^2) ≥ 0 ↔ x = 0 ∨ x = 1 ∨ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_discriminant_nonnegativity_l3445_344510


namespace NUMINAMATH_CALUDE_subtraction_result_l3445_344550

theorem subtraction_result : -3.219 - 7.305 = -10.524 := by sorry

end NUMINAMATH_CALUDE_subtraction_result_l3445_344550


namespace NUMINAMATH_CALUDE_solve_for_y_l3445_344568

theorem solve_for_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 18) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l3445_344568


namespace NUMINAMATH_CALUDE_probability_multiple_3_or_4_in_30_l3445_344527

def is_multiple_of_3_or_4 (n : ℕ) : Bool :=
  n % 3 = 0 || n % 4 = 0

def count_multiples (n : ℕ) : ℕ :=
  (List.range n).filter is_multiple_of_3_or_4 |>.length

theorem probability_multiple_3_or_4_in_30 :
  count_multiples 30 / 30 = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_probability_multiple_3_or_4_in_30_l3445_344527


namespace NUMINAMATH_CALUDE_rational_function_simplification_and_evaluation_l3445_344507

theorem rational_function_simplification_and_evaluation :
  ∀ x : ℝ, x ≠ 5 →
  (x^2 - 3*x - 10) / (x - 5) = x + 2 ∧
  (4^2 - 3*4 - 10) / (4 - 5) = 6 := by
sorry

end NUMINAMATH_CALUDE_rational_function_simplification_and_evaluation_l3445_344507


namespace NUMINAMATH_CALUDE_checkerboard_area_equality_l3445_344505

-- Define a convex quadrilateral
structure ConvexQuadrilateral where
  vertices : Fin 4 → ℝ × ℝ
  is_convex : sorry -- Condition for convexity

-- Define the division points on the sides
def division_points (q : ConvexQuadrilateral) : Fin 4 → Fin 8 → ℝ × ℝ :=
  sorry -- Function that returns the division points on each side

-- Define the cells formed by connecting corresponding division points
def cells (q : ConvexQuadrilateral) : List (List (ℝ × ℝ)) :=
  sorry -- List of cells, each cell represented by its vertices

-- Define the area of a cell
def cell_area (cell : List (ℝ × ℝ)) : ℝ :=
  sorry -- Function to calculate the area of a cell

-- Define the sum of areas of alternating cells (checkerboard pattern)
def alternating_sum (cells : List (List (ℝ × ℝ))) : ℝ :=
  sorry -- Sum of areas of alternating cells

-- The theorem to be proved
theorem checkerboard_area_equality (q : ConvexQuadrilateral) :
  let c := cells q
  alternating_sum c = alternating_sum (List.drop 1 c) :=
sorry

end NUMINAMATH_CALUDE_checkerboard_area_equality_l3445_344505


namespace NUMINAMATH_CALUDE_divisibility_of_expression_l3445_344506

theorem divisibility_of_expression (p : ℕ) (h_prime : Nat.Prime p) (h_gt_two : p > 2) :
  ∃ k : ℤ, (⌊(2 + Real.sqrt 5)^p⌋ : ℤ) - 2^(p + 1) = k * p :=
sorry

end NUMINAMATH_CALUDE_divisibility_of_expression_l3445_344506


namespace NUMINAMATH_CALUDE_siblings_age_ratio_l3445_344585

theorem siblings_age_ratio : 
  ∀ (aaron_age henry_age sister_age : ℕ),
  aaron_age = 15 →
  sister_age = 3 * aaron_age →
  aaron_age + henry_age + sister_age = 240 →
  henry_age / sister_age = 4 := by
sorry

end NUMINAMATH_CALUDE_siblings_age_ratio_l3445_344585


namespace NUMINAMATH_CALUDE_simplify_expression_calculate_expression_l3445_344543

-- Part 1
theorem simplify_expression (a b : ℝ) (h : a > 0 ∧ b > 0) :
  (2 * a^(3/2) * b^(1/2)) * (-6 * a^(1/2) * b^(1/3)) / (-3 * a^(1/6) * b^(5/6)) = 4 * a^(11/6) := by
  sorry

-- Part 2
theorem calculate_expression :
  (2^(1/3) * 3^(1/2))^6 + (2^(1/2) * 2^(1/4))^(4/3) - 2^(1/4) * 8^(1/4) - (-2005)^0 = 100 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_calculate_expression_l3445_344543


namespace NUMINAMATH_CALUDE_sum_of_common_divisors_l3445_344562

def number_list : List Int := [48, 96, -16, 144, 192]

def is_common_divisor (d : Nat) : Bool :=
  number_list.all (fun n => n % d = 0)

def common_divisors : List Nat :=
  (List.range 193).filter is_common_divisor

theorem sum_of_common_divisors : (common_divisors.sum) = 15 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_common_divisors_l3445_344562


namespace NUMINAMATH_CALUDE_finite_ring_power_equality_l3445_344514

theorem finite_ring_power_equality (A : Type) [Ring A] [Fintype A] :
  ∃ (m p : ℕ), m > p ∧ p ≥ 1 ∧ ∀ (a : A), a^m = a^p := by
  sorry

end NUMINAMATH_CALUDE_finite_ring_power_equality_l3445_344514


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3445_344558

theorem sum_of_squares_of_roots (p q r : ℝ) : 
  (3 * p^3 - 4 * p^2 + 7 * p - 9 = 0) →
  (3 * q^3 - 4 * q^2 + 7 * q - 9 = 0) →
  (3 * r^3 - 4 * r^2 + 7 * r - 9 = 0) →
  p^2 + q^2 + r^2 = -26/9 := by
sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l3445_344558


namespace NUMINAMATH_CALUDE_sheila_weekly_earnings_l3445_344584

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hourlyWage : ℝ
  hoursMonWedFri : ℝ
  hoursTueThu : ℝ
  daysWithLongHours : ℕ
  daysWithShortHours : ℕ

/-- Calculates the weekly earnings based on the work schedule --/
def weeklyEarnings (schedule : WorkSchedule) : ℝ :=
  (schedule.hourlyWage * schedule.hoursMonWedFri * schedule.daysWithLongHours) +
  (schedule.hourlyWage * schedule.hoursTueThu * schedule.daysWithShortHours)

/-- Theorem stating that Sheila's weekly earnings are $288 --/
theorem sheila_weekly_earnings :
  let schedule : WorkSchedule := {
    hourlyWage := 8,
    hoursMonWedFri := 8,
    hoursTueThu := 6,
    daysWithLongHours := 3,
    daysWithShortHours := 2
  }
  weeklyEarnings schedule = 288 := by
  sorry

end NUMINAMATH_CALUDE_sheila_weekly_earnings_l3445_344584


namespace NUMINAMATH_CALUDE_circle_path_in_triangle_l3445_344590

theorem circle_path_in_triangle (a b c : ℝ) (h_right : a^2 + b^2 = c^2) 
  (h_sides : a = 9 ∧ b = 12 ∧ c = 15) (r : ℝ) (h_radius : r = 2) :
  let p := a + b + c
  let s := (c - 2*r) / c
  (s * p) = 26.4 := by sorry

end NUMINAMATH_CALUDE_circle_path_in_triangle_l3445_344590


namespace NUMINAMATH_CALUDE_f_strictly_increasing_after_one_l3445_344520

/-- The quadratic function f(x) = (x-1)^2 + 5 -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 5

/-- Theorem: f(x) is strictly increasing for all x > 1 -/
theorem f_strictly_increasing_after_one :
  ∀ x₁ x₂ : ℝ, 1 < x₁ → x₁ < x₂ → f x₁ < f x₂ :=
by
  sorry

end NUMINAMATH_CALUDE_f_strictly_increasing_after_one_l3445_344520


namespace NUMINAMATH_CALUDE_stacy_heather_walking_problem_l3445_344564

/-- The problem of Stacy and Heather walking towards each other -/
theorem stacy_heather_walking_problem 
  (total_distance : ℝ) 
  (heather_speed : ℝ) 
  (stacy_speed : ℝ) 
  (heather_distance : ℝ) :
  total_distance = 15 →
  heather_speed = 5 →
  stacy_speed = heather_speed + 1 →
  heather_distance = 5.7272727272727275 →
  ∃ (time_difference : ℝ), 
    time_difference = 24 / 60 ∧ 
    time_difference * stacy_speed = total_distance - (heather_distance + stacy_speed * (heather_distance / heather_speed)) :=
by sorry

end NUMINAMATH_CALUDE_stacy_heather_walking_problem_l3445_344564


namespace NUMINAMATH_CALUDE_sum_of_interior_angles_octagon_l3445_344504

theorem sum_of_interior_angles_octagon (a : ℝ) : a = 1080 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_interior_angles_octagon_l3445_344504


namespace NUMINAMATH_CALUDE_ap_sequence_a_equals_one_l3445_344538

/-- Given a sequence 1, 6 + 2a, 10 + 5a, ..., if it forms an arithmetic progression, then a = 1 -/
theorem ap_sequence_a_equals_one (a : ℝ) :
  (∀ n : ℕ, (fun i => if i = 0 then 1 else if i = 1 then 6 + 2*a else 10 + 5*a) n.succ - 
             (fun i => if i = 0 then 1 else if i = 1 then 6 + 2*a else 10 + 5*a) n = 
             (fun i => if i = 0 then 1 else if i = 1 then 6 + 2*a else 10 + 5*a) 1 - 
             (fun i => if i = 0 then 1 else if i = 1 then 6 + 2*a else 10 + 5*a) 0) →
  a = 1 := by
sorry


end NUMINAMATH_CALUDE_ap_sequence_a_equals_one_l3445_344538


namespace NUMINAMATH_CALUDE_inequality_proof_l3445_344509

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (z^2 - x^2) / (x + y) + (x^2 - y^2) / (y + z) + (y^2 - z^2) / (z + x) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3445_344509


namespace NUMINAMATH_CALUDE_angle_sum_in_circle_l3445_344524

theorem angle_sum_in_circle (x : ℝ) : 
  (6*x + 7*x + 3*x + 2*x + 4*x = 360) → x = 180/11 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_in_circle_l3445_344524


namespace NUMINAMATH_CALUDE_linear_equation_solution_l3445_344552

theorem linear_equation_solution (m : ℤ) :
  (∃ x : ℚ, x^(|m|) - m*x + 1 = 0 ∧ ∃ a b : ℚ, a ≠ 0 ∧ a*x + b = 0) →
  (∃ x : ℚ, x^(|m|) - m*x + 1 = 0 ∧ x = -1/2) :=
by sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l3445_344552


namespace NUMINAMATH_CALUDE_quadratic_root_two_l3445_344517

theorem quadratic_root_two (c : ℝ) : (2 : ℝ)^2 = c → c = 4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_two_l3445_344517


namespace NUMINAMATH_CALUDE_digits_of_3_pow_10_times_5_pow_6_l3445_344513

theorem digits_of_3_pow_10_times_5_pow_6 :
  (Nat.digits 10 (3^10 * 5^6)).length = 9 := by
  sorry

end NUMINAMATH_CALUDE_digits_of_3_pow_10_times_5_pow_6_l3445_344513


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l3445_344545

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x : ℝ | -1 ≤ x ∧ x < 1}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l3445_344545


namespace NUMINAMATH_CALUDE_odd_iff_a_eq_zero_l3445_344583

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  3 * Real.log (x + Real.sqrt (x^2 + 1)) + a * (7^x + 7^(-x))

def isOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem odd_iff_a_eq_zero (a : ℝ) :
  isOdd (f a) ↔ a = 0 :=
sorry

end NUMINAMATH_CALUDE_odd_iff_a_eq_zero_l3445_344583


namespace NUMINAMATH_CALUDE_division_cannot_be_operation_l3445_344551

def P : Set ℤ := {x | ∃ k : ℤ, x = 2 * k}

theorem division_cannot_be_operation :
  ¬(∀ a b : ℤ, a ∈ P → b ∈ P → (a / b) ∈ P) :=
by
  sorry

end NUMINAMATH_CALUDE_division_cannot_be_operation_l3445_344551


namespace NUMINAMATH_CALUDE_bob_water_usage_percentage_l3445_344555

-- Define the farmers
inductive Farmer
| Bob
| Brenda
| Bernie

-- Define the crop types
inductive Crop
| Corn
| Cotton
| Beans

-- Define the acreage for each farmer and crop
def acreage : Farmer → Crop → ℕ
  | Farmer.Bob, Crop.Corn => 3
  | Farmer.Bob, Crop.Cotton => 9
  | Farmer.Bob, Crop.Beans => 12
  | Farmer.Brenda, Crop.Corn => 6
  | Farmer.Brenda, Crop.Cotton => 7
  | Farmer.Brenda, Crop.Beans => 14
  | Farmer.Bernie, Crop.Corn => 2
  | Farmer.Bernie, Crop.Cotton => 12
  | Farmer.Bernie, Crop.Beans => 0

-- Define water requirements for each crop (in gallons per acre)
def waterPerAcre : Crop → ℕ
  | Crop.Corn => 20
  | Crop.Cotton => 80
  | Crop.Beans => 40  -- Twice as much as corn

-- Calculate total water used by a farmer
def farmerWaterUsage (f : Farmer) : ℕ :=
  (acreage f Crop.Corn * waterPerAcre Crop.Corn) +
  (acreage f Crop.Cotton * waterPerAcre Crop.Cotton) +
  (acreage f Crop.Beans * waterPerAcre Crop.Beans)

-- Calculate total water used by all farmers
def totalWaterUsage : ℕ :=
  farmerWaterUsage Farmer.Bob +
  farmerWaterUsage Farmer.Brenda +
  farmerWaterUsage Farmer.Bernie

-- Theorem: Bob's water usage is 36% of total water usage
theorem bob_water_usage_percentage :
  (farmerWaterUsage Farmer.Bob : ℚ) / totalWaterUsage * 100 = 36 := by
  sorry

end NUMINAMATH_CALUDE_bob_water_usage_percentage_l3445_344555


namespace NUMINAMATH_CALUDE_three_identical_digits_divisible_by_37_l3445_344579

theorem three_identical_digits_divisible_by_37 (A : ℕ) (h : A < 10) :
  ∃ k : ℕ, 111 * A = 37 * k := by
  sorry

end NUMINAMATH_CALUDE_three_identical_digits_divisible_by_37_l3445_344579


namespace NUMINAMATH_CALUDE_cannot_determine_package_size_l3445_344582

/-- Represents the number of candies in a package -/
def CandiesPerPackage := ℕ

/-- Represents the state of candies on the desk -/
structure CandyPile :=
  (initial : ℕ)
  (added : ℕ)
  (final : ℕ)

/-- Given a candy pile state, it's not possible to determine the number of candies per package -/
theorem cannot_determine_package_size (pile : CandyPile) : 
  pile.initial = 6 → pile.added = 4 → pile.final = 10 → 
  ¬∃ (package_size : CandiesPerPackage), ∀ (other_size : CandiesPerPackage), package_size = other_size :=
by sorry

end NUMINAMATH_CALUDE_cannot_determine_package_size_l3445_344582


namespace NUMINAMATH_CALUDE_vector_2016_coordinates_l3445_344521

def matrix_transformation (x_n y_n : ℝ) : ℝ × ℝ :=
  (x_n, x_n + y_n)

def vector_sequence (n : ℕ) : ℝ × ℝ :=
  match n with
  | 0 => (2, 0)
  | n + 1 => matrix_transformation (vector_sequence n).1 (vector_sequence n).2

theorem vector_2016_coordinates :
  vector_sequence 2015 = (2, 4030) := by
  sorry

end NUMINAMATH_CALUDE_vector_2016_coordinates_l3445_344521


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l3445_344536

noncomputable def f (x : ℝ) : ℝ := Real.exp (abs x) * Real.sin x

theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Ioo (-π/4) (3*π/4)) :=
sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l3445_344536


namespace NUMINAMATH_CALUDE_cos_transformation_l3445_344566

theorem cos_transformation (x : ℝ) : 
  Real.sqrt 2 * Real.cos (3 * x) = Real.sqrt 2 * Real.cos ((3 / 2) * (2 * x)) := by
  sorry

end NUMINAMATH_CALUDE_cos_transformation_l3445_344566


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3445_344563

/-- Definition of an ellipse with foci F₁ and F₂ -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ

/-- The maximum distance from a point on the ellipse to F₁ -/
def max_distance (E : Ellipse) : ℝ := 7

/-- The minimum distance from a point on the ellipse to F₁ -/
def min_distance (E : Ellipse) : ℝ := 1

/-- The eccentricity of an ellipse -/
def eccentricity (E : Ellipse) : ℝ := sorry

/-- Theorem: The square root of the eccentricity of the ellipse E is √3/2 -/
theorem ellipse_eccentricity (E : Ellipse) :
  Real.sqrt (eccentricity E) = Real.sqrt 3 / 2 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3445_344563
